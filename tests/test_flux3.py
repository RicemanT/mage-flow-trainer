import pytest
import torch

from trainer.modeling.flux3 import Flux3, Flux3Params
from trainer.training.params import (
    ComponentLRs,
    adapter_target_names,
    build_param_groups,
)


def tiny():
    torch.set_num_threads(2)
    return Flux3(
        Flux3Params(
            hidden=32,
            heads=2,
            depth=1,
            depth_single=2,
            axes_dim=(4, 4, 4, 4),
            channels=8,
            text_dim=16,
        )
    )


def test_checkpoint_backward_and_native_resolution_isolation():
    torch.manual_seed(3)
    m = tiny().train()
    xs = [torch.randn(1, 8, 1, 3, 4), torch.randn(1, 8, 1, 2, 3)]
    text = torch.randn(2, 5, 16)
    mask = torch.tensor([[1] * 5, [1, 1, 1, 0, 0]], dtype=torch.bool)
    t = torch.tensor([0.3, 0.7])
    m.configure_execution(False)
    ys = m(xs, t, (text, mask))[0]
    loss = sum(y.square().mean() for y in ys)
    loss.backward()
    grads = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    m.zero_grad()
    m.configure_execution(True)
    zs = m(xs, t, (text, mask))[0]
    sum(y.square().mean() for y in zs).backward()
    for y, z in zip(ys, zs):
        torch.testing.assert_close(y, z)
    for n, p in m.named_parameters():
        if n in grads:
            torch.testing.assert_close(p.grad, grads[n])
        if ".image_cond." in n:
            assert p.grad is None
    text[1] *= 100
    torch.testing.assert_close(m(xs, t, (text, mask))[0][0], ys[0])


def test_targets_and_unused_editing_branch():
    m = tiny()
    targets = adapter_target_names(m, ["image_attn", "text_attn", "mlp"])
    assert len(targets) == 24
    assert all("modulation" not in n and "image_cond" not in n for n in targets)
    report = build_param_groups(m, ComponentLRs(adaln=0), 1e-5)
    assert report.groups
    assert all(
        not p.requires_grad
        for n, p in m.named_parameters()
        if ".image_cond." in n or "modulations." in n
    )


def test_flux3_resolver_and_guards(tmp_path):
    import json

    from trainer.data.dataset import DatasetConfig
    from trainer.modeling.loader import resolve_model_family
    from trainer.training.config import Config, validate_model_options

    (tmp_path / "config.json").write_text(json.dumps({"model": "flux-3-image-klein"}))
    assert resolve_model_family(tmp_path) == "flux3"
    c = Config(dataset=DatasetConfig(path=str(tmp_path)))
    c.train.model_path = str(tmp_path)
    c.train.attention_backend = "sdpa"
    c.train.compile_text_encoder = False
    validate_model_options(c)
    assert c.train.flux2_vae
    c.train.attention_backend = "torch_varlen"
    validate_model_options(c)
    c.flow.dual_timestep = True
    with pytest.raises(ValueError, match="Experimental Flux3"):
        validate_model_options(c)


def test_text_layers_padding_and_batch_independence():
    from types import SimpleNamespace

    from trainer.modeling.flux3_loader import TEXT_LAYERS, encode_prompts

    class Tokenizer:
        def __call__(self, prompt, **kwargs):
            n = 81 if "long" in prompt else 13
            return SimpleNamespace(input_ids=torch.ones(1, n, dtype=torch.long))

    class Encoder:
        def __call__(self, input_ids, attention_mask, position_ids, **kwargs):
            assert (position_ids[~attention_mask] == 1).all()
            assert input_ids.shape[1] % 80 == 0
            return SimpleNamespace(
                hidden_states=tuple(
                    torch.full((*input_ids.shape, 2), float(i)) for i in range(37)
                )
            )

    c = SimpleNamespace(tokenizer=Tokenizer(), text_encoder=Encoder())
    h, m = encode_prompts(c, ["short", "long"], "cpu", 512)
    assert m.sum(1).tolist() == [80, 160]
    assert h.shape == (2, 160, 16)
    torch.testing.assert_close(
        h[0, 0], torch.tensor(TEXT_LAYERS).float().repeat_interleave(2)
    )
    one, _mask = encode_prompts(c, ["short"], "cpu", 512)
    torch.testing.assert_close(h[:1, :80], one)


@pytest.mark.parametrize(
    "length,cap,expected",
    [(13, 512, 80), (81, 128, 128), (512, 512, 512), (13, 32, 32), (80, 80, 80)],
)
def test_text_window_never_exceeds_limit(length, cap, expected):
    from types import SimpleNamespace

    from trainer.modeling.flux3_loader import encode_prompts

    class Tokenizer:
        def __call__(self, prompt, **kwargs):
            return SimpleNamespace(
                input_ids=torch.ones(
                    1, min(length, kwargs["max_length"]), dtype=torch.long
                )
            )

    class Encoder:
        def __call__(self, input_ids, attention_mask, position_ids, **kwargs):
            assert input_ids.shape[1] == expected
            assert (position_ids[~attention_mask] == 1).all()
            return SimpleNamespace(
                hidden_states=tuple(torch.zeros(*input_ids.shape, 2) for _ in range(37))
            )

    h, m = encode_prompts(
        SimpleNamespace(tokenizer=Tokenizer(), text_encoder=Encoder()),
        ["x"],
        "cpu",
        cap,
    )
    assert h.shape[1] == expected and m.sum() == expected


def test_packed_varlen_forward_backward_and_isolation(monkeypatch):
    from torch.nn.attention import varlen

    def reference(q, k, v, cq, ck, mq, mk):
        parts = []
        for start, end in zip(cq.tolist()[:-1], cq.tolist()[1:]):
            parts.append(
                torch.nn.functional.scaled_dot_product_attention(
                    q[start:end].transpose(0, 1)[None],
                    k[start:end].transpose(0, 1)[None],
                    v[start:end].transpose(0, 1)[None],
                )[0].transpose(0, 1)
            )
        return torch.cat(parts)

    monkeypatch.setattr(varlen, "varlen_attn", reference)
    torch.manual_seed(17)
    m = tiny().train()
    xs = [torch.randn(1, 8, 1, 3, 4), torch.randn(1, 8, 1, 2, 3)]
    text = torch.randn(2, 5, 16)
    mask = torch.tensor([[1] * 5, [1, 1, 1, 0, 0]], dtype=torch.bool)
    t = torch.tensor([0.2, 0.8])
    m.configure_execution(False)
    expected = m(xs, t, (text, mask))[0]
    sum(y.square().mean() for y in expected).backward()
    grads = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    m.zero_grad(set_to_none=True)
    m.configure_execution(True, attention_backend="torch_varlen")
    actual = m(xs, t, (text, mask))[0]
    sum(y.square().mean() for y in actual).backward()
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    for n, p in m.named_parameters():
        if n in grads:
            torch.testing.assert_close(p.grad, grads[n], atol=1e-6, rtol=1e-4)
    text[1] *= 10
    torch.testing.assert_close(m(xs, t, (text, mask))[0][0], actual[0])
