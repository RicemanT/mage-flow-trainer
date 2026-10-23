"""Configuration gates and optional CUDA ConvRot gradient regression."""
from types import SimpleNamespace
import os

import pytest
import torch
import toml

from trainer.training.config import load_config
from trainer.training.quant import QuantConfig, quantize_module, quantized_layer_report


@pytest.mark.parametrize("section,key,value", [
    ("quant", "mode", "training"), ("train", "model_family", "mage_flow"),
    ("adapter", "lycoris_bypass", False), ("adapter", "lycoris_algo", "dora"),
    ("adapter", "init", "pissa"), ("train", "dtype", "float32"),
])
def test_reject_unsupported_convrot(tmp_path, section, key, value):
    config = dict(dataset=dict(path="/images"), train=dict(model_family="moss_image"),
                  adapter=dict(kind="lycoris_lora", dtype="bfloat16"), quant=dict(mode="frozen", backend="convrot"))
    config[section][key] = value
    path = tmp_path / "config.toml"
    path.write_text(toml.dumps(config))
    with pytest.raises(ValueError):
        load_config(path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_convrot_keeps_base_frozen_and_propagates_input_gradient():
    pytest.importorskip("comfy_kitchen")
    model = torch.nn.Module()
    model.model_family = "moss_image"
    model.params = SimpleNamespace(num_layers=4)
    model.layers = torch.nn.ModuleList([torch.nn.Linear(256, 256, bias=False) for _ in range(4)])
    model.to(device="cuda", dtype=torch.bfloat16)
    quantize_module(model, QuantConfig(mode="frozen", backend="convrot"),
                    torch.device("cuda"), torch.bfloat16)
    assert quantized_layer_report(model)[:2] == (1, 4)
    layer = model.layers[1]
    weight = layer.weight.dequantize()
    x = torch.randn(16, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    grad_output = torch.randn_like(x)
    layer(x).backward(grad_output)
    expected = grad_output @ weight
    assert x.grad.isfinite().all() and x.grad.norm() > 0
    torch.testing.assert_close(x.grad, expected, rtol=.03, atol=.03)
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())


@pytest.mark.skipif(os.environ.get("MAGEFLOW_GPU_TESTS") != "1" or os.environ.get("WORLD_SIZE") != "2",
                    reason="requires two-rank GPU opt-in")
def test_convrot_distributed_storage_and_adapter_updates():
    import torch.distributed as dist
    from trainer.training.distributed import configure_quantized_ddp, model_storage, sync_quantized_model

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model_family = "moss_image"
            self.params = SimpleNamespace(num_layers=4)
            self.layers = torch.nn.ModuleList([torch.nn.Linear(256, 256) for _ in range(4)])

        def forward(self, x):
            # A trainable path before the quantized layer must receive gradients.
            return self.layers[1](self.adapter(x)).float().square().mean()

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl")
    try:
        torch.manual_seed(100 + rank)
        model = Model().to(device=device, dtype=torch.bfloat16)
        quantize_module(model, QuantConfig(mode="frozen", backend="convrot"), device, torch.bfloat16)
        model.adapter = torch.nn.Linear(256, 256, bias=False).to(device=device, dtype=torch.bfloat16)
        sync_quantized_model(model)

        def check_equal():
            for name, tensor in model_storage(model):
                reference = tensor.detach().clone().contiguous()
                dist.broadcast(reference, src=0)
                assert torch.equal(reference, tensor), name

        check_equal()
        frozen = {name: value.detach().clone() for name, value in model_storage(model)
                  if name.startswith("layers.")}
        before = model.adapter.weight.detach().clone()
        ddp = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[rank], **configure_quantized_ddp().to_kwargs())
        optimizer = torch.optim.SGD(model.adapter.parameters(), lr=.1)
        ddp(torch.randn(16, 256, device=device, dtype=torch.bfloat16)).backward()
        assert model.adapter.weight.grad.isfinite().all() and model.adapter.weight.grad.norm() > 0
        optimizer.step()
        check_equal()
        assert not torch.equal(before, model.adapter.weight)
        for name, value in model_storage(model):
            if name in frozen:
                assert torch.equal(frozen[name], value), name
    finally:
        dist.destroy_process_group()
