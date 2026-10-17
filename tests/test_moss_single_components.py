import json
import re

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from trainer.modeling.flux2_vae_loader import load_flux2_vae
from trainer.modeling.moss_text_loader import (
    load_single_text_encoder,
    read_weight,
    regular_hadamard,
)


def test_convrot_decode_and_invalid_format(tmp_path):
    torch.set_num_threads(2)
    q = torch.randint(-127, 128, (520, 16), dtype=torch.int8)
    scale = torch.rand(520, 1) * 0.01
    path = tmp_path / "int8.safetensors"
    conf = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 16}

    def write():
        save_file(
            {
                "layer.weight": q,
                "layer.weight_scale": scale,
                "layer.comfy_quant": torch.tensor(
                    list(json.dumps(conf).encode()), dtype=torch.uint8
                ),
            },
            path,
        )

    write()
    with safe_open(path, framework="pt") as reader:
        actual = read_weight(reader, "layer.weight", torch.float32)
    h = regular_hadamard(16)
    torch.testing.assert_close(actual @ h.T, q.float() * scale)
    conf["format"] = "unsupported"
    write()
    with (
        safe_open(path, framework="pt") as reader,
        pytest.raises(ValueError, match="unsupported quantization"),
    ):
        read_weight(reader, "layer.weight", torch.float32)


def test_single_text_matches_native_text_model(tmp_path):
    from transformers import Qwen3_5TextConfig, Qwen3_5TextModel

    torch.set_num_threads(2)
    cfg = Qwen3_5TextConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=64,
        layer_types=["full_attention"] * 2,
    )
    cfg._attn_implementation = "sdpa"
    original = Qwen3_5TextModel(cfg).eval()
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"text_config": cfg.to_dict()}))
    state = {"model.language_model." + k: v for k, v in original.state_dict().items()}
    state["lm_head.weight"] = torch.zeros(64, 32)
    path = tmp_path / "text.safetensors"
    save_file(state, path)
    actual = load_single_text_encoder(path, config, torch.float32)
    ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        torch.testing.assert_close(
            actual(ids, use_cache=False).last_hidden_state,
            original(ids, use_cache=False).last_hidden_state,
        )
    del state["model.language_model.norm.weight"]
    save_file(state, path)
    with pytest.raises(ValueError, match="missing"):
        load_single_text_encoder(path, config, torch.float32)


def original_vae_key(k):
    k = (
        k.replace("quant_conv.", "encoder.quant_conv.")
        if k.startswith("quant_conv.")
        else k
    )
    k = (
        k.replace("post_quant_conv.", "decoder.post_quant_conv.")
        if k.startswith("post_quant_conv.")
        else k
    )
    k = k.replace("conv_norm_out.", "norm_out.")
    k = re.sub(r"encoder.down_blocks.(\d+).resnets.", r"encoder.down.\1.block.", k)
    k = re.sub(
        r"encoder.down_blocks.(\d+).downsamplers.0.", r"encoder.down.\1.downsample.", k
    )
    k = re.sub(
        r"decoder.up_blocks.(\d+).resnets.",
        lambda m: f"decoder.up.{3 - int(m[1])}.block.",
        k,
    )
    k = re.sub(
        r"decoder.up_blocks.(\d+).upsamplers.0.",
        lambda m: f"decoder.up.{3 - int(m[1])}.upsample.",
        k,
    )
    k = re.sub(r"mid_block.resnets.(\d+).", lambda m: f"mid.block_{int(m[1]) + 1}.", k)
    k = k.replace("conv_shortcut.", "nin_shortcut.")
    for a, b in [
        ("group_norm", "norm"),
        ("to_q", "q"),
        ("to_k", "k"),
        ("to_v", "v"),
        ("to_out.0", "proj_out"),
    ]:
        k = k.replace("mid_block.attentions.0." + a + ".", "mid.attn_1." + b + ".")
    return k


@pytest.mark.parametrize("decoder", [(32, 32, 32, 32), (32, 32, 64, 64)])
def test_vae_original_and_diffusers_weights_match(tmp_path, decoder):
    from diffusers import AutoencoderKLFlux2

    from trainer.data.cache import LatentCacher

    torch.set_num_threads(2)
    original = AutoencoderKLFlux2(
        block_out_channels=(32, 32, 64, 64), decoder_block_out_channels=decoder
    ).eval()
    x = torch.randn(1, 3, 1, 32, 32)
    expected = LatentCacher(original, device="cpu", flux2_vae=True).encode_tensor(x)
    for ldm in [False, True]:
        state = {}
        for k, v in original.state_dict().items():
            name = original_vae_key(k) if ldm else k
            if ldm and "mid.attn_1." in name and v.ndim == 2:
                v = v[:, :, None, None]
            state[name] = v
        path = tmp_path / ("ldm.safetensors" if ldm else "diffusers.safetensors")
        save_file(state, path)
        actual = load_flux2_vae(path).eval()
        assert list(actual.config.decoder_block_out_channels) == list(decoder)
        for k, v in original.state_dict().items():
            torch.testing.assert_close(v, actual.state_dict()[k], rtol=0, atol=0)
        torch.testing.assert_close(
            LatentCacher(actual, device="cpu", flux2_vae=True).encode_tensor(x),
            expected,
        )


def test_original_vae_cache_cli_stays_in_trainer(tmp_path, monkeypatch):
    import os
    import sys

    from diffusers import AutoencoderKLFlux2
    from PIL import Image

    from trainer.tools.cache_latents import main

    torch.set_num_threads(2)
    vae = AutoencoderKLFlux2(block_out_channels=(32, 32, 32, 32))
    state = {}
    for key, value in vae.state_dict().items():
        name = original_vae_key(key)
        if "mid.attn_1." in name and value.ndim == 2:
            value = value[:, :, None, None]
        state[name] = value
    weights = tmp_path / "vae.safetensors"
    save_file(state, weights)
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (64, 64)).save(images / "test.png")
    (images / "test.txt").write_text("test")

    def forbidden(*args):
        raise AssertionError("Cache must not restart under another interpreter")

    monkeypatch.setattr(os, "execv", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cache_latents",
            "cache",
            str(images),
            "--model-path",
            str(tmp_path),
            "--model-family",
            "moss_image",
            "--vae-path",
            str(weights),
            "--resolution",
            "64",
            "--min-bucket-reso",
            "64",
            "--max-bucket-reso",
            "64",
            "--device",
            "cpu",
        ],
    )
    assert main() == 0
    caches = list(images.glob("*_trainer.safetensors"))
    assert len(caches) == 1
    with safe_open(caches[0], framework="pt") as f:
        keys = f.keys()
        assert all(f.get_tensor(k).isfinite().all() for k in keys)
    assert not any(
        k == "comfy" or k.startswith(("comfy.", "comfy_kitchen")) for k in sys.modules
    )


def test_moss_bundled_sources_and_explicit_precedence(tmp_path):
    from trainer.modeling.moss_loader import ASSETS, component_config, text_sources

    missing = tmp_path / "no-model-package"
    encoder = tmp_path / "encoder.safetensors"
    assert (
        component_config(missing, "text_encoder") == ASSETS / "text_encoder_config.json"
    )
    assert (
        component_config(missing, "transformer") == ASSETS / "transformer_config.json"
    )
    assert text_sources(missing, encoder) == (encoder, ASSETS)
    explicit = tmp_path / "explicit-tokenizer"
    assert text_sources(missing, encoder, explicit) == (encoder, explicit)
    (missing / "text_encoder").mkdir(parents=True)
    config = missing / "text_encoder/config.json"
    config.write_text("{}")
    (missing / "tokenizer").mkdir()
    assert component_config(missing, "text_encoder") == config
    assert text_sources(missing, encoder)[1] == missing / "tokenizer"


def test_no_model_package_uses_bundled_text_config_and_tokenizer(tmp_path, monkeypatch):
    from trainer.modeling import moss_text_loader
    from trainer.modeling.loader import load_components
    from trainer.modeling.moss_loader import ASSETS

    encoder = tmp_path / "encoder.safetensors"
    encoder.touch()
    calls = []

    def load(source, config, dtype):
        calls.append((source, config))
        return torch.nn.Linear(1, 1)

    monkeypatch.setattr(moss_text_loader, "load_single_text_encoder", load)
    components = load_components(
        tmp_path / "missing",
        model_family="moss_image",
        text_encoder_path=encoder,
        load_transformer=False,
        load_vae=False,
    )
    assert calls == [(encoder, ASSETS / "text_encoder_config.json")]
    prompt = components.tokenizer.apply_chat_template(
        [{"role": "user", "content": "a cat"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    assert components.tokenizer(prompt, return_tensors="pt").input_ids.numel() > 0


def test_bundled_transformer_config_fallback(tmp_path, monkeypatch):
    from dataclasses import asdict

    from trainer.modeling import moss_loader
    from trainer.modeling.moss_image import MOSSImage, MOSSParams

    params = MOSSParams(
        dim=32, ffn_dim=48, num_heads=2, num_layers=1, text_dim=16, latent_channels=8
    )
    model = MOSSImage(params)
    source = tmp_path / "transformer.safetensors"
    save_file(model.state_dict(), source)
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "transformer_config.json").write_text(json.dumps(asdict(params)))
    monkeypatch.setattr(moss_loader, "ASSETS", assets)
    actual = moss_loader.load_transformer(tmp_path / "missing", source, torch.float32)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, actual.state_dict()[key], rtol=0, atol=0)
