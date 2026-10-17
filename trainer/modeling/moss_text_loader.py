"""Qwen3.5 text-only safetensors loading without ComfyUI runtime dependencies."""

import json
from functools import lru_cache
from pathlib import Path

import torch
from safetensors import safe_open


@lru_cache(maxsize=8)
def regular_hadamard(size):
    """Normalized regular Hadamard used by the ConvRot checkpoint format."""
    if size < 4 or size & (size - 1) or (size.bit_length() - 1) % 2:
        raise ValueError(f"ConvRot group size must be a power of four, got {size}")
    seed = torch.ones(4, 4)
    seed[torch.arange(4), torch.arange(3, -1, -1)] = -1
    matrix = seed
    while matrix.shape[0] < size:
        matrix = torch.kron(matrix, seed)
    return matrix / size**0.5


def read_weight(reader, key, dtype):
    """Decode one weight, chunking the vocabulary table to bound temporary RAM."""
    prefix = key.removesuffix(".weight")
    quant_key = prefix + ".comfy_quant"
    keys = reader.keys()
    if quant_key not in keys:
        value = reader.get_tensor(key)
        if not value.is_floating_point() or value.element_size() < 2:
            raise ValueError(
                f"{key}: unsupported quantization without comfy_quant metadata"
            )
        return value.to(dtype)
    conf = json.loads(bytes(reader.get_tensor(quant_key).tolist()).decode("utf-8"))
    if conf.get("format") != "int8_tensorwise":
        fmt = conf.get("format")
        low_bit = fmt in {"asym_w4a8_int8", "convrot_w4a4", "nvfp4", "fp4", "int4"}
        detail = (
            " This is a 4-bit weight format (even if its name contains 'int8')."
            " Four-bit quantization can alter caption embeddings more than INT8;"
            " converting it to BF16 cannot recover the discarded precision."
            if low_bit else ""
        )
        raise ValueError(
            f"MOSS text encoder: unsupported quantization format {fmt!r} at {key}."
            f"{detail} Set train.text_encoder_path to Qwen3.5-9B BF16/FP16 weights"
            " or an int8_tensorwise checkpoint (ConvRot is supported)."
            " This concerns the text encoder, not the transformer."
        )
    view = reader.get_slice(key)
    shape = view.get_shape()
    if view.get_dtype() != "I8" or len(shape) != 2:
        raise ValueError(f"{key}: expected a two-dimensional INT8 weight")
    scale = reader.get_tensor(prefix + ".weight_scale").float()
    if scale.numel() == 1:
        scale = scale.reshape(1, 1)
    elif tuple(scale.shape) != (shape[0], 1):
        raise ValueError(f"{key}: unsupported INT8 scale shape {tuple(scale.shape)}")
    rotation = None
    if conf.get("convrot", False):
        group = int(conf.get("convrot_groupsize", 256))
        rotation = regular_hadamard(group)
        if shape[1] % group:
            raise ValueError(f"{key}: width is not divisible by ConvRot group size")
    out = torch.empty(shape, dtype=dtype)
    for start in range(0, shape[0], 512):
        stop = min(start + 512, shape[0])
        part = view[start:stop].float() * (
            scale if scale.numel() == 1 else scale[start:stop]
        )
        if rotation is not None:
            part = (part.reshape(-1, group) @ rotation.T).reshape(
                stop - start, shape[1]
            )
        out[start:stop].copy_(part)
    return out


def load_single_text_encoder(source, config_source, dtype):
    from accelerate import init_empty_weights
    from transformers import Qwen3_5TextConfig, Qwen3_5TextModel

    data = json.loads(Path(config_source).read_text(encoding="utf-8"))
    config = Qwen3_5TextConfig(**data.get("text_config", data))
    config._attn_implementation = "sdpa"
    with init_empty_weights():
        model = Qwen3_5TextModel(config)
    with safe_open(str(source), framework="pt") as reader:
        keys = set(reader.keys())
        prefixes = ["model.language_model.", "language_model.", "model.", ""]
        prefix = next((p for p in prefixes if p + "embed_tokens.weight" in keys), None)
        if prefix is None:
            raise ValueError(
                "Qwen3.5 checkpoint has no recognized text embedding table"
            )
        expected = model.state_dict()
        used = set()
        for name, param in expected.items():
            key = prefix + name
            if key not in keys:
                raise ValueError(f"Qwen3.5 checkpoint is missing {key}")
            value = read_weight(reader, key, dtype)
            if value.shape != param.shape:
                raise ValueError(
                    f"{key}: expected {tuple(param.shape)}, got {tuple(value.shape)}"
                )
            parent, _, leaf = name.rpartition(".")
            module = model.get_submodule(parent) if parent else model
            setattr(module, leaf, torch.nn.Parameter(value, requires_grad=False))
            used.add(key)
            if name.endswith(".weight"):
                used.update({key[:-7] + ".comfy_quant", key[:-7] + ".weight_scale"})
        # Vision, generation head and multi-token prediction are unused for captions.
        extra = {
            k
            for k in keys - used
            if not k.startswith(
                ("model.visual.", "visual.", "lm_head.", "mtp.", "model.mtp.")
            )
        }
        if extra:
            raise ValueError(f"Unexpected Qwen3.5 text tensors: {sorted(extra)[:10]}")
    return model.requires_grad_(False).eval()
