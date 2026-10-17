"""Standalone FLUX.2 VAE loading, including original LDM safetensors exports."""

from pathlib import Path

import torch
from safetensors.torch import load_file


def load_flux2_vae(source, dtype=torch.float32):
    from diffusers import AutoencoderKLFlux2
    from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint

    source = Path(source)
    if source.is_dir():
        return AutoencoderKLFlux2.from_pretrained(
            source, torch_dtype=dtype, local_files_only=True
        )
    state = load_file(str(source))
    state = {
        k.removeprefix("first_stage_model.").removeprefix("module."): v
        for k, v in state.items()
    }
    original = "encoder.down.0.block.0.conv1.weight" in state
    if original:
        enc = [
            state[f"encoder.down.{i}.block.0.conv1.weight"].shape[0] for i in range(4)
        ]
        dec = [state[f"decoder.up.{i}.block.0.conv1.weight"].shape[0] for i in range(4)]
    else:
        enc = [
            state[f"encoder.down_blocks.{i}.resnets.0.conv1.weight"].shape[0]
            for i in range(4)
        ]
        dec = [
            state[f"decoder.up_blocks.{i}.resnets.0.conv1.weight"].shape[0]
            for i in reversed(range(4))
        ]
    config = {
        "block_out_channels": enc,
        "decoder_block_out_channels": dec,
        "latent_channels": 32,
        "layers_per_block": 2,
        "batch_norm_eps": 1e-4,
        "down_block_types": ["DownEncoderBlock2D"] * 4,
        "up_block_types": ["UpDecoderBlock2D"] * 4,
    }
    if original:
        relocated = {
            k.replace("encoder.quant_conv.", "quant_conv.").replace(
                "decoder.post_quant_conv.", "post_quant_conv."
            ): v
            for k, v in state.items()
        }
        converted = convert_ldm_vae_checkpoint(relocated, config)
        converted.update({k: v for k, v in state.items() if k.startswith("bn.")})
        if len(converted) != len(state):
            raise ValueError(
                "Unsupported FLUX.2 VAE tensors: conversion did not preserve all keys"
            )
        state = converted
    with torch.device("meta"):
        vae = AutoencoderKLFlux2(**config)
    vae.load_state_dict(state, strict=True, assign=True)
    return vae.to(dtype=dtype)
