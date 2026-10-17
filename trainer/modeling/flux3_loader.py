"""Flux3 package loading and text conditioning, separate from Mage-Flow."""

from dataclasses import fields
import json
from pathlib import Path
import torch
from safetensors.torch import load_file
from .flux3 import Flux3, Flux3Params

TEXT_TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
TEXT_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)


def load_transformer(root, source, dtype):
    root = Path(root)
    source = Path(source) if source else root / "model.safetensors"
    if not source.is_file():
        raise ValueError(
            "Flux3 currently requires the native BF16 model.safetensors, not diffusers shards"
        )
    config = json.loads((root / "config.json").read_text())
    if config.get("quantization") is not None:
        raise ValueError(
            "Load the BF16 Flux3 package, then quantize with SDNQ; inference FP8/NVFP4 imports are not supported"
        )
    dit = config["dit"]
    args = {f.name: dit[f.name] for f in fields(Flux3Params) if f.name in dit}
    args.update(
        channels=dit["in_channels"]["image"], text_dim=dit["in_channels"]["text"]
    )
    with torch.device("meta"):
        model = Flux3(Flux3Params(**args))
    state = load_file(str(source))
    for prefix in ("model.diffusion_model.", "diffusion_model."):
        if state and all(k.startswith(prefix) for k in state):
            state = {k[len(prefix) :]: v for k, v in state.items()}
    if "ref_time_in.in_layer.weight" in state:
        raise ValueError("Distilled Flux3 flow-map checkpoints are unsupported")
    if any(v.dtype in (torch.float8_e4m3fn, torch.uint8) for v in state.values()):
        raise ValueError(
            "Prequantized Flux3 inference weights cannot be used as BF16 training weights"
        )
    model.load_state_dict(state, strict=True, assign=True)
    model.to(dtype=dtype)
    return model


@torch.no_grad()
def encode_prompts(components, prompts, device, max_length):
    # Encode independently: reference padding rounds each prompt to 80, and those
    # pad hidden states are conditioning tokens, not discarded by the DiT.
    # The official pipeline caps the rounded window at max_length.
    values = []
    for prompt in prompts:
        tokens = components.tokenizer(
            TEXT_TEMPLATE.format(prompt),
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        ids = tokens.input_ids.to(device)
        n = ids.shape[1]
        pad = min(((n + 79) // 80) * 80, max_length) - n
        ids = torch.nn.functional.pad(ids, (0, pad), value=151643)
        mask = torch.arange(ids.shape[1], device=device)[None] < n
        positions = (mask.long().cumsum(-1) - 1).masked_fill(~mask, 1)
        # Match Comfy's list-of-layers semantics: intermediate layers are
        # unnormalized; the last hidden state retains the encoder's final norm.
        out = components.text_encoder(
            input_ids=ids,
            attention_mask=mask,
            position_ids=positions,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
            logits_to_keep=1,
        )
        states = out.hidden_states
        values.append(torch.cat([states[i] for i in TEXT_LAYERS], -1)[0])
    hidden = torch.nn.utils.rnn.pad_sequence(values, batch_first=True)
    lengths = torch.tensor([len(v) for v in values], device=device)
    return hidden, torch.arange(hidden.shape[1], device=device)[None] < lengths[:, None]
