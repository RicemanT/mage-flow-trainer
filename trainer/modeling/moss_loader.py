"""Local MOSS-Image components; no model-code execution or network downloads."""

import json
from dataclasses import fields
from pathlib import Path

import torch
from safetensors.torch import load_file

from .moss_image import MOSSImage, MOSSParams

TEXT_TEMPLATE_VERSION = "moss-chat-thinking-v1"
ASSETS = Path(__file__).with_name("assets") / "moss_image"


def component_config(root, component):
    local = Path(root) / component / "config.json"
    return local if local.is_file() else ASSETS / f"{component}_config.json"


def text_sources(root, text_encoder_path=None, tokenizer_path=None):
    root = Path(root)
    encoder = Path(text_encoder_path) if text_encoder_path else root / "text_encoder"
    tokenizer = Path(tokenizer_path) if tokenizer_path else root / "tokenizer"
    if not tokenizer_path and not tokenizer.is_dir():
        tokenizer = ASSETS
    return encoder, tokenizer


def weight_files(source):
    source = Path(source)
    if source.is_file():
        return [source]
    single = source / "diffusion_pytorch_model.safetensors"
    if single.is_file():
        return [single]
    index = source / "diffusion_pytorch_model.safetensors.index.json"
    names = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    paths = [(source / name).resolve() for name in names]
    if any(not p.is_relative_to(source.resolve()) for p in paths):
        raise ValueError(
            "Checkpoint shard paths must stay inside their model directory"
        )
    return paths


def load_transformer(root, source, dtype):
    root = Path(root)
    config = json.loads(
        component_config(root, "transformer").read_text(encoding="utf-8")
    )
    params = MOSSParams(
        **{f.name: config[f.name] for f in fields(MOSSParams) if f.name in config}
    )
    with torch.device("meta"):
        model = MOSSImage(params)
    state = {}
    for path in weight_files(source or root / "transformer"):
        shard = load_file(str(path))
        if state.keys() & shard.keys():
            raise ValueError("Duplicate tensor keys in MOSS shards")
        state.update(shard)
    for prefix in ("model.diffusion_model.", "diffusion_model."):
        if state and all(k.startswith(prefix) for k in state):
            state = {k[len(prefix) :]: v for k, v in state.items()}
            break
    if any(
        v.is_floating_point() is False or v.element_size() < 2 for v in state.values()
    ):
        raise ValueError("Load unquantized MOSS weights, then apply SDNQ")
    # ComfyUI BF16 exports concatenate Q, K, V along the output dimension.
    # Text-refiner qkv tensors are already native and must remain fused.
    converted = 0
    for i in range(params.num_layers):
        prefix = f"layers.{i}.proj."
        key = prefix + "qkv.weight"
        if key not in state:
            continue
        weight = state.pop(key)
        if tuple(weight.shape) != (3 * params.dim, params.dim):
            raise ValueError(f"{key}: expected {(3 * params.dim, params.dim)}, got {tuple(weight.shape)}")
        for name, part in zip(("q", "k", "v"), weight.chunk(3, dim=0)):
            target = prefix + name + ".weight"
            if target in state:
                raise ValueError(f"Ambiguous MOSS checkpoint: both {key} and {target}")
            state[target] = part.clone()
        converted += 1
    if converted:
        print(f"MOSS transformer: split fused Q/K/V weights in {converted} blocks into native training tensors")
    # ComfyUI exports may omit this derived constant.
    if "rope.bands" not in state:
        state["rope.bands"] = model.rope._default_bands(torch.device("cpu"))
    model.load_state_dict(state, strict=True, assign=True)
    return model.to(dtype=dtype)


@torch.no_grad()
def encode_prompts(components, prompts, device, max_length):
    rows = []
    for prompt in prompts:
        text = components.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        inputs = components.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length
        ).to(device)
        result = components.text_encoder(**inputs, use_cache=False, return_dict=True)
        rows.append(result.last_hidden_state[0])
    hidden = torch.nn.utils.rnn.pad_sequence(rows, batch_first=True)
    lengths = torch.tensor([len(row) for row in rows], device=device)
    return hidden, torch.arange(hidden.shape[1], device=device)[None] < lengths[:, None]
