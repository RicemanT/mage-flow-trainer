"""Lossless export layouts for MOSS joint-block Q/K/V projections."""
import re
import torch

_PROJECTION = re.compile(r'^(?:(?:diffusion_model\.)?layers\.\d+\.proj\.|lora_unet_layers_\d+_proj_)')


def fuse_qkv(state, *, adapter=False):
    """Return a new state dictionary; native text-refiner QKV stays unchanged.

    Independent low-rank updates require block-diagonal up matrices and stacked
    down matrices, not simply stacking both factors with the original rank.
    """
    result = dict(state)
    groups = sorted({m.group() for key in state if (m := _PROJECTION.match(key))
                     and key[len(m.group()):].split('.')[0] in ('q', 'k', 'v')})
    ranks = {}
    for prefix in groups:
        if any(key.startswith(prefix + 'qkv.') for key in state):
            raise ValueError(f'{prefix}: ambiguous separate and fused QKV tensors')
        if not adapter:
            names = [prefix + p + '.weight' for p in ('q', 'k', 'v')]
            if not all(k in state for k in names):
                raise ValueError(f'{prefix}: fused export requires all Q/K/V weights')
            result[prefix + 'qkv.weight'] = torch.cat([result.pop(k) for k in names], dim=0)
            continue
        down_key, up_key = ('.lora_down.weight', '.lora_up.weight') if prefix + 'q.lora_down.weight' in state else ('.lora_A.weight', '.lora_B.weight')
        expected = {prefix + p + suffix for p in ('q', 'k', 'v')
                    for suffix in (down_key, up_key, '.alpha')}
        actual = {k for k in state if any(k.startswith(prefix + p + '.') for p in ('q', 'k', 'v'))}
        if actual != expected:
            raise ValueError(f'{prefix}: fused export supports complete plain LoRA/LoCon Q/K/V factors only')
        downs = [state[prefix + p + down_key] for p in ('q', 'k', 'v')]
        ups = [state[prefix + p + up_key] for p in ('q', 'k', 'v')]
        if any(d.ndim != 2 or u.ndim != 2 or d.shape[0] != u.shape[1] for d, u in zip(downs, ups)):
            raise ValueError(f'{prefix}: expected compatible linear LoRA factors')
        scales = [state[prefix + p + '.alpha'].item() / d.shape[0] for p, d in zip(('q', 'k', 'v'), downs)]
        if len(set(scales)) != 1:
            raise ValueError(f'{prefix}: unequal Q/K/V scaling requires separated export')
        rank = sum(d.shape[0] for d in downs)
        # An F64 scalar avoids another FP32 alpha rounding when rank is tripled.
        alpha = torch.tensor(scales[0] * rank, dtype=torch.float64)
        for key in expected:
            del result[key]
        result[prefix + 'qkv' + down_key] = torch.cat(downs, dim=0)
        result[prefix + 'qkv' + up_key] = torch.block_diag(*ups)
        result[prefix + 'qkv.alpha'] = alpha
        ranks[prefix + 'qkv'] = rank
    return result, {'joint_blocks': len(groups), 'fused_adapter_ranks': ranks,
                    'method': 'block-diagonal-up/stacked-down' if adapter else 'concatenate-output-rows'}


def separate_adapter_qkv(state):
    """Split output rows, keeping the full rank and alpha; no lossy rank reduction."""
    result = dict(state)
    groups = sorted({m.group() for key in state if (m := _PROJECTION.match(key))
                     and key[len(m.group()):].startswith('qkv.')})
    for prefix in groups:
        down_key, up_key = ('.lora_down.weight', '.lora_up.weight') if prefix + 'qkv.lora_down.weight' in state else ('.lora_A.weight', '.lora_B.weight')
        expected = {prefix + 'qkv' + s for s in (down_key, up_key, '.alpha')}
        actual = {k for k in state if k.startswith(prefix + 'qkv.')}
        if actual != expected or any(k.startswith(prefix + p + '.') for k in state for p in ('q','k','v')):
            raise ValueError(f'{prefix}: expected only a plain fused LoRA/LoCon adapter')
        down, up = state[prefix+'qkv'+down_key], state[prefix+'qkv'+up_key]
        if down.ndim != 2 or up.ndim != 2 or up.shape[0] % 3 or up.shape[1] != down.shape[0]:
            raise ValueError(f'{prefix}: expected three equal Q/K/V output sections')
        for key in expected:
            del result[key]
        for p, part in zip(('q','k','v'), up.chunk(3,dim=0)):
            result[prefix+p+down_key] = down.clone()
            result[prefix+p+up_key] = part.contiguous().clone()
            result[prefix+p+'.alpha'] = state[prefix+'qkv.alpha'].clone()
    return result, {'joint_blocks': len(groups), 'method': 'split-output-rows/retain-full-rank'}
