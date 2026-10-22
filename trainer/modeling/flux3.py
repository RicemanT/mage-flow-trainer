"""Experimental text-to-image Flux3, using native BFL checkpoint names.

Training uses out-of-place residuals and ordinary PyTorch autograd. Image editing
and distilled flow-map checkpoints are deliberately not supported yet.
"""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from trainer.training.activation_offload import (
    checkpoint_with_offload, configure_activation_offload,
)


@dataclass
class Flux3Params:
    hidden: int = 3072
    heads: int = 24
    depth: int = 5
    depth_single: int = 28
    mlp_ratio: float = 3.0
    axes_dim: tuple = (32, 32, 32, 32)
    theta: int = 10000
    channels: int = 128
    text_dim: int = 20480
    image_rope_scale: float = 0.5


def timestep_embedding(t, dim=256):
    freq = torch.exp(
        -math.log(10000) * torch.arange(dim // 2, device=t.device).float() / (dim // 2)
    )
    phase = (t * 1000).float()[:, None] * freq[None]
    return torch.cat((phase.cos(), phase.sin()), dim=-1).to(t.dtype)


class MLPEmbedder(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.in_layer = nn.Linear(256, d, bias=False)
        self.out_layer = nn.Linear(d, d, bias=False)

    def forward(self, x):
        return self.out_layer(F.silu(self.in_layer(x)))


class Modulation(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, 3 * d, bias=False)

    def forward(self, v):
        return self.lin(F.silu(v))[:, None].chunk(3, -1)


def rotate(x, angles):
    pairs = x.float().reshape(*x.shape[:-1], -1, 2)
    c, s = angles.cos()[None, :, None], angles.sin()[None, :, None]
    a, b = pairs.unbind(-1)
    return torch.stack((a * c - b * s, a * s + b * c), -1).flatten(-2).to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return (
            x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        ).to(x.dtype) * self.scale


class Block(nn.Module):
    def __init__(self, p):
        super().__init__()
        d = p.hidden
        self.heads = p.heads
        for name in ("q_proj", "k_proj", "v_proj", "attn_out"):
            setattr(self, name, nn.Linear(d, d, bias=False))
        self.mlp_in = nn.Linear(d, int(d * p.mlp_ratio) * 2, bias=False)
        self.mlp_out = nn.Linear(int(d * p.mlp_ratio), d, bias=False)
        self.norm = nn.Module()
        self.norm.query_norm = RMSNorm(d // p.heads)
        self.norm.key_norm = RMSNorm(d // p.heads)
        self.pre_norm = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)

    def forward(self, x, mod, angles, packed=None):
        shift, scale, gate = mod
        z = self.pre_norm(x) * (1 + scale) + shift
        b, n, d = z.shape
        q = self.norm.query_norm(self.q_proj(z).reshape(b, n, self.heads, -1))
        k = self.norm.key_norm(self.k_proj(z).reshape(b, n, self.heads, -1))
        v = self.v_proj(z).reshape(b, n, self.heads, -1)
        q, k = rotate(q, angles), rotate(k, angles)
        if packed is None:
            attn = (
                F.scaled_dot_product_attention(
                    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
                )
                .transpose(1, 2)
                .reshape(b, n, d)
            )
        else:
            from torch.nn.attention.varlen import varlen_attn

            cu, longest = packed
            attn = varlen_attn(
                q.squeeze(0).contiguous(),
                k.squeeze(0).contiguous(),
                v.squeeze(0).contiguous(),
                cu,
                cu,
                longest,
                longest,
            ).reshape(b, n, d)
        # Native SwiGLU order is gate first, value second.
        gate_mlp, value = self.mlp_in(z).chunk(2, -1)
        update = self.attn_out(attn) + self.mlp_out(F.silu(gate_mlp) * value)
        return x + gate * update


class LastLayer(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.norm_final = nn.LayerNorm(p.hidden, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(p.hidden, 2 * p.hidden, bias=False)
        )
        self.linear = nn.Linear(p.hidden, p.channels, bias=False)

    def forward(self, x, v):
        shift, scale = self.adaLN_modulation(v)[:, None].chunk(2, -1)
        return self.linear(self.norm_final(x) * (1 + scale) + shift)


class Flux3(nn.Module):
    model_family = "flux3"
    patch_size = 1

    def __init__(self, p):
        super().__init__()
        self.params = p
        if (
            p.hidden % p.heads
            or sum(p.axes_dim) != p.hidden // p.heads
            or any(d % 2 for d in p.axes_dim)
        ):
            raise ValueError("Invalid Flux3 RoPE/head dimensions")
        self.emb_in = nn.ModuleDict(
            {
                m: nn.Linear(p.channels, p.hidden, bias=False)
                for m in ("image", "image_cond")
            }
        )
        self.txt_in = nn.Linear(p.text_dim, p.hidden, bias=False)
        self.time_in = MLPEmbedder(p.hidden)
        for name in ("early_stream_modulations", "single_stream_modulations"):
            setattr(
                self,
                name,
                nn.ModuleDict(
                    {m: Modulation(p.hidden) for m in ("image", "image_cond", "txt")}
                ),
            )
        self.content_mode_blocks = nn.ModuleDict(
            {
                m: nn.ModuleList([Block(p) for _ in range(p.depth)])
                for m in ("image", "image_cond")
            }
        )
        self.txt_mode_blocks = nn.ModuleList([Block(p) for _ in range(p.depth)])
        self.single_blocks = nn.ModuleList([Block(p) for _ in range(p.depth_single)])
        self.final_layer = nn.ModuleDict({"image": LastLayer(p)})
        self.configure_execution()

    def configure_execution(
        self,
        gradient_checkpointing=True,
        checkpoint_blocks=None,
        compile_mode=None,
        compile_dynamic=True,
        attention_backend="sdpa",
        activation_offload=False,
        activation_offload_budget_mb=1024,
        activation_offload_pin_memory=False,
        activation_offload_prefetch=False,
    ):
        configure_activation_offload(
            self, enabled=activation_offload, budget_mb=activation_offload_budget_mb,
            pin_memory=activation_offload_pin_memory, prefetch=activation_offload_prefetch,
            gradient_checkpointing=gradient_checkpointing, compile_mode=compile_mode,
        )
        if attention_backend not in ("sdpa", "torch_varlen"):
            raise ValueError(
                'Flux3 requires attention_backend="sdpa" or "torch_varlen"'
            )
        if checkpoint_blocks is not None:
            raise ValueError("Flux3 currently supports all-block checkpointing only")
        self.attention_backend = attention_backend
        self.checkpointing = gradient_checkpointing
        self.block_forward = (
            Block.forward
            if compile_mode is None
            else torch.compile(
                Block.forward, mode=compile_mode, dynamic=compile_dynamic
            )
        )

    def add_adapter(self, config):
        from peft import inject_adapter_in_model

        inject_adapter_in_model(config, self)

    def angles(self, ids):
        return torch.cat(
            [
                ids[:, i, None].float()
                * self.params.theta
                ** (-torch.arange(0, d, 2, device=ids.device).float() / d)
                for i, d in enumerate(self.params.axes_dim)
            ],
            -1,
        )

    def run_blocks(self, blocks, x, mod, angles, packed=None):
        for block in blocks:
            if self.checkpointing and self.training and torch.is_grad_enabled():
                x = checkpoint_with_offload(
                    self, self.block_forward,
                    block,
                    x,
                    mod,
                    angles,
                    packed,
                )
            else:
                x = self.block_forward(block, x, mod, angles, packed)
        return x

    def forward_one(self, x, t, context):
        if x.ndim != 5 or x.shape[2] != 1:
            raise ValueError("Flux3 expects image latents [B,128,1,H,W]")
        b, c, _, h, w = x.shape
        img = self.emb_in["image"](x.squeeze(2).flatten(2).transpose(1, 2))
        txt = self.txt_in(context)
        vec = self.time_in(timestep_embedding(t.to(x.dtype)))
        zero = self.time_in(timestep_embedding(torch.zeros_like(t).to(x.dtype)))
        ids = torch.zeros(h, w, 4, device=x.device)
        ids[..., 1] = (
            torch.arange(h, device=x.device)[:, None] * self.params.image_rope_scale
        )
        ids[..., 2] = (
            torch.arange(w, device=x.device)[None, :] * self.params.image_rope_scale
        )
        ti = torch.zeros(txt.shape[1], 4, device=x.device)
        ti[:, 3] = torch.arange(txt.shape[1], device=x.device)
        ia, ta = self.angles(ids.flatten(0, 1)), self.angles(ti)
        img = self.run_blocks(
            self.content_mode_blocks["image"],
            img,
            self.early_stream_modulations["image"](vec),
            ia,
        )
        txt = self.run_blocks(
            self.txt_mode_blocks, txt, self.early_stream_modulations["txt"](zero), ta
        )
        im, tm = (
            self.single_stream_modulations["image"](vec),
            self.single_stream_modulations["txt"](zero),
        )
        mod = tuple(
            torch.cat((a.expand(-1, h * w, -1), b_.expand(-1, txt.shape[1], -1)), 1)
            for a, b_ in zip(im, tm)
        )
        joint = self.run_blocks(
            self.single_blocks, torch.cat((img, txt), 1), mod, torch.cat((ia, ta))
        )
        return (
            self.final_layer["image"](joint[:, : h * w], vec)
            .transpose(1, 2)
            .reshape(b, c, 1, h, w)
        )

    def forward_packed(self, xs, t, contexts):
        # Each attention segment belongs to one example. Image and text streams
        # have separate boundaries until the joint blocks concatenate each pair.
        shapes = [x.shape for x in xs]
        if any(len(shape) != 5 or shape[0] != 1 or shape[2] != 1 for shape in shapes):
            raise ValueError("Packed Flux3 expects individual [1,C,1,H,W] latents")
        ilens = [shape[-2] * shape[-1] for shape in shapes]
        tlens = [ctx.shape[1] for ctx in contexts]
        device = xs[0].device

        def boundaries(lengths):
            cu = F.pad(
                torch.tensor(lengths, device=device, dtype=torch.int32).cumsum(
                    0, dtype=torch.int32
                ),
                (1, 0),
            )
            return cu, max(lengths)

        def expand_mod(mod, lengths):
            return tuple(
                torch.cat(
                    [
                        v[i : i + 1].expand(1, length, -1)
                        for i, length in enumerate(lengths)
                    ],
                    1,
                )
                for v in mod
            )

        vec = self.time_in(timestep_embedding(t.to(xs[0].dtype)))
        zero = self.time_in(timestep_embedding(torch.zeros_like(t).to(xs[0].dtype)))
        img = self.emb_in["image"](
            torch.cat([x.squeeze(2).flatten(2).transpose(1, 2) for x in xs], 1)
        )
        txt = self.txt_in(torch.cat(contexts, 1))
        image_angles, text_angles = [], []
        for shape, length in zip(shapes, tlens):
            h, w = shape[-2:]
            ids = torch.zeros(h, w, 4, device=device)
            ids[..., 1] = (
                torch.arange(h, device=device)[:, None] * self.params.image_rope_scale
            )
            ids[..., 2] = (
                torch.arange(w, device=device)[None, :] * self.params.image_rope_scale
            )
            ti = torch.zeros(length, 4, device=device)
            ti[:, 3] = torch.arange(length, device=device)
            image_angles.append(self.angles(ids.flatten(0, 1)))
            text_angles.append(self.angles(ti))
        img = self.run_blocks(
            self.content_mode_blocks["image"],
            img,
            expand_mod(self.early_stream_modulations["image"](vec), ilens),
            torch.cat(image_angles),
            boundaries(ilens),
        )
        txt = self.run_blocks(
            self.txt_mode_blocks,
            txt,
            expand_mod(self.early_stream_modulations["txt"](zero), tlens),
            torch.cat(text_angles),
            boundaries(tlens),
        )
        images, texts = img.split(ilens, 1), txt.split(tlens, 1)
        im, tm = (
            self.single_stream_modulations["image"](vec),
            self.single_stream_modulations["txt"](zero),
        )
        mod = tuple(
            torch.cat(
                [
                    torch.cat(
                        (
                            a[i : i + 1].expand(1, ni, -1),
                            b[i : i + 1].expand(1, nt, -1),
                        ),
                        1,
                    )
                    for i, (ni, nt) in enumerate(zip(ilens, tlens))
                ],
                1,
            )
            for a, b in zip(im, tm)
        )
        lengths = [ni + nt for ni, nt in zip(ilens, tlens)]
        joint = self.run_blocks(
            self.single_blocks,
            torch.cat([torch.cat((a, b), 1) for a, b in zip(images, texts)], 1),
            mod,
            torch.cat([torch.cat((a, b)) for a, b in zip(image_angles, text_angles)]),
            boundaries(lengths),
        )
        return [
            self.final_layer["image"](part[:, :ni], vec[i : i + 1])
            .transpose(1, 2)
            .reshape(shape)
            for i, (part, ni, shape) in enumerate(
                zip(joint.split(lengths, 1), ilens, shapes)
            )
        ]

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        return_dict=False,
        **kwargs,
    ):
        if kwargs:
            raise ValueError(f"Unsupported Flux3 forward arguments: {list(kwargs)}")
        context, mask = encoder_hidden_states
        xs = (
            hidden_states
            if isinstance(hidden_states, list)
            else list(hidden_states.split(1))
        )
        if self.attention_backend == "torch_varlen":
            ys = self.forward_packed(
                xs,
                timestep,
                [context[i : i + 1, mask[i].bool()] for i in range(len(xs))],
            )
            return (ys if isinstance(hidden_states, list) else torch.cat(ys),)
        # Per-image SDPA avoids attention leakage and supports native resolutions.
        ys = [
            self.forward_one(x, timestep[i : i + 1], context[i : i + 1, mask[i].bool()])
            for i, x in enumerate(xs)
        ]
        return (ys if isinstance(hidden_states, list) else torch.cat(ys),)
