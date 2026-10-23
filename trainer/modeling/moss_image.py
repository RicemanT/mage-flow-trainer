"""MOSS-Image training kernels adapted from OpenMOSS-Team/MOSS-Image.

Apache-2.0; see LICENSE.moss-image. Packed attention, checkpointing and the
trainer interface are local additions. Native checkpoint names are preserved.
"""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from trainer.training.activation_offload import (
    checkpoint_with_offload, configure_activation_offload,
)


class _RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        return y.to(x.dtype) * self.weight


def _situ(hidden, gate):
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (
        25.0 * torch.tanh(hidden / 25.0)
    )


def _rotate(x, rope):
    sin, cos = rope.chunk(2, -1)
    sin = torch.cat((sin, sin), -1).unsqueeze(-2)
    cos = torch.cat((cos, cos), -1).unsqueeze(-2)
    a, b = x.chunk(2, -1)
    return x * cos + torch.cat((-b, a), -1) * sin


def _attention(q, k, v, rope, heads, packed):
    shape = (*q.shape[:2], heads, q.shape[-1] // heads)
    q = _rotate(q.reshape(shape), rope).transpose(1, 2)
    k = _rotate(k.reshape(shape), rope).transpose(1, 2)
    v = v.reshape(shape).transpose(1, 2)
    lengths, cu, longest = packed
    if cu is not None:
        from torch.nn.attention.varlen import varlen_attn

        return varlen_attn(
            q[0].transpose(0, 1).contiguous().to(v.dtype),
            k[0].transpose(0, 1).contiguous().to(v.dtype),
            v[0].transpose(0, 1).contiguous(),
            cu,
            cu,
            longest,
            longest,
        ).flatten(1)[None]
    return (
        torch.cat(
            [
                F.scaled_dot_product_attention(a.to(c.dtype), b.to(c.dtype), c)
                for a, b, c in zip(
                    q.split(lengths, 2), k.split(lengths, 2), v.split(lengths, 2)
                )
            ],
            2,
        )
        .transpose(1, 2)
        .flatten(2)
    )


class _TextBlock(nn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = _RMSNorm(dim), _RMSNorm(dim)
        self.q_norm, self.k_norm = _RMSNorm(dim // heads), _RMSNorm(dim // heads)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ffn_gate, self.ffn_up = nn.Linear(dim, ffn_dim), nn.Linear(dim, ffn_dim)
        self.ffn_down = nn.Linear(ffn_dim, dim)

    def forward(self, x, rope, packed):
        q, k, v = self.qkv(self.norm1(x)).chunk(3, -1)
        q = self.q_norm(q.unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(k.unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj(_attention(q, k, v, rope, self.heads, packed))
        y = self.norm2(x)
        return x + self.ffn_down(_situ(self.ffn_up(y), self.ffn_gate(y)))


class _TextProjection(nn.Module):
    def __init__(self, text_dim, dim, ffn_dim, heads):
        super().__init__()
        self.proj = nn.Linear(text_dim, dim)
        self.blocks = nn.ModuleList([_TextBlock(dim, ffn_dim, heads) for _ in range(2)])

    def forward(self, x, rope, packed):
        x = self.proj(x)
        for block in self.blocks:
            x = block(x, rope, packed)
        return x


class _Block(nn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = _RMSNorm(dim), _RMSNorm(dim)
        self.q_norm, self.k_norm = _RMSNorm(dim // heads), _RMSNorm(dim // heads)
        self.proj = nn.ModuleDict(
            {name: nn.Linear(dim, dim, bias=False) for name in ("q", "k", "v", "o")}
        )
        self.proj["ffn_up_gate"] = nn.Linear(dim, 2 * ffn_dim, bias=False)
        self.proj["ffn_down"] = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x, rope, modulation, packed):
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.unbind(1)
        y = self.norm1(x) * (1 + scale_a) + shift_a
        dtype = x.dtype
        y = y.to(dtype)
        q = self.q_norm(self.proj["q"](y).unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(self.proj["k"](y).unflatten(-1, (self.heads, -1))).flatten(2)
        x = (
            x
            + self.proj["o"](
                _attention(q, k, self.proj["v"](y), rope, self.heads, packed)
            )
            * gate_a
        )
        y = self.norm2(x) * (1 + scale_f) + shift_f
        hidden, gate = self.proj["ffn_up_gate"](y.to(dtype)).chunk(2, -1)
        return x + self.proj["ffn_down"](_situ(hidden, gate)) * gate_f


class _RoPE(nn.Module):
    def __init__(self, head_dim, base):
        super().__init__()
        self.num_bands = head_dim // 8
        self.base = base
        self.register_buffer("bands", self._default_bands(torch.device("cpu")))

    def _default_bands(self, device):
        # Match training's CPU fp32 construction, including under meta init.
        exponent = (
            torch.arange(self.num_bands, device="cpu", dtype=torch.float32)
            / self.num_bands
        )
        return (1 / (self.base**exponent)).to(device)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        # .float() cannot recover frequencies already rounded by .to(bf16).
        self.bands = self._default_bands(self.bands.device)
        return self

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        # Frequencies are derived constants; retain the key for checkpoint compatibility.
        self.bands = self._default_bands(self.bands.device)

    def forward(self, height, width, text_length, device):
        # Diffusers/Accelerate can install tensors directly, bypassing load hooks.
        if self.bands.dtype != torch.float32:
            self.bands = self._default_bands(device)
        # Coordinates: (r, f, h, w); sequence order is image followed by text.
        h, w = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij",
        )
        image = (
            torch.stack(
                (torch.zeros_like(h), torch.full_like(h, text_length), h, w), -1
            )
            .flatten(0, 1)
            .float()
        )
        image[:, 2] -= (height - 1) / 2
        image[:, 3] -= (width - 1) / 2
        text = torch.zeros(text_length, 4, device=device, dtype=torch.float32)
        text[:, 1] = torch.arange(text_length, device=device)
        phase = torch.cat((image, text), 0).unsqueeze(-1) * self.bands
        return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2).unsqueeze(0)


@dataclass
class MOSSParams:
    dim: int = 3072
    ffn_dim: int = 8192
    num_heads: int = 24
    num_layers: int = 42
    text_dim: int = 4096
    freq_dim: int = 256
    latent_channels: int = 128
    rope_base: float = 10000.0


class MOSSImage(nn.Module):
    model_family = "moss_image"
    patch_size = 1

    def __init__(self, p):
        super().__init__()
        self.params = p
        if p.dim % p.num_heads or (p.dim // p.num_heads) % 8 or p.freq_dim % 2:
            raise ValueError("Invalid MOSS head or timestep dimensions")
        self.video_embedder = nn.Linear(p.latent_channels, p.dim)
        self.text_embedder = _TextProjection(p.text_dim, p.dim, p.ffn_dim, p.num_heads)
        self.rope = _RoPE(p.dim // p.num_heads, p.rope_base)
        self.time_embedding = nn.Sequential(
            nn.Linear(p.freq_dim, p.dim), nn.SiLU(), nn.Linear(p.dim, p.dim)
        )
        self.shared_block_modulation = nn.Linear(p.dim, 6 * p.dim)
        self.layers = nn.ModuleList(
            [_Block(p.dim, p.ffn_dim, p.num_heads) for _ in range(p.num_layers)]
        )
        self.final_norm_video = _RMSNorm(p.dim)
        self.final_linear_video = nn.Linear(p.dim, p.latent_channels, bias=False)
        self.video_head_modulation = nn.Linear(p.dim, 2 * p.dim)
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
        if (
            attention_backend not in ("sdpa", "torch_varlen")
            or checkpoint_blocks is not None
        ):
            raise ValueError(
                "MOSS requires SDPA or torch_varlen and all-block checkpointing"
            )
        self.checkpointing = gradient_checkpointing
        self.attention_backend = attention_backend
        compile_options = None
        if compile_mode is not None:
            compile_options = dict(torch._inductor.list_mode_options(compile_mode))
            # MOSS RMSNorm backward at longer packed sequence lengths can produce
            # mixed-order reduction kernels exceeding Ada's shared-memory limit.
            # Keep other compilation optimizations, using separate reductions.
            compile_options["triton.mix_order_reduction"] = False
        self.block_forward = (
            _Block.forward
            if compile_mode is None
            else torch.compile(
                _Block.forward, options=compile_options, dynamic=compile_dynamic
            )
        )
        self.text_forward = (
            _TextBlock.forward
            if compile_mode is None
            else torch.compile(
                _TextBlock.forward, options=compile_options, dynamic=compile_dynamic
            )
        )

    def add_adapter(self, config):
        from peft import inject_adapter_in_model

        inject_adapter_in_model(config, self)

    def _run(self, fn, block, *args):
        if self.checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint_with_offload(self, fn, block, *args)
        return fn(block, *args)

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        return_dict=False,
        **kwargs,
    ):
        if kwargs:
            raise ValueError(f"Unsupported MOSS forward arguments: {list(kwargs)}")
        xs = (
            hidden_states
            if isinstance(hidden_states, list)
            else list(hidden_states.split(1))
        )
        context, mask = encoder_hidden_states
        shapes = [x.shape for x in xs]
        if any(len(s) != 5 or s[0] != 1 or s[2] != 1 for s in shapes):
            raise ValueError("MOSS expects [1,C,1,H,W] image latents")
        contexts = [context[i : i + 1, mask[i].bool()] for i in range(len(xs))]
        ilens = [s[-2] * s[-1] for s in shapes]
        tlens = [x.shape[1] for x in contexts]
        if not all(tlens):
            raise ValueError("Each MOSS example needs at least one text token")
        lengths = [a + b for a, b in zip(ilens, tlens)]

        def packed(lens):
            cu = None
            if self.attention_backend == "torch_varlen":
                cu = F.pad(
                    torch.tensor(lens, device=xs[0].device, dtype=torch.int32).cumsum(
                        0, dtype=torch.int32
                    ),
                    (1, 0),
                )
            return lens, cu, max(lens)

        # Trainer timesteps are normalized sigma; the native MOSS sinusoid uses 0..1000.
        times = timestep.float() * 1000
        half = self.params.freq_dim // 2
        freq = 10000 ** (
            -torch.arange(half, device=times.device, dtype=torch.float64) / half
        )
        phase = times.double()[:, None] * freq
        emb = torch.cat((phase.cos(), phase.sin()), -1).float()
        time = self.time_embedding(emb.to(xs[0].dtype))
        mod = self.shared_block_modulation(time).view(len(xs), 6, -1)
        head = self.video_head_modulation(time).view(len(xs), 2, -1)
        video = self.video_embedder(
            torch.cat([x.squeeze(2).flatten(2).transpose(1, 2) for x in xs], 1)
        )
        ropes = [
            self.rope(s[-2], s[-1], nt, video.device).to(video.dtype)
            for s, nt in zip(shapes, tlens)
        ]
        text = self.text_embedder.proj(torch.cat(contexts, 1).to(video.dtype))
        text_rope = torch.cat([r[:, ni:] for r, ni in zip(ropes, ilens)], 1)
        tp = packed(tlens)
        for block in self.text_embedder.blocks:
            text = self._run(self.text_forward, block, text, text_rope, tp)
        x = torch.cat(
            [
                torch.cat((a, b), 1)
                for a, b in zip(video.split(ilens, 1), text.split(tlens, 1))
            ],
            1,
        )
        modulation = torch.cat(
            [
                mod[i : i + 1, :, None, :].expand(1, 6, n, -1)
                for i, n in enumerate(lengths)
            ],
            2,
        )
        rope = torch.cat(ropes, 1)
        jp = packed(lengths)
        for block in self.layers:
            x = self._run(self.block_forward, block, x, rope, modulation, jp)
        outputs = []
        for i, (part, ni, shape) in enumerate(zip(x.split(lengths, 1), ilens, shapes)):
            y = self.final_norm_video(part[:, :ni])
            y = y * (1 + head[i : i + 1, 1, None]) + head[i : i + 1, 0, None]
            outputs.append(self.final_linear_video(y).transpose(1, 2).reshape(shape))
        return (outputs if isinstance(hidden_states, list) else torch.cat(outputs),)
