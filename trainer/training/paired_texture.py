"""Paired latent texture loss, ported from sd_scripts/library/paired_texture_loss.py.

Mage-Flow uses the same noisy - sigma * velocity clean-latent reconstruction.
"""
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PairedTextureConfig:
    scale: float = 0.0
    levels: int = 3
    patch: int = 3
    tokens: int = 512
    features: int = 128
    seed: int = 1729
    snr_cut: float = 1.0 / 9.0
    min_gate: float = 0.0

    def __post_init__(self):
        if not math.isfinite(self.scale) or self.scale < 0:
            raise ValueError("texture scale must be finite and nonnegative")
        for name in ("levels", "patch", "tokens", "features"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"texture {name} must be a positive integer")
        if self.patch % 2 != 1:
            raise ValueError("texture patch must be odd")
        if not math.isfinite(self.snr_cut) or self.snr_cut <= 0:
            raise ValueError("texture snr_cut must be finite and positive")
        if not math.isfinite(self.min_gate) or not 0 <= self.min_gate <= 1:
            raise ValueError("texture min_gate must be in [0, 1]")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("texture seed must be in [0, 2**63)")

class PairedTextureLoss:
    def __init__(self, config):
        self.config = config
        self._projections = {}
        self._locations = {}

    def gate(self, sigma):
        sigma = sigma.detach().float().reshape(-1).clamp(0, 1)
        snr = ((1 - sigma) / sigma.clamp_min(1e-6)).square()
        return (snr / self.config.snr_cut).clamp(self.config.min_gate, 1)

    def bands(self, x):
        # Gaussian reduction creates octaves; raw band values are never pooled
        # to meet the token budget. Omit the coarse residual (mostly content).
        kernel = x.new_tensor([1, 4, 6, 4, 1]) / 16
        kernel = (kernel[:, None] * kernel[None, :])[None, None]
        kernel = kernel.expand(x.shape[1], 1, 5, 5)
        for _ in range(self.config.levels):
            if min(x.shape[-2:]) < 2:
                break
            low = F.conv2d(F.pad(x, (2, 2, 2, 2), mode="replicate"),
                           kernel, stride=2, groups=x.shape[1])
            yield x - F.interpolate(low, size=x.shape[-2:], mode="bilinear", align_corners=False)
            x = low

    def patches(self, x):
        b, c, h, w = x.shape
        key = (h, w, x.device)
        if key not in self._locations:
            generator = torch.Generator(device="cpu").manual_seed(self.config.seed)
            idx = torch.randperm(h * w, generator=generator)[:self.config.tokens].to(x.device)
            self._locations[key] = (idx // w, idx % w)
        rows, cols = self._locations[key]
        p = self.config.patch
        x = F.pad(x, (p // 2,) * 4, mode="replicate")
        windows = x.unfold(2, p, 1).unfold(3, p, 1)
        return windows[:, :, rows, cols].permute(0, 2, 1, 3, 4).reshape(b, -1, c * p * p)

    def embedding(self, tokens):
        dim = tokens.shape[-1]
        key = (dim, tokens.device)
        if key not in self._projections:
            generator = torch.Generator(device="cpu").manual_seed(self.config.seed)
            frequencies = torch.randn(dim, self.config.features, generator=generator)
            self._projections[key] = frequencies.to(tokens.device) / math.sqrt(dim)
        projected = tokens @ self._projections[key]
        # Paired sine/cosine features give a unit-norm map. Three bandwidths,
        # relative to sqrt(patch dimension) after clean-only normalization.
        return torch.cat([f(projected / width).mean(dim=1)
                          for width in (0.5, 1.0, 2.0) for f in (torch.cos, torch.sin)], dim=-1) / math.sqrt(3 * self.config.features)

    def per_sample(self, prediction, clean):
        if prediction.shape != clean.shape or prediction.ndim != 4:
            raise ValueError("texture loss expects matching [B,C,H,W] tensors")
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            prediction = prediction.float()
            clean = clean.detach().float()
            losses = []
            for pred_band, clean_band in zip(self.bands(prediction), self.bands(clean)):
                # The same detached target transform is applied to both sides.
                mean = clean_band.mean(dim=(-2, -1), keepdim=True)
                std = clean_band.var(dim=(-2, -1), unbiased=False, keepdim=True).sqrt().clamp_min(0.05)
                pred_features = self.embedding(self.patches((pred_band - mean) / std))
                with torch.no_grad():
                    clean_features = self.embedding(self.patches((clean_band - mean) / std))
                losses.append((pred_features - clean_features).square().sum(dim=-1))
            return torch.stack(losses).mean(dim=0) if losses else prediction.flatten(1).sum(1) * 0

    def apply(self, loss, velocity, clean, noisy, sigma):
        if self.config.scale == 0:
            return loss, None
        if velocity.ndim == 5:
            if velocity.shape[2] != 1:
                raise ValueError("paired texture loss requires one latent frame")
            velocity, clean, noisy = velocity.squeeze(2), clean.squeeze(2), noisy.squeeze(2)
        if velocity.ndim != 4 or velocity.shape != clean.shape or noisy.shape != clean.shape:
            raise ValueError("paired texture loss requires matching latent shapes")
        if sigma.numel() != clean.shape[0]:
            raise ValueError("paired texture loss requires one timestep per image")
        with torch.autocast(device_type=velocity.device.type, enabled=False):
            t = sigma.detach().float().reshape(-1, 1, 1, 1)
            predicted_clean = noisy.detach().float() - t * velocity.float()
            term = self.config.scale * (self.per_sample(predicted_clean, clean) * self.gate(sigma)).mean()
        return loss + term, term.detach()
