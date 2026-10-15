"""Optional decoded LPIPS supervision through a frozen FLUX.2/VSF3 decoder."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import os
import math
import json
import torch
from torch.utils.checkpoint import checkpoint


@dataclass
class PerceptualConfig:
    enabled: bool = False
    weight: float = 0.01
    max_t: float = 0.5
    every: int = 1
    cache_dir: str = 'output/perceptual-targets'
    eye_manifest: str = ''
    dedicated_low_noise: bool = False

    def __post_init__(self):
        if not math.isfinite(self.weight) or self.weight < 0:
            raise ValueError('perceptual.weight must be finite and nonnegative')
        if not 0 < self.max_t <= 1 or type(self.every) is not int or self.every < 1:
            raise ValueError('perceptual requires 0 < max_t <= 1 and every >= 1')
        if self.dedicated_low_noise and (not self.enabled or not self.eye_manifest
                                        or self.every != 1 or self.max_t <= .05):
            raise ValueError('Dedicated low-noise mode requires enabled eye supervision, every=1 and max_t>0.05')


def dedicated_flow_batch(clean, seed, ordinal, max_t):
    """Independent RNG leaves ordinary flow draws unchanged between matched runs."""
    generator = torch.Generator(device=clean.device).manual_seed(seed + 1000003 + ordinal)
    t = .05 + (max_t-.05)*torch.rand(1,device=clean.device,generator=generator)
    noise = torch.randn(clean.shape,device=clean.device,dtype=clean.dtype,generator=generator)
    return (1-t)*clean+t*noise, t, noise-clean


def unpack_flux2_latent(latent, vae):
    """Inverse of LatentCacher normalization, preserving autograd."""
    x = latent.squeeze(2)
    if getattr(vae, 'outputs_packed_flux2', False):
        return x
    bn = vae.bn
    x = x.float() * (bn.running_var[None,:,None,None].float() + bn.eps).sqrt()
    x = x + bn.running_mean[None,:,None,None].float()
    return torch.nn.functional.pixel_shuffle(x, 2)


def load_eye_regions(manifest_path):
    """Index approved boxes in exact training-bucket coordinates by latent path."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    regions = {}
    for row in manifest['images']:
        annotation = json.loads((manifest_path.parent / row['annotation']).read_text())
        source = Path(row['source'])
        if hashlib.sha256(source.read_bytes()).hexdigest() != annotation['source_sha256']:
            raise ValueError(f'Eye annotation source changed: {source}')
        for training in annotation['training']:
            width, height = training['size']
            eyes = training['eyes']
            if not 1 <= len(eyes) <= 2:
                raise ValueError(f'Eye manifest must contain only approved 1–2 eye entries: {source}')
            boxes = []
            for eye in eyes:
                x1, y1, x2, y2 = eye['xyxy']
                if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                    raise ValueError(f'Invalid eye box: {source}')
                ew, eh = x2-x1, y2-y1
                boxes.append((max(0, math.floor(x1-.5*ew)), max(0, math.floor(y1-.75*eh)),
                              min(width, math.ceil(x2+.5*ew)), min(height, math.ceil(y2+.75*eh))))
            latent = source.with_name(f'{source.stem}_{width:04d}x{height:04d}_trainer.safetensors')
            regions[str(latent.resolve())] = ((height, width), boxes)
    if not regions:
        raise ValueError('Eye manifest contains no approved regions')
    return regions


def eye_crop_loss(metric, prediction, target, boxes):
    """Equal eye weighting, native pixels, padding only for AlexNet minimum size."""
    terms = []
    for x1, y1, x2, y2 in boxes:
        a, b = prediction[..., y1:y2, x1:x2], target[..., y1:y2, x1:x2]
        pad = (0, max(0,64-a.shape[-1]), 0, max(0,64-a.shape[-2]))
        a = torch.nn.functional.pad(a, pad, mode='replicate')
        b = torch.nn.functional.pad(b, pad, mode='replicate')
        terms.append(metric(a,b).float().mean())
    return torch.stack(terms).mean()


class DecodedPerceptualLoss:
    def __init__(self, train, cfg, device):
        from ..modeling.loader import load_components, model_load_kwargs
        from ..modeling.checkpoint_metadata import checkpoint_identity
        import json
        if cfg.weight > 0:
            try:
                import lpips
            except ImportError as e:
                raise RuntimeError('Decoded perceptual loss requires pip install lpips') from e
        self.cfg, self.device = cfg, device
        self.eye_regions = load_eye_regions(cfg.eye_manifest) if cfg.eye_manifest else None
        # Model initialization must not perturb timestep/noise RNG of the baseline.
        with torch.random.fork_rng(devices=[device.index if device.index is not None else torch.cuda.current_device()]):
            self.vae = load_components(train.model_path, torch.bfloat16,
                **model_load_kwargs(train), load_transformer=False,
                load_text_encoder=False, load_tokenizers=False, load_vae=True).vae.to(device)
            self.metric = lpips.LPIPS(net='alex').to(device).eval().requires_grad_(False) if cfg.weight > 0 else None
        self.vae.eval().requires_grad_(False)
        if hasattr(self.vae, 'enable_gradient_checkpointing'):
            self.vae.enable_gradient_checkpointing()
        source = train.vae_path or str(Path(train.model_path)/'vae')
        identity = checkpoint_identity(source)
        signature = json.dumps({'source':identity['md5'], 'format':'vsf3-lpips-target-v1',
                                'dtype':'bfloat16'},sort_keys=True)
        self.cache = Path(cfg.cache_dir)/hashlib.sha256(signature.encode()).hexdigest()[:20]
        self.cache.mkdir(parents=True,exist_ok=True)
        self.micro = 0
        self.applied = 0
        self.raw_sum = 0.0
        self.last_term = None
        self.last_raw = None

    def decode(self, latent):
        x = unpack_flux2_latent(latent, self.vae).to(next(self.vae.parameters()).dtype)
        result = self.vae.decode(x)
        return (result.sample if hasattr(result, 'sample') else result).float()

    def target(self, latent):
        cpu = latent.detach().float().cpu().contiguous()
        key = hashlib.sha256(str(tuple(cpu.shape)).encode()+cpu.numpy().tobytes()).hexdigest()
        path = self.cache/(key+'.pt')
        if path.exists():
            return torch.load(path, map_location=self.device, weights_only=True).float()
        with torch.no_grad():
            target = self.decode(latent).detach()
        tmp = path.with_suffix(f'.{os.getpid()}.tmp')
        torch.save(target.cpu(),tmp)
        os.replace(tmp,path)
        return target

    def loss(self, predictions, noisy, clean, times, paths=None):
        self.micro += 1
        self.last_term = self.last_raw = None
        if self.cfg.weight == 0:
            return predictions[0].new_zeros(())
        if (self.micro-1) % self.cfg.every:
            return predictions[0].new_zeros(())
        eligible = [i for i,t in enumerate(times) if float(t.detach().item()) <= self.cfg.max_t]
        if self.eye_regions is not None:
            if paths is None:
                raise ValueError('Eye-local LPIPS requires batch paths')
            eligible = [i for i in eligible if str(Path(paths[i]).resolve()) in self.eye_regions]
        if not eligible:
            return predictions[0].new_zeros(())
        i = eligible[(self.micro-1) % len(eligible)]
        target = self.target(clean[i])
        boxes = None
        if self.eye_regions is not None:
            size, boxes = self.eye_regions[str(Path(paths[i]).resolve())]
            if tuple(target.shape[-2:]) != tuple(size):
                raise ValueError(f'Eye annotation size does not match decoded latent: {paths[i]}')
        z0 = noisy[i].float() - times[i].float().reshape(-1,1,1,1,1)*predictions[i].float()
        # Whole-branch checkpoint bounds live activations alongside the DiT graph.
        def objective(z):
            decoded = self.decode(z)
            if boxes is not None:
                return eye_crop_loss(self.metric, decoded, target, boxes)
            return self.metric(decoded,target).float().mean()
        raw = checkpoint(objective,z0,use_reentrant=False)
        self.last_raw = raw.detach().item()
        if not math.isfinite(self.last_raw):
            raise FloatingPointError('Nonfinite decoded perceptual loss')
        self.applied += 1
        self.raw_sum += self.last_raw
        self.last_term = self.cfg.weight*raw
        return self.last_term
