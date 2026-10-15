"""Experimental eye-local logistic GAN objective through a frozen decoder."""
from dataclasses import dataclass
from pathlib import Path
import math
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm
from torch.utils.checkpoint import checkpoint

@dataclass
class EyeAdversarialConfig:
    enabled: bool = False
    weight: float = .001
    discriminator_lr: float = .0001
    warm_start: str = ''

    def __post_init__(self):
        if not math.isfinite(self.weight) or self.weight < 0:
            raise ValueError('adversarial.weight must be finite and nonnegative')
        if not math.isfinite(self.discriminator_lr) or self.discriminator_lr <= 0:
            raise ValueError('adversarial.discriminator_lr must be finite and positive')

class EyeDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        layers=[]
        for a,b in [(3,32),(32,64),(64,128),(128,128)]:
            layers.extend([spectral_norm(nn.Conv2d(a,b,3,2,1)),nn.LeakyReLU(.2)])
        self.features=nn.Sequential(*layers)
        self.head=spectral_norm(nn.Linear(128*4*4,1))

    def forward(self,x):
        return self.head(F.adaptive_avg_pool2d(self.features(x),(4,4)).flatten(1)).flatten()


def eye_crops(image,boxes):
    return torch.cat([F.interpolate(image[...,y1:y2,x1:x2],(128,128),mode='bilinear',
                                   align_corners=False,antialias=True)
                      for x1,y1,x2,y2 in boxes])


def augment_pair(real,fake,seed):
    # Same spatial/color transform on paired real/fake crops, independent of training RNG.
    g=torch.Generator(device=real.device).manual_seed(seed)
    flip=bool(torch.rand((),device=real.device,generator=g)<.5)
    brightness=(torch.rand((),device=real.device,generator=g)-.5)*.1
    contrast=.9+torch.rand((),device=real.device,generator=g)*.2
    offsets=torch.randint(0,9,(2,),device=real.device,generator=g).tolist()
    def transform(x):
        if flip:x=x.flip(-1)
        x=F.pad(x,(4,4,4,4),mode='reflect')
        y,z=offsets
        return x[...,y:y+128,z:z+128]*contrast+brightness
    return transform(real),transform(fake)


def discriminator_update(model,optimizer,real,fake):
    model.train().requires_grad_(True)
    optimizer.zero_grad(set_to_none=True)
    real_score=model(real.detach());fake_score=model(fake.detach())
    loss=F.softplus(-real_score).mean()+F.softplus(fake_score).mean()
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite discriminator loss')
    loss.backward()
    if dist.is_initialized():
        for p in model.parameters():
            dist.all_reduce(p.grad);p.grad.div_(dist.get_world_size())
    norm=nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    model.eval().requires_grad_(False)
    return dict(d_loss=loss.item(),real_score=real_score.mean().item(),
                fake_score=fake_score.mean().item(),d_grad_norm=float(norm))


class EyeAdversarialLoss:
    def __init__(self,decoder,cfg,device):
        from safetensors.torch import load_file
        self.decoder,self.cfg,self.device=decoder,cfg,device
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(20260923)
            self.discriminator=EyeDiscriminator().to(device)
        if cfg.warm_start:
            self.discriminator.load_state_dict(load_file(cfg.warm_start,device=str(device)))
        self.discriminator.eval().requires_grad_(False)
        self.optimizer=torch.optim.Adam(self.discriminator.parameters(),lr=cfg.discriminator_lr,betas=(0.,.99))
        self.micro=0;self.last_term=None;self.last_raw=None;self.last_stats={}

    def loss(self,prediction,noisy,clean,t,path):
        self.micro+=1;self.last_term=self.last_raw=None
        size,boxes=self.decoder.eye_regions[str(Path(path).resolve())]
        target=self.decoder.target(clean)
        if tuple(target.shape[-2:])!=tuple(size):raise ValueError('Adversarial annotation size mismatch')
        real=eye_crops(target,boxes)
        z0=noisy.float()-t.float().reshape(-1,1,1,1,1)*prediction.float()
        with torch.no_grad():fake=eye_crops(self.decoder.decode(z0),boxes)
        rank=dist.get_rank() if dist.is_initialized() else 0
        aug_seed=700001+self.micro*101+rank
        real_aug,fake_aug=augment_pair(real,fake,aug_seed)
        self.last_stats=discriminator_update(self.discriminator,self.optimizer,real_aug,fake_aug)
        if self.cfg.weight==0:return prediction.new_zeros(())
        # D is frozen and eval through both the first pass and checkpoint recomputation.
        def objective(z):
            fake=eye_crops(self.decoder.decode(z),boxes)
            _,fake=augment_pair(real,fake,aug_seed)
            return F.softplus(-self.discriminator(fake)).mean()
        raw=checkpoint(objective,z0,use_reentrant=False)
        if not torch.isfinite(raw):raise FloatingPointError('Nonfinite generator adversarial loss')
        self.last_raw=raw.detach().item();self.last_term=self.cfg.weight*raw
        self.last_stats['g_loss']=self.last_raw
        return self.last_term
