"""MOSS conditioning must retain the FP32 times used for noising."""
from types import SimpleNamespace

import pytest
import torch

from trainer.training import train
from trainer.training.config import Config
from trainer.data.dataset import DatasetConfig


@pytest.mark.parametrize('packed', [False, True])
def test_moss_training_passes_unrounded_timesteps(monkeypatch, packed):
    trainer = train.Trainer.__new__(train.Trainer)
    from trainer.training.flow_rng import FlowRNG
    trainer.flow_rng = FlowRNG(42, 0, 1, 'cpu')
    trainer.cfg = Config(dataset=DatasetConfig(path="/test"))
    trainer.cfg.train.model_family = 'moss_image'
    trainer.cfg.train.pack_resolutions = packed
    trainer.dtype = torch.bfloat16
    trainer.accelerator = SimpleNamespace(device=torch.device('cpu'))
    trainer.self_flow_ema = trainer.phase = trainer.paired_texture = None
    trainer.hf_patch = trainer.rti_interface = trainer.adversarial = trainer.perceptual = None
    trainer._encode = lambda captions: None
    trainer._preserve_term = lambda: 0
    times_used = []

    def prepare(latents, cfg, **kwargs):
        t = torch.full((latents.shape[0],), .501, dtype=torch.float32)
        times_used.append(t)
        target = torch.ones_like(latents)
        noisy = latents + t.view(-1, 1, 1, 1, 1) * target
        return noisy, t, target, None, None

    monkeypatch.setattr(train, 'prepare_training_flow_batch', prepare)

    def forward(hidden_states, timestep, **kwargs):
        assert timestep.dtype == torch.float32
        torch.testing.assert_close(timestep, torch.cat(times_used), rtol=0, atol=0)
        torch.testing.assert_close(timestep * 1000, torch.full_like(timestep, 501))
        assert (timestep.bfloat16().float() * 1000 == 500).all()
        if packed:
            return ([torch.zeros_like(x) for x in hidden_states],)
        return (torch.zeros_like(hidden_states),)

    trainer.transformer = forward
    x = torch.zeros(2, 128, 1, 2, 2)
    batch = dict(latents=list(x.split(1)) if packed else x,
                 captions=['a', 'b'], paths=['a', 'b'])
    loss, _ = trainer._step(batch)
    assert torch.isfinite(loss)
