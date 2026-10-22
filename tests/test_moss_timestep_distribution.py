"""Static MOSS recipe, RNG isolation, and config safeguards."""
import math

import pytest
import torch

from trainer.training.flow import FlowConfig, sample_timesteps
from trainer.training.moss_timesteps import probability_table
from trainer.training.config import load_config


def config(**kwargs):
    return FlowConfig(**dict(timestep_sample_method="moss", shift=1.0, **kwargs))


def test_default_recipe_probability_and_high_noise_tail():
    p = probability_table()
    t = torch.arange(1, 1001) / 1000
    assert p.dtype == torch.float32 and p.shape == (1000,)
    assert (p > 0).all()
    assert p.sum().item() == pytest.approx(1, abs=2e-7)
    # Checked against the reference's static default table, not its staged recipe.
    assert (t * p).sum().item() == pytest.approx(.74416944, abs=2e-7)
    assert p[950:].sum().item() == pytest.approx(.09250056, abs=2e-7)
    assert sample_timesteps(config(), 1, 8, 8, "cpu", quantile=.5).item() == pytest.approx(.781)
    assert sample_timesteps(config(), 1, 8, 8, "cpu", quantile=1).item() == 1
    assert p[-1] > .001  # pure-noise bin is present even with p_full=0


def test_untruncated_recipe_matches_logistic_normal_bin_integrals():
    p = probability_table(0, 1, 1, 0, 0, 32).double()
    expected = [0.0]
    for j in range(1, 32):
        t = j / 32
        expected.append(.5 * math.erfc(-math.log(t / (1 - t)) / math.sqrt(2)))
    expected.append(1.0)
    torch.testing.assert_close(p, torch.tensor(expected, dtype=torch.float64).diff(),
                               atol=1e-8, rtol=1e-6)


def test_uniform_mixture_and_extra_pure_noise_mass():
    p = probability_table(uniform_mix=1, p_full=.2, steps=10)
    torch.testing.assert_close(p, torch.tensor([.08] * 9 + [.28]))
    torch.testing.assert_close(sample_timesteps(config(moss_p_full=1), 24, 8, 8, "cpu"),
                               torch.ones(24))


@pytest.mark.parametrize("method", ["moss", "logit_normal", "uniform"])
def test_explicit_generator_reproducible_and_independent_of_global_rng(method):
    cfg = FlowConfig(timestep_sample_method=method, shift=1)
    state = torch.random.get_rng_state()
    a = sample_timesteps(cfg, 1000, 8, 8, "cpu", generator=torch.Generator().manual_seed(37))
    assert torch.equal(torch.random.get_rng_state(), state)
    b = sample_timesteps(cfg, 1000, 8, 8, "cpu", generator=torch.Generator().manual_seed(37))
    assert torch.equal(a, b)
    c = sample_timesteps(cfg, 1000, 8, 8, "cpu", generator=torch.Generator().manual_seed(38))
    assert not torch.equal(a, c)


@pytest.mark.parametrize("mapping", ["rescale", "truncate"])
def test_range_mapping(mapping):
    samples = sample_timesteps(config(phase_mapping=mapping), 200, 8, 8, "cpu",
                               generator=torch.Generator().manual_seed(42), t_range=(.2, .8))
    assert ((samples >= .2) & (samples <= .8)).all()


@pytest.mark.parametrize("options", [
    {"shift": 3}, {"sigmoid_scale": 3}, {"shift": None, "flux_shift": True},
    {"moss_std": 0}, {"moss_std": float("nan")}, {"moss_mu": float("inf")},
    {"moss_cutoff": 0}, {"moss_cutoff": 1.01}, {"moss_uniform_mix": -.1},
    {"moss_p_full": 1.01}, {"moss_steps": 1}, {"moss_steps": 1000.5},
])
def test_invalid_configuration(options):
    with pytest.raises(ValueError):
        FlowConfig(**({"timestep_sample_method": "moss", "shift": 1} | options))


def test_toml_defaults_and_conflict(tmp_path):
    path = tmp_path / "test.toml"
    content = '[dataset]\npath="/images"\n[flow]\ntimestep_sample_method="moss"\n'
    path.write_text(content)
    cfg = load_config(path)
    assert cfg.flow.shift == 1 and cfg.flow.sigmoid_scale == 1
    path.write_text(content + 'shift=3\n')
    with pytest.raises(ValueError, match="locks"):
        load_config(path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_generator_and_cpu_generator_can_sample_on_cuda():
    for device in ("cpu", "cuda"):
        a = sample_timesteps(config(), 32, 8, 8, "cuda",
                             generator=torch.Generator(device).manual_seed(21))
        b = sample_timesteps(config(), 32, 8, 8, "cuda",
                             generator=torch.Generator(device).manual_seed(21))
        assert a.is_cuda and a.dtype == torch.float32
        assert torch.equal(a, b)
