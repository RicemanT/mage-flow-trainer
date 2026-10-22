import math
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from trainer.gui.timestep_plot import distribution
from trainer.training.flow import FlowConfig


@pytest.mark.parametrize("method", ["uniform", "logit_normal", "moss"])
def test_preview_probabilities_normalize(method):
    preview = distribution(FlowConfig(timestep_sample_method=method, shift=1))
    assert len(preview.probabilities) == 40
    assert sum(preview.probabilities) == pytest.approx(1, abs=1e-7)
    assert min(preview.probabilities) >= 0


def test_preview_uniform_and_shifted_uniform_analytic():
    p = distribution(FlowConfig(timestep_sample_method="uniform", shift=1))
    assert p.probabilities == pytest.approx([1 / 40] * 40)
    assert p.high_noise == pytest.approx(.1)
    p = distribution(FlowConfig(timestep_sample_method="uniform", shift=3))
    assert p.median == .75
    assert p.high_noise == pytest.approx(.25)


def test_preview_logit_normal_tail_and_flux_grid():
    p = distribution(FlowConfig(shift=1))
    assert p.high_noise == pytest.approx(.5 * math.erfc(math.log(9) / math.sqrt(2)))
    cfg = FlowConfig(shift=None, flux_shift=True)
    low, high = distribution(cfg, 16, 16), distribution(cfg, 64, 64)
    assert high.median > low.median
    assert high.high_noise > low.high_noise


def test_preview_discrete_mass_at_pure_noise_and_no_extra_atom():
    cfg = FlowConfig(timestep_sample_method="moss", shift=1, moss_p_full=1)
    p = distribution(cfg)
    assert p.median == 1 and p.high_noise == 1
    assert p.probabilities == (0,) * 39 + (1,)
    regular = distribution(FlowConfig(timestep_sample_method="moss", shift=1))
    assert regular.probabilities[-1] > .04
