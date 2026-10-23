"""Discrete timestep distribution using the static Kohaku MOSS training recipe.

Independent numerical implementation of the probability law: condition
sigmoid(N(-mu, std)) on [0, cutoff], rescale to [0, 1], then quantize upwards
onto a grid. Mix with a uniform grid distribution and optional mass at t=1.
This is a static recipe, not the reference trainer's staged training schedule.
"""

import math
from functools import lru_cache

import torch


@lru_cache(maxsize=32)
def probability_table(mu=-1.09, std=0.95, cutoff=0.95, uniform_mix=0.01,
                      p_full=0.0, steps=1000):
    """Return CPU FP32 probabilities; callers must not mutate this cached tensor.

    Integrate normal probabilities in double precision, using the survival
    function in the upper tail to avoid cancellation near CDF=1.
    """
    def normal_coordinate(x):
        if x == 0:
            return -math.inf
        if x == 1:
            return math.inf
        return (math.log(x) - math.log1p(-x) + mu) / std

    edges = [normal_coordinate(cutoff * i / steps) for i in range(steps + 1)]
    root_two = math.sqrt(2)
    masses = []
    for left, right in zip(edges, edges[1:]):
        if left >= 0:
            mass = (math.erfc(left / root_two) - math.erfc(right / root_two)) / 2
        else:
            mass = (math.erfc(-right / root_two) - math.erfc(-left / root_two)) / 2
        masses.append(max(0.0, mass))
    total = math.fsum(masses)
    if total <= 0 or not math.isfinite(total):
        raise ValueError("MOSS timestep parameters leave no numerically representable mass below cutoff")
    probabilities = [(1 - p_full) * ((1 - uniform_mix) * m / total + uniform_mix / steps)
                     for m in masses]
    probabilities[-1] += p_full
    return torch.tensor(probabilities, dtype=torch.float32)


@lru_cache(maxsize=32)
def _cdf(mu, std, cutoff, uniform_mix, p_full, steps):
    cdf = probability_table(mu, std, cutoff, uniform_mix, p_full, steps).cumsum(0)
    cdf[-1] = 1.0
    return cdf


def inverse_cdf(quantiles, cfg):
    cdf = _cdf(cfg.moss_mu, cfg.moss_std, cfg.moss_cutoff, cfg.moss_uniform_mix,
               cfg.moss_p_full, cfg.moss_steps).to(quantiles.device)
    indices = torch.searchsorted(cdf, quantiles.contiguous()).clamp_max(cfg.moss_steps - 1)
    return (indices.float() + 1) / cfg.moss_steps
