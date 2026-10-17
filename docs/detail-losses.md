# SNR-gated HF and paired latent texture losses

Ported from `LoRA_Easy_Training_Scripts/backend/sd_scripts/library/hf_token_loss.py`
and `paired_texture_loss.py`. These are experimental auxiliary objectives for
Mage-Flow. Neither has been shown to improve eye quality in this trainer yet.
Both scales default to zero; importing this update does not enable either loss.

## HF loss: high-noise attenuation

The existing HF objective now has the source implementation's one-sided SNR gate.
For the flow timestep convention `z_t = (1-t) * clean + t * noise`,
`SNR = ((1-t)/t)^2`. Only the auxiliary HF term is multiplied by
`clamp(min(1, SNR/cutoff)^power, minimum_gate, 1)`.
The ordinary flow MSE is unchanged.

```toml
[flow]
hf_scale = 0.0 # opt in with a separately calibrated positive weight
hf_exponent = 1.0
hf_high_noise_snr_cut = 0.1111111111111111
hf_high_noise_min_weight = 0.1
hf_high_noise_power = 1.0
```

The defaults retain full HF weight through `t=0.75` and attenuate higher noise,
with a floor of 10%. **Existing configurations with HF enabled pick up this new
attenuation.** Set `hf_high_noise_min_weight = 1.0` to reproduce the previous
ungated HF objective. Spatial timesteps from dual-timestep noising receive a
spatial gate. Mage-Flow's HF token patch size remains 1.

GUI: **Flow**, alongside HF scale and exponent.

## Paired latent texture loss

The source's `anima_texture_*` options are exposed as `[paired_texture]`, with the
prefix removed (for example `anima_texture_scale` becomes `scale`). This is a
separate objective from `[dataset.texture]`, which controls crop preparation.

```toml
[paired_texture]
scale = 0.0 # disabled; a positive weight enables the experiment
levels = 3
patch = 3
tokens = 512
features = 128
seed = 1729
snr_cut = 0.1111111111111111
min_gate = 0.0
```

The loss reconstructs `clean_prediction = noisy - t * velocity`, forms Laplacian
detail bands, and compares fixed random Fourier statistics of signed local
patches against the clean target's statistics. Target-derived normalization is
detached. The coarse residual is omitted. Its independent SNR gate suppresses
high-noise supervision and reaches zero at pure noise by default.

It uses FP32 loss arithmetic, deterministic local random generators, and no VAE,
additional encoder, or extra transformer forward. Different native resolutions
are weighted equally per image. The `texture` training-log value includes the
configured scale. Configuration is included in the usual checkpoint metadata.

GUI: **Method → Paired Latent Texture Loss**. Start by testing this objective
alone against a matched control; a useful weight needs gradient calibration,
not just matching the scalar flow-loss value. Enabling both losses immediately
would make any improvement or regression harder to attribute.

Current support is for scalar timesteps and unmasked image batches, in either
packed or ordinary training. Config validation rejects combining it with
Self-Flow, dual-timestep noising, or texture curriculum phases. Scale zero skips
all auxiliary computation and projection allocation. The local patch size 3 is
independent of the transformer's token patch size.

## Validation

`tests/test_detail_losses.py` covers gates, gradient propagation, target
detachment, disabled behavior, small shapes, packed/unpacked training hooks,
configuration guards, and GUI serialization. `benchmarks/detail-loss-port/`
contains CPU/CUDA comparisons against the local reference, including a BF16
backward check. These validate implementation behavior, not generated quality or
full-training VRAM requirements.
