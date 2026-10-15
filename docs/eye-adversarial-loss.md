# Experimental eye-local adversarial supervision

This experiment keeps the transformer's ordinary flow objective, a frozen VSF3
VAE, and the dedicated low-noise example path. LPIPS weight must be zero: its
network is not loaded. A small FP32 convolutional discriminator learns from
matching eye-context crops of decoded clean targets and decoded predicted clean
latents. The adversarial gradient travels through the frozen decoder to LoCon.
No discriminator is needed for inference.

The discriminator has four stride-2 convolution layers (32, 64, 128, 128 channels),
LeakyReLU, spectral normalization, a 4x4 pooled representation and a scalar head.
Eye-context crops use the existing approved annotations and are bilinearly resized
to 128x128 with antialiasing on both sides. Both eyes receive equal weight.
Paired augmentation applies identical flip, small translation, brightness and
contrast changes to real/fake crops using an independent RNG.

The discriminator uses logistic softplus loss and FP32 Adam (betas 0, 0.99), with
its gradient norm capped at 1. It updates once per dedicated example. Under DDP,
its gradients are explicitly averaged across ranks; it has no BatchNorm. Spectral
normalization buffers stop updating during the generator pass. The discriminator
is frozen during the generator backward and its parameters are excluded from the
adapter optimizer and export. A final checksum verifies that rank replicas agree.

The generator uses `softplus(-D(fake))`. Its frozen-decoder branch is activation
checkpointed. A zero adversarial weight trains the shadow discriminator but adds
no discriminator-derived gradient to the transformer. This supports a matched
control. LPIPS stays off in both cases.

```toml
[perceptual]
enabled = true # shared decoded-supervision infrastructure
weight = 0.0 # LPIPS off, network not loaded
max_t = 0.5
every = 1
dedicated_low_noise = true
eye_manifest = "path/to/approved-training-manifest.json"
cache_dir = "output/perceptual-targets"

[adversarial]
enabled = true
weight = 0.00035 # experimental Houki calibration, not a general default
discriminator_lr = 0.0001
warm_start = "path/to/eye-discriminator.safetensors"
```

The GUI exposes the adversarial section and the shared decoded-supervision controls.
The dedicated path requires fresh runs, integer batch size and accumulation 1;
resume is not supported. The final discriminator is saved separately as
`eye-discriminator.safetensors`; adapter exports remain standard LoCon checkpoints.
Generator/optimizer settings are included in checkpoint metadata. D warm-start
provenance, split and audit records are stored with this experiment's artifacts.

## Houki trial protocol

151 source IDs were grouped by the filename before `_face-N`. There were no repeated
source IDs in this set. With fixed split seed 20260923, 30 sources were held out
from all LoCon and discriminator training. This is a holdout for this experiment;
it makes no claim about the base model's prior training data.

Training retains 121 ordinary images, of which 111 have approved eye annotations.
The holdout has 30 images, 27 with approved annotations. No source data was changed;
separate directories contain symlinks to original images, captions and cached latents.

The D warmup uses only training-source crops for 300 small batches of controlled
blur, resampling and smooth local shifts. Evaluation ranks intact above corrupted
crops on separate held-out sources. Passing this test establishes synthetic-defect
sensitivity, not eye-anatomy understanding or generation quality. Both training
runs start from the same warmup state and then train D against actual predictions.

Calibration through the actual VAE and adapter on three batches gave weighted
adversarial/flow gradient-norm ratios 0.0043, 0.0163 and 0.0286 at weight 0.00035.
The weight is fixed thereafter; these initial ratios are not guaranteed throughout
training. The same base, LoCon rank/alpha 64, Adafactor RMS LR 2e-4, warmup 10,
batch 10/GPU plus one dedicated example, and five-epoch schedule are used in both
runs. The held-out split makes this 35 updates per run, not the older trial's 40.

The ordinary flow mean is unchanged. The extra flow term has coefficient 0.1,
and only the experimental run adds the adversarial term. Audit hashes compare
all ordinary and extra noisy inputs, target velocities, captions, paths, timesteps,
initial adapter state and CUDA RNG state. Evaluation must compare this new control
against this new adversarial adapter. The old full-dataset control is secondary.

Artifacts and reproduction scripts: `benchmarks/houki-eye-adversarial/`.

Both trials completed 35 updates. All 420 adapter tensors were finite, discriminator
replicas matched across ranks, and the complete input audit passed. Peak allocated
VRAM was 18.5/18.5 decimal GB. Held-out reconstruction eye MSE was about 0.5–0.6%
higher with adversarial loss, while discriminator scores slightly preferred it.
Inspected crops showed only subtle differences; generation quality remains unproven.
See `benchmarks/houki-eye-adversarial/README.md` for checkpoints and evaluation.
