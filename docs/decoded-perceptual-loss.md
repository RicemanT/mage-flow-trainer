# Experimental decoded LPIPS supervision

Keep ordinary flow velocity MSE, and optionally add LPIPS on the decoded clean
prediction (`z0 = zt - t * velocity`). The FLUX.2/VSF3 VAE and AlexNet LPIPS
parameters remain frozen. Autograd passes through both to the transformer or
adapter. No inference architecture changes are needed.

Install the optional `perceptual` dependency group (or `pip install lpips==0.1.4`).
AlexNet pretrained weights must be available in the Torch cache or downloaded on
first use. The GUI exposes the settings under Method → Experimental Decoded
Perceptual Loss. They are also included in checkpoint config metadata.

```toml
[perceptual]
enabled = true
weight = 0.01 # illustrative; calibrate against adapter gradients
max_t = 0.5  # 0 is clean, 1 is noise
every = 1  # frequency per microbatch, per GPU
cache_dir = "output/perceptual-targets"
```

This initial implementation requires packed cached latents and `train.flux2_vae`.
It rejects dual timestep, Self-Flow, RTI, curriculum, preservation, HF loss, OT and
gradient release so the first experiment remains interpretable. It uses the full
image by default, with no face detector or eye annotation requirement.

One image is selected from timesteps at or below `max_t` on scheduled microbatches.
If none qualifies, only flow MSE applies. It does not resample timesteps or add
extra noisy examples. LPIPS is averaged over the selected image, independent of
microbatch size; it is not divided by the number of images in the batch.
Accelerate still averages gradients over accumulation and ranks. Different batch
sizes therefore change exposure to auxiliary supervision, even at equal effective
batch size. `lpips` in the training log is raw, unweighted LPIPS on that rank.

Targets are decoded from the actual clean cached latent, not the original source
image. They are cached lazily as FP32 tensors on disk, keyed by VAE checksum,
decoding-format version and latent contents/shape. At 1024 square this costs about
12 MiB per image plus file overhead. The target branch has no gradient. The
predicted branch uses activation checkpointing; this adds compute and decoder
memory. Model initialization preserves the baseline's training RNG state.

A low-noise auxiliary loss is a hypothesis for improving visible detail, not a
proven eye-geometry fix. Compare matched checkpoints and held-out prompts. An
improved training LPIPS score alone does not establish better generation quality.

## Houki first experiment (2026-09-23)

Completed 40 DDP updates (5 epochs), LoCon rank/alpha 64, Adafactor RMS at
2e-4, batch 10 per GPU, same VSF3-adapted base and fixed caption cache as the
baseline. Auxiliary weight 0.1, maximum timestep 0.5. A calibration sample at
t=0.3 measured a weighted LPIPS/flow adapter-gradient norm ratio of 0.00845
at weight 0.01; weight 0.1 makes that approximately 0.0845 on that sample.
This single-image calibration does not estimate the ratio for every packed batch.

Peak allocated memory: 17.4/17.3 decimal GB. Median step: 4.54 s
including the full step list (compile outliers are present). Rank 0 logged LPIPS
on 13 of 40 updates; rank 1 independently selects eligible images and its
LPIPS count is not logged. This is sparse auxiliary supervision, not 40 LPIPS
updates per image. All final adapter tensors are finite; the checkpoint embeds
the perceptual settings and exports no decoder/LPIPS weights.

The selected face's fresh encoding exactly matched its cached latent. Focused
tests: 14 passed. GUI parity: 338/338. No visual-quality improvement is claimed
until matched inference comparisons are evaluated.

## Eye-local supervision

Set `perceptual.eye_manifest` to an approved JSON manifest produced by
`trainer.tools.detect_eyes` to replace whole-image LPIPS with native-resolution,
padded eye crops. The GUI exposes this as **Eye annotation manifest**. An empty
path retains whole-image behavior.

The manifest indexes exact latent filenames and bucket dimensions. Original image
hashes are checked when loading annotations, and decoded dimensions must match.
Each approved entry must have one or two eyes. Boxes receive half an eye-width
of horizontal context on each side and three-quarters of an eye-height vertically.
Crops smaller than 64 pixels are replicate-padded for AlexNet; they are not
rescaled. Both eyes are equally weighted, then averaged to one per-image term.
This is box-local supervision, not a segmentation-masked LPIPS loss.

Only approved, low-noise samples are eligible for the auxiliary term. Images absent
from the manifest still receive ordinary flow loss. Startup reports annotation
coverage; zero coverage is an error. This allows the original dataset and ordinary
flow sampling to remain unchanged for matched experiments. The end-of-run summary
reports microbatch counts, auxiliary application counts and raw loss sums per rank.

The Houki diagnostic used ten reconstructed training images and found LPIPS
sensitive to synthetic blur and shifts, with much stronger pixel gradients from
local scoring. This does not establish a generation-quality benefit. Recalibrate
through the decoder and adapter when changing the crop policy or loss weight.

The eye-local trial config is `configs/houki-faces-LoCon-eye-lpips-1024.toml`:
151 ordinary-flow images, 138 approved eye annotations, weight 0.025, same
five-epoch/40-update DDP schedule as the previous baseline. Three eligible
calibration batches at weight 0.01 gave adapter-gradient ratios 0.0264–0.0381
(median 0.0303); weight 0.025 scales these to approximately 0.066–0.095.
These initial measurements do not describe every timestep or training stage.

The eye-local trial completed successfully: 40 updates, 17.4/17.3 decimal GB peak
allocated memory, median logged step 4.425 seconds. The auxiliary branch ran
12/40 and 11/40 microbatches on the two ranks. All 420 exported tensors were
finite. Subsequent user evaluation of the two portrait examples found no convincing
improvement over whole-image LPIPS. This motivates the dedicated low-noise pair. Full records live in `benchmarks/houki-eye-lpips/`.

## Dedicated low-noise matched experiment

`perceptual.dedicated_low_noise = true` adds one approved annotation example to
each packed forward. Ordinary examples retain their original timestep/noise draws
and mean flow loss. The extra example uses an independent RNG, uniform timestep
in `[0.05, max_t)`, and contributes flow MSE divided by configured batch size.
Eye LPIPS applies only to this extra example. With `weight = 0`, this is the
matched extra-flow-only control. Both runs use the same shuffled approved-image
order, sharded across ranks, and the same extra noise.

The experiment requires `enabled = true`, an eye manifest, `every = 1`, an integer
batch size and accumulation 1. Resume is currently rejected because the dedicated
sample counter is not checkpointed. Set the extra example's LPIPS weight separately
from its flow contribution; the first paired Houki test retains weight 0.025.

For the 10-image ordinary batch, the objective is:
`mean(ordinary flow losses) + 0.1 * extra flow loss + weight * extra eye LPIPS`.
The partial final batch retains the same 0.1 extra-flow coefficient. This preserves
the ordinary objective scale rather than averaging all 11 examples together.

The audit launcher at `benchmarks/houki-eye-dedicated/run.py` records initial
adapter hashes, source paths, caption hashes, timesteps, noisy-input and velocity
target hashes, plus CUDA RNG state before/after each forward. These logs allow
checking the matched-input claim after both runs. Audit hashing synchronizes and
copies tensors to CPU, so their step times are not a pure throughput benchmark.

The dedicated pair completed 40 updates each, with identical audited inputs and
initial adapter hashes. Both reached 18.5/18.4 decimal GB peak allocated VRAM.
The treatment applied eye LPIPS 40 times per rank (80 distinct extra images total),
versus zero for the control. All exported tensors were finite. Matched inference
comparison remains pending. Records: `benchmarks/houki-eye-dedicated/README.md`.
