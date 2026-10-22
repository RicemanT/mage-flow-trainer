# Experimental MOSS-Image training

Native MOSS-Image Base/SFT text-to-image weights can be finetuned or adapted from
the GUI and CLI. Select `moss_image` as the model family (the official package's
`model_index.json` is also detected automatically).

**Support may be temporary.** Mage-Flow remains this trainer's primary focus.
If MOSS-Image sees little adoption or is abandoned by the open-source community,
its experimental support may be discontinued or removed. Long-term maintenance
is not guaranteed.

## Start

Use `configs/moss-image-finetune-experimental.toml` for full finetuning or
`configs/moss-image-locon-experimental.toml` for LoCon. Both presets default to
1024px resolution and a 1024 maximum bucket resolution; the memory measurements
below remain the original 512px tests. Set the dataset path and
`train.model_path` to a local package containing `transformer/`, `text_encoder/`,
`tokenizer/`, and `vae/`. Native single-file and sharded safetensors transformers
are accepted. `train.transformer_path` can override the transformer; the package
supplies its architecture configuration when present; otherwise the bundled
standard MOSS-Image configuration is used. Unquantized ComfyUI exports with fused
Q/K/V weights are split losslessly into native training tensors during loading.
Prequantized transformer exports are not accepted as training sources.

Separate components are also supported through the GUI's text encoder and VAE
file fields or `train.text_encoder_path` and `train.vae_path`:

- Qwen3.5 text encoders accept local Hugging Face directories or single-file
  BF16/FP32 and Comfy INT8 `int8_tensorwise` safetensors, including ConvRot.
  Single files use `model_path/text_encoder/config.json` when present, otherwise
  the bundled Qwen3.5-9B configuration. The tokenizer defaults to the package's
  `tokenizer/` or the bundled MOSS tokenizer and chat template; an explicit
  `train.tokenizer_path` takes precedence. Thus separate standard MOSS transformer,
  text encoder and VAE files do not require a downloaded model package. Select
  `model_family = "moss_image"` when no package is provided.
  The text-only model omits vision, generation-head,
  and multi-token prediction weights. INT8 weights are dequantized and ConvRot
  is reversed in FP32, then stored in the requested training dtype. This is
  **not INT8 execution**: BF16 text weights occupy approximately 15 GiB. Optional
  SDNQ text-encoder quantization can reduce memory, with additional quantization
  error. Cached Text Encoder mode releases the encoder after caching.
- FLUX.2 VAEs accept Diffusers directories and standalone Diffusers/original-LDM
  safetensors. Encoder and decoder widths are inferred independently, supporting
  both the regular and MOSS small decoder. No ComfyUI or `comfy_kitchen` import is
  needed. Latent packing and batch-normalization statistics remain unchanged.

Use the VAE that matches the base model, and rebuild latents when changing VAEs.
Filenames alone do not establish the decoder architecture or checkpoint identity.

```bash
python -m trainer.tools.cache_latents cache-config configs/moss-image-finetune-experimental.toml --batch-size 1
python -m trainer.training.train configs/moss-image-finetune-experimental.toml
```

Select GPUs in the GUI, or launch distributed training with Accelerate. Cached
latents and Cached Text Encoder embeddings allow the encoder and VAE to be
unloaded during training. Loaded Text Encoder operation is also implemented, but
its 9B encoder adds substantial VRAM; start with caching.

## Model and cache details

- Native 5.082B single-stream DiT, 42 joint blocks and two text-refinement blocks.
- Shared block modulation is trainable in full finetuning, with the existing
  AdaLN toggle and LR control; adapters exclude it.
- Qwen3.5-9B final hidden state: 4,096 features/token, with the official thinking
  chat template. Each caption is encoded without padding before cache storage;
  padded batch positions are excluded from transformer attention and RoPE.
- BF16 embeddings cost 8 KiB/token, or 4 MiB per unique 512-token caption.
  Caption augmentation and exact-caption deduplication use the existing SQLite
  cache with a distinct MOSS fingerprint. The preset cap is 512 tokens; the
  upstream model supports 4,096. Raising the cap can increase storage and VRAM.
- Packaged FLUX.2 VAE: 32 channels at /8, then 2x2 packing and BN normalization
  yield 128 channels at /16. The decoder uses widths 96/192/384/384; the encoder
  uses 128/256/512/512. VAE execution is FP32. **f8 is spatial compression, not FP8.**
- Use fresh caches with this package's VAE. Do not assume Mage/VSF3 caches match.

SDPA and PyTorch varlen attention support mixed resolutions and caption lengths.
Checkpointing and regional compile cover text-refinement and joint blocks.
MOSS compilation disables Inductor's mixed-order reduction fusion: at longer
packed sequence lengths its fused RMSNorm backward kernel can exceed the RTX
4090's per-block shared-memory limit (`No valid triton configs`). Compilation
and varlen remain enabled; this workaround does not change the loss or precision.
The transformer converts the trainer's normalized timestep to the native
0–1000 sinusoid input; RoPE frequencies remain FP32 after BF16 conversion.

SDNQ supports frozen mode for adapters and training mode for full finetuning.
The initial conservative policy protects conditioning, modulation, first and last
two joint blocks, and output layers: 234/269 Linear modules are quantized.
Quantized matmul is disabled in the presets. Optimizer state quantization and CPU
offload remain separate choices; neither is enabled in these presets.
Frozen SDNQ quantized matmul is rejected for adapter training: its inference
forward does not provide the required input gradients. This restriction does
not disable storage-only INT8/UINT8 or the separate experimental ConvRot backend.

The preset uses continuous logit-normal sampling with shift 1 as an initial
finetuning choice. It does **not** reproduce the paper's staged, truncated,
discrete logit-normal/uniform mixture. The inference scheduler's shift 5 is not
an instruction to use shift 5 for training.

## Pipeline audit against KohakuMossTrainer (2026-10-22)

Compared with KohakuMossTrainer commit
`da5dd97e15492e4dfb3caa171f023388004884ff` and the official MOSS pipeline.
The comparison covered loading, VAE normalization, conditioning, timesteps,
attention, loss reduction, adapters, optimizers, DDP, and checkpoint export.

The core model passes forward and parameter-gradient comparisons with the
official reference on a small FP32 model. The VAE mean-posterior encoding,
2x2 channel packing, BN normalization, Qwen thinking template, and final-hidden-state
conditioning agree with the reference design. MOSS timesteps remain FP32 before
the native 0–1000 conversion. Our default 512-token cap differs from Kohaku's
4096; longer captions require an explicit cap change and rebuilt text caches.

Recipe differences are intentional, not mathematical parity claims:

- Kohaku implements a discrete, truncated logit-normal mixture and staged
  schedules; our preset uses continuous logit-normal sampling.
- Kohaku normalizes by global token count. Our packed flow loss gives each image
  equal weight, so larger images do not automatically contribute more loss.
- Kohaku exposes FP32 residual streams and broader adapter target presets.
  Our BF16 execution and conservative target/quantization policy remain in place.
- Our MOSS encoder processes prompts individually to avoid padding effects;
  increasing the cache batch size does not turn that path into a padded Qwen batch.

The audit corrected three training issues:

1. Optimizer stochastic-rounding RNG is shared across DDP ranks for ordinary
   BF16 adapters as well as quantized full finetuning. Rank-specific data/noise
   RNG is preserved. Previously identical reduced gradients could produce
   different adapter weights on each GPU.
2. A partial final accumulation group is normalized by its actual microbatch
   count. Previously Accelerate's full-group divisor reduced that last update.
3. Explicit frozen SDNQ matmul is rejected because it lacks the input-gradient
   path needed by adapters. The normal storage-only presets remain supported.

When global gradient clipping is enabled, logs show the pre-clip `grad` norm
and `clip` multiplier. `clip 1.000` means no reduction. This measures the global
gradient cap, not SDNQ's internal normalized-update clamp.

LyCORIS is pinned to **4.1.3**, commit
`7abd5709f74f9159246ff9c6cc83bc0d5118b639`. Its upstream changes include isolated
tuning RNG and corrected BF16 kernel-selection checks. See the
[upstream changelog](https://github.com/KohakuBlueleaf/LyCORIS/blob/7abd5709f74f9159246ff9c6cc83bc0d5118b639/Change.md).
This update does not change the PyTorch or SDNQ pins, and is not a claim of
faster convergence or universally faster LoCon training.

Validation on two RTX 4090s with PyTorch 2.10 and SDNQ 0.2.4 included:

- Four-update, 512px MOSS LoCon runs with old and new LyCORIS on the same isolated
  eight-image subset, fresh latents, and cached captions. Losses closely agreed.
- A four-update MOSS DDP run with partial accumulation groups and LyCORIS 4.1.3;
  all trainable adapter tensors were byte-identical across ranks at export.
- Separate two-GPU BF16 SDNQ AdamW and Adafactor replica checks.
- Compiled, checkpointed native varlen versus SDPA comparisons on a small MOSS
  model: aggregate gradient differences were about 0.33% for full finetuning and
  0.58% for LoCon/LoKr in BF16, with finite gradients throughout.
- Compiled SDNQ adapter forward/backward and export/reload regression tests.

These are correctness smoke tests, not long-run convergence or visual-quality
benchmarks. Full-size finetuning, Windows fused kernels, and optimizer-state
resume were not newly GPU-validated in this audit. Latent caches still do not
embed a VAE checkpoint hash: rebuild them when changing the VAE. Local audit
scripts, measurements, and logs are under `benchmarks/moss-pipeline-audit/`.

## Scope

PEFT LoRA and LyCORIS LoCon/LoKr/DoRA reuse the trainer's adapter machinery with
native MOSS layer targets. Export records training
settings and source identity. Sharded sources record each file's MD5 plus an
explicitly labelled aggregate manifest checksum.

RTI, Self-Flow, dual-timestep noising, decoded perceptual/adversarial losses,
preservation, curricula, compiled text encoding, and the Mage embedding-only
shortcut are rejected. Reference-image editing and distilled variants are not
implemented. No convergence or image-quality claim follows from smoke tests.

## QKV saving format and existing LoRAs

**Training → Checkpoints → MOSS Q/K/V saving format** controls inference exports:

```toml
[train]
moss_qkv_format = "auto"  # default: fused for MOSS finetunes and plain LoRA/LoCon
```

Choose `separated` explicitly for native Q/K/V names, or `fused` to require fused
export. The `auto` default selects fused for MOSS full finetunes, PEFT LoRA, and
LyCORIS LoCon; LoKr, DoRA and other model families keep their native layout.
Explicit fused export with LoKr/DoRA is rejected. Training parameters and native resume state retain
their original layout. The selected format is recorded in checkpoint metadata.
Full checkpoints concatenate Q/K/V output rows; the trainer can reload either
layout.

For adapters, fusion stacks the down factors and uses a block-diagonal up factor
to preserve all three independent updates. Equal-rank Q/K/V adapters therefore
have three times the rank in their fused projection, with alpha adjusted to keep
the same scaling. Files grow; this does not increase the intended LoRA strength.
Unequal Q/K/V alpha-to-rank ratios are rejected rather than silently rescaled.

The repository-root utility converts existing plain MOSS LoRA/LoCon checkpoints:

```bash
python lora_qkv_convert.py                         # opens a small GUI
python lora_qkv_convert.py model.safetensors --format fused
python lora_qkv_convert.py model-qkv-fused.safetensors --format separated
```

Use `--output path.safetensors` to choose a filename. Otherwise it adds
`-qkv-fused` or `-qkv-separated` to the source name. Existing files are never
overwritten. Training metadata is preserved and conversion details are added.
Splitting retains the fused rank rather than applying a lossy rank reduction,
so converting back need not recover the original file size.

The audited local Forge loader needs fused adapters for its fused MOSS base.
The audited local ComfyUI loader already accepts both layouts. This is a loader
compatibility option, not a training-strength setting. Tests cover numerical
equivalence, full-checkpoint reload, metadata, GUI/TOML round trips, and preserving
the source file. The standalone converter also reproduced all 534 tensors of
the independently verified Modan epoch-35 fused test copy exactly.

## References

- [OpenMOSS MOSS-Image](https://github.com/OpenMOSS/MOSS-Image): architecture,
  pipeline, technical report and Apache-2.0 source (vendored license in
  `trainer/modeling/LICENSE.moss-image`).
- [AI Toolkit reference commit](https://github.com/ostris/ai-toolkit/commit/e05a3a4f2cc9bbbdd93407ad295f19edb6f89876): training integration cross-check.
- [ComfyUI MOSS support PR](https://github.com/Comfy-Org/ComfyUI/pull/17095): native
  checkpoint detection and inference integration. Apply or update ComfyUI to a
  version containing this support to load exported models.

## Initial validation (RTX 4090)

- Native FP32 output and parameter-gradient parity against the local upstream
  pipeline passed on a small model. Mixed-resolution/caption-length isolation
  and checkpoint parity also passed.
- Compiled BF16 varlen gradients differed from SDPA by 0.358% relative L2 on the
  small CUDA test; all gradients were finite.
- Real packaged VAE caching and Qwen3.5-9B text caching completed. The smoke
  image's caption retained 17 tokens (empty conditioning retained 10).
- All 5.082B parameters completed two single-GPU 512px updates with compiled
  varlen, checkpointing, Adafactor RMS clip and training-mode UINT8 SDNQ.
  The trainer reported 16.9 GB peak allocated memory and 1.22 s for the second
  update. The final native checkpoint exported successfully: 463 finite tensors.
- Rank-4 LoCon completed two single-GPU updates, with 6.6 GB reported peak
  allocation and 0.69 s for the second update. All 786 exported tensors were finite.
  PEFT LoRA and LyCORIS LoCon/LoKr/DoRA also passed small-model backward tests;
  modulation stayed frozen in every adapter test.
- Full finetuning completed **10 DDP updates**, batch 1 per GPU at 512px, with
  **15.72 GiB allocated / 16.09 GiB reserved on each GPU** and median 3.08 s/update
  after the first step. No CPU optimizer offload, first moment, Kahan or optimizer
  state quantization was used. DDP checkpoint saving was disabled for measurement.
- The exported full checkpoint's keys and shapes match the patched ComfyUI MOSS
  loader. Full exported-checkpoint image generation has not been tested.
- 77 regression tests and 10 subtests passed; GUI construction and both presets
  validated. Logs and detailed local results are in `benchmarks/moss-smoke/`.

These runs repeat one cached image and a short caption. They validate execution,
not convergence, larger-batch capacity or memory at the preset's 512-token cap.

## Global image weighting, invalid updates and cache identity

Every image now has equal weight across the complete optimizer update, including
unequal bucket batches on different GPUs and partial accumulation groups. Losses
are accumulated as image sums; gradients are divided by the actual global image
count before clipping. Logged flow and HF losses use the same weighting. This can
change training relative to the previous average of rank/microbatch means when
batch lengths differ. It does not change the objective to equal-token weighting.

Ordinary optimizer updates check the accumulated losses and global gradient norm
for NaN/Inf even when clipping is disabled. A failure on any DDP rank skips the
main optimizer, scheduler and EMA update on all ranks and clears gradients.
`train.nonfinite_patience = 3` stops after three consecutive failures. Checkpoints
record attempted steps, successful updates and skipped updates; scheduler time
advances only on successful updates. Auxiliary adversarial discriminator updates
are separate and are not rolled back by this guard.

Optimi gradient release has already updated earlier parameters when a later
backward hook detects a bad gradient. Its guards therefore **abort immediately**
without a final checkpoint rather than pretending to skip or roll back the update.

New latent caches record a SHA-256 VAE identity (a content manifest for directory
checkpoints), encoding precision, packing/normalization convention, preprocessing
version, posterior-mean mode, FP32 storage and source-image content hash. Cache
writes publish atomically. GUI caching and `cache-config` pass the validation
policy through to the CLI.

- `dataset.cache_validation = "warn"` is the default: legacy/unverifiable caches
  warn, while known VAE/convention mismatches fail. Source content changes also
  fail when the cache command checks an existing file.
- `"strict"` requires provenance and an available matching local VAE.
- `"off"` explicitly bypasses validation.

Moving the same VAE file preserves its identity. Converting its storage format or
changing its decoder/config can change the identity even with an equivalent
encoder: the check is deliberately conservative. Missing local VAE files remain
supported for latents-only training under `warn`, but cannot verify the intended
VAE. Known mixed-VAE caches are rejected. Re-cache with `--overwrite` to regenerate
provenance; old cache identities are never guessed or filled in retroactively.

## Bounded checkpoint-input offload

Available in the GUI under **Memory / Precision**, and for Mage-Flow, MOSS and
Flux3 through TOML:

```toml
[train]
gradient_checkpointing = true
compile = false
activation_offload = true
activation_offload_budget_mb = 1024
activation_offload_pin_memory = false
activation_offload_prefetch = false
```

The budget is live CPU tensor payload **per rank**, so two GPUs can use twice the
configured amount. It does not cap the whole process, allocator caches, temporary
copies or total system RAM. Inputs that exceed the budget remain on the GPU.
Identical saved views are deduplicated, parameter/buffer storage is excluded, and
nested checkpoint inputs are flattened so Python closures do not retain them on
the GPU. Logs report peak host payload and budget fallbacks.

Pinning is optional and disabled by default. Prefetch requires pinning and restores
one preceding checkpoint boundary ahead of backward; this can use additional
transient GPU memory. CPU copies are synchronous. There is no speedup guarantee:
this trades transfer time and RAM for activation VRAM. Compilation and Self-Flow's
alternate forward are currently incompatible and rejected explicitly.

## Other useful Kohaku ideas, not implemented by this change

The local reference review identified these independent follow-ups:

1. **Fixed validation at several noise levels.** Reuse held-out batches, sigma
   values and noise seeds; report each sigma separately. This distinguishes
   learning from changes in timestep difficulty. Lower loss still does not prove
   better visual quality.
2. **Atomic training checkpoint publication.** Dedicated per-rank timestep,
   noise and mask generators are now implemented and saved separately, with exact
   next-draw resume tests on CPU DDP and CUDA. Atomic publication of the complete
   model/optimizer checkpoint remains a separate crash-recovery improvement.
3. **A resolved adapter/group manifest.** List trained tensor names, counts, LR and
   decay explicitly. This would have made the 262-versus-263 target difference in
   our comparison obvious. Excluding norm/scalar parameters from weight decay is
   an optional recipe, not a proven correction.
4. **FP32 residual streams.** Keep residual additions in FP32 while linear/attention
   compute remains BF16. This may preserve small additions but costs activation
   bandwidth/memory; improved LoRA quality has not been established.
5. **MOSS-specific timestep curricula.** The static Kohaku recipe is now available
   as `flow.timestep_sample_method="moss"`, with a GUI probability preview and locked
   shared shift/scale. Staged schedules remain unimplemented; long-run stage
   boundaries should not be copied unchanged into short adapter runs.
6. **LoRA+ factor-specific LR.** Give the up/zero-initialized factor its own LR
   multiplier while preserving component/scheduler ratios. Ratio 1 reproduces
   ordinary behavior; improved convergence or mixing still needs measurement.

These follow-ups should be tested individually. Kohaku's custom optimizer/fused
rollback engine, FSDP and pipeline infrastructure would be much larger ports.

Validation for these changes: 207 tests and 27 subtests passed, with 14 opt-in
checks skipped. A separate two-rank GPU test matched concatenated-batch gradients
with unequal image counts and partial accumulation, and rejected a failure on
only one rank. A real MOSS UINT8-storage LoCon smoke run completed three DDP
updates at batch 2/rank and accumulation 2. Peak allocation was 12.2/11.9 GB;
peak live host payload was 1023.4/1020.1 MiB under a 1024 MiB/rank budget. This was
an execution check, not a convergence or offload-speed benchmark. Local artifacts:
`benchmarks/kohaku-safety-smoke/`.
