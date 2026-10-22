# Kohaku review follow-ups

Updated 2026-10-22. These are separate experiments and reliability improvements,
not a promise that copying them improves image quality.

## Completed foundation

- Equal-per-image gradient normalization across GPUs and accumulation, including uneven batches.
- Coordinated NaN/Inf rejection for ordinary optimizer updates; fail-fast guards for gradient release.
- Latent-cache VAE/content provenance and atomic cache writes, with warn/strict/off policies.
- Bounded checkpoint-input CPU offload, exact-view deduplication, parameter/buffer exclusions and optional prefetch.
- Dedicated timestep/noise/mask streams per rank, honored generator arguments and
  rank-specific resume files. RNG changes pushed as `a5527c2`; 235 tests and 27
  subtests passed, plus exact next-draw CPU DDP and CUDA resume probes.

Validation: 207 tests and 27 subtests passed; 14 opt-in tests skipped. Separate
DDP gradient checks passed, and a three-update MOSS LoCon smoke run completed on
two GPUs. See [implementation details](moss-image-experimental.md#global-image-weighting-invalid-updates-and-cache-identity).

## Remaining candidates, in suggested order


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

## Follow-up review after the RNG fix

Reviewed the local reference implementation again on 2026-10-22. No new training
recipe changes were made during this review.

### Reliability: atomic checkpoint publication

Our `Trainer.save_full_state` writes Accelerate state into its final directory;
`export_checkpoint` writes the final inference filename directly; `state.json`
is written after inference export. A new interrupted save normally lacks the
state marker, but overwriting an existing tag can retain the old marker alongside
partially replaced files. An external checkpoint sync can also see a file while
it is still being written. This is a reliability gap, not a learning-strength
explanation.

Kohaku's `training/checkpoint/manager.py` writes a `.partial` directory, waits for
rank files, then publishes the completed directory. Its replacement path deletes
an existing final directory before rename; copying that behavior would leave a
replacement gap. Our port should preserve the previous complete checkpoint until
the replacement is ready, publish each inference file with an atomic rename,
and prune only after successful publication. Cross-file atomicity and power-loss
durability must not be claimed merely because individual renames are atomic.

Acceptance tests: interrupted writes leave the old checkpoint usable; partial
directories never appear resumable; all rank RNG files exist before publication;
failed saves never trigger pruning; repeated tags have an explicit safe policy.

### Diagnostics: resolved adapter/group manifest

Our optimizer metadata records group settings and parameter counts, but not the
tensor names. Kohaku's `training/optim/groups.py` retains those names. Record each
trainable tensor's name, shape, dtype, component, effective initial LR and decay,
plus adapter targets and the inference QKV layout. Assert that trainable parameters
appear exactly once in optimizer groups and frozen parameters appear zero times.
Keep full lists in a run artifact and a checksum/summary in checkpoint metadata.
This can expose target/group differences; it cannot verify an external loader's
application of the exported adapter without a separate loader check.

### Measurement: fixed held-out validation

We have no ordinary held-out validation callback; concept preservation is a
training regularizer, not a replacement. Kohaku's `callbacks/evaluate.py` evaluates
fixed batches at fixed sigma values with repeatable noise and restores model mode.
A suitable initial port would use cached held-out latents and fixed caption
embeddings, literal sigmas such as 0.25/0.5/0.75, and noise keyed by sample identity.
Report each sigma's ordinary flow MSE and its mean separately from training losses.

Do not directly reuse Kohaku's mean-of-batch-loss reduction: with unequal batch
sizes, our equal-per-image objective requires summed per-image losses divided by
the actual global image count. Avoid distributed padding duplicates in validation.
Validation must restore model mode, leave training RNG/optimizer/scheduler state
unchanged, and support uneven validation shards without mismatched DDP collectives.
Start periodically and with a small set; compare wall-clock cost as well as steps.
Validation loss complements fixed-seed image comparisons rather than proving eye
quality or visual strength by itself.

### Experiments that should remain opt-in

- **FP32 residuals:** Kohaku supports `fp32_residual`, but its BF16 training script
  defaults to `FP16_FIX="none"`. It is not automatically an explanation for a
  trainer gap. A trial must keep residual storage/additions FP32 while explicitly
  casting inputs back to the linear compute dtype, and measure peak VRAM and step
  time. Keeping select conditioning modules FP32 is a separate precision choice.
- **MOSS timestep recipe:** Kohaku's sampler uses a discrete grid, logit-normal
  parameters, a uniform mixture, optional pure-noise mass and staged tables. This
  changes the training objective's sampling distribution. Its report schedule has
  boundaries at 140k, 207k and 274k updates; those cannot serve as a useful schedule
  unchanged for a short LoRA run. Compare static recipes first, without changing LR.
- **LoRA+:** Kohaku's adapter grouping applies `LORAPLUS_RATIO` to kind-specific
  zero-initialized factors; its default is 1.0 (off). Our adapter factors currently
  share their component LR. A port should use explicit PEFT/LyCORIS factor mappings,
  preserve component/scheduler ratios and resume groups, and reproduce ordinary
  training at ratio 1. It is an optimization experiment, not a missing correctness
  requirement or a reason to change every user's defaults.

Recommended implementation order: atomic saves and the manifest for reliability
and visibility, fixed validation for controlled comparisons, then one numerical
or optimizer experiment at a time. None of the remaining recipe options has yet
been shown to explain the observed difference in LoRA strength.
