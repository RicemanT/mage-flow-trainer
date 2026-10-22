# Kohaku review follow-ups

Updated 2026-10-22. These are separate experiments and reliability improvements,
not a promise that copying them improves image quality.

## Completed foundation

- Equal-per-image gradient normalization across GPUs and accumulation, including uneven batches.
- Coordinated NaN/Inf rejection for ordinary optimizer updates; fail-fast guards for gradient release.
- Latent-cache VAE/content provenance and atomic cache writes, with warn/strict/off policies.
- Bounded checkpoint-input CPU offload, exact-view deduplication, parameter/buffer exclusions and optional prefetch.

Validation: 207 tests and 27 subtests passed; 14 opt-in tests skipped. Separate
DDP gradient checks passed, and a three-update MOSS LoCon smoke run completed on
two GPUs. See [implementation details](moss-image-experimental.md#global-image-weighting-invalid-updates-and-cache-identity).

## Remaining candidates, in suggested order


The local reference review identified these independent follow-ups:

1. **Fixed validation at several noise levels.** Reuse held-out batches, sigma
   values and noise seeds; report each sigma separately. This distinguishes
   learning from changes in timestep difficulty. Lower loss still does not prove
   better visual quality.
2. **Atomic training checkpoints and isolated RNG streams.** Our Accelerate resume
   already restores optimizer/scheduler/RNG state. Atomic checkpoint publication,
   dedicated sigma/noise generators and uninterrupted-versus-resumed tests would
   strengthen crash recovery and experimental control.
3. **A resolved adapter/group manifest.** List trained tensor names, counts, LR and
   decay explicitly. This would have made the 262-versus-263 target difference in
   our comparison obvious. Excluding norm/scalar parameters from weight decay is
   an optional recipe, not a proven correction.
4. **FP32 residual streams.** Keep residual additions in FP32 while linear/attention
   compute remains BF16. This may preserve small additions but costs activation
   bandwidth/memory; improved LoRA quality has not been established.
5. **MOSS-specific timestep curricula.** Discrete grids, uniform mixtures and
   staged distributions are experimental alternatives to our valid continuous
   logit-normal sampler. Long-run stage boundaries should not be copied unchanged
   into short adapter runs.
6. **LoRA+ factor-specific LR.** Give the up/zero-initialized factor its own LR
   multiplier while preserving component/scheduler ratios. Ratio 1 reproduces
   ordinary behavior; improved convergence or mixing still needs measurement.

These follow-ups should be tested individually. Kohaku's custom optimizer/fused
rollback engine, FSDP and pipeline infrastructure would be much larger ports.
