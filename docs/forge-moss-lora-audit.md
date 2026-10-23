# Forge MOSS split-QKV LoRA loading audit

Checked 2026-10-22 against local Forge revision
`2c24de162d08a70829cd56bf6cd4bb4f60258b68` in
`/home/bluvoll/sd-webui-forge-classic/`. Forge source was not modified.

The saved Forge configuration selects `moss_image_base_bf16.safetensors`.
Its actual header contains `layers.*.proj.qkv.weight`: the joint blocks are fused.
Forge supports separate base Q/K/V weights too, but its local generic LoRA key
mapping does not route separate Q/K/V adapters into slices of a fused QKV weight.

Using Forge's unchanged key-mapping and LoRA-parser functions (isolated from UI
and GPU startup), the original Modan epochs 1, 10 and 35 each load 136 of 262
adapter modules against the fused layout. All 126 separate Q/K/V modules are
omitted. Against the native separate layout, all 262 map successfully.

The real Forge base header was also checked: epoch 35 leaves 378 unmatched tensor
keys. Forge logs unmatched keys at info level, but its rejection threshold is
more than half the file's tensor keys; 378/786 falls below that threshold. Thus
this is not literally an absence of logging, although loading proceeds partially.

## Magnitudes, not visual-strength percentages

Effective update norms include alpha/rank scaling and use FP64 low-rank Gram
products. Q/K/V matrices are disjoint portions of the parameter vector.

| Modan epoch | Full update L2 | Retained L2 | Omitted squared-norm share | Total L2 reduction |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1.398803 | 1.189304 | 27.71% | 14.98% |
| 10 | 37.578185 | 30.152598 | 35.62% | 19.76% |
| 35 | 63.630441 | 51.126584 | 35.44% | 19.65% |

These do not imply corresponding percentages of visual influence. Increasing
strength scales the surviving updates; it cannot restore missing Q/K/V updates.
No generation-based quality comparison has been performed in this audit.

## Requested epoch-35 test copy

Original preserved:
`output/modan-OpenMOSS-LoCON/modan-OpenMOSS-LoCON-epoch035.safetensors`

Converted file:
`output/modan-OpenMOSS-LoCON/modan-OpenMOSS-LoCON-epoch035-forge-fused-qkv.safetensors`

Also copied into Forge's `models/Lora/` under that converted basename.

For each of 42 blocks, concatenate the three down matrices along rank and place
three up matrices on a block diagonal. This produces rank 48 from three rank-16
adapters. Alpha becomes 15.59999942779541 (stored as a float64 scalar) so
alpha/rank exactly preserves the source scale 0.32499998807907104. No SVD,
truncation, retraining or strength adjustment was performed. All other 408 tensor
keys are unchanged.

All 178 resulting modules map and parse successfully against the actual Forge
base, with zero unmatched keys and matching base-weight dimensions. They cover
all original 262 module updates: three separate adapters now form one fused
adapter in each joint block. FP64 random-input projection probes across all 42
blocks agreed within 7.8e-16 absolute error. This verifies mathematical conversion
and loader coverage, not GPU bitwise identity or visual quality.

Compare original versus converted at identical LoRA strength, base, VAE, prompt,
seed, sampler, steps and CFG. Do not load both simultaneously. The converted file
is intended for the fused-QKV inference layout, not for resuming trainer weights.

Local scripts and JSON results: `benchmarks/forge-moss-lora-audit/`.

## ComfyUI comparison

The installed working copy at `/home/bluvoll/ComfyUI/` was checked separately.
Unlike the audited Forge copy, `comfy/lora.py` has MOSS-specific aliases from
separate Q/K/V names to row slices of a fused QKV weight.

A CPU-only test imported the real installed ComfyUI modules under its `venv` and
used the actual fused base's tensor shapes with a MOSS model shell:

- Original Modan epoch 35: all 262 patches parsed and registered, no unmatched-key warnings.
- All 126 Q/K/V patches target the correct three slices across 42 fused weights.
- The converted file loads all 178 combined modules.
- Original separate patches versus the converted fused patch produced identical
  FP32 merged updates in tested joint blocks 0, 20 and 41 (maximum absolute difference zero).
- The original also maps all 262 modules against the separate base layout.

This verifies standard loader mapping, patch registration and sampled numerical
merging, not a full image generation or every third-party loader node. ComfyUI
source and installed LoRAs were not changed. The original epoch-35 LoRA is already
appropriate for this installed standard ComfyUI loader.

Script and JSON results: `benchmarks/comfy-moss-lora-audit/`.
