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
still supplies its architecture configuration. Fused-qkv and prequantized
inference exports are not accepted as training sources.

Separate components are also supported through the GUI's text encoder and VAE
file fields or `train.text_encoder_path` and `train.vae_path`:

- Qwen3.5 text encoders accept local Hugging Face directories or single-file
  BF16/FP32 and Comfy INT8 `int8_tensorwise` safetensors, including ConvRot.
  Single files use `model_path/text_encoder/config.json` and the package tokenizer
  (or `train.tokenizer_path`). The text-only model omits vision, generation-head,
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

The preset uses continuous logit-normal sampling with shift 1 as an initial
finetuning choice. It does **not** reproduce the paper's staged, truncated,
discrete logit-normal/uniform mixture. The inference scheduler's shift 5 is not
an instruction to use shift 5 for training.

## Scope

PEFT LoRA and LyCORIS LoCon/LoKr/DoRA reuse the trainer's adapter machinery with
native MOSS layer targets. Export retains native tensor names and records training
settings and source identity. Sharded sources record each file's MD5 plus an
explicitly labelled aggregate manifest checksum.

RTI, Self-Flow, dual-timestep noising, decoded perceptual/adversarial losses,
preservation, curricula, compiled text encoding, and the Mage embedding-only
shortcut are rejected. Reference-image editing and distilled variants are not
implemented. No convergence or image-quality claim follows from smoke tests.

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
