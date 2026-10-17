# Experimental Flux3 training

The trainer can load the native **BF16 Flux 3 Image Klein Base** package and train
text-to-image adapters or full transformer weights. This is an initial integration;
short smoke tests do not establish convergence or generation quality. Image editing/reference inputs and distilled
flow-map checkpoints are not supported.

## Configuration

Start with `configs/flux3-locon-experimental.toml`. Set the dataset path and the
package directory (`train.model_path`). The package must contain:

- `config.json` and native `model.safetensors`;
- `text_encoder/` with Qwen3-VL weights and tokenizer;
- `vae/` with the Diffusers AutoencoderKLFlux2 configuration and weights.

Set `train.model_family = "flux3"`, also available in the GUI's model section.
Auto detection recognizes the root package configuration. A separate native BF16
transformer can be selected with `train.transformer_path`; keep `model_path`
pointing to the original package for architecture and component configuration.
Diffusers transformer shard imports and prequantized inference exports are not
implemented. The existing Mage-Flow loader remains available.

```sh
python -m trainer.tools.cache_latents cache-config configs/flux3-locon-experimental.toml --batch-size 1
python -m trainer.training.train configs/flux3-locon-experimental.toml
```

**Use separate latent caches for Flux3.** The packaged FLUX.2 VAE is loaded in
FP32, with 2x2 latent packing and its own batch-normalization statistics. Existing
Mage-VAE or VSF3 caches are not interchangeable. This integration does not
retroactively verify which VAE produced pre-existing cache files.

Qwen conditioning uses layers 4, 8, 12, 16, 20, 24, 28, 32, concatenated in that
order. Captions use the user/assistant template without a think block; each prompt
is padded to a multiple of 80, capped at `max_text_tokens`. The DiT retains those padding embeddings. Text
caching has a separate fingerprint, preventing reuse of Mage-Flow embeddings.
Encoding currently processes captions individually for padding consistency;
`compile_text_encoder` and the Mage embedding-only shortcut are rejected for this experimental path.

## Training and memory

- `attention_backend = "torch_varlen"` packs examples into PyTorch varlen attention,
  with separate sequence boundaries in image, text, and joint blocks. Each example
  retains its own RoPE coordinates and timestep modulation. Retained text-padding
  embeddings are preserved. `pack_resolutions = true` allows mixed image sizes.
- `attention_backend = "sdpa"` remains available: examples are evaluated separately
  at their native latent dimensions. Both paths average the image losses equally.
- Non-reentrant block checkpointing and optional block compilation are supported.
- PEFT LoRA and LyCORIS LoCon/LoKr/DoRA use native projection names. Modulation is
  excluded from adapters. The unused reference-image branch stays frozen in
  text-to-image finetuning and receives no adapters.
- Full finetuning uses `adapter.kind = "none"` and, for quantized training,
  `quant.mode = "training"`. LoRA uses `quant.mode = "frozen"`.
- SDNQ converts **BF16 source weights** using its own training/frozen machinery.
  The supplied FP8/NVFP4 packages are inference formats, not directly importable
  SDNQ weights. Their exclusion lists informed a conservative policy: protect
  inputs/outputs, time/modulation, text blocks, unused reference blocks, the first
  image block, and the final four joint blocks. Inspect the runtime layer report.
  This protection is applied in addition to SDNQ's generic exclusions.
- Start with storage-only UINT8 and quantized matmul off. No claim is made that
  the vendor's FP8/NVFP4 calibration transfers to SDNQ or that NVFP4 training works.
- HF and paired texture operate on the common latent layout, but their quality
  benefit has not been validated for Flux3.
- RTI, Self-Flow, dual timesteps, decoded perceptual/adversarial losses,
  preservation probes, and curriculum phases are rejected for now.

Checkpoint exports retain native transformer names and include `model=flux3` in
metadata. Load adapters on the matching Flux3 base in an inference implementation
that supports native Flux3 adapter keys; an actual ComfyUI adapter-generation
check is still needed. Full exports keep the inactive editing weights but have
not been trained or evaluated for editing.

The model package has its own Flux Kommunity License; weights are not included in
this repository. Respect the license distributed with the downloaded package.

## Initial validation

- All 357 native checkpoint tensor keys and shapes match the local BF16 package.
- Small FP32 transformer forward agrees with ComfyUI to maximum absolute error
  4.8e-7 (relative L2 1.4e-7). This is not full-size BF16 numerical parity.
- Real 5.5B model: 256px VAE cache, Qwen cache, two compiled SDNQ UINT8 LoCon
  trainer updates and export completed on one 4090. 168/271 Linear layers were
  quantized, peak allocated VRAM was 8.9 GB, and all 684 adapter tensors were finite.
- Small-model full finetuning passed two SDNQ training-mode UINT8 / Adafactor
  updates with finite gradients. Full-size 1024px finetuning also completed 10 updates with cached short captions,
  UINT8 SDNQ, Adafactor, checkpointing and compile: 17.19 GiB allocated on one GPU,
  and 17.93/17.92 GiB on two GPUs with batch 1 per GPU (SDPA, no CPU offload).
  Checkpoint saving was excluded from the memory measurements.
- CPU regression suite: 43 tests passed, including text padding/layer selection,
  backward checkpoint parity, native-resolution isolation and adapter exclusions.
  GUI construction and serialization checks also passed.
- ComfyUI generation with the exported adapter remains unverified. No image-quality
  improvement is claimed.

## Official recipe cross-check

Checked against the [Hugging Face Flux3 article](https://huggingface.co/blog/flux-3)
and its [DreamBooth training script](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora_flux3_klein.py).
The Hugging Face DreamBooth script defaults to uniform draws (`weighting_scheme="none"`)
and also supports logit-normal draws. This does not establish BFL's original
pretraining distribution: BFL's released repository contains inference code.
Its evenly spaced schedule with fixed shift 6.93 is an **inference schedule**, not
a requirement to draw uniform random training timesteps.

Our example uses `timestep_sample_method="logit_normal"` and configurable shift
6.93. This retains the trainer's usual training distribution rather than claiming
to reproduce BFL pretraining. The previous switch to uniform was unnecessary.
Diffusers 0.41.0.dev0, revision `7ecb1e1b4fe500be7fb7381a7bb79a24d7fa3941`,
was installed and checked directly; the environment lockfile records that revision.


The prompt template matches the packaged tokenizer. Text windows are rounded to
80 tokens and capped at the requested maximum; changing this policy updates only
the Flux3 embedding-cache fingerprint, because the DiT attends to retained pads.
Mage text caches are unaffected.

The article describes joint ordering as text/image/reference, whereas our native
path follows ComfyUI's image/reference/text order. Full, non-causal attention is
permutation-equivariant when RoPE positions, modulation and output selection move
with each stream, so this ordering alone does not require a model change.

## Packed varlen validation

At 512px, full text-to-image finetuning completed 10 updates at batch 2 per GPU
with Cached Text Encoder embeddings, UINT8 SDNQ, Adafactor RMS clip, checkpointing
and regional compile. No CPU offload was used. Two RTX 4090s peaked at
17.27 / 17.30 GiB allocated and 17.45 / 17.51 GiB reserved; median step time
after the first compiled step was 2.93 seconds (effective batch 4).

The controlled dataset repeats one cached image with an 80-token caption; the
configured text cap was 128. Checkpoint saving was disabled. This is a memory and
execution check, not evidence of convergence or a guarantee for longer captions.
Single-GPU batch-2 SDPA and varlen both peaked at 17.18 GiB allocated. Varlen
measured 1.32 s/step versus 1.61 s for SDPA on different cards, so those timings
are indicative rather than a controlled same-device speedup measurement.

CPU tests check mixed image/caption lengths, forward and gradient parity,
checkpointing and example isolation. A compiled BF16 CUDA test against SDPA
passed with 0.234% relative L2 gradient difference.
