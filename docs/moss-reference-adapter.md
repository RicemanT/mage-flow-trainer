# MOSS-Image reference adapter (plan and status)

Goal: give MOSS-Image one reference image and a prompt that only describes the scene; the adapter keeps **who**
is drawn (character design) and **how** (the artist's style), while pose, background and composition follow the
prompt. Inspired by [Anima Reference Adapter](https://github.com/ElectronGirl/ComfyUI-Anima-Ref-Adapter)
(Apache-2.0 nodes, technical report), adapted to MOSS and to real, curated data instead of synthetic pairs.

## Layout: ComfyUI's MOSS reference latents (no custom model code at inference)

ComfyUI's MOSS-Image model already reads reference latents (`comfy/ldm/moss_image/model.py`). We train in its
**Moss-Image-Edit layout** (`reference_latents_method = "index_timestep_zero_text_start"`):

| | image being generated | text | reference i |
|---|---|---|---|
| RoPE (r, f, h, w) | (0, T, h, w), centered | (0, 0..T-1, 0, 0) | (i + 1, 0, h, w), centered on its own grid |
| timestep | t | t | 0 (clean) |
| attends to | image, text, references | image, text, references | its example's references only |
| loss | yes | - | - |

The references attend only to each other, so their keys and values do not depend on the image: ComfyUI computes them
once per generation (`preprocess_refs`) and appends them to every block's keys and values. The image and text keep
exactly the positions of plain text-to-image, which is also what this trainer uses, and the references do not go
through the text encoder, so cached text embeddings keep working.

Inference in stock ComfyUI: **Text Encode MOSS-Image Edit** (format `Moss-Image-Edit`, with the VAE and the reference
image) for the positive and the negative prompt, the adapter LoRA through a normal LoRA loader. Optional later: a
guider with separate text and reference guidance plus APG, as the Anima adapter's (Apache-2.0).

Anima's adapter shifts the reference's RoPE by one image height so its video prior cannot copy the layout. MOSS keeps
references on their own `r` axis instead; layout copying is avoided by training only on pairs whose scenes differ and
by giving references their own size (about 512 px area), so there is no pixel correspondence to copy.

**Status: done in the model.** `MOSSImage.forward(..., ref_latents=[[ref, ...] or None per example])` runs a reference
pass at timestep 0 (references packed per example, attending only to each other, keys and values kept per block) and
the main pass with each example's reference keys and values appended (`_attention(ref=...)`, varlen and SDPA).
`tests/test_moss_reference.py` checks it against one dense sequence with per-token timesteps and a mask (equal to
1e-10), packing with and without references, and gradients through the reference pass.

## Data: real pairs from the curated dataset

Pairs come from the Studio's curated folders (pilot: the MossAD-Demo groups, 3,685 images, 57 artists; 445 named
characters with at least 3 images, about 27k ordered pairs). One reference slot carries whatever the prompt leaves out:

| pair | reference | target prompt keeps | the reference teaches |
|---|---|---|---|
| same character, same artist | another image of it | scene tags | identity + style |
| same character, other artist | another artist's image of it | the target's artist trigger + scene | identity |
| same artist, other character | another image by the artist | the characters + scene | style |

Prompt reduction (as Anima's 70 / 20 / 10): scene only / plus a random 30-70% of the reduced tags / minimal. Reduced
tags: character and copyright tags, the artist trigger when the reference supplies the style, and the character's
usual look (tags on at least 40% of its images, as the Studio's character facts) and outfit (30%). The layout files'
character boxes allow cropping one character out of a group image as a clean reference. Hold out about 15 characters
and 3 artists for evaluation.

## Next steps

1. Studio: a pair-builder export (`pairs.jsonl`: target, reference, optional crop box, reduced captions).
2. Trainer: a reference dataset and step (target latents at the training tier, reference latents at ~512 px area,
   captions from the manifest), LoRA on MOSS, ComfyUI-loadable export.
3. Pilot: LoRA rank 32-64, about 2-3k steps, one 80-96 GB GPU (~10-15 h); evaluate held-out characters as Anima's
   report does (reference / adapter / no adapter / full tags), then retrain on the full MossAD checkpoint later.
