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
by giving references their own size, so there is no pixel correspondence to copy.

Reference size: the Edit node uses references at the size they come in, so the workflow scales them first (Scale
Image to Total Pixels). Anima-Ref v2 keeps its reference at ~1 MP for every target size and recommends 1 MP; we train
with references between ~0.5 and 1 MP (random per step) and recommend 1 MP, so smaller references still work.

The stock LoRA loader keeps the LoRA on in every call, the reference stream included, so we train with the LoRA in
the reference pass too (Anima v2's `ref_lora`). Unlike Anima's hooks it is also on when a call has no reference;
about 10% of steps drop the reference so that call stays close to the base model, which a separate reference
guidance (an extra call without the reference) needs.

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

## What Anima-Ref v2 taught (2026-10-29)

v2 ([nodes](https://github.com/ElectronGirl/ComfyUI-Anima-Ref-Adapter), training code Apache-2.0 at
[anima-ref-adapter-research](https://github.com/ElectronGirl/anima-ref-adapter-research), log in its
`docs/experiments.md`) moved to the same asymmetric design as ours: reference self-attention only at timestep 0,
keys and values cached once per image (0.28 s per step against 0.58 s for its two-frame pilot, 0.24 s without a
reference).

- **Recipe of the released v2:** LoRA rank 64 / alpha 64 on attention and MLP, AdamW 1e-4, 100 warmup steps, cosine
  to 0.1x, batch 4, grad clip 1, flow shift 3; 15,600 images of 2,300 characters in 1,952 styles; targets 67% at
  ~1 MP and 33% at ~0.56 MP. It was released at only 1,000 of 3,000 steps (~1.1 h on an RTX PRO 6000). MOSS is 2.5x
  larger at the same token count, so expect ~3 h per 1,000 steps.
- **Style is the hard part.** Two style-only adapters (style slot; reference = same artist, other character) did not
  copy unseen styles. Two reasons: the flow-matching MSE barely sees style (reference effects sit at high sigma), and
  Anima's own renders of different artist tags look alike. Our data is real art, so the second reason does not
  apply; the first does. Identity transfers much better than style, and painterly styles barely transfer.
- **Contradictory pairs average out.** When the same reference setup sometimes meant "copy the style" and sometimes
  "don't", the adapter learned a mix. A prompt switch works: with the target's artist tags in the prompt the adapter
  keeps only the identity and follows the tags (their char_v2). Our `identity` pairs (other artist, trigger kept) and
  `identity_style` pairs (same artist, no trigger) are exactly that switch. Our `style` pairs are the kind that failed
  for them, so the pilot gives them a small share (~15%) and the evaluation decides whether they stay.
- **Measure with loss gaps, not only images:** held-out denoising loss at fixed sigmas (0.15, 0.35, 0.6, 0.85) with
  fixed noise. The identity gap is the loss with another character's reference (same style) minus the loss with the
  right character (other style); the copy gap is right character, other artist minus same character and artist.
- **Two characters** worked untrained from two references stitched side by side and a prompt saying who is on the
  left and who is on the right. MOSS takes several references natively (`r` = 1, 2, ...), and our layout files box each
  character, so a later version can train two-reference targets directly.
- Their planned fixes for style: pick pairs whose styles clearly differ (style-embedding distance), a style loss on
  the predicted x0 with a style encoder, or style-encoder tokens (IP-Adapter style).

## Next steps

1. Studio: a pair-builder export (`pairs.jsonl`: target, reference, optional crop box, reduced captions).
2. Trainer: a reference dataset and step (targets ~1 MP and ~0.56 MP, references 0.5-1 MP, ~10% reference dropout,
   prompts from the pair manifest), LoRA on MOSS (reference pass included), ComfyUI-loadable export, loss-gap eval.
3. Pilot: v2's recipe (rank 64, 1e-4, batch 4), checkpoints at 1,000 / 2,000 / 3,000 steps, one 96 GB GPU (~3 h per
   1,000 steps). Evaluate held-out characters with loss gaps and images (reference / adapter / no adapter / full tags).
   Pilot data: the MossAD-Demo groups. Their 57 artists are enough for identity but few for style; adding the
   accepted Danbooru batch 1 folders (pairs need tags, not captions) gives several hundred artists.
