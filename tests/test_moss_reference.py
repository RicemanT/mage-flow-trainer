"""MOSS-Image reference latents: the packed two-pass forward equals one dense sequence in which the references ride
along at timestep 0 and attend only to each other (ComfyUI's MOSS reference layout, as Moss-Image-Edit uses it)."""
import torch
import torch.nn.functional as F

from trainer.modeling.moss_image import MOSSImage, MOSSParams, _rotate, _situ

P = MOSSParams(dim=64, ffn_dim=96, num_heads=4, num_layers=2, text_dim=32, freq_dim=32, latent_channels=8)


def model():
    torch.manual_seed(0)
    m = MOSSImage(P).double()
    for p in m.parameters():   # non-zero modulation so timestep 0 and t really differ
        torch.nn.init.normal_(p, std=0.2)
    m.configure_execution(gradient_checkpointing=False, attention_backend="sdpa")
    return m.eval()


def time_modulation(m, t):
    half = P.freq_dim // 2
    freq = 10000 ** (-torch.arange(half, dtype=torch.float64) / half)
    phase = (t * 1000).double()[:, None] * freq
    time = m.time_embedding(torch.cat((phase.cos(), phase.sin()), -1))
    return time, m.shared_block_modulation(time).view(1, 6, 1, -1)


def dense(m, latent, t, context, refs):
    """One sequence [image, text, refs]: per-token modulation (refs at t = 0) and a mask (refs see only refs)."""
    H, W = latent.shape[-2:]
    T = context.shape[1]
    video = m.video_embedder(latent.squeeze(2).flatten(2).transpose(1, 2))
    rope_main = m.rope(H, W, T, latent.device)
    text = m.text_embedder.proj(context)
    for block in m.text_embedder.blocks:
        text = block(text, rope_main[:, H * W:], ([T], None, T))
    ref_tokens = m.video_embedder(torch.cat([r.squeeze(2).flatten(2).transpose(1, 2) for r in refs], 1))
    rope = torch.cat((rope_main, m.rope.references([r.shape[-2:] for r in refs], latent.device)), 1)
    x = torch.cat((video, text, ref_tokens), 1)
    main, R = H * W + T, ref_tokens.shape[1]
    time, mod_t = time_modulation(m, t)
    _, mod_0 = time_modulation(m, torch.zeros(1))
    modulation = torch.cat((mod_t.expand(1, 6, main, -1), mod_0.expand(1, 6, R, -1)), 2)
    mask = torch.ones(main + R, main + R, dtype=torch.bool)
    mask[main:, :main] = False
    for block in m.layers:
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.unbind(1)
        y = block.norm1(x) * (1 + scale_a) + shift_a
        q = block.q_norm(block.proj["q"](y).unflatten(-1, (block.heads, -1)))
        k = block.k_norm(block.proj["k"](y).unflatten(-1, (block.heads, -1)))
        v = block.proj["v"](y).unflatten(-1, (block.heads, -1))
        q, k = _rotate(q, rope).transpose(1, 2), _rotate(k, rope).transpose(1, 2)
        attended = F.scaled_dot_product_attention(q, k, v.transpose(1, 2), attn_mask=mask).transpose(1, 2).flatten(2)
        x = x + block.proj["o"](attended) * gate_a
        y = block.norm2(x) * (1 + scale_f) + shift_f
        hidden, gate = block.proj["ffn_up_gate"](y).chunk(2, -1)
        x = x + block.proj["ffn_down"](_situ(hidden, gate)) * gate_f
    head = m.video_head_modulation(time).view(1, 2, -1)
    out = m.final_norm_video(x[:, :H * W]) * (1 + head[:, 1, None]) + head[:, 0, None]
    return m.final_linear_video(out).transpose(1, 2).reshape(latent.shape)


def inputs(seed, h=3, w=4, text=5):
    g = torch.Generator().manual_seed(seed)
    latent = torch.randn(1, P.latent_channels, 1, h, w, generator=g, dtype=torch.float64)
    context = torch.randn(1, text, P.text_dim, generator=g, dtype=torch.float64)
    refs = [torch.randn(1, P.latent_channels, 1, 2, 3, generator=g, dtype=torch.float64),
            torch.randn(1, P.latent_channels, 1, 4, 2, generator=g, dtype=torch.float64)]
    return latent, context, refs


@torch.no_grad()
def test_packed_references_match_the_dense_sequence():
    m = model()
    latent, context, refs = inputs(1)
    t = torch.tensor([0.6], dtype=torch.float64)
    mask = torch.ones(1, context.shape[1], dtype=torch.bool)
    ours = m(latent, t, (context, mask), ref_latents=[refs])[0]
    assert torch.allclose(ours, dense(m, latent, t, context, refs), atol=1e-10)
    plain = m(latent, t, (context, mask))[0]
    assert not torch.allclose(ours, plain), "the references change the prediction"
    assert torch.equal(plain, m(latent, t, (context, mask), ref_latents=[None])[0])


@torch.no_grad()
def test_examples_with_and_without_references_pack_together():
    m = model()
    a, ctx_a, refs = inputs(2)
    b, ctx_b, _ = inputs(3, h=2, w=2, text=7)
    times = torch.tensor([0.3, 0.8], dtype=torch.float64)
    width = max(ctx_a.shape[1], ctx_b.shape[1])
    context = torch.zeros(2, width, P.text_dim, dtype=torch.float64)
    mask = torch.zeros(2, width, dtype=torch.bool)
    for i, ctx in enumerate((ctx_a, ctx_b)):
        context[i, :ctx.shape[1]], mask[i, :ctx.shape[1]] = ctx[0], True
    together = m([a, b], times, (context, mask), ref_latents=[refs, None])[0]
    alone_a = m(a, times[:1], (ctx_a, torch.ones(1, ctx_a.shape[1], dtype=torch.bool)), ref_latents=[refs])[0]
    alone_b = m(b, times[1:], (ctx_b, torch.ones(1, ctx_b.shape[1], dtype=torch.bool)))[0]
    assert torch.allclose(together[0], alone_a, atol=1e-10)
    assert torch.allclose(together[1], alone_b, atol=1e-10)


def test_gradients_reach_the_reference_pass():
    m = model().train()
    latent, context, refs = inputs(4)
    out = m(latent, torch.tensor([0.5], dtype=torch.float64), (context, torch.ones(1, 5, dtype=torch.bool)), ref_latents=[refs])[0]
    out.square().mean().backward()
    assert m.video_embedder.weight.grad is not None and m.video_embedder.weight.grad.abs().sum() > 0
