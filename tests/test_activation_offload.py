"""Boundary offload tests; CPU simulation exercises payload lifetime/budget too."""

import gc
import copy
import weakref

import pytest
import torch

from trainer.training.activation_offload import (
    ActivationOffloader, configure_activation_offload, checkpoint_with_offload,
)


def cpu_manager(model, budget=4096):
    manager = ActivationOffloader(model, budget)
    manager._eligible = lambda tensor: tensor.layout == torch.strided and tensor.numel() > 0
    model._activation_offloader = manager
    return manager


def test_budget_dedup_views_and_storage_exclusion():
    model = torch.nn.Linear(4, 4)
    model.register_buffer("test_buffer", torch.ones(4))
    manager = cpu_manager(model, 32)
    x = torch.arange(8.).reshape(2, 4)
    with manager.scope():
        saved = manager.pack(x)
        duplicate = manager.pack(x.view(2, 4))
        assert saved is duplicate
        assert manager.live_bytes == 32
        assert manager.pack(x[:, :2]) is not saved  # different view, no budget left
        assert manager.pack(model.weight.detach()) is not saved
        assert manager.pack(model.test_buffer) is model.test_buffer
        assert manager.budget_fallbacks == 1
        torch.testing.assert_close(manager.unpack(saved), x)
    assert manager.peak_bytes == 32
    del saved, duplicate
    gc.collect()
    assert manager.live_bytes == 0


def test_saved_input_does_not_keep_original_tensor_alive():
    model = torch.nn.Identity()
    manager = cpu_manager(model)
    x = torch.arange(8.)
    reference = weakref.ref(x)
    with manager.scope():
        saved = manager.pack(x)
    del x
    gc.collect()
    assert reference() is None
    torch.testing.assert_close(manager.unpack(saved), torch.arange(8.))


def test_changed_inplace_version_is_not_deduplicated():
    model = torch.nn.Identity()
    manager = cpu_manager(model)
    x = torch.ones(8)
    with manager.scope():
        first = manager.pack(x)
        x.add_(2)
        second = manager.pack(x)
    assert first is not second
    torch.testing.assert_close(manager.unpack(first), torch.ones(8))
    torch.testing.assert_close(manager.unpack(second), torch.full((8,), 3.))


def test_checkpoint_nested_inputs_gradient_parity_and_only_boundaries():
    torch.manual_seed(2)
    model = torch.nn.Linear(4, 4)
    manager = cpu_manager(model)
    x = torch.randn(3, 4, requires_grad=True)
    y = torch.randn(3, 4, requires_grad=True)

    def block(a, nested):
        return (model(a).sin() * nested["value"][0].cos()).square().sum()

    expected = block(x, {"value": [y]})
    reference = torch.autograd.grad(expected, (x, y, model.weight, model.bias))
    actual = checkpoint_with_offload(model, block, x, {"value": [y]})
    assert manager.offloaded_tensors == 2  # no linear/sin/cos intermediates
    grads = torch.autograd.grad(actual, (x, y, model.weight, model.bias))
    torch.testing.assert_close(actual, expected)
    for got, want in zip(grads, reference):
        torch.testing.assert_close(got, want)
    del actual
    gc.collect()
    assert manager.live_bytes == 0


def test_budget_fallback_and_noncontiguous_input_preserve_gradients():
    model = torch.nn.Identity()
    manager = cpu_manager(model, 16)
    x = torch.arange(12.).reshape(3, 4).t().requires_grad_()
    y = checkpoint_with_offload(model, lambda a: a.square().sum(), x)
    y.backward()
    torch.testing.assert_close(x.grad, 2 * x)
    assert manager.live_bytes == manager.peak_bytes == 0
    assert manager.budget_fallbacks == 1


def test_noncontiguous_offloaded_view_restores_values():
    model = torch.nn.Identity()
    manager = cpu_manager(model)
    x = torch.arange(12.).reshape(3, 4).t().requires_grad_()
    y = checkpoint_with_offload(model, lambda a: a.square().sum(), x)
    y.backward()
    torch.testing.assert_close(x.grad, 2 * x)
    assert manager.offloaded_tensors == 1


def test_config_gates_and_cpu_noop():
    model = torch.nn.Identity()
    configure_activation_offload(model)
    assert model._activation_offloader is None
    for kwargs in ({"gradient_checkpointing": False}, {"compile_mode": "default"},
                   {"budget_mb": 0}, {"budget_mb": float("inf")}):
        with pytest.raises(ValueError):
            configure_activation_offload(model, enabled=True, **kwargs)
    configure_activation_offload(model, enabled=True)
    x = torch.ones(3, requires_grad=True)
    checkpoint_with_offload(model, lambda a: a.square().sum(), x).backward()
    torch.testing.assert_close(x.grad, 2 * x)
    assert model._activation_offloader.offloaded_tensors == 0


def test_deepcopy_creates_independent_idle_manager_and_rejects_live_graph():
    model = torch.nn.Linear(4, 4)
    configure_activation_offload(model, enabled=True)
    cloned = copy.deepcopy(model)
    assert cloned._activation_offloader is not model._activation_offloader
    assert cloned._activation_offloader._model() is cloned
    assert cloned._activation_offloader.live_bytes == 0
    manager = cpu_manager(model)
    x = torch.ones(4)
    with manager.scope():
        saved = manager.pack(x)
    with pytest.raises(RuntimeError, match="live activation"):
        copy.deepcopy(model)
    del saved


@pytest.mark.parametrize("family", ["moss", "flux3", "mage"])
def test_integrated_checkpoint_models_cpu_gradient_parity(family):
    torch.set_num_threads(2)
    torch.manual_seed(18)
    if family == "moss":
        from trainer.modeling.moss_image import MOSSImage, MOSSParams
        model = MOSSImage(MOSSParams(dim=32, ffn_dim=48, num_heads=2,
                                    num_layers=2, text_dim=16, latent_channels=8))
        channels, text_dim = 8, 16
    elif family == "flux3":
        from trainer.modeling.flux3 import Flux3, Flux3Params
        model = Flux3(Flux3Params(hidden=32, heads=2, depth=1, depth_single=2,
                                 axes_dim=(4, 4, 4, 4), channels=8, text_dim=16))
        channels, text_dim = 8, 16
    else:
        from trainer.modeling.mage_flow import MageFlow, MageFlowParams
        model = MageFlow(MageFlowParams(128, 128, 24, 32, 4, 2, [2, 2, 4], False))
        channels, text_dim = 128, 24
    model.train()
    inputs = dict(hidden_states=torch.randn(2, channels, 1, 2, 3),
                  timestep=torch.tensor([0.2, 0.7]),
                  encoder_hidden_states=(torch.randn(2, 4, text_dim), torch.ones(2, 4, dtype=torch.bool)))
    model.configure_execution(gradient_checkpointing=True)
    expected = model(**inputs)[0]
    expected.square().mean().backward()
    reference = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    model.configure_execution(gradient_checkpointing=True, activation_offload=True)
    manager = cpu_manager(model, 16 * 1024 * 1024)
    actual = model(**inputs)[0]
    actual.square().mean().backward()
    torch.testing.assert_close(actual, expected)
    for name, param in model.named_parameters():
        if name in reference:
            torch.testing.assert_close(param.grad, reference[name], rtol=2e-4, atol=2e-6)
        else:
            assert param.grad is None
    assert manager.offloaded_tensors > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("prefetch", [False, True])
def test_cuda_boundary_offload_forward_backward_parity(prefetch):
    torch.manual_seed(9)
    device = "cuda"
    model = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.Linear(16, 16)).to(device)
    x = torch.randn(8, 16, device=device, requires_grad=True)
    expected = model[1](model[0](x).sin()).square().mean()
    targets = (x, *model.parameters())
    reference = torch.autograd.grad(expected, targets)
    configure_activation_offload(model, enabled=True, budget_mb=1,
                                 pin_memory=prefetch, prefetch=prefetch)
    hidden = checkpoint_with_offload(model, lambda a: model[0](a).sin(), x)
    actual = checkpoint_with_offload(model, lambda a: model[1](a).square().mean(), hidden)
    grads = torch.autograd.grad(actual, targets)
    torch.testing.assert_close(actual, expected)
    for got, want in zip(grads, reference):
        torch.testing.assert_close(got, want)
    assert model._activation_offloader.offloaded_tensors == 2
    assert model._activation_offloader.peak_bytes <= 1024 * 1024
