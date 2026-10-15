import torch
import pytest
from trainer.training.eye_adversarial import EyeDiscriminator,augment_pair,discriminator_update,eye_crops,EyeAdversarialConfig


def test_crop_and_augmentation_gradient_and_shared_transform():
    image=torch.randn(1,3,160,160,requires_grad=True)
    crop=eye_crops(image,[(10,20,90,80)])
    state=torch.random.get_rng_state().clone()
    a,b=augment_pair(crop,crop,123)
    torch.testing.assert_close(a,b)
    torch.testing.assert_close(state,torch.random.get_rng_state())
    a.square().mean().backward()
    assert image.grad.isfinite().all() and image.grad.abs().sum()>0
    assert image.grad[...,:20,:].count_nonzero()==0


def test_discriminator_step_and_frozen_generator_gradient():
    torch.set_num_threads(2)
    model=EyeDiscriminator()
    opt=torch.optim.Adam(model.parameters(),lr=1e-4)
    real=torch.randn(2,3,128,128);fake=torch.randn_like(real,requires_grad=True)
    stats=discriminator_update(model,opt,real,fake)
    assert stats['d_grad_norm']>0 and fake.grad is None
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    loss=torch.nn.functional.softplus(-model(fake)).mean()
    loss.backward()
    assert fake.grad.isfinite().all() and fake.grad.abs().sum()>0
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize('kw',[{'weight':float('nan')},{'weight':-1},{'discriminator_lr':0}])
def test_invalid_adversarial_config(kw):
    with pytest.raises(ValueError):EyeAdversarialConfig(**kw)
