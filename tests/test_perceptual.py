import pytest
import torch
from types import SimpleNamespace
from trainer.training.perceptual import unpack_flux2_latent,PerceptualConfig
from trainer.training.config import Config,validate_model_options

def test_inverse_normalization_and_gradients():
    raw=torch.randn(2,32,12,16,requires_grad=True)
    mean=torch.randn(128);var=torch.rand(128)+.2;eps=1e-5
    vae=SimpleNamespace(bn=SimpleNamespace(running_mean=mean,running_var=var,eps=eps))
    packed=torch.nn.functional.pixel_unshuffle(raw,2)
    z=(packed-mean[None,:,None,None])/(var[None,:,None,None]+eps).sqrt()
    decoded=unpack_flux2_latent(z.unsqueeze(2),vae)
    torch.testing.assert_close(decoded,raw)
    decoded.sum().backward();torch.testing.assert_close(raw.grad,torch.ones_like(raw))

@pytest.mark.parametrize('kwargs',[{'weight':float('nan')},{'max_t':0},{'every':0}])
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):PerceptualConfig(**kwargs)

def test_unsupported_combinations():
    from trainer.data.dataset import DatasetConfig
    c=Config(dataset=DatasetConfig(path='/unused'));c.perceptual.enabled=True
    with pytest.raises(ValueError):validate_model_options(c)
    c.train.flux2_vae=True;c.train.pack_resolutions=True;c.dataset.source='latents'
    validate_model_options(c)
    c.flow.dual_timestep=True
    with pytest.raises(ValueError):validate_model_options(c)


def test_eye_crop_gradient_is_local_and_eyes_are_averaged():
    from trainer.training.perceptual import eye_crop_loss
    prediction = torch.ones(1,3,128,192,requires_grad=True)
    target = torch.zeros_like(prediction)
    metric = lambda a,b: (a-b).square().mean()
    boxes = [(0,0,64,64),(128,64,192,128)]
    loss = eye_crop_loss(metric,prediction,target,boxes)
    torch.testing.assert_close(loss,torch.tensor(1.))
    loss.backward()
    assert prediction.grad[..., :64, :64].abs().sum() > 0
    assert prediction.grad[..., 64:, 128:].abs().sum() > 0
    assert prediction.grad[..., :, 64:128].count_nonzero() == 0
    torch.testing.assert_close(prediction.grad.sum(),torch.tensor(2.))


def test_eye_manifest_integrity_and_bucket_mapping(tmp_path):
    import hashlib,json
    from trainer.training.perceptual import load_eye_regions
    source=tmp_path/'face.png';source.write_bytes(b'source')
    annotation={'source_sha256':hashlib.sha256(b'source').hexdigest(),
                'training':[{'size':[128,96],'eyes':[{'xyxy':[40,30,60,40]}]}]}
    (tmp_path/'a.json').write_text(json.dumps(annotation))
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'images':[{'source':str(source),'annotation':'a.json'}]}))
    regions=load_eye_regions(manifest)
    size,boxes=regions[str(tmp_path/'face_0128x0096_trainer.safetensors')]
    assert size==(96,128) and boxes==[(30,22,70,48)]
    source.write_bytes(b'changed')
    with pytest.raises(ValueError,match='source changed'):load_eye_regions(manifest)


def test_eye_loss_skips_unapproved_annotations():
    from trainer.training.perceptual import DecodedPerceptualLoss
    obj=DecodedPerceptualLoss.__new__(DecodedPerceptualLoss)
    obj.cfg=PerceptualConfig();obj.micro=0;obj.eye_regions={}
    prediction=torch.ones(1,128,1,4,4,requires_grad=True)
    zero=obj.loss([prediction],[prediction],[prediction],[torch.tensor(.1)],['/unapproved'])
    assert zero.item()==0 and obj.last_term is None


def test_dedicated_noise_is_repeatable_and_preserves_rng():
    from trainer.training.perceptual import dedicated_flow_batch
    clean=torch.randn(1,4,1,8,8)
    state=torch.random.get_rng_state().clone()
    a=dedicated_flow_batch(clean,42,7,.5)
    torch.testing.assert_close(state,torch.random.get_rng_state())
    b=dedicated_flow_batch(clean,42,7,.5)
    for x,y in zip(a,b):torch.testing.assert_close(x,y)
    noisy,t,target=a
    assert .05 <= t.item() < .5
    torch.testing.assert_close(noisy-t*target,clean)
    assert not torch.equal(a[0],dedicated_flow_batch(clean,42,8,.5)[0])


@pytest.mark.parametrize('extra',[{'every':2},{'max_t':.05},{'eye_manifest':''},{'enabled':False}])
def test_invalid_dedicated_settings(extra):
    values=dict(enabled=True,eye_manifest='manifest.json',dedicated_low_noise=True)
    values.update(extra)
    with pytest.raises(ValueError):PerceptualConfig(**values)
