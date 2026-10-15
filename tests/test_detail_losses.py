from trainer.data.dataset import DatasetConfig
import pytest
import torch
import torch.nn.functional as F
from trainer.training.flow import FlowConfig,hf_loss,hf_snr_gate
from trainer.training.paired_texture import PairedTextureConfig,PairedTextureLoss


def test_hf_gate_boundary_floor_and_power():
    t=torch.tensor([0.,.5,.75,.9,1.],requires_grad=True)
    torch.testing.assert_close(hf_snr_gate(t),torch.tensor([1.,1.,1.,1/9,.1]))
    torch.testing.assert_close(hf_snr_gate(t,min_gate=0,power=2),torch.tensor([1.,1.,1.,1/81,0.]))
    assert not hf_snr_gate(t).requires_grad


def test_hf_scalar_gate_and_mask_not_normalized_away():
    x=torch.randn(2,4,1,8,10);p=torch.randn_like(x,requires_grad=True)
    t=torch.tensor([.2,.9]);n=torch.randn_like(x)
    for mask in [None,torch.ones(2,1,1,8,10)]:
        expected=torch.stack([hf_loss(p[i:i+1],n[i:i+1],x[i:i+1],t[i:i+1],1,1.,
                             mask=None if mask is None else mask[i:i+1])*hf_snr_gate(t[i:i+1])[0] for i in range(2)]).mean()
        actual=hf_loss(p,n,x,t,1,1.,mask=mask,snr_cut=1/9)
        torch.testing.assert_close(actual,expected)
    torch.testing.assert_close(hf_loss(p,n,x,t,1,1.,snr_cut=1/9,min_gate=1),hf_loss(p,n,x,t,1,1.),rtol=0,atol=0)
    actual.backward();assert p.grad.isfinite().all()


def test_hf_spatial_gate_zeroes_high_noise_gradient():
    p=torch.ones(1,4,1,8,8,requires_grad=True);x=torch.zeros_like(p);t=torch.ones(1,1,1,8,8);t[...,:4]=.5
    value=hf_loss(p,x,x,t,1,1.,snr_cut=1/9,min_gate=0)
    value.backward()
    assert p.grad[...,:4].abs().sum()>0
    assert p.grad[...,4:].count_nonzero()==0


def test_texture_identity_blur_and_target_detachment():
    torch.set_num_threads(2)
    op=PairedTextureLoss(PairedTextureConfig(scale=.1,tokens=64,features=16))
    clean=torch.randn(2,4,17,21,requires_grad=True)
    same=clean.detach().clone().requires_grad_();v=op.per_sample(same,clean).sum()
    assert v.item()==0
    v.backward();assert clean.grad is None and same.grad.count_nonzero()==0
    pred=F.avg_pool2d(clean.detach(),3,1,1).requires_grad_()
    state=torch.get_rng_state().clone();values=op.per_sample(pred,clean)
    assert torch.equal(state,torch.get_rng_state())
    torch.testing.assert_close(values,torch.cat([op.per_sample(a,b) for a,b in zip(pred.split(1),clean.split(1))]))
    assert values.mean()>0
    values.sum().backward();assert pred.grad.isfinite().all() and pred.grad.abs().sum()>0 and clean.grad is None


def test_texture_flow_5d_and_disabled():
    cfg=PairedTextureConfig(scale=.1,tokens=32,features=8);op=PairedTextureLoss(cfg)
    x=torch.randn(2,4,1,8,10);n=torch.randn_like(x);t=torch.tensor([.2,.9]);tv=t[:,None,None,None,None]
    pred=(n-x).requires_grad_()
    loss,term=op.apply(torch.tensor(1.),pred,x,(1-tv)*x+tv*n,t)
    assert term.abs()<1e-10
    loss.backward();assert pred.grad.isfinite().all()
    base=torch.tensor(1.,requires_grad=True)
    disabled=PairedTextureLoss(PairedTextureConfig())
    result,value=disabled.apply(base,None,None,None,None)
    assert result is base and value is None and not disabled._projections and not disabled._locations
    torch.testing.assert_close(op.gate(torch.tensor([.75,.9,1.])),torch.tensor([1.,1/9,0.]))
    with pytest.raises(ValueError,match='one timestep'):op.apply(base,pred,x,n,torch.ones(2,1,1,8,10))


@pytest.mark.parametrize('shape',[(1,4,1,1),(1,4,2,3),(1,4,15,17)])
def test_texture_tiny_flat(shape):
    op=PairedTextureLoss(PairedTextureConfig(tokens=32,features=8))
    p=torch.ones(shape,requires_grad=True);clean=torch.zeros(shape,requires_grad=True)
    value=op.per_sample(p,clean).sum();value.backward()
    assert torch.isfinite(value) and p.grad.isfinite().all() and clean.grad is None


@pytest.mark.parametrize('kw',[{'hf_high_noise_snr_cut':0},{'hf_high_noise_min_weight':1.1},{'hf_high_noise_power':float('nan')},{'hf_scale':float('nan')}])
def test_invalid_hf(kw):
    with pytest.raises(ValueError):FlowConfig(**kw)


@pytest.mark.parametrize('kw',[{'scale':float('nan')},{'patch':2},{'tokens':False},{'snr_cut':0},{'seed':1.5},{'min_gate':-1}])
def test_invalid_texture(kw):
    with pytest.raises(ValueError):PairedTextureConfig(**kw)


@pytest.mark.parametrize('packed',[False,True])
def test_trainer_detail_loss_integration(monkeypatch,packed):
    from types import SimpleNamespace
    from trainer.training.config import Config
    from trainer.training.train import Trainer
    import trainer.training.train as module
    from trainer.training.flow import flow_loss
    cfg=Config(dataset=DatasetConfig(path="/tmp"));cfg.train.pack_resolutions=packed;cfg.flow.hf_scale=.2
    cfg.paired_texture=PairedTextureConfig(scale=.1,tokens=16,features=8)
    trainer=Trainer.__new__(Trainer)
    trainer.cfg=cfg;trainer.accelerator=SimpleNamespace(device=torch.device('cpu'))
    trainer.dtype=torch.float32;trainer.phase=None;trainer.self_flow_ema=None
    trainer.rti_interface=None;trainer.adversarial=None;trainer.perceptual=None
    trainer.hf_patch=1;trainer.paired_texture=PairedTextureLoss(cfg.paired_texture)
    trainer._encode=lambda captions: None;trainer._preserve_term=lambda: 0.
    weight=torch.tensor(.7,requires_grad=True)
    def forward(hidden_states,**kwargs):
        return ([x*weight for x in hidden_states] if isinstance(hidden_states,list) else hidden_states*weight,)
    trainer.transformer=forward
    def prepare(x,*args,**kwargs):
        t=x.new_full((x.shape[0],),.9)
        return x*.5,t,-x,None,None
    monkeypatch.setattr(module,'prepare_training_flow_batch',prepare)
    xs=[torch.randn(1,4,1,8,10),torch.randn(1,4,1,12 if packed else 8,10)]
    batch={'latents':xs if packed else torch.cat(xs),'captions':['a','b'],'paths':['a','b']}
    actual,hf=trainer._step(batch)
    expected=[]
    for x in xs:
        n,t,target,_,_=prepare(x);pred=n*weight
        base=flow_loss(pred,target)
        term=trainer.paired_texture.apply(base,pred,x,n,t)[0]
        expected.append(term+cfg.flow.hf_scale*hf_loss(pred,n,x,t,1,cfg.flow.hf_exponent,**cfg.flow.hf_gate_kwargs()))
    expected=torch.stack(expected).mean()
    torch.testing.assert_close(actual,expected)
    grad=torch.autograd.grad(actual,weight,retain_graph=True)[0]
    torch.testing.assert_close(grad,torch.autograd.grad(expected,weight)[0])
    assert torch.isfinite(grad) and 'texture' in trainer.last_stats


def test_detail_config_gui_roundtrip_and_guards():
    import tomllib
    from trainer.gui.bridge import dump_toml,flatten
    from trainer.training.config import Config,validate_model_options
    values={'paired_texture.scale':.2,'paired_texture.seed':2**40,
            'flow.hf_high_noise_min_weight':.3,'flow.hf_high_noise_power':2.}
    restored=flatten(tomllib.loads(dump_toml(values)))
    for k,v in values.items():assert restored[k]==v
    for setting in ('dual_timestep','self_flow','masked'):
        cfg=Config(dataset=DatasetConfig(path="/tmp"));cfg.paired_texture=PairedTextureConfig(scale=.1)
        if setting=='dual_timestep':cfg.flow.dual_timestep=True
        elif setting=='self_flow':cfg.self_flow.enabled=True
        else:
            from types import SimpleNamespace
            cfg.curriculum.phases=[SimpleNamespace(mode='texture')]
        with pytest.raises(ValueError,match='scalar timesteps'):validate_model_options(cfg)
