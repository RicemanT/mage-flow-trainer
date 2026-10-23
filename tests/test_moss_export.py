import json
from dataclasses import asdict
import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from test_moss_image import tiny
from trainer.data.dataset import DatasetConfig
from trainer.training.config import Config, TrainConfig, load_config
from trainer.training.params import AdapterConfig, apply_adapter
from trainer.modeling.export import export_checkpoint
from trainer.modeling.moss_export import fuse_qkv
from trainer.modeling.moss_loader import load_transformer


@pytest.mark.parametrize('kind', ['lora', 'lycoris_lora'])
def test_adapter_export_preserves_qkv_updates_and_other_tensors(tmp_path, kind):
    torch.manual_seed(7)
    model = tiny()
    cfg = Config(dataset=DatasetConfig(path=str(tmp_path)))
    cfg.train.model_family = 'moss_image'
    cfg.adapter = AdapterConfig(kind=kind, rank=2, alpha=1.7)
    apply_adapter(model, cfg.adapter)
    with torch.no_grad():
        for n,p in model.named_parameters():
            if 'lora_up' in n or 'lora_B' in n:
                p.normal_()
    original_parameters = {n:p.detach().clone() for n,p in model.named_parameters()}
    cfg.train.moss_qkv_format = 'separated'
    export_checkpoint(model,tmp_path,'separate',cfg,torch.bfloat16,5)
    cfg.train.moss_qkv_format = TrainConfig().moss_qkv_format
    export_checkpoint(model,tmp_path,'fused',cfg,torch.bfloat16,5)
    separate, fused = [load_file(tmp_path/(name+'.safetensors')) for name in ('separate','fused')]
    down,up = ('.lora_down.weight','.lora_up.weight') if kind == 'lycoris_lora' else ('.lora_A.weight','.lora_B.weight')
    for i in range(2):
        prefix=f'diffusion_model.layers.{i}.proj.'
        x=torch.randn(3,32,dtype=torch.float64)
        expected=torch.cat([(x@separate[prefix+p+down].double().T)@separate[prefix+p+up].double().T * (separate[prefix+p+'.alpha'].item()/2) for p in ('q','k','v')],dim=-1)
        actual=(x@fused[prefix+'qkv'+down].double().T)@fused[prefix+'qkv'+up].double().T * (fused[prefix+'qkv.alpha'].item()/6)
        torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
        assert fused[prefix+'qkv'+down].shape==(6,32)
        assert fused[prefix+'qkv'+up].shape==(96,6)
    for k in set(separate)&set(fused):
        assert torch.equal(separate[k],fused[k])
    for n,p in model.named_parameters():
        assert torch.equal(original_parameters[n],p)
    with safe_open(tmp_path/'fused.safetensors',framework='pt') as f:
        meta=f.metadata()
        assert meta['moss_qkv_format']=='fused'
        assert json.loads(meta['moss_qkv_conversion'])['joint_blocks']==2
        assert json.loads(meta['training_config'])['train']['moss_qkv_format']=='auto'


@pytest.mark.parametrize('single_file', [True,False])
def test_full_fused_export_reloads_native_weights(tmp_path,single_file):
    model=tiny()
    root=tmp_path/'base'; (root/'transformer').mkdir(parents=True)
    (root/'transformer/config.json').write_text(json.dumps(asdict(model.params)))
    cfg=Config(dataset=DatasetConfig(path=str(tmp_path)))
    cfg.adapter.kind='none'
    cfg.train.model_family='moss_image'
    cfg.train.save_native=single_file
    export_checkpoint(model,tmp_path,'fused',cfg,torch.float32,2)
    source=tmp_path/'fused.safetensors' if single_file else tmp_path/'transformer'
    reloaded=load_transformer(root,source,torch.float32)
    for k,v in model.state_dict().items():
        assert torch.equal(v,reloaded.state_dict()[k]),k
    file=source if single_file else source/'diffusion_pytorch_model.safetensors'
    state=load_file(file)
    assert state['layers.0.proj.qkv.weight'].shape==(96,32)
    assert 'layers.0.proj.q.weight' not in state
    assert torch.equal(state['text_embedder.blocks.0.qkv.weight'],model.state_dict()['text_embedder.blocks.0.qkv.weight'])


def test_export_rejects_incomplete_or_scaled_differently():
    state={}
    for p in ('q','k','v'):
        state[f'layers.0.proj.{p}.lora_up.weight']=torch.ones(4,2)
        state[f'layers.0.proj.{p}.lora_down.weight']=torch.ones(2,4)
        state[f'layers.0.proj.{p}.alpha']=torch.tensor(2.)
    state['layers.0.proj.q.alpha']=torch.tensor(1.)
    with pytest.raises(ValueError,match='unequal'):
        fuse_qkv(state,adapter=True)
    del state['layers.0.proj.v.alpha']
    with pytest.raises(ValueError,match='complete'):
        fuse_qkv(state,adapter=True)


@pytest.mark.parametrize('algo', ['lokr', 'dora'])
def test_default_preserves_native_layout_for_other_moss_adapters(tmp_path, algo):
    model = tiny()
    cfg = Config(dataset=DatasetConfig(path=str(tmp_path)))
    cfg.train.model_family = 'moss_image'
    cfg.adapter = AdapterConfig(kind='lycoris_lora', lycoris_algo=algo, rank=2, alpha=2)
    apply_adapter(model, cfg.adapter)
    export_checkpoint(model, tmp_path, 'native', cfg, torch.bfloat16, 1)
    with safe_open(tmp_path/'native.safetensors', framework='pt') as reader:
        assert reader.metadata()['moss_qkv_format'] == 'separated'
        assert any(key.startswith('diffusion_model.layers.0.proj.q.') for key in reader.keys())
        assert not any(key.startswith('diffusion_model.layers.0.proj.qkv.') for key in reader.keys())


@pytest.mark.parametrize('family,kind,algo',[('mage_flow','lora','locon'),('moss_image','lokr','locon'),('moss_image','lycoris_lora','dora'),('moss_image','lycoris_lora','lokr')])
def test_config_rejects_unsupported_export_before_training(tmp_path,family,kind,algo):
    path=tmp_path/'bad.toml'
    path.write_text(f'[dataset]\npath="/images"\n[train]\nmodel_family="{family}"\nmoss_qkv_format="fused"\n[adapter]\nkind="{kind}"\nlycoris_algo="{algo}"\n')
    with pytest.raises(ValueError,match='Fused QKV export'):
        load_config(path)
    with pytest.raises(ValueError,match='moss_qkv_format'):
        TrainConfig(moss_qkv_format='typo')


@pytest.mark.parametrize('naming', ['diffusion_model.layers.0.proj.', 'lora_unet_layers_0_proj_'])
def test_standalone_conversion_roundtrip_and_no_overwrite(tmp_path,naming):
    import hashlib
    from lora_qkv_convert import convert_file
    from safetensors.torch import save_file
    state={}
    for p in ('q','k','v'):
        state[naming+p+'.lora_down.weight']=torch.randn(2,4).bfloat16()
        state[naming+p+'.lora_up.weight']=torch.randn(4,2).bfloat16()
        state[naming+p+'.alpha']=torch.tensor(1.7)
    state['diffusion_model.text_embedder.blocks.0.qkv.alpha']=torch.tensor(2.)
    source=tmp_path/'original.safetensors'
    save_file(state,source,metadata={'training_config':'original training settings'})
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    fused,details=convert_file(source)
    assert details['joint_blocks']==1
    with pytest.raises(ValueError,match='never overwritten'):
        convert_file(source,fused)
    with pytest.raises(ValueError,match='never overwritten'):
        convert_file(source,source)
    separated,_=convert_file(fused,layout='separated')
    restored=load_file(separated)
    for p in ('q','k','v'):
        target=naming+p
        expected=(state[target+'.lora_up.weight'].double()@state[target+'.lora_down.weight'].double())*(state[target+'.alpha'].item()/2)
        actual=(restored[target+'.lora_up.weight'].double()@restored[target+'.lora_down.weight'].double())*(restored[target+'.alpha'].item()/6)
        torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
    with safe_open(separated,framework='pt') as f:
        assert f.metadata()['training_config']=='original training settings'
        assert f.metadata()['moss_qkv_format']=='separated'
    with pytest.raises(ValueError,match='No convertible'):
        convert_file(fused,tmp_path/'already.safetensors')


def test_standalone_gui_opens(monkeypatch):
    import os
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6 import QtCore,QtWidgets
    from lora_qkv_convert import gui
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    QtCore.QTimer.singleShot(100,app.quit)
    assert gui()==0
