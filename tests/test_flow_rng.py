"""Flow draws are rank-specific, isolated, and restored with training checkpoints."""
from types import SimpleNamespace as NS
import json
import pytest
import torch
from trainer.training.flow import FlowConfig, sample_timesteps, prepare_training_flow_batch
from trainer.training.flow_rng import FlowRNG


@pytest.mark.parametrize('method', ['uniform', 'logit_normal'])
@pytest.mark.parametrize('mapping', ['rescale', 'truncate'])
def test_explicit_timestep_generator_is_used(method, mapping):
    cfg = FlowConfig(timestep_sample_method=method, phase_mapping=mapping)
    gen = torch.Generator().manual_seed(12)
    initial = gen.get_state().clone()
    global_state = torch.get_rng_state().clone()
    first = sample_timesteps(cfg, 32, 8, 8, 'cpu', generator=gen, t_range=(.2,.8))
    assert not torch.equal(initial, gen.get_state())
    assert torch.equal(global_state, torch.get_rng_state())
    torch.rand(1000)
    gen.set_state(initial)
    second = sample_timesteps(cfg, 32, 8, 8, 'cpu', generator=gen, t_range=(.2,.8))
    assert torch.equal(first, second)
    assert ((first >= .2) & (first <= .8)).all()


def draws(rng, *, height=8, dual=True):
    return prepare_training_flow_batch(torch.zeros(4, 4, 1, height, 8),
                                      FlowConfig(dual_timestep=dual, use_ot=False), **rng.kwargs())


def assert_same(left, right):
    for a,b in zip(left,right):
        if a is None: assert b is None
        else: assert torch.equal(a,b)


def test_streams_are_independent_and_ranks_distinct():
    a,b = FlowRNG(42,0,2,'cpu'),FlowRNG(42,0,2,'cpu')
    first = draws(a)
    torch.randn(500)
    assert_same(first, draws(b))
    other = draws(FlowRNG(42,1,2,'cpu'))
    assert not torch.equal(first[1], other[1])  # time
    assert not torch.equal(first[2], other[2])  # noise
    assert not torch.equal(first[4], other[4])  # mask
    # Image size/noise consumption must not alter later timestep draws.
    a,b = FlowRNG(42,0,1,'cpu'),FlowRNG(42,0,1,'cpu')
    draws(a,height=4); draws(b,height=12)
    assert torch.equal(draws(a)[1], draws(b)[1])


def test_rank_files_restore_without_resetting_global_rng(tmp_path):
    results = []
    for rank in (0,1):
        rng = FlowRNG(42,rank,2,'cpu')
        draws(rng); rng.save(tmp_path)
        results.append(draws(rng))
    for rank in (0,1):
        rng = FlowRNG(999,rank,2,'cpu')
        global_state = torch.get_rng_state().clone()
        assert rng.load(tmp_path, required=True)
        assert torch.equal(global_state,torch.get_rng_state())
        assert_same(results[rank],draws(rng))
    with pytest.raises(ValueError, match='same world size'):
        FlowRNG(42,0,1,'cpu').load(tmp_path)
    (tmp_path/'flow_rng_rank1.safetensors').unlink()
    with pytest.raises(FileNotFoundError):
        FlowRNG(42,1,2,'cpu').load(tmp_path)


def test_legacy_resume_warns_but_missing_new_state_fails(tmp_path):
    rng = FlowRNG(42,0,1,'cpu')
    with pytest.warns(UserWarning, match='Legacy checkpoint'):
        assert not rng.load(tmp_path)
    with pytest.raises(FileNotFoundError):
        rng.load(tmp_path,required=True)


def test_trainer_save_resume_restores_next_draws(tmp_path):
    from accelerate import Accelerator
    from trainer.training.train import Trainer
    acc = Accelerator(cpu=True)
    model = torch.nn.Linear(2,2)
    opt = torch.optim.AdamW(model.parameters())
    model, opt = acc.prepare(model,opt)
    trainer = Trainer.__new__(Trainer)
    trainer.accelerator=acc
    trainer.cfg=NS(train=NS(run_name='rng',gradient_accumulation_steps=1),quant=NS(mode='none'))
    trainer.out_dir=tmp_path
    trainer.flow_rng=FlowRNG(42,0,1,'cpu')
    trainer.source_checkpoint=None
    trainer.steps_per_epoch=10
    draws(trainer.flow_rng)
    trainer.save_full_state('step000001')
    folder=tmp_path/'rng-step000001-state'
    (folder/'state.json').write_text(json.dumps({'global_step':1,'flow_rng_version':1}))
    expected=draws(trainer.flow_rng)
    trainer.flow_rng=FlowRNG(99,0,1,'cpu')
    trainer._resume(folder)
    assert_same(expected,draws(trainer.flow_rng))
    acc.free_memory()
