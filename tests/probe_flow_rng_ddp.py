"""CPU probe: torchrun --standalone --nproc_per_node=2 tests/probe_flow_rng_ddp.py OUTPUT

Writes only a caller-provided temporary output directory; no model downloads.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS
import torch
import torch.distributed as dist
from accelerate import Accelerator
from trainer.training.flow import FlowConfig, prepare_training_flow_batch
from trainer.training.flow_rng import FlowRNG
from trainer.training.train import Trainer


def main():
    acc=Accelerator(cpu=True)
    rank,world=acc.process_index,acc.num_processes
    assert world==2
    model=torch.nn.Linear(2,2)
    optimizer=torch.optim.AdamW(model.parameters())
    model,optimizer=acc.prepare(model,optimizer)
    trainer=Trainer.__new__(Trainer)
    trainer.accelerator=acc
    trainer.cfg=NS(train=NS(run_name='rng',gradient_accumulation_steps=1),quant=NS(mode='none'))
    trainer.out_dir=Path(sys.argv[1])
    trainer.flow_rng=FlowRNG(42,rank,world,'cpu')
    trainer.source_checkpoint=None
    trainer.steps_per_epoch=10
    def draw():
        return prepare_training_flow_batch(torch.zeros(4,4,1,8,8),
                  FlowConfig(dual_timestep=True,use_ot=False),**trainer.flow_rng.kwargs())
    draw()
    trainer.save_full_state('step000001')
    folder=trainer.out_dir/'rng-step000001-state'
    if acc.is_main_process:
        (folder/'state.json').write_text(json.dumps({'global_step':1,'flow_rng_version':1}))
    acc.wait_for_everyone()
    expected=draw()
    trainer.flow_rng=FlowRNG(999,rank,world,'cpu')
    torch.rand(100+rank)
    trainer._resume(folder)
    actual=draw()
    for a,b in zip(expected,actual):
        assert torch.equal(a,b)
    times=[None]*world
    dist.all_gather_object(times,actual[1].tolist())
    assert times[0]!=times[1]
    if acc.is_main_process:
        print('PASS: both ranks restore exact next noise, times and masks; streams remain distinct.',flush=True)
    acc.wait_for_everyone()
    dist.destroy_process_group()

if __name__=='__main__':
    main()
