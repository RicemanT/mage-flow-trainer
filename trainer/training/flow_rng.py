"""Rank-local flow randomness, independent of model, loader and optimizer RNGs."""
import hashlib
import os
from pathlib import Path
import tempfile
import warnings

import torch
from safetensors import safe_open
from safetensors.torch import save_file


class FlowRNG:
    version = 1

    def __init__(self, seed, rank, world_size, device):
        self.rank, self.world_size = rank, world_size
        self.timesteps = self._generator(seed, rank, 'timesteps', 'cpu')
        self.noise = self._generator(seed, rank, 'noise', device)
        self.mask = self._generator(seed, rank, 'mask', device)

    @staticmethod
    def _generator(seed, rank, stream, device):
        key = f'mage-flow-rng-v1:{seed}:{rank}:{stream}'.encode()
        value = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), 'little')
        return torch.Generator(device=device).manual_seed(value)

    def kwargs(self):
        return dict(generator=self.noise, timestep_generator=self.timesteps, mask_generator=self.mask)

    def _path(self, folder):
        return Path(folder) / f'flow_rng_rank{self.rank}.safetensors'

    def save(self, folder):
        path = self._path(folder)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
        os.close(fd)
        try:
            save_file({name: getattr(self, name).get_state() for name in ('timesteps', 'noise', 'mask')},
                      temp, metadata=dict(version=str(self.version), rank=str(self.rank),
                                          world_size=str(self.world_size), device_type=self.noise.device.type))
            os.replace(temp, path)
        finally:
            Path(temp).unlink(missing_ok=True)

    def load(self, folder, *, required=False):
        path = self._path(folder)
        if not path.exists():
            if required or any(Path(folder).glob('flow_rng_rank*.safetensors')):
                raise FileNotFoundError(f'Missing flow RNG state for rank {self.rank}: {path}')
            warnings.warn('Legacy checkpoint has no dedicated flow RNG states; starting new rank-local '
                          'streams. Noise/timestep draws will not reproduce the old run.', UserWarning)
            return False
        with safe_open(path, framework='pt', device='cpu') as reader:
            meta = reader.metadata() or {}
            expected = dict(version=str(self.version), rank=str(self.rank), world_size=str(self.world_size),
                            device_type=self.noise.device.type)
            if any(meta.get(k) != v for k,v in expected.items()):
                raise ValueError(f'{path}: incompatible flow RNG state; resume requires the same world size '
                                 'and device type, and a matching RNG format')
            states = {name: reader.get_tensor(name) for name in ('timesteps', 'noise', 'mask')}
        # Validate all states before mutating any live stream.
        for name, state in states.items():
            torch.Generator(device=getattr(self, name).device).set_state(state)
        for name, state in states.items():
            getattr(self, name).set_state(state)
        return True
