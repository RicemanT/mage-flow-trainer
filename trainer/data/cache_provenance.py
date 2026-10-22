"""Content identities for latent caches; independent of machine-specific paths."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import tempfile
import warnings


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def vae_identity(model_path, vae_path=None, model_family='auto', flux2_vae=False):
    from ..modeling.loader import resolve_model_family
    family = resolve_model_family(model_path, model_family)
    flux = flux2_vae or family in ('flux3', 'moss_image')
    source = Path(vae_path) if vae_path else Path(model_path) / (
        'vae' if flux else 'vae/diffusion_pytorch_model.safetensors')
    identity = dict(schema=1, layout='128x1xh16xw16', posterior='mean',
                    normalization='flux2-pack2-bn' if flux else 'mage-native',
                    preprocessing='rgb-lanczos-center-v1', storage_dtype='float32')
    files = []
    if source.is_file():
        files = [source]
    elif source.is_dir():
        single = source / 'diffusion_pytorch_model.safetensors'
        index = source / 'diffusion_pytorch_model.safetensors.index.json'
        if single.is_file():
            files = [single]
        elif index.is_file():
            files = [source / name for name in sorted(set(json.loads(index.read_text())['weight_map'].values()))]
        if files and (source / 'config.json').is_file():
            files.append(source / 'config.json')
    if files and all(p.is_file() for p in files):
        # Content rather than absolute path: moving a checkpoint does not invalidate it.
        hashes = sorted(file_sha256(p) for p in files)
        identity['vae_sha256'] = hashes[0] if len(hashes) == 1 else hashlib.sha256(''.join(hashes).encode()).hexdigest()
    return identity


def validate_cache(path, expected=None, policy='warn', source=None):
    """Known mismatches always fail unless explicitly disabled; legacy caches can warn."""
    if policy not in ('warn', 'strict', 'off'):
        raise ValueError('cache_validation must be warn, strict or off')
    if policy == 'off':
        return None
    from .cache import read_cache_metadata
    meta = read_cache_metadata(path)
    raw = meta.get('cache_provenance')
    actual = json.loads(raw) if raw else None
    if actual is not None and expected:
        for key, value in expected.items():
            if key in actual and actual[key] != value:
                raise ValueError(f'{path}: latent cache {key} mismatch; rebuild with --overwrite using the intended VAE')
    if source is not None and meta.get('source_sha256') and file_sha256(source) != meta['source_sha256']:
        raise ValueError(f'{path}: source image content changed; rebuild with --overwrite')
    unknown = (actual is None or not actual.get('vae_sha256') or
               (expected is not None and not expected.get('vae_sha256')) or
               (expected is not None and any(k not in actual for k in expected)))
    if unknown:
        message = 'Latent cache provenance is unverified (legacy cache or unavailable VAE). Re-cache to verify its identity.'
        if policy == 'strict':
            raise ValueError(f'{path}: {message}')
        warnings.warn(message, UserWarning, stacklevel=2)
    return actual


def validate_dataset_caches(dataset, train, policy='warn'):
    if policy == 'off' or not dataset.require_cache:
        return
    expected = vae_identity(train.model_path, train.vae_path, train.model_family, train.flux2_vae)
    identities = set()
    for path in sorted({e.latent_path for e in dataset.entries}):
        actual = validate_cache(path, expected, policy)
        if actual and actual.get('vae_sha256'):
            identities.add(actual['vae_sha256'])
    if len(identities) > 1:
        raise ValueError('Dataset mixes latent caches from different VAEs; rebuild with one VAE or explicitly disable cache validation')


def atomic_save(tensors, path, metadata):
    from safetensors.torch import save_file
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    os.close(fd)
    try:
        save_file(tensors, temporary, metadata=metadata)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
