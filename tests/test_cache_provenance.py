import json
from types import SimpleNamespace as NS
import pytest
import torch
from safetensors.torch import save_file
from trainer.data.cache_provenance import vae_identity, validate_cache, validate_dataset_caches, atomic_save


def test_content_identity_move_change_and_mode(tmp_path):
    vae = tmp_path / 'vae.safetensors'
    vae.write_bytes(b'weights-a')
    expected = vae_identity('', vae, 'moss_image')
    moved = tmp_path / 'moved.safetensors'
    moved.write_bytes(vae.read_bytes())
    assert expected == vae_identity('', moved, 'moss_image')
    cache = tmp_path / 'cache.safetensors'
    atomic_save({'latents': torch.ones(1)}, cache, {'cache_provenance': json.dumps(expected)})
    validate_cache(cache, expected, 'strict')
    vae.write_bytes(b'weights-b')
    with pytest.raises(ValueError, match='vae_sha256 mismatch'):
        validate_cache(cache, vae_identity('', vae, 'moss_image'))
    with pytest.raises(ValueError, match='normalization mismatch'):
        validate_cache(cache, vae_identity('', moved, 'mage_flow'))


def test_legacy_policy_and_unavailable_vae(tmp_path):
    cache = tmp_path / 'cache.safetensors'
    save_file({'latents': torch.ones(1)}, cache)
    with pytest.warns(UserWarning, match='unverified'):
        validate_cache(cache, {}, 'warn')
    with pytest.raises(ValueError, match='unverified'):
        validate_cache(cache, {}, 'strict')
    validate_cache(cache, {}, 'off')
    expected = vae_identity('', tmp_path / 'missing', 'moss_image')
    known = dict(expected, vae_sha256='abc')
    atomic_save({'latents': torch.ones(1)}, cache, {'cache_provenance': json.dumps(known)})
    with pytest.warns(UserWarning):
        validate_cache(cache, expected)
    with pytest.raises(ValueError, match='unverified'):
        validate_cache(cache, expected, 'strict')


def test_source_change_and_mixed_cache_rejected(tmp_path):
    from trainer.data.cache_provenance import file_sha256
    source = tmp_path / 'image.png'
    source.write_bytes(b'first')
    expected = vae_identity('', tmp_path / 'missing', 'moss_image')
    paths = [tmp_path / 'a.safetensors', tmp_path / 'b.safetensors']
    for path, digest in zip(paths, ('a', 'b')):
        atomic_save({'latents': torch.ones(1)}, path, {
            'cache_provenance': json.dumps(dict(expected, vae_sha256=digest)),
            'source_sha256': file_sha256(source)})
    source.write_bytes(b'changed')
    with pytest.raises(ValueError, match='source image content changed'):
        validate_cache(paths[0], source=source)
    data = NS(require_cache=True, entries=[NS(latent_path=p) for p in paths])
    train = NS(model_path='', vae_path=str(tmp_path / 'missing'), model_family='moss_image', flux2_vae=False)
    with pytest.warns(UserWarning), pytest.raises(ValueError, match='different VAEs'):
        validate_dataset_caches(data, train)


def test_writer_records_actual_vae_and_source_and_refuses_reuse(tmp_path):
    from PIL import Image
    from trainer.data.cache import LatentCacher, read_cache_metadata
    from trainer.data.bucket import BucketManager
    class VAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
        def encode(self, x):
            return torch.zeros(len(x), 128, x.shape[-2] // 16, x.shape[-1] // 16)
    image = tmp_path / 'image.png'
    Image.new('RGB', (256, 256)).save(image)
    provenance = dict(vae_sha256='test-vae', normalization='mage-native')
    cacher = LatentCacher(VAE(), device='cpu', provenance=provenance, cache_validation='strict')
    from trainer.data.dataset import DatasetConfig
    manager = DatasetConfig(path=str(tmp_path), resolution=256, min_bucket_reso=256).build_bucket_manager(256)
    path, _ = cacher.cache_image(image, manager)
    meta = read_cache_metadata(path)
    assert json.loads(meta['cache_provenance'])['encode_dtype'] == 'float32'
    assert meta['source_sha256']
    cacher.cache_image(image, manager)
    cacher.provenance['vae_sha256'] = 'different'
    with pytest.raises(ValueError, match='mismatch'):
        cacher.cache_batch([image], manager)
    cacher.cache_image(image, manager, overwrite=True)
    assert json.loads(read_cache_metadata(path)['cache_provenance'])['vae_sha256'] == 'different'
