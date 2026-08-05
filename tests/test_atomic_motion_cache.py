import numpy as np
import pytest

from yolomg_motion.run_artifacts import write_npz_atomic


def test_roundtrip_unicode_cache_without_pickle(tmp_path):
    path = tmp_path / '缓存' / '000002.npz'
    write_npz_atomic(path, center_index=np.int64(2), difference=np.ones((4, 4), np.float32))
    with np.load(path, allow_pickle=False) as data:
        assert data['center_index'].item() == 2
        np.testing.assert_equal(data['difference'], np.ones((4, 4)))
    assert list(path.parent.iterdir()) == [path]


def test_interrupted_compression_preserves_previous_cache(tmp_path, monkeypatch):
    path = tmp_path / 'motion.npz'
    path.write_bytes(b'old complete file')
    def fail(stream, **arrays):
        stream.write(b'partial zip')
        raise KeyboardInterrupt()
    monkeypatch.setattr(np, 'savez_compressed', fail)
    with pytest.raises(KeyboardInterrupt):
        write_npz_atomic(path, difference=np.zeros((4, 4)))
    assert path.read_bytes() == b'old complete file'
    assert list(tmp_path.iterdir()) == [path]
