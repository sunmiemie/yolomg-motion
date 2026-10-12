import cv2
import numpy as np
import pytest

from yolomg_motion import run_artifacts


def test_unicode_snapshot_roundtrip(tmp_path):
    path = tmp_path / '预览.png'
    pixels = np.arange(144, dtype=np.uint8).reshape(8, 6, 3)
    run_artifacts.write_png_atomic(path, pixels)
    decoded = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    np.testing.assert_array_equal(decoded, pixels)
    assert list(tmp_path.iterdir()) == [path]


def test_publish_failure_keeps_previous_preview(tmp_path, monkeypatch):
    path = tmp_path / 'preview.png'
    path.write_bytes(b'previous')
    def fail(*args):
        raise PermissionError('destination is in use')
    monkeypatch.setattr(run_artifacts.os, 'replace', fail)
    with pytest.raises(PermissionError):
        run_artifacts.write_png_atomic(path, np.zeros((4, 4, 3), np.uint8))
    assert path.read_bytes() == b'previous'
    assert list(tmp_path.iterdir()) == [path]
