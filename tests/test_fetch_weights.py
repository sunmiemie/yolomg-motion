import hashlib
import io

import pytest

from tools import fetch_weights


def test_verified_download_is_published_and_temporary_removed(tmp_path, monkeypatch):
    payload = b'checkpoint' * 100
    monkeypatch.setattr(fetch_weights, 'EXPECTED_SHA256', hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(fetch_weights.urllib.request, 'urlopen', lambda *a, **k: io.BytesIO(payload))
    path = tmp_path / 'model.pt'
    fetch_weights.fetch_weights(path)
    assert path.read_bytes() == payload
    assert list(tmp_path.iterdir()) == [path]
    monkeypatch.setattr(fetch_weights.urllib.request, 'urlopen', lambda *a, **k: pytest.fail('no re-download'))
    fetch_weights.fetch_weights(path)


def test_checksum_failure_never_publishes(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_weights.urllib.request, 'urlopen', lambda *a, **k: io.BytesIO(b'wrong bytes'))
    with pytest.raises(RuntimeError, match='SHA-256'):
        fetch_weights.fetch_weights(tmp_path / 'model.pt')
    assert not list(tmp_path.iterdir())


def test_concurrent_destination_is_not_overwritten(tmp_path, monkeypatch):
    payload = b'valid'
    path = tmp_path / 'model.pt'
    monkeypatch.setattr(fetch_weights, 'EXPECTED_SHA256', hashlib.sha256(payload).hexdigest())
    def response(*args, **kwargs):
        path.write_bytes(b'another process')
        return io.BytesIO(payload)
    monkeypatch.setattr(fetch_weights.urllib.request, 'urlopen', response)
    with pytest.raises(FileExistsError):
        fetch_weights.fetch_weights(path)
    assert path.read_bytes() == b'another process'
    assert list(tmp_path.iterdir()) == [path]
