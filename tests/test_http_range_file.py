import io

import pytest

from tools.fetch_ard100_archive import HTTPRangeFile


class Response:
    status_code = 206
    def __init__(self, content, headers):
        self.raw = io.BytesIO(content)
        self.headers = headers
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def raise_for_status(self): pass


def test_seek_read_and_closed_contract(monkeypatch):
    remote = HTTPRangeFile('https://example.test/archive', 4)
    requests = []
    def get(url, **kwargs):
        requests.append(kwargs)
        return Response(b'ab', {'Content-Range': 'bytes 0-1/4'})
    monkeypatch.setattr(remote.session, 'get', get)
    assert remote.read(2) == b'ab'
    assert requests[0]['headers']['Accept-Encoding'] == 'identity'
    assert remote.tell() == 2
    with pytest.raises(ValueError): remote.seek(0, 7)
    with pytest.raises(TypeError): remote.read(1.5)
    assert remote.seek(1, 2) == 5
    assert remote.read() == b''
    remote.close()
    for operation in (remote.tell, remote.read, lambda: remote.seek(0)):
        with pytest.raises(ValueError): operation()


def test_encoded_or_truncated_response_does_not_advance(monkeypatch):
    with HTTPRangeFile('https://example.test/archive', 4) as remote:
        monkeypatch.setattr(remote.session, 'get', lambda *a, **k: Response(
            b'abcd', {'Content-Range': 'bytes 0-3/4', 'Content-Encoding': 'gzip'}))
        with pytest.raises(IOError, match='encoded'): remote.read()
        assert remote.tell() == 0
        monkeypatch.setattr(remote.session, 'get', lambda *a, **k: Response(
            b'ab', {'Content-Range': 'bytes 0-3/4'}))
        with pytest.raises(IOError, match='truncated'): remote.read()
        assert remote.tell() == 0


def test_invalid_archive_size_rejected():
    with pytest.raises(ValueError): HTTPRangeFile('unused', -1)


def test_ranges_cannot_mix_different_archive_versions(monkeypatch):
    with HTTPRangeFile('https://example.test/archive', 4) as remote:
        calls = []
        def get(url, **kwargs):
            calls.append(kwargs['headers'])
            if len(calls) == 1:
                return Response(b'ab', {'Content-Range': 'bytes 0-1/4', 'ETag': '"original"'})
            return Response(b'cd', {'Content-Range': 'bytes 2-3/4', 'ETag': '"changed"'})
        monkeypatch.setattr(remote.session, 'get', get)
        assert remote.read(2) == b'ab'
        with pytest.raises(IOError, match='changed'): remote.read(2)
        assert calls[1]['If-Range'] == '"original"'
        assert remote.tell() == 2
