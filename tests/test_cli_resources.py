from pathlib import Path
import pytest

from yolomg_motion import cli, track_cli, yolo_detector


class Capture:
    released = False
    def isOpened(self): return True
    def get(self, key): return 30.0
    def release(self): self.released = True


@pytest.mark.parametrize('entry', [cli.main, track_cli.main])
def test_output_creation_failure_releases_capture(tmp_path, monkeypatch, entry):
    video = tmp_path / 'input.mp4'
    video.touch()
    capture = Capture()
    monkeypatch.setattr(cli.cv2, 'VideoCapture', lambda *a: capture)
    monkeypatch.setattr(yolo_detector, 'YOLOMGDetector', lambda **k: object())
    monkeypatch.setattr(Path, 'mkdir', lambda *a, **k: (_ for _ in ()).throw(OSError('read-only output')))
    with pytest.raises(OSError, match='read-only'):
        entry(['--input', str(video), '--output', str(tmp_path / 'output')])
    assert capture.released


def test_invalid_fps_does_not_create_motion_output(tmp_path, monkeypatch):
    video = tmp_path / 'input.mp4'
    video.touch()
    capture = Capture()
    monkeypatch.setattr(capture, 'get', lambda key: float('nan'))
    monkeypatch.setattr(cli.cv2, 'VideoCapture', lambda *a: capture)
    output = tmp_path / 'output'
    with pytest.raises(SystemExit): cli.main(['--input', str(video), '--output', str(output)])
    assert capture.released and not output.exists()
