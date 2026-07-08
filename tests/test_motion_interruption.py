import json
import numpy as np
import pytest
from yolomg_motion import cli


@pytest.mark.parametrize('failure', [RuntimeError('encoder failed'), KeyboardInterrupt()])
def test_motion_failure_saves_partial_diagnostics(tmp_path, monkeypatch, failure):
    video = tmp_path / 'input.mp4'
    video.touch()
    frame = np.zeros((64, 64, 3), np.uint8)
    class Capture:
        released = False
        def isOpened(self): return True
        def get(self, key): return 30
        def read(self): return True, frame.copy()
        def release(self): self.released = True
    cap = Capture()
    monkeypatch.setattr(cli.cv2, 'VideoCapture', lambda *a: cap)
    monkeypatch.setattr(cli, 'preview', lambda *a: (_ for _ in ()).throw(failure))
    output = tmp_path / 'run'
    with pytest.raises(type(failure)):
        cli.main(['--input', str(video), '--output', str(output), '--max-frames', '5'])
    summary = json.loads((output / 'summary.json').read_text())
    rows = json.loads((output / 'diagnostics.json').read_text())
    assert cap.released and len(rows) == summary['output_frames'] == 1
    assert summary['partial'] and not summary['completed']
    assert summary['error']['type'] == type(failure).__name__
    assert summary['status'] == ('interrupted' if isinstance(failure, KeyboardInterrupt) else 'failed')
    assert not list(output.glob('*.tmp'))
