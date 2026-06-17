import numpy as np
import pytest

from yolomg_motion import tracking


def test_preprocessing_failure_does_not_consume_frame_index(monkeypatch):
    tracker = tracking.SingleTargetTracker(confirmations=1)
    frame = np.zeros((32, 32, 3), np.uint8)
    convert = tracking.cv2.cvtColor
    monkeypatch.setattr(tracking.cv2, 'cvtColor', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('device failed')))
    with pytest.raises(RuntimeError): tracker.update(frame, [], 4)
    assert tracker._last_frame_index is None
    monkeypatch.setattr(tracking.cv2, 'cvtColor', convert)
    assert tracker.update(frame, [tracking.Detection((2, 2, 10, 10), .9)], 4).state == 'LOCKED'


def test_failed_detection_iteration_does_not_consume_frame_index():
    tracker = tracking.SingleTargetTracker()
    frame = np.zeros((32, 32, 3), np.uint8)
    def broken():
        raise ValueError('invalid batch')
        yield
    with pytest.raises(ValueError): tracker.update(frame, broken(), 2)
    assert tracker.update(frame, [], 2).state == 'SEARCHING'
