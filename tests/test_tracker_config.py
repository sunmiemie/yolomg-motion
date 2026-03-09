import numpy as np
import pytest

from yolomg_motion.tracking import SingleTargetTracker


@pytest.mark.parametrize('settings', [
    {'confirmations': 1.9}, {'max_detector_gap': True}, {'search_radius': float('inf')},
    {'min_template_width': 2.5}, {'min_template_height': '16'},
    {'max_size_ratio': float('nan')}, {'min_template_std': float('inf')},
    {'min_confidence': True}, {'appearance_threshold': '0.5'},
])
def test_rejects_invalid_settings_before_state_creation(settings):
    with pytest.raises(ValueError, match=next(iter(settings))):
        SingleTargetTracker(**settings)


def test_accepts_numpy_numeric_settings():
    tracker = SingleTargetTracker(confirmations=np.int64(2), min_template_std=np.float32(2))
    assert tracker.confirmations == 2


def test_boolean_index_does_not_consume_first_frame():
    tracker = SingleTargetTracker()
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match='frame_index'):
        tracker.update(frame, [], True)
    assert tracker.update(frame, [], 0).state == 'SEARCHING'
