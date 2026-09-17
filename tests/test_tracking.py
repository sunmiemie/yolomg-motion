"""Synthetic tracking checks; no particular video coordinates or weights."""
import numpy as np
import pytest

from yolomg_motion.tracking import Detection, SingleTargetTracker


WIDTH, HEIGHT = 240, 128


def _frame(box=None, distractor=None):
    rng = np.random.default_rng(392)
    frame = rng.integers(35, 42, (HEIGHT, WIDTH, 3), dtype=np.uint8)
    for target in (box, distractor):
        if target is None:
            continue
        x1, y1, x2, y2 = map(int, target)
        frame[y1:y2, x1:x2] = (230, 220, 250)
        frame[y1:y2, x1 + 2:x1 + 4] = (70, 100, 160)
    return frame


def _box(index):
    return (48 + 2 * index, 52, 58 + 2 * index, 55)


def _confirm(tracker):
    first = tracker.update(_frame(_box(0)), [Detection(_box(0), 0.8)], 0)
    assert first.state == 'CONFIRMING'
    assert first.track_id is None
    second = tracker.update(_frame(_box(1)), [Detection(_box(1), 0.8)], 1)
    assert second.state == 'LOCKED'
    assert second.track_id == 1
    return second


def test_detector_confirmation_and_tiny_box_template_intervals():
    tracker = SingleTargetTracker()
    _confirm(tracker)

    for index in range(2, 8):
        result = tracker.update(_frame(_box(index)), None, index)
        assert result.state == 'LOCKED'
        assert result.source == 'template'
        assert result.track_id == 1
        assert result.detector_confidence is None
        assert result.appearance_score >= tracker.appearance_threshold
        np.testing.assert_allclose(result.bbox, _box(index), atol=1.0)
        # A 24 x 16 or larger appearance patch must not inflate the tiny box.
        assert result.bbox[2] - result.bbox[0] == 10
        assert result.bbox[3] - result.bbox[1] == 3


def test_empty_detector_run_breaks_confirmation_but_skipped_run_does_not():
    tracker = SingleTargetTracker()
    first = tracker.update(_frame(_box(0)), [Detection(_box(0), 0.8)], 0)
    assert first.state == 'CONFIRMING'
    skipped = tracker.update(_frame(_box(1)), None, 1)
    assert skipped.state == 'CONFIRMING'
    missing = tracker.update(_frame(_box(2)), [], 2)
    assert missing.state == 'SEARCHING'
    renewed = tracker.update(_frame(_box(3)), [Detection(_box(3), 0.8)], 3)
    assert renewed.state == 'CONFIRMING'
    confirmed = tracker.update(_frame(_box(4)), [Detection(_box(4), 0.8)], 4)
    assert confirmed.state == 'LOCKED'


def test_local_association_rejects_distant_higher_confidence_target():
    tracker = SingleTargetTracker()
    _confirm(tracker)
    distant = (178, 80, 188, 83)
    result = tracker.update(_frame(_box(2), distant),
                            [Detection(distant, 0.99), Detection(_box(2), 0.55)], 2)
    assert result.state == 'LOCKED'
    assert result.source == 'detector'
    assert result.detector_confidence == 0.55
    np.testing.assert_allclose(result.bbox, _box(2))
    # When only the distant target is detected, retain the locally supported
    # old target rather than assigning its identity to the higher score.
    result = tracker.update(_frame(_box(3), distant), [Detection(distant, 0.99)], 3)
    assert result.state == 'LOCKED'
    assert result.source == 'template'
    assert result.track_id == 1
    np.testing.assert_allclose(result.bbox, _box(3), atol=1.0)


def test_template_cannot_extend_detector_gap_forever_and_reacquires_new_id():
    tracker = SingleTargetTracker(max_detector_gap=4)
    _confirm(tracker)
    for index in range(2, 6):
        result = tracker.update(_frame(_box(index)), [], index)
        assert result.state == 'LOCKED'
    expired = tracker.update(_frame(_box(6)), None, 6)
    assert expired.state == 'LOST'
    assert expired.bbox is None
    assert expired.source == 'none'
    # Even a visible target requires new detector confirmations after expiry.
    pending = tracker.update(_frame(_box(7)), [Detection(_box(7), 0.8)], 7)
    assert pending.state == 'CONFIRMING'
    reacquired = tracker.update(_frame(_box(8)), [Detection(_box(8), 0.8)], 8)
    assert reacquired.state == 'LOCKED'
    assert reacquired.track_id == 2


def test_disappearing_target_does_not_return_predicted_phantom_box():
    tracker = SingleTargetTracker()
    _confirm(tracker)
    disappeared = tracker.update(_frame(), [], 2)
    assert disappeared.state == 'LOST'
    assert disappeared.bbox is None
    assert disappeared.source == 'none'


def test_low_confidence_and_invalid_boxes_cannot_start_track():
    tracker = SingleTargetTracker()
    result = tracker.update(_frame(_box(0)), [Detection(_box(0), 0.01),
                            Detection((20, 20, 10, 30), 0.9),
                            Detection((np.nan, 0, 10, 10), 0.9)], 0)
    assert result.state == 'SEARCHING'
    assert result.bbox is None


def test_source_frame_indices_control_expiry_and_must_increase():
    tracker = SingleTargetTracker(max_detector_gap=4)
    _confirm(tracker)
    # Skipping source indices still consumes the detector gap budget.
    result = tracker.update(_frame(_box(6)), None, 6)
    assert result.state == 'LOST'
    assert result.bbox is None
    with pytest.raises(ValueError, match='strictly increase'):
        tracker.update(_frame(), None, 6)
