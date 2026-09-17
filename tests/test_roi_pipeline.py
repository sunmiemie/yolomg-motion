"""State/scheduling tests plus a real local matcher; no downloaded model needed."""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from yolomg_motion.roi_pipeline import ROIPipeline, timing_statistics
from yolomg_motion.tracking import Detection, SingleTargetTracker


def frame(index, target=True, shape=(128, 240)):
    image = np.full((*shape, 3), 40, np.uint8)
    if target:
        x = 40 + index
        image[50:56, x:x+12] = 230
        image[50:56, x+3:x+5] = 70
    return image


class MotionStub:
    config = SimpleNamespace(frame_step=2)
    def __init__(self):
        self.calls = 0
        self.valid = True

    def process_triplet(self, previous, current, following):
        self.calls += 1
        return SimpleNamespace(success=self.valid, difference=np.zeros(current.shape[:2], np.float32))


class DetectorStub:
    def __init__(self):
        self.calls = 0
        self.empty = False

    def detect(self, image, motion):
        self.calls += 1
        if self.empty:
            return []
        ys, xs = np.nonzero(image[:, :, 0] > 200)
        if not len(xs):
            return []
        return [SimpleNamespace(bbox_xyxy=(float(xs.min()), float(ys.min()), float(xs.max()+1), float(ys.max()+1)), confidence=.9)]


def pipeline(**kwargs):
    detector, motion = DetectorStub(), MotionStub()
    tracker = SingleTargetTracker(max_detector_gap=None, roi_only=True)
    return ROIPipeline(detector, motion, tracker, **kwargs), detector, motion


def test_lock_skips_all_motion_and_detection_without_old_gap_expiry():
    p, d, m = pipeline(detect_every=3)
    observations = [r for i in range(100) if (r := p.push(frame(i))) is not None]
    assert [r.frame_index for r in observations] == list(range(2, 98))
    assert m.calls == d.calls == 2
    assert observations[3].frame_index == 5 and observations[3].result.state == 'LOCKED'
    for r in observations[4:]:
        assert r.result.state == 'LOCKED' and r.result.source == 'template'
        assert not r.details['motion_computed'] and not r.details['detector_ran']
        assert r.details['motion_valid'] is None and r.details['roi_tracking']
        np.testing.assert_allclose(r.result.bbox, (40+r.frame_index, 50, 52+r.frame_index, 56), atol=1)


def test_disappearance_returns_no_box_then_reacquires_new_identity():
    p, d, m = pipeline()
    observations = [r for i in range(45) if (r := p.push(frame(i, not 18 <= i < 27))) is not None]
    disappeared = [r for r in observations if 18 <= r.frame_index < 27]
    assert all(r.result.state == 'LOST' and r.result.bbox is None for r in disappeared)
    after = [r for r in observations if r.frame_index >= 32]
    assert all(r.result.state == 'LOCKED' and r.result.track_id == 2 for r in after)
    assert any(r.details['detection_reason'] == 'reacquisition' for r in disappeared)


def test_verification_failure_breaks_visual_lock():
    p, d, m = pipeline(verify_every=6)
    rows = []
    for i in range(17):
        if i == 13:  # output center 11, six frames after confirmation
            d.empty = True
        r = p.push(frame(i))
        if r is not None:
            rows.append(r)
    verify = next(r for r in rows if r.details['detection_reason'] == 'verification')
    assert verify.frame_index == 11 and verify.result.state == 'LOST'
    assert verify.result.bbox is None


def test_invalid_motion_never_calls_detector_or_fakes_valid_observation():
    p, d, m = pipeline()
    m.valid = False
    rows = [r for i in range(11) if (r := p.push(frame(i))) is not None]
    assert m.calls == 3 and d.calls == 0
    assert all(r.result.state == 'SEARCHING' for r in rows)
    assert [r.details['motion_valid'] for r in rows] == [False, None, None, False, None, None, False]


def test_2k_roi_never_converts_full_frame_and_handles_image_edge(monkeypatch):
    tracker = SingleTargetTracker(confirmations=1, max_detector_gap=None, roi_only=True)
    image = frame(0, shape=(1440, 2560))
    shapes = []
    original = cv2.cvtColor
    def observe(image, code, *args, **kwargs):
        shapes.append(image.shape)
        return original(image, code, *args, **kwargs)
    monkeypatch.setattr(cv2, 'cvtColor', observe)
    tracker.update(image, [Detection((40, 50, 52, 56), .9)], 0)
    for i in range(1, 10):
        result = tracker.update(frame(i, shape=(1440, 2560)), None, i)
        assert result.state == 'LOCKED'
    assert shapes and all(h < 128 and w < 128 for h, w, _ in shapes)
    tracker.reset()
    edge = np.full((1440, 2560, 3), 40, np.uint8)
    edge[2:8, 1:13] = 220
    assert tracker.update(edge, [Detection((1, 2, 13, 8), .9)], 0).state == 'LOCKED'
    assert tracker.update(edge, None, 1).state == 'LOCKED'
    assert min(tracker.last_search_roi) >= 0


def test_validation_and_timing_statistics():
    p, _, _ = pipeline()
    p.push(frame(0))
    with pytest.raises(ValueError, match='shape changed'):
        p.push(frame(1, shape=(1440, 2560)))
    with pytest.raises(ValueError):
        pipeline(verify_every=-1)
    with pytest.raises(ValueError):
        SingleTargetTracker(roi_only=True, backend='opencl')
    assert timing_statistics([])['samples'] == 0
    report = timing_statistics([.001, .002, .030])
    assert report['over_budget_frames'] == 1 and report['max_ms'] == 30


def test_tight_roi_tracks_target_over_fixed_textured_background():
    rng = np.random.default_rng(217)
    background = rng.integers(30, 180, (180, 400, 3), dtype=np.uint8)
    target = rng.integers(190, 255, (8, 12, 3), dtype=np.uint8)
    target[:, 4:6] = 20
    tracker = SingleTargetTracker(confirmations=1, max_detector_gap=None, roi_only=True,
                                  min_template_width=8, min_template_height=8, template_padding=2)
    for i in range(80):
        image = background.copy()
        x, y = 60+i*2, 70
        image[y:y+8, x:x+12] = target
        detections = [Detection((x, y, x+12, y+8), .9)] if i == 0 else None
        result = tracker.update(image, detections, i)
        assert result.state == 'LOCKED'
        np.testing.assert_allclose(result.bbox, (x, y, x+12, y+8), atol=1)
