"""Zoom acquisition must work without an initial whole-frame YOLO detection."""
from types import SimpleNamespace

import numpy as np
import pytest

from yolomg_motion.tracking import SingleTargetTracker
from yolomg_motion.zoom_pipeline import ZoomROIPipeline


def image(index=0, target=True):
    frame = np.full((256, 384, 3), 40, np.uint8)
    if target:
        x = 80 + index
        frame[110:118, x:x+12] = 230
        frame[110:118, x+3:x+5] = 80
    return frame


class Motion:
    config = SimpleNamespace(frame_step=2)

    def __init__(self):
        self.shapes = []
        self.valid = True
        self.no_proposals = False

    def process_triplet(self, previous, current, following):
        assert previous.shape == current.shape == following.shape
        self.shapes.append(current.shape[:2])
        difference = np.maximum(current[:, :, 0].astype(np.float32)-40, 0)
        if self.no_proposals:
            difference[:] = 0
        return SimpleNamespace(success=self.valid, difference=difference,
                               valid_mask=np.ones(current.shape[:2], bool))


class CropDetector:
    """Only finds the object in a crop; deliberately rejects whole images."""
    last_input_shape = (1, 3, 128, 128)

    def __init__(self):
        self.shapes = []
        self.empty = False

    def detect(self, rgb, motion):
        assert motion.shape == rgb.shape[:2]
        assert motion.dtype == np.float32
        assert max(rgb.shape[:2]) <= 128, 'Whole-frame detection is not allowed'
        self.shapes.append(rgb.shape[:2])
        if self.empty:
            return []
        ys, xs = np.nonzero(rgb[:, :, 0] > 200)
        if not len(xs):
            return []
        # The two channels represent the same location before model resizing.
        assert motion[ys, xs].mean() > 100
        return [SimpleNamespace(bbox_xyxy=(float(xs.min()), float(ys.min()),
                                          float(xs.max()+1), float(ys.max()+1)), confidence=.9)]


def pipeline(**kwargs):
    full, local, detector = Motion(), Motion(), CropDetector()
    tracker = SingleTargetTracker(max_detector_gap=None, roi_only=True,
                                  min_template_width=8, min_template_height=8,
                                  template_padding=2, min_confidence=.1)
    p = ZoomROIPipeline(detector, full, tracker, local_extractor=local,
                        detect_every=4, roi_size=64, motion_context=128, **kwargs)
    return p, detector, full, local


def test_motion_proposal_acquires_then_only_local_context_is_processed():
    p, detector, full, local = pipeline()
    rows = [r for i in range(40) if (r := p.push(image(i))) is not None]
    assert [r.frame_index for r in rows] == list(range(2, 38))
    assert rows[4].result.state == 'LOCKED'
    assert full.shapes == [(256, 384)]
    assert local.shapes and all(shape == (128, 128) for shape in local.shapes)
    assert all(shape == (64, 64) for shape in detector.shapes)
    assert len(detector.shapes) == sum(r.details['detector_calls'] for r in rows)
    for r in rows[4:]:
        assert r.result.state == 'LOCKED'
        np.testing.assert_allclose(r.result.bbox, (80+r.frame_index, 110, 92+r.frame_index, 118), atol=1)
        if not r.details['detector_ran']:
            assert not r.details['motion_computed']
            assert r.details['motion_valid'] is None
            assert r.details['detections'] is None
        else:
            assert r.result.source == 'detector'
            assert r.details['motion_scope'] == 'local'
            assert r.details['network_input_shapes'] == [[1, 3, 128, 128]]


def test_failed_verification_stops_stale_box_and_never_bypasses_budget():
    p, detector, full, local = pipeline()
    rows = []
    for i in range(32):
        if i == 12:  # next center is 10, a scheduled local verification
            detector.empty = True
        r = p.push(image(i))
        if r is not None:
            rows.append(r)
    failed = next(r for r in rows if r.frame_index == 10)
    assert failed.result.state == 'LOST' and failed.result.bbox is None
    assert all(r.result.bbox is None for r in rows if r.frame_index >= 10)
    attempts = [r.frame_index for r in rows if r.details['motion_computed']]
    assert all(b-a >= 4 for a, b in zip(attempts, attempts[1:]))
    recoveries = [r for r in rows if r.details['detection_reason'] == 'expanded_reacquisition']
    assert len(recoveries) == 2
    assert [r.details['detection_windows'][0][2]-r.details['detection_windows'][0][0]
            for r in recoveries] == [96, 128]
    assert any(r.details['motion_scope'] == 'full' for r in rows if r.frame_index > 18)
    assert all(r.details['detector_calls'] <= 2 for r in rows)


def test_invalid_motion_does_not_feed_zero_map_to_model_and_is_rate_limited():
    p, detector, full, local = pipeline()
    full.valid = False
    rows = [r for i in range(20) if (r := p.push(image(i))) is not None]
    assert not detector.shapes and not local.shapes
    assert len(full.shapes) == 4
    assert all(r.result.state == 'SEARCHING' for r in rows)
    assert all(r.details['motion_valid'] is False for r in rows if r.details['motion_computed'])


def test_weak_motion_uses_bounded_round_robin_scan_without_fake_target():
    p, detector, full, local = pipeline(max_rois=1)
    full.no_proposals = True
    rows = [r for i in range(24) if (r := p.push(image(i, target=False))) is not None]
    attempts = [r for r in rows if r.details['motion_computed']]
    assert all(r.details['proposal_count'] == 0 for r in attempts)
    assert all(r.details['detector_calls'] == 1 for r in attempts)
    assert len({tuple(r.details['detection_windows'][0]) for r in attempts}) == len(attempts)
    assert all(r.result.bbox is None for r in rows)


def test_periodic_scan_not_starved_by_strong_motion_candidates():
    p, detector, full, local = pipeline(max_rois=1, scan_every=2)
    detector.empty = True
    rows = [r for i in range(16) if (r := p.push(image(i))) is not None]
    attempts = [r for r in rows if r.details['motion_computed']]
    assert attempts[0].details['proposal_count'] == 1
    assert attempts[1].details['detection_windows'] == [[0, 0, 64, 64]]
    assert all(r.details['detector_calls'] == 1 for r in attempts)


def test_scan_replaces_near_duplicate_proposal_and_keeps_distinct_window(monkeypatch):
    p, detector, _, _ = pipeline(max_rois=2, scan_every=1)
    detector.empty = True
    # First proposal differs from the scheduled scan by only one pixel. The
    # remaining inference slot should cover the second, distinct proposal.
    monkeypatch.setattr('yolomg_motion.zoom_pipeline.motion_windows',
                        lambda *args, **kwargs: [(1, 1, 65, 65), (180, 100, 244, 164)])
    for i in range(5):
        row = p.push(image(i, target=False))
    assert row.details['proposal_count'] == 2
    assert row.details['detection_windows'] == [[180, 100, 244, 164], [0, 0, 64, 64]]
    assert row.details['detector_calls'] == 2
    assert len(detector.shapes) == 2
    assert row.result.bbox is None


def test_shape_validation_and_prediction_is_not_an_observation():
    p, _, _, _ = pipeline()
    p.push(image())
    with pytest.raises(ValueError, match='shape changed'):
        p.push(np.zeros((128, 256, 3), np.uint8))
    for value in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            ZoomROIPipeline(CropDetector(), Motion(), detect_every=value)
    with pytest.raises(ValueError):
        ZoomROIPipeline(CropDetector(), Motion(), motion_context=100, roi_size=320)
    assert p.tracker.predicted_box(0) is None


def test_artificial_context_edge_is_not_treated_as_physical_frame_edge():
    p, detector, _, _ = pipeline()
    for i in range(12):
        p.push(image(i))
    assert p.tracker.state == 'LOCKED'
    # Reacquisition can expand the detector window to the entire context.
    p.motion_context = p.roi_size
    def truncated(rgb, motion):
        return [SimpleNamespace(bbox_xyxy=(0., 20., 12., 28.), confidence=.99)]
    detector.detect = truncated
    r = p.push(image(12))  # source frame 10: scheduled local verification
    assert r.details['detector_calls'] == 1
    assert r.details['detections'] == []
    assert r.result.state == 'LOST' and r.result.bbox is None
