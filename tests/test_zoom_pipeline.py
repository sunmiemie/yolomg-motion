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


def test_contrast_search_can_acquire_weak_motion_target_without_extra_budget():
    p, detector, full, local = pipeline(contrast_search=True)
    full.no_proposals = True

    def detect(rgb, motion):
        assert motion.shape == rgb.shape[:2]
        ys, xs = np.nonzero(rgb[:, :, 0] > 200)
        return [] if not len(xs) else [SimpleNamespace(
            bbox_xyxy=(float(xs.min()), float(ys.min()), float(xs.max()+1), float(ys.max()+1)),
            confidence=.9)]

    detector.detect = detect
    rows = [r for i in range(20) if (r := p.push(image(i))) is not None]
    assert rows[0].details['proposal_count'] == 0
    assert rows[0].details['contrast_window'] is not None
    assert rows[0].result.source == 'detector'
    assert rows[4].result.state == 'LOCKED'
    assert all(r.result.state == 'LOCKED' for r in rows[4:])
    attempts = [r.frame_index for r in rows if r.details['detector_ran']]
    assert all(b - a == 4 for a, b in zip(attempts, attempts[1:]))
    assert all(r.details['detector_calls'] <= 2 for r in rows)
    assert all(r.details['contrast_window'] is None for r in rows[1:])


def test_contrast_candidates_cannot_lock_without_yolo_and_cannot_starve_scan():
    p, detector, full, _ = pipeline(contrast_search=True, max_rois=1, scan_every=3)
    full.no_proposals = True
    detector.empty = True
    rows = [r for i in range(28) if (r := p.push(image(i))) is not None]
    attempts = [r for r in rows if r.details['detector_ran']]
    assert attempts[0].details['contrast_window'] is not None
    assert attempts[2].details['contrast_window'] is None
    assert attempts[5].details['contrast_window'] is None
    assert all(r.details['detector_calls'] == 1 for r in attempts)
    assert all(r.result.bbox is None for r in rows)


def test_contrast_search_cannot_bypass_failed_camera_registration():
    p, detector, full, _ = pipeline(contrast_search=True)
    full.valid = False
    rows = [r for i in range(16) if (r := p.push(image(i))) is not None]
    assert not detector.shapes
    assert all(r.details['contrast_window'] is None for r in rows)


@pytest.mark.parametrize('budget', [1, 2])
def test_contrast_keeps_periodic_scan_slot_and_never_skips_unexecuted_tiles(monkeypatch, budget):
    p, detector, _, _ = pipeline(contrast_search=True, max_rois=budget, scan_every=2)
    detector.empty = True
    monkeypatch.setattr('yolomg_motion.zoom_pipeline.motion_windows',
                        lambda *args, **kwargs: [(200, 120, 264, 184), (280, 120, 344, 184)][:budget])
    monkeypatch.setattr('yolomg_motion.zoom_pipeline.contrast_windows',
                        lambda *args, **kwargs: [(80, 100, 144, 164)])
    rows = [r for i in range(28) if (r := p.push(image(i))) is not None]
    attempts = [r for r in rows if r.details['detector_ran']]
    for number, row in enumerate(attempts[1::2]):
        assert [48 * number, 0, 48 * number + 64, 64] in row.details['detection_windows']
        assert row.details['detector_calls'] == budget
        if budget == 2:
            assert [200, 120, 264, 184] in row.details['detection_windows']
    assert p._scan_cursor == len(attempts[1::2])


def test_scale_retry_only_on_empty_local_detection_reuses_context_and_budget():
    p, detector, _, local = pipeline(scale_retry=True)
    for i in range(12):
        p.push(image(i))
    before = len(local.shapes)
    real_detect = detector.detect
    detector.detect = lambda rgb, motion: [] if rgb.shape[0] == 64 else real_detect(rgb, motion)
    row = p.push(image(12))
    assert row.result.state == 'LOCKED' and row.result.source == 'detector'
    assert row.details['scale_retry_used']
    assert row.details['detector_calls'] == 2
    assert [w[2] - w[0] for w in row.details['detection_windows']] == [64, 32]
    assert len(local.shapes) == before + 1


def test_scale_retry_cannot_exceed_one_slot_or_run_on_intermediate_frames():
    p, detector, _, _ = pipeline(scale_retry=True, max_rois=1)
    for i in range(12):
        row = p.push(image(i))
        if row:
            assert not row.details['scale_retry_used']
    detector.empty = True
    row = p.push(image(12))
    assert row.details['detector_calls'] == 1
    assert not row.details['scale_retry_used'] and row.result.state == 'LOST'


def _recent_search_with_competing_contrast(monkeypatch, max_rois):
    from yolomg_motion.predictive_tracking import PredictiveTargetTracker

    full, local, detector = Motion(), Motion(), CropDetector()
    tracker = PredictiveTargetTracker(
        max_detector_gap=None, roi_only=True, min_template_width=8,
        min_template_height=8, template_padding=2, min_confidence=.1,
        prediction_max_gap=9, verification_grace=0)
    p = ZoomROIPipeline(
        detector, full, tracker, local_extractor=local, detect_every=4,
        roi_size=64, motion_context=128, max_rois=max_rois, scan_every=2,
        recent_search_attempts=2)
    monkeypatch.setattr('yolomg_motion.zoom_pipeline.contrast_windows',
                        lambda *args, **kwargs: [(240, 120, 304, 184)])
    rows = {}
    for i in range(29):
        if i == 16:
            # A real confirmed identity loses detection, then its prediction
            # expires. The two recent-target attempts compete with contrast.
            detector.empty = True
            full.no_proposals = True
            p.contrast_search = True
        row = p.push(image(i))
        if row is not None:
            rows[row.frame_index] = row
    assert rows[10].result.state == 'LOCKED'
    assert rows[14].result.state == 'LOST'
    assert rows[22].details['prediction'] is None
    return p, rows


def test_two_slots_preserve_periodic_scan_and_recent_target_ahead_of_contrast(monkeypatch):
    p, rows = _recent_search_with_competing_contrast(monkeypatch, max_rois=2)
    row = rows[22]
    # The original detector box was (90, 110, 102, 118); its 64px crop
    # remains available alongside the first global scan, without a third call.
    assert row.details['detection_windows'] == [[0, 0, 64, 64], [64, 82, 128, 146]]
    assert row.details['detection_reason'] == 'recent_target_search'
    assert row.details['contrast_window'] is None
    assert row.details['detector_calls'] == 2
    assert row.result.bbox is None
    assert p._scan_cursor == 1


def test_one_slot_reports_actual_scan_then_recent_target_without_skipping_tiles(monkeypatch):
    p, rows = _recent_search_with_competing_contrast(monkeypatch, max_rois=1)
    scan, recent = rows[22], rows[26]
    assert scan.details['detection_windows'] == [[0, 0, 64, 64]]
    assert scan.details['detection_reason'] == 'motion_search'
    assert recent.details['detection_windows'] == [[64, 82, 128, 146]]
    assert recent.details['detection_reason'] == 'recent_target_search'
    for row in (scan, recent):
        assert row.details['contrast_window'] is None
        assert row.details['detector_calls'] == 1
        assert row.result.bbox is None
    assert p._scan_cursor == 1


@pytest.mark.parametrize('offset,expected_state', [(0, 'LOCKED'), (64, 'LOST')])
def test_scale_retry_rejects_distant_background_but_accepts_nearby_candidate(offset, expected_state):
    full, local = Motion(), Motion()
    detector = SimpleNamespace(last_input_shape=(1, 3, 640, 640))

    def visible_target(rgb, motion):
        ys, xs = np.nonzero(rgb[:, :, 0] > 200)
        return [] if not len(xs) else [SimpleNamespace(
            bbox_xyxy=(float(xs.min()), float(ys.min()), float(xs.max()+1), float(ys.max()+1)),
            confidence=.9)]

    detector.detect = visible_target
    p = ZoomROIPipeline(detector, full, local_extractor=local, detect_every=4,
                        roi_size=320, motion_context=384, scale_retry=True)

    def padded_image(index):
        return np.pad(image(index), ((128, 128), (128, 128), (0, 0)),
                      mode='constant', constant_values=40)

    for i in range(12):
        p.push(padded_image(i))
    assert p.tracker.state == 'LOCKED'

    def retry_candidate(rgb, motion):
        if rgb.shape[0] == 320:
            return []
        assert rgb.shape[:2] == (224, 224)
        # Both boxes fit the tight crop and have high confidence. Only the
        # nearby candidate is consistent with the previous local track.
        cx, cy = 112 + offset, 112
        return [SimpleNamespace(bbox_xyxy=(cx-6., cy-4., cx+6., cy+4.), confidence=.99)]

    detector.detect = retry_candidate
    row = p.push(padded_image(12))
    assert row.details['scale_retry_used'] and row.details['detector_calls'] == 2
    assert row.result.state == expected_state
    if offset:
        assert row.details['detections'] == [] and row.result.bbox is None
    else:
        assert row.result.source == 'detector' and len(row.details['detections']) == 1


@pytest.mark.parametrize('reason', ['textured_background', 'large_target'])
def test_scale_retry_does_not_spend_spare_call_when_background_or_target_is_unsuitable(reason):
    p, detector, _, _ = pipeline(scale_retry=True)

    def scene(index):
        frame = image(index)
        if reason == 'large_target':
            x = 80 + index
            frame[110:118, x:x+40] = 230
            frame[110:118, x+3:x+5] = 80
        if reason == 'textured_background' and index >= 10:
            # Texture is outside the central target patch so high target
            # contrast itself does not incorrectly disable the retry.
            yy, xx = np.mgrid[82:146, 64:128]
            ring = (abs(xx-96) >= 16) | (abs(yy-114) >= 16)
            texture = np.where((xx//4 + yy//4) % 2, 30, 120).astype(np.uint8)
            frame[82:146, 64:128][ring] = texture[ring, None]
        return frame

    for i in range(10):
        p.push(scene(i))
    assert p.tracker.state == 'LOCKED'
    detector.empty = True
    for i in range(10, 13):
        row = p.push(scene(i))
    assert row.frame_index == 10
    assert row.details['detector_calls'] == 1
    assert not row.details['scale_retry_used']
    assert row.result.bbox is None and row.result.state == 'LOST'


@pytest.mark.parametrize('side,center,expected', [(16, (8, 8), False),
                                                (64, (4, 4), True),
                                                (64, (60, 60), True)])
def test_retry_background_mask_handles_small_frames_and_targets_at_image_edges(side, center, expected):
    p, _, _, _ = pipeline(scale_retry=True)
    frame = np.full((side, side, 3), 40, np.uint8)
    cx, cy = center
    box = (cx-4, cy-4, cx+4, cy+4)
    frame[box[1]:box[3], box[0]:box[2]] = 230
    assert p._smooth_retry_background(frame, center, box) is expected
