"""Prediction may move a crop, but cannot change the detector call budget."""
import numpy as np
import pytest

from yolomg_motion.predictive_tracking import PredictiveTargetTracker
from yolomg_motion.tracking import TrackResult
from yolomg_motion.zoom_pipeline import ZoomROIPipeline
from test_zoom_pipeline import Motion, CropDetector, image


def pipeline(strict=True, **kwargs):
    full, local, detector = Motion(), Motion(), CropDetector()
    tracker = PredictiveTargetTracker(max_detector_gap=None, roi_only=True,
                                      min_template_width=8, min_template_height=8,
                                      template_padding=2, min_confidence=.1,
                                      prediction_max_gap=9, verification_grace=0 if strict else 1)
    p = ZoomROIPipeline(detector, full, tracker, local_extractor=local,
                        detect_every=4, roi_size=64, motion_context=64,
                        prediction_max_roi=128, recent_search_attempts=0 if strict else 2, **kwargs)
    return p, detector, full, local


def test_moving_recovery_crop_is_bounded_and_expiry_returns_to_global_search():
    p, detector, full, local = pipeline()
    rows = []
    for i in range(34):
        if i == 12:
            detector.empty = True
        r = p.push(image(i))
        if r is not None:
            rows.append(r)
    failed = next(r for r in rows if r.frame_index == 10)
    assert failed.result.state == 'LOST' and failed.result.bbox is None
    assert failed.details['prediction']['coasting']
    recovery = next(r for r in rows if r.frame_index == 14)
    assert recovery.details['detection_reason'] == 'predicted_reacquisition'
    assert recovery.details['prediction']['center'][0] > failed.details['prediction']['center'][0]
    roi = recovery.details['detection_windows'][0]
    assert 64 < roi[2]-roi[0] <= 128
    context = recovery.details['motion_context_window']
    assert context[0] <= roi[0] and context[1] <= roi[1]
    assert context[2] >= roi[2] and context[3] >= roi[3]
    expired = next(r for r in rows if r.frame_index == 18)
    assert expired.details['prediction'] is None
    assert expired.details['prediction_roi'] is None
    assert expired.details['motion_scope'] == 'full'
    attempts = [r.frame_index for r in rows if r.details['motion_computed']]
    assert all(b-a >= 4 for a, b in zip(attempts, attempts[1:]))
    assert all(r.details['detector_calls'] <= 2 for r in rows)
    assert all(r.result.bbox is None for r in rows if r.frame_index >= 10)


@pytest.mark.parametrize('strict', [True, False])
def test_invalid_local_motion_cannot_correct_prediction_from_a_template(strict):
    p, detector, full, local = pipeline(strict=strict)
    for i in range(16):
        p.push(image(i))
    prior = p.tracker.prediction_details(14)
    local.valid = False
    row = p.push(image(16))
    assert row.frame_index == 14
    assert row.result.state == 'LOST' and row.result.bbox is None
    assert row.details['detector_calls'] == 0
    np.testing.assert_allclose(row.details['prediction']['center'], prior['center'])
    np.testing.assert_allclose(row.details['prediction']['covariance'], prior['covariance'])


def test_detector_observations_restore_lock_after_short_dropout():
    p, detector, _, _ = pipeline()
    rows = []
    for i in range(26):
        detector.empty = 12 <= i < 16
        r = p.push(image(i))
        if r is not None:
            rows.append(r)
    assert next(r for r in rows if r.frame_index == 10).result.state == 'LOST'
    resumed = next(r for r in rows if r.frame_index == 14)
    assert resumed.result.state == 'CONFIRMING' and resumed.result.source == 'detector'
    assert not resumed.details['prediction']['coasting']
    assert next(r for r in rows if r.frame_index == 18).result.state == 'LOCKED'


def test_detector_interval_cannot_exceed_prediction_lifetime():
    tracker = PredictiveTargetTracker(max_detector_gap=None, roi_only=True, prediction_max_gap=9)
    with pytest.raises(ValueError, match='prediction_max_gap must be >= detect_every'):
        ZoomROIPipeline(CropDetector(), Motion(), tracker, detect_every=12)


def test_grace_preserves_identity_and_nominal_crop_without_more_calls():
    p, detector, _, _ = pipeline(strict=False)
    rows = []
    for i in range(26):
        detector.empty = 16 <= i < 20
        row = p.push(image(i))
        if row:
            rows.append(row)
    for row in rows:
        if row.frame_index >= 6:
            assert row.result.state == 'LOCKED' and row.result.track_id == 1
        assert row.details['verification_pending'] == (14 <= row.frame_index < 18)
        if row.frame_index >= 6 and row.details['detector_ran']:
            assert row.details['detector_calls'] == 1
            window = row.details['detection_windows'][0]
            assert window[2] - window[0] == 64


def test_recent_search_is_bounded_and_uses_existing_budget_when_motion_is_weak():
    p, detector, full, _ = pipeline(strict=False)
    rows = []
    for i in range(38):
        if i >= 16:
            detector.empty = True
            full.no_proposals = True
        row = p.push(image(i))
        if row:
            rows.append(row)
    assert next(r for r in rows if r.frame_index == 14).result.state == 'LOCKED'
    assert next(r for r in rows if r.frame_index == 18).result.state == 'LOST'
    recent = [r for r in rows if r.details['detection_reason'] == 'recent_target_search']
    assert [r.frame_index for r in recent] == [22, 26]
    for row in recent:
        assert row.details['prediction'] is None and row.result.bbox is None
        assert row.details['motion_scope'] == 'full'
        window = row.details['detection_windows'][0]
        assert window[0] <= 96 <= window[2] and window[1] <= 114 <= window[3]
    assert next(r for r in rows if r.frame_index == 30).details['detection_reason'] == 'motion_search'
    attempts = [r.frame_index for r in rows if r.details['motion_computed']]
    assert all(b - a == 4 for a, b in zip(attempts, attempts[1:]))
    assert all(r.details['detector_calls'] <= 2 for r in rows)


@pytest.mark.parametrize('attempts', [-1, 3, True, .5])
def test_recent_search_rejects_unbounded_configuration(attempts):
    with pytest.raises(ValueError, match='recent_search_attempts'):
        ZoomROIPipeline(CropDetector(), Motion(), recent_search_attempts=attempts)


def test_adaptive_roi_waits_for_confirmation_and_preserves_original_coordinates():
    p, detector, _, local = pipeline(adaptive_roi=True, scale_retry=True)
    detector.image_size = 128
    rows = [r for i in range(24) if (r := p.push(image(i))) is not None]
    attempts = [r for r in rows if r.details['detector_ran']]
    assert [r.frame_index for r in attempts] == [2, 6, 10, 14, 18]
    assert all(not r.details['adaptive_roi_used'] for r in attempts[:2])
    assert attempts[1].result.state == 'LOCKED'
    for r in attempts[2:]:
        assert r.details['adaptive_roi_used']
        assert r.details['detector_calls'] == 1
        window = r.details['detection_windows'][0]
        assert window[2] - window[0] == 48
        assert r.result.source == 'detector' and r.result.state == 'LOCKED'
        np.testing.assert_allclose(r.result.bbox, (80+r.frame_index, 110, 92+r.frame_index, 118))
    # Magnification changes neither the network tensor nor registration work.
    assert local.shapes == [(64, 64)] * 4
    assert all(r.details['network_input_shapes'] == [[1, 3, 128, 128]] for r in attempts)


def test_adaptive_size_ignores_template_and_weak_measurements_and_limits_scale_jumps():
    p, detector, _, _ = pipeline(adaptive_roi=True)
    p.roi_size = p._adaptive_side = 320
    detector.image_size = 640
    small = TrackResult('LOCKED', (50., 60., 58., 65.), 1, 'detector', .9)
    observed = []
    for _ in range(4):
        p._update_adaptive_size(small, 'LOCKED')
        observed.append(p._adaptive_side)
    assert all(224 <= side <= 320 for side in observed)
    assert all(0 <= a-b <= 32 for a, b in zip([320] + observed, observed))
    assert observed[-1] == 224
    history = tuple(p._target_sizes)
    p._update_adaptive_size(TrackResult('LOCKED', (30., 40., 90., 80.), 1, 'template'), 'LOCKED')
    p.tracker.low_confidence_used = True
    p._update_adaptive_size(TrackResult('LOCKED', (30., 40., 90., 80.), 1, 'detector', .05), 'LOCKED')
    assert tuple(p._target_sizes) == history and p._adaptive_side == 224
    p.tracker.low_confidence_used = False
    # A single oversized detector box cannot change the median scale.
    large = TrackResult('LOCKED', (30., 40., 90., 80.), 1, 'detector', .9)
    p._update_adaptive_size(large, 'LOCKED')
    assert p._adaptive_side == 224
    p._update_adaptive_size(large, 'LOCKED')
    assert 224 < p._adaptive_side <= 256
    # A fresh identity cannot inherit the previous target's tight scale.
    p._update_adaptive_size(TrackResult('CONFIRMING', small.bbox, None, 'detector', .9), 'LOST')
    assert p._adaptive_side == 320 and len(p._target_sizes) == 1


@pytest.mark.parametrize('max_rois', [1, 2])
def test_empty_adaptive_crop_falls_back_with_existing_budget_and_motion_context(max_rois):
    p, detector, _, local = pipeline(adaptive_roi=True, scale_retry=True, max_rois=max_rois)
    detector.image_size = 128
    for i in range(12):
        p.push(image(i))
    detect = detector.detect

    def nominal_only(rgb, motion):
        if rgb.shape[0] < 64:
            detector.shapes.append(rgb.shape[:2])
            return []
        return detect(rgb, motion)

    detector.detect = nominal_only
    calls_before, motion_before = len(detector.shapes), len(local.shapes)
    row = p.push(image(12))
    assert row.frame_index == 10 and row.details['adaptive_roi_used']
    assert row.details['detector_calls'] == max_rois
    assert len(detector.shapes) - calls_before == max_rois
    assert len(local.shapes) - motion_before == 1
    context = row.details['motion_context_window']
    assert context[2]-context[0] == 64
    for x, y, right, bottom in row.details['detection_windows']:
        assert context[0] <= x < right <= context[2]
        assert context[1] <= y < bottom <= context[3]
    assert [w[2]-w[0] for w in row.details['detection_windows']] == ([48, 64] if max_rois == 2 else [48])
    assert row.details['scale_retry_used'] == (max_rois == 2)
    if max_rois == 2:
        assert row.result.state == 'LOCKED'
        np.testing.assert_allclose(row.result.bbox, (90., 110., 102., 118.))
    else:
        assert row.result.state == 'LOST' and row.result.bbox is None


def test_adaptive_crop_returns_to_nominal_expansion_after_loss():
    p, detector, _, _ = pipeline(adaptive_roi=True, scale_retry=True)
    detector.image_size = 128
    for i in range(12):
        p.push(image(i))
    detector.empty = True
    rows = [p.push(image(i)) for i in range(12, 17)]
    assert rows[0].result.state == 'LOST'
    recovery = rows[-1]
    assert recovery.frame_index == 14
    assert not recovery.details['adaptive_roi_used']
    assert recovery.details['detection_reason'] == 'predicted_reacquisition'
    first = recovery.details['detection_windows'][0]
    assert 64 < first[2]-first[0] <= 128
    assert recovery.result.bbox is None


def test_adaptive_crop_keeps_nominal_scale_on_textured_background():
    p, detector, _, _ = pipeline(adaptive_roi=True, scale_retry=True)
    detector.image_size = 128
    for i in range(13):
        frame = image(i)
        if i == 10:
            checker = ((np.indices(frame.shape[:2]).sum(axis=0) % 2) * 80 + 40).astype(np.uint8)
            frame[:] = checker[:, :, None]
            frame[110:118, 90:102] = image(10)[110:118, 90:102]
        row = p.push(frame)
    assert row.frame_index == 10 and row.result.state == 'LOCKED'
    assert not row.details['adaptive_roi_used']
    assert row.details['detector_calls'] == 1
    window = row.details['detection_windows'][0]
    assert window[2]-window[0] == 64
