"""Prediction may move a crop, but cannot change the detector call budget."""
import numpy as np
import pytest

from yolomg_motion.predictive_tracking import PredictiveTargetTracker
from yolomg_motion.zoom_pipeline import ZoomROIPipeline
from test_zoom_pipeline import Motion, CropDetector, image


def pipeline():
    full, local, detector = Motion(), Motion(), CropDetector()
    tracker = PredictiveTargetTracker(max_detector_gap=None, roi_only=True,
                                      min_template_width=8, min_template_height=8,
                                      template_padding=2, min_confidence=.1,
                                      prediction_max_gap=9)
    p = ZoomROIPipeline(detector, full, tracker, local_extractor=local,
                        detect_every=4, roi_size=64, motion_context=64,
                        prediction_max_roi=128)
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


def test_invalid_local_motion_cannot_correct_prediction_from_a_template():
    p, detector, full, local = pipeline()
    for i in range(12):
        p.push(image(i))
    prior = p.tracker.prediction_details(10)
    local.valid = False
    row = p.push(image(12))
    assert row.frame_index == 10
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
