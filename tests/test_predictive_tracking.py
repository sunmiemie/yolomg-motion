"""Motion continuity must never manufacture detector observations or locks."""
import numpy as np
import pytest

from yolomg_motion.predictive_tracking import PredictiveTargetTracker
from yolomg_motion.tracking import Detection


def box(index):
    return (48 + 2 * index, 52, 58 + 2 * index, 58)


def frame(index=None):
    image = np.full((160, 320, 3), 40, np.uint8)
    if index is not None:
        x1, y1, x2, y2 = box(index)
        image[y1:y2, x1:x2] = 230
        image[y1:y2, x1+2:x1+4] = 70
    return image


def tracker(**kwargs):
    return PredictiveTargetTracker(max_detector_gap=None, roi_only=True,
                                   min_template_width=8, min_template_height=8,
                                   template_padding=2, **kwargs)


def establish(t):
    assert t.update(frame(0), [Detection(box(0), .9)], 0).state == 'CONFIRMING'
    assert t.update(frame(5), [Detection(box(5), .9)], 5).state == 'LOCKED'


def test_failed_detection_keeps_moving_hint_but_no_observed_box_or_template_update():
    t = tracker()
    establish(t)
    prior = t.prediction_details(10)
    result = t.update(frame(10), [], 10)  # target visibly present; verification failed
    assert result.state == 'LOST' and result.bbox is None and result.source == 'none'
    after = t.prediction_details(10)
    np.testing.assert_allclose(after['center'], prior['center'])
    np.testing.assert_allclose(after['covariance'], prior['covariance'])
    assert after['coasting'] and after['velocity'][0] > 1.5
    next_hint = t.predicted_box(15)
    assert next_hint[0] > t.predicted_box(10)[0] + 8
    assert t.update(frame(), None, 11).bbox is None


def test_coasting_reacquires_only_motion_consistent_detection_and_reconfirms():
    t = tracker()
    establish(t)
    t.update(frame(), [], 10)
    distant = Detection((240, 90, 250, 96), .99)
    result = t.update(frame(15), [distant, Detection(box(15), .7)], 15)
    assert result.state == 'CONFIRMING' and result.source == 'detector'
    assert result.bbox == box(15) and result.detector_confidence == .7
    assert t.prediction_details(15)['velocity'][0] > 1.5
    result = t.update(frame(20), [Detection(box(20), .8)], 20)
    assert result.state == 'LOCKED' and result.track_id == 2


def test_distant_false_detection_cannot_restart_identity_during_coast():
    t = tracker()
    establish(t)
    result = t.update(frame(), [Detection((240, 90, 250, 96), .99)], 10)
    assert result.state == 'LOST' and result.bbox is None
    assert t.prediction_details(10)['age_frames'] == 5


def test_template_observations_do_not_extend_detector_anchored_lifetime():
    t = tracker(prediction_max_gap=8)
    establish(t)
    for index in range(6, 14):
        result = t.update(frame(index), None, index)
        assert result.state == 'LOCKED' and result.source == 'template'
    assert t.predicted_box(14) is None
    result = t.update(frame(14), None, 14)
    assert result.state == 'LOST' and result.bbox is None
    assert not t.predictor.active and t.prediction_details(14) is None


def test_expired_hint_allows_new_global_candidate_and_reset_clears_motion():
    t = tracker(prediction_max_gap=8)
    establish(t)
    new_box = (240, 90, 250, 96)
    result = t.update(frame(), [Detection(new_box, .9)], 14)
    assert result.state == 'CONFIRMING' and result.bbox == new_box
    assert t.prediction_details(14)['age_frames'] == 0
    np.testing.assert_allclose(t.prediction_details(14)['velocity'], [0, 0])
    t.reset()
    assert t.state == 'SEARCHING' and not t.predictor.active
    assert not t.detector_predictor.active
    assert t.predicted_box(0) is None


def test_inconsistent_template_is_rejected_by_innovation_gate(monkeypatch):
    from yolomg_motion.tracking import SingleTargetTracker
    t = tracker()
    establish(t)
    monkeypatch.setattr(SingleTargetTracker, '_match_template',
                        lambda *args: ((240, 90, 250, 96), .99))
    result = t.update(frame(), None, 6)
    assert result.state == 'LOST' and result.bbox is None
    assert t.prediction_details(6)['velocity'][0] > 1.5


@pytest.mark.parametrize('options', [dict(prediction_max_gap=0), dict(prediction_max_gap=True),
                                   dict(prediction_gate=float('nan')), dict(process_noise=-1)])
def test_invalid_prediction_options(options):
    with pytest.raises(ValueError):
        tracker(**options)


def test_invalid_update_does_not_mutate_motion_state():
    t = tracker()
    establish(t)
    before = t.prediction_details(10)
    with pytest.raises(ValueError):
        t.update(frame().astype(np.float32), None, 30)
    assert t.state == 'LOCKED'
    assert t.prediction_details(10) == before
    with pytest.raises(ValueError, match='strictly increase'):
        t.update(frame(), [], 5)


def test_future_horizon_is_bounded_by_remaining_detector_lifetime():
    t = tracker(prediction_max_gap=20)
    establish(t)
    hint = t.prediction_details(24, horizon_frames=10)
    assert hint['age_frames'] == 19 and hint['horizon_frames'] == 1
    at_limit = t.prediction_details(25, horizon_frames=10)
    assert at_limit['horizon_frames'] == 0
    assert at_limit['future_center'] == at_limit['center']
    with pytest.raises(ValueError, match='horizon_frames'):
        t.prediction_details(24, horizon_frames=-1)


def test_many_biased_templates_cannot_block_detector_grounded_correction(monkeypatch):
    from yolomg_motion.tracking import SingleTargetTracker
    t = tracker()
    establish(t)
    # Simulate gradual drift that passes each local template check. The KF
    # guiding template search becomes confident in the wrong location.
    for index in range(6, 15):
        drifted = np.array(box(index), float) + np.array([3., 0., 3., 0.]) * (index-5)
        monkeypatch.setattr(SingleTargetTracker, '_match_template',
                            lambda *args, b=tuple(drifted): (b, .99))
        t.update(frame(index), None, index)
    template_prior = t.predictor.predict(15)
    detector_prior = t.detector_predictor.predict(15)
    assert template_prior['center'][0] > detector_prior['center'][0] + 20
    result = t.update(frame(15), [Detection(box(15), .9)], 15)
    assert result.state == 'LOCKED' and result.source == 'detector'
    assert result.bbox == box(15)
    assert t.predictor.predict(15) == t.detector_predictor.predict(15)
