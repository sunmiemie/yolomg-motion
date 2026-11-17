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


def establish_trusted(t):
    for index in (0, 2, 5):
        t.update(frame(index), [Detection(box(index), .9)], index)
    assert t.state == 'LOCKED'


def test_failed_detection_keeps_moving_hint_but_no_observed_box_or_template_update():
    t = tracker(verification_grace=0)
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
                                   dict(prediction_gate=float('nan')), dict(process_noise=-1),
                                   dict(verification_grace=-1), dict(verification_grace=2),
                                   dict(verification_grace=True), dict(verification_grace=.5)])
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


def test_one_empty_verification_bridges_with_observation_but_second_loses():
    t = tracker()
    establish_trusted(t)
    template = t._template.copy()
    for index in range(6, 16):
        result = t.update(frame(index), [] if index == 10 else None, index)
        assert result.state == 'LOCKED' and result.source == 'template'
        assert result.track_id == 1
        np.testing.assert_allclose(result.bbox, box(index), atol=1)
        assert t.verification_pending == (index >= 10)
        assert t.last_detection_index == 5
        assert t.prediction_details(index)['age_frames'] == index - 5
        np.testing.assert_array_equal(t._template, template)
    result = t.update(frame(16), [], 16)
    assert result.state == 'LOST' and result.bbox is None
    assert not t.verification_pending


def test_successful_detector_clears_grace_without_changing_identity():
    t = tracker()
    establish_trusted(t)
    t.update(frame(10), [], 10)
    assert t.verification_pending
    result = t.update(frame(15), [Detection(box(15), .9)], 15)
    assert result.state == 'LOCKED' and result.track_id == 1 and result.source == 'detector'
    assert not t.verification_pending and t.verification_misses == 0
    assert t.last_detection_index == 15


@pytest.mark.parametrize('failure_index', [10, 11])
def test_grace_requires_visible_target_on_every_frame(failure_index):
    t = tracker()
    establish_trusted(t)
    for index in range(10, failure_index + 1):
        result = t.update(frame() if index == failure_index else frame(index),
                          [] if index == 10 else None, index)
    assert result.state == 'LOST' and result.bbox is None


def test_grace_does_not_confirm_candidate_or_override_conflicting_detection():
    t = tracker()
    t.update(frame(0), [Detection(box(0), .9)], 0)
    assert t.update(frame(5), [], 5).state == 'LOST'
    t.reset()
    establish_trusted(t)
    result = t.update(frame(10), [Detection((240, 90, 250, 96), .99)], 10)
    assert result.state == 'LOST' and result.bbox is None


@pytest.mark.parametrize('drift,score', [(0., .79), (30., .99)])
def test_grace_checks_score_and_detector_anchor_each_frame(monkeypatch, drift, score):
    from yolomg_motion.tracking import SingleTargetTracker
    t = tracker()
    establish_trusted(t)
    assert t.update(frame(10), [], 10).state == 'LOCKED'
    candidate = tuple(np.array(box(11)) + [drift, 0, drift, 0])
    monkeypatch.setattr(SingleTargetTracker, '_match_template', lambda *args: (candidate, score))
    # Even a template-fed predictor agreeing with a distractor cannot bypass
    # the independent detector-only anchor during verification grace.
    monkeypatch.setattr(t.predictor, 'gate_distance', lambda *args, **kwargs: 0.)
    result = t.update(frame(11), None, 11)
    assert result.state == 'LOST' and result.bbox is None


def test_grace_cannot_extend_hard_detector_lifetime():
    t = tracker(prediction_max_gap=8)
    establish_trusted(t)
    assert t.update(frame(10), [], 10).state == 'LOCKED'
    assert t.update(frame(13), None, 13).state == 'LOCKED'
    assert t.update(frame(14), None, 14).state == 'LOST'
    assert t.prediction_details(14) is None


@pytest.mark.parametrize('confidences', [(.9, .9), (.9, .29, .9), (.9, .9, .29)])
def test_new_or_weakly_detected_identity_cannot_use_grace(confidences):
    t = tracker()
    for index, confidence in enumerate(confidences):
        t.update(frame(index), [Detection(box(index), confidence)], index)
    assert t.state == 'LOCKED'
    result = t.update(frame(5), [], 5)
    assert result.state == 'LOST' and result.bbox is None


def test_repeated_dropout_requires_three_fresh_detector_hits():
    t = tracker()
    establish_trusted(t)
    assert t.update(frame(10), [], 10).state == 'LOCKED'
    assert t.update(frame(15), [Detection(box(15), .9)], 15).state == 'LOCKED'
    assert t.update(frame(20), [], 20).state == 'LOST'


def test_weak_detection_rescue_keeps_confidence_and_strong_reference_lifetime():
    t = tracker(low_confidence=.04, verification_grace=0)
    establish_trusted(t)
    template = t._template.copy()
    anchor = t.detector_predictor.predict(10)
    result = t.update(frame(10), [Detection(box(10), .08)], 10)
    assert result.state == 'LOCKED' and result.source == 'detector'
    assert result.bbox == box(10) and result.track_id == 1
    assert result.detector_confidence == .08 and result.appearance_score >= .8
    assert t.low_confidence_used and t.verification_pending
    assert t.min_confidence == .2
    assert t.last_detection_index == t._last_motion_detection_index == 5
    assert t.prediction_details(10)['age_frames'] == 5
    assert t.detector_predictor.predict(10) == anchor
    assert t._consecutive_detector_hits == 0
    np.testing.assert_array_equal(t._template, template)
    assert t.update(frame(11), None, 11).state == 'LOCKED'
    assert not t.low_confidence_used
    assert t.last_detection_index == 5
    np.testing.assert_array_equal(t._template, template)


def test_weak_support_cannot_extend_strong_detector_hard_ttl():
    t = tracker(low_confidence=.04, prediction_max_gap=8, verification_grace=0)
    establish_trusted(t)
    assert t.update(frame(10), [Detection(box(10), .08)], 10).state == 'LOCKED'
    assert t.update(frame(13), None, 13).state == 'LOCKED'
    result = t.update(frame(14), [Detection(box(14), .08)], 14)
    assert result.state == 'LOST' and result.bbox is None
    assert not t.low_confidence_used and t.prediction_details(14) is None


def test_weak_support_cannot_bypass_base_detector_gap():
    t = PredictiveTargetTracker(low_confidence=.04, max_detector_gap=6,
                                roi_only=True, verification_grace=0)
    establish_trusted(t)
    result = t.update(frame(12), [Detection(box(12), .08)], 12)
    assert result.state == 'LOST' and result.bbox is None
    assert not t.low_confidence_used


def test_only_one_weak_rescue_until_three_fresh_regular_detections():
    t = tracker(low_confidence=.04, verification_grace=0)
    establish_trusted(t)
    assert t.update(frame(10), [Detection(box(10), .08)], 10).state == 'LOCKED'
    assert t.update(frame(11), [Detection(box(11), .08)], 11).state == 'LOST'
    assert not t.low_confidence_used
    t.reset()
    establish_trusted(t)
    t.update(frame(10), [Detection(box(10), .08)], 10)
    for index in (11, 12, 13):
        result = t.update(frame(index), [Detection(box(index), .9)], index)
        assert result.state == 'LOCKED' and result.track_id == 1
        assert not t.low_confidence_used and not t.verification_pending
    result = t.update(frame(14), [Detection(box(14), .08)], 14)
    assert result.state == 'LOCKED' and t.low_confidence_used


@pytest.mark.parametrize('prior_state', ['SEARCHING', 'CONFIRMING', 'LOCKED_NEW', 'LOST'])
def test_weak_detections_cannot_acquire_confirm_or_reacquire(prior_state):
    t = tracker(low_confidence=.04, verification_grace=0)
    if prior_state == 'CONFIRMING':
        t.update(frame(0), [Detection(box(0), .9)], 0)
    elif prior_state == 'LOCKED_NEW':
        establish(t)
    elif prior_state == 'LOST':
        establish_trusted(t)
        t.update(frame(6), [], 6)
    result = t.update(frame(10), [Detection(box(10), .08)], 10)
    assert result.state in ('SEARCHING', 'LOST')
    assert result.bbox is None and not t.low_confidence_used


def test_regular_detection_wins_and_conflicting_regular_cannot_be_bypassed():
    t = tracker(low_confidence=.04, verification_grace=0)
    establish_trusted(t)
    result = t.update(frame(10), [Detection(box(10), .08), Detection(box(10), .3)], 10)
    assert result.detector_confidence == .3 and result.state == 'LOCKED'
    assert not t.low_confidence_used and t.last_detection_index == 10
    result = t.update(frame(15), [Detection(box(15), .08),
                                  Detection((240, 90, 250, 96), .99)], 15)
    assert result.state == 'LOST' and result.bbox is None
    assert not t.low_confidence_used


@pytest.mark.parametrize('candidate', [
    Detection((200, 52, 210, 58), .08),  # outside detector anchor
    Detection((64, 49, 92, 61), .08),    # implausibly different dimensions
    Detection(box(10), .039),           # below guarded lower floor
])
def test_weak_candidate_geometry_and_floor_are_not_bypassed(candidate):
    t = tracker(low_confidence=.04, verification_grace=0)
    establish_trusted(t)
    result = t.update(frame(10), [candidate], 10)
    assert result.state == 'LOST' and not t.low_confidence_used


def test_weak_requires_actual_target_pixels_not_only_background_match(monkeypatch):
    t = tracker(low_confidence=.04, verification_grace=0)
    establish_trusted(t)
    # A context template can be dominated by stable background. Even a high
    # mocked context score is insufficient when candidate pixels are empty.
    monkeypatch.setattr(t, '_match_template', lambda *args: (box(10), .99))
    result = t.update(frame(), [Detection(box(10), .08)], 10)
    assert result.state == 'LOST' and not t.low_confidence_used


def test_weak_requires_high_context_score_and_candidate_agreement(monkeypatch):
    t = tracker(low_confidence=.04, verification_grace=0)
    establish_trusted(t)
    monkeypatch.setattr(t, '_match_template', lambda *args: (box(10), .79))
    assert t.update(frame(10), [Detection(box(10), .08)], 10).state == 'LOST'
    t.reset()
    establish_trusted(t)
    monkeypatch.setattr(t, '_match_template', lambda *args: (box(15), .99))
    assert t.update(frame(10), [Detection(box(10), .08)], 10).state == 'LOST'


def test_weak_mode_is_opt_in_and_reset_clears_its_state():
    t = tracker(verification_grace=0)
    establish_trusted(t)
    assert t.update(frame(10), [Detection(box(10), .08)], 10).state == 'LOST'
    assert not t.low_confidence_used
    t = tracker(low_confidence=.04, verification_grace=0)
    establish_trusted(t)
    t.update(frame(10), [Detection(box(10), .08)], 10)
    t.reset()
    assert not t.low_confidence_used and not t._weak_rescue_used
    assert t._weak_detection is None


@pytest.mark.parametrize('value', [True, False, .039, .2, .8, float('nan'),
                                  float('inf'), '.04'])
def test_invalid_low_confidence_options(value):
    with pytest.raises(ValueError, match='low_confidence'):
        tracker(low_confidence=value)
