"""Small-state motion prediction, gating, and covariance guarantees."""
import numpy as np
import pytest

from yolomg_motion.motion_prediction import MotionPredictor


def box(cx, cy, width=10., height=6.):
    return (cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2)


def test_explicit_initialization_and_reset_preserve_no_observation_semantics():
    predictor = MotionPredictor()
    assert not predictor.active
    assert predictor.predict(0) is None
    assert np.isinf(predictor.gate_distance(box(10, 20), 0))
    assert not predictor.correct(box(10, 20), 0)
    predictor.initialize(box(10, 20), 0)
    assert predictor.active
    initial = predictor.predict(0)
    assert initial['center'] == [10., 20.]
    assert initial['velocity'] == [0., 0.]
    assert initial['std_xy'] == [4., 4.]
    predictor.reset()
    assert not predictor.active
    assert predictor.predict(20) is None
    predictor.initialize(box(200, 300), 20)
    assert predictor.predict(20)['center'] == [200., 300.]


def test_constant_velocity_converges_and_extrapolates_in_source_frame_units():
    predictor = MotionPredictor()
    predictor.initialize(box(100, 200), 0)
    for index in range(1, 31):
        assert predictor.correct(box(100 + 3 * index, 200 - 2 * index), index)
    prediction = predictor.predict(35)
    np.testing.assert_allclose(prediction['center'], [205., 130.], atol=.1)
    np.testing.assert_allclose(prediction['velocity'], [3., -2.], atol=.02)
    np.testing.assert_allclose(prediction['bbox'], box(205, 130), atol=.1)


def test_repeated_queries_and_mutated_return_values_do_not_advance_state():
    predictor = MotionPredictor()
    predictor.initialize(box(100, 200), 2)
    assert predictor.correct(box(106, 196), 4)
    initial = predictor.predict(4)
    future = predictor.predict(20)
    assert predictor.predict(20) == future
    assert predictor.predict(7) == predictor.predict(7)
    assert predictor.predict(4) == initial
    predictor.gate_distance(box(500, 600), 30)
    assert predictor.predict(4) == initial
    future['center'][0] = -999
    future['bbox'][0] = -999
    future['velocity'][0] = -999
    future['covariance'][0][0] = -999
    assert predictor.predict(20)['center'][0] != -999
    assert predictor.predict(20)['covariance'][0][0] > 0
    assert predictor.correct(box(109, 194), 5)


def test_variable_intervals_learn_velocity_without_treating_calls_as_frames():
    predictor = MotionPredictor()
    predictor.initialize(box(50, 100), 0)
    for index in (3, 8, 15, 22, 40):
        assert predictor.correct(box(50 + 2 * index, 100 - .75 * index), index)
    prediction = predictor.predict(47)
    np.testing.assert_allclose(prediction['center'], [144., 64.75], atol=.15)
    np.testing.assert_allclose(prediction['velocity'], [2., -.75], atol=.03)


def test_process_covariance_uses_exact_elapsed_frame_integral():
    predictor = MotionPredictor(process_noise=.6)
    predictor.initialize(box(50, 100), 10)
    # Position variance starts at 16, propagated velocity variance is 25*dt^2,
    # and continuous white-acceleration contributes q*dt^3/3.
    prediction = predictor.predict(14)
    expected_variance = 16. + 25. * 4. ** 2 + .6 * 4. ** 3 / 3.
    np.testing.assert_allclose(prediction['covariance'], np.eye(2) * expected_variance)
    np.testing.assert_allclose(prediction['std_xy'], np.sqrt([expected_variance] * 2))


def test_outlier_rejection_leaves_time_state_size_and_covariance_unchanged():
    predictor = MotionPredictor()
    predictor.initialize(box(100, 100), 0)
    assert predictor.correct(box(103, 102), 1)
    before = predictor.predict(5)
    outlier = box(1500, -2000, width=80, height=90)
    assert predictor.gate_distance(outlier, 5) > predictor.gate_threshold
    assert not predictor.correct(outlier, 5)
    assert predictor.predict(5) == before
    # A rejected observation at 5 did not prevent a legitimate update at 2.
    assert predictor.correct(box(106, 104), 2)
    prediction = predictor.predict(2)
    np.testing.assert_allclose(np.subtract(prediction['bbox'][2:], prediction['bbox'][:2]), [10, 6])


def test_gate_uses_observation_noise_and_accepts_practical_initial_motion():
    predictor = MotionPredictor()
    predictor.initialize(box(100, 100), 0)
    candidate = box(120, 100)
    detector_gate = predictor.gate_distance(candidate, 1)
    template_gate = predictor.gate_distance(candidate, 1, source='template')
    expected = 20. ** 2 / (16. + 25. + .25 / 3. + 4.)
    assert detector_gate == pytest.approx(expected)
    assert template_gate < detector_gate < predictor.gate_threshold
    assert predictor.correct(candidate, 1)


def test_template_is_weaker_and_cannot_change_detector_dimensions():
    detector, template = MotionPredictor(), MotionPredictor()
    for predictor in (detector, template):
        predictor.initialize(box(100, 100), 0)
    observation = box(108, 104, width=30, height=20)
    assert detector.correct(observation, 1, source='detector')
    assert template.correct(observation, 1, source='template')
    detector_prediction, template_prediction = detector.predict(1), template.predict(1)
    assert 100 < template_prediction['center'][0] < detector_prediction['center'][0] < 108
    assert template_prediction['velocity'][0] < detector_prediction['velocity'][0]
    assert template_prediction['covariance'][0][0] > detector_prediction['covariance'][0][0]
    np.testing.assert_allclose(np.subtract(template_prediction['bbox'][2:], template_prediction['bbox'][:2]), [10, 6])
    np.testing.assert_allclose(np.subtract(detector_prediction['bbox'][2:], detector_prediction['bbox'][:2]), [30, 20])


def test_joseph_covariance_stays_symmetric_positive_semidefinite():
    predictor = MotionPredictor(process_noise=.001, detector_noise=.01, template_noise=100.)
    predictor.initialize(box(1000, 1000), 0)
    rng = np.random.default_rng(42)
    index = 0
    for iteration in range(600):
        index += int(rng.integers(1, 5))
        source = 'detector' if iteration % 7 == 0 else 'template'
        noise = rng.normal(0, .02, size=2)
        assert predictor.correct(box(1000 + .3 * index + noise[0],
                                     1000 - .1 * index + noise[1]), index, source)
        np.testing.assert_allclose(predictor._covariance, predictor._covariance.T, atol=1e-12)
        assert np.linalg.eigvalsh(predictor._covariance).min() >= -1e-12
        prediction = predictor.predict(index + 20)
        assert np.linalg.eigvalsh(prediction['covariance']).min() >= 0
        assert np.isfinite(prediction['std_xy']).all()


@pytest.mark.parametrize('name', ['process_noise', 'detector_noise', 'template_noise', 'gate_threshold'])
@pytest.mark.parametrize('value', [0, -1, np.nan, np.inf, True, '4'])
def test_invalid_configuration(name, value):
    with pytest.raises(ValueError):
        MotionPredictor(**{name: value})


@pytest.mark.parametrize('value', [-1, 1.5, np.nan, np.inf, True, '2'])
def test_invalid_frame_indices(value):
    predictor = MotionPredictor()
    for call in (lambda: predictor.initialize(box(10, 20), value),
                 lambda: predictor.predict(value),
                 lambda: predictor.gate_distance(box(10, 20), value),
                 lambda: predictor.correct(box(10, 20), value)):
        with pytest.raises(ValueError):
            call()


@pytest.mark.parametrize('value', [
    (0, 0, 0, 1), (1, 0, 0, 1), (0, 0, 1, 0),
    (0, 0, np.nan, 1), (0, 0, 1, np.inf), (0, 1, 2), None,
    (-1e308, 0, 1e308, 1),
])
def test_invalid_boxes(value):
    predictor = MotionPredictor()
    for call in (lambda: predictor.initialize(value, 0),
                 lambda: predictor.gate_distance(value, 0),
                 lambda: predictor.correct(value, 0)):
        with pytest.raises(ValueError):
            call()
    assert not predictor.active


def test_source_and_temporal_order_validation_do_not_corrupt_state():
    predictor = MotionPredictor()
    predictor.initialize(box(10, 20), 5)
    original = predictor.predict(5)
    for source in ('none', 'prediction', None, True):
        with pytest.raises(ValueError):
            predictor.correct(box(10, 20), 6, source)
        with pytest.raises(ValueError):
            predictor.gate_distance(box(10, 20), 6, source)
    for call in (lambda: predictor.predict(4),
                 lambda: predictor.gate_distance(box(10, 20), 4),
                 lambda: predictor.correct(box(10, 20), 4),
                 lambda: predictor.correct(box(10, 20), 5)):
        with pytest.raises(ValueError):
            call()
    assert predictor.predict(5) == original


def test_extreme_interval_does_not_poison_filter_with_infinite_values():
    predictor = MotionPredictor()
    predictor.initialize(box(10, 20), 0)
    with pytest.raises(ValueError):
        predictor.predict(10 ** 200)
    with pytest.raises(ValueError):
        predictor.predict(10 ** 1000)
    assert predictor.predict(0)['center'] == [10., 20.]
