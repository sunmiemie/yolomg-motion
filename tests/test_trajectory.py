"""Observed history stays distinct from forecasts, bounded and non-mutating."""
from copy import deepcopy

import numpy as np
import pytest

from yolomg_motion.tracking import TrackResult
from yolomg_motion.trajectory import TrajectoryRenderer


def locked(index, identity=1, source='detector'):
    return TrackResult('LOCKED', (20.+index, 75., 30.+index, 85.), identity, source)


def prediction():
    return dict(bbox=[75., 75., 85., 85.], center=[80., 80.], velocity=[3., 0.],
                std_xy=[4., 4.], covariance=np.eye(4).tolist(), age_frames=3,
                coasting=True, horizon_frames=10, future_center=[110., 80.])


def test_renderer_keeps_input_and_predictions_unchanged_and_draws_overlays():
    frame = np.zeros((200, 320, 3), np.uint8)
    original = frame.copy()
    details = dict(prediction=prediction(), prediction_roi=[45, 45, 130, 120],
                   detection_windows=[[40, 40, 140, 125]])
    saved = deepcopy(details)
    renderer = TrajectoryRenderer(show_prediction=True)
    canvas = renderer.render(frame, TrackResult('LOST'), 2, details, fps=30.)
    assert np.array_equal(frame, original)
    assert details == saved
    assert not np.shares_memory(canvas, frame)
    assert np.any(canvas[35:128])  # Search boxes and forecast, outside the legend.
    assert not renderer.points  # A forecast never becomes observed history.


def test_trail_is_bounded_and_resets_on_identity_or_discontinuity():
    frame = np.zeros((200, 320, 3), np.uint8)
    renderer = TrajectoryRenderer()
    for index in range(80):
        renderer.render(frame, locked(index, source='template' if index % 2 else 'detector'), index)
    assert len(renderer.points) == 60
    assert renderer.points[0][0] == 20 and renderer.points[-1][0] == 79
    renderer.render(frame, locked(80, identity=2), 80)
    assert len(renderer.points) == 1
    renderer.render(frame, locked(90, identity=2), 90)
    assert len(renderer.points) == 1 and renderer.points[0][0] == 90
    renderer.render(frame, locked(90, identity=2), 90)
    assert len(renderer.points) == 1  # Replayed frame cannot double the trail.
    renderer.reset()
    assert renderer.points == ()


def test_coasting_does_not_extend_solid_trail_and_reacquisition_breaks_line():
    frame = np.zeros((200, 320, 3), np.uint8)
    renderer = TrajectoryRenderer()
    renderer.render(frame, locked(0), 0)
    renderer.render(frame, locked(1), 1)
    observed = renderer.points
    renderer.render(frame, TrackResult('LOST'), 2, dict(prediction=prediction()))
    assert renderer.points == observed
    renderer.render(frame, locked(3), 3)
    assert len(renderer.points) == 1
    renderer.render(frame, TrackResult('LOST'), 4)
    assert not renderer.points


def test_out_of_frame_forecast_is_clipped_for_drawing_and_shape_change_resets():
    frame = np.zeros((200, 320, 3), np.uint8)
    renderer = TrajectoryRenderer(show_prediction=True)
    renderer.render(frame, locked(0), 0)
    forecast = prediction()
    forecast.update(center=[1.e8, -1.e8], future_center=[2.e8, -2.e8],
                    bbox=[1.e8, -1.e8, 1.e8+10, -1.e8+10])
    out = renderer.render(frame[:100].copy(), TrackResult('LOST'), 1,
                          dict(prediction=forecast, prediction_roi=[-500, -500, 900, 900]))
    assert out.shape == (100, 320, 3)
    assert not renderer.points


@pytest.mark.parametrize('value', [0, 61, True, 2.5])
def test_renderer_rejects_unbounded_history(value):
    with pytest.raises(ValueError):
        TrajectoryRenderer(value)
