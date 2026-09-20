"""Prediction is opt-in; output distinguishes coasted locations from locks."""
import cv2
import numpy as np
import pytest

from yolomg_motion import roi_cli, yolo_detector, predictive_tracking, trajectory
from yolomg_motion.roi_pipeline import ROIObservation
from yolomg_motion.run_artifacts import read_tracking_run
from yolomg_motion.tracking import TrackResult, SingleTargetTracker


def make_video(path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 30., (320, 200))
    assert writer.isOpened()
    for index in range(12):
        frame = np.full((200, 320, 3), 40+index, np.uint8)
        writer.write(frame)
    writer.release()


def install_fake_pipeline(monkeypatch):
    observed = {}

    class Detector:
        def __init__(self, **kwargs):
            observed['detector_options'] = kwargs

    class Pipeline:
        def __init__(self, detector, extractor, tracker, **kwargs):
            observed.update(tracker=tracker, pipeline_options=kwargs)
            self.frame_shape, self.count = None, 0

        def push(self, frame):
            self.frame_shape, self.count = frame.shape, self.count+1
            if self.count < 5:
                return None
            index = self.count-3
            coasting = index >= 4
            prediction = dict(bbox=[20., 70., 30., 80.], center=[25., 75.], velocity=[1., 0.],
                              std_xy=[3., 3.], covariance=(np.eye(2)*9).tolist(), age_frames=index-2,
                              coasting=coasting, horizon_frames=10, future_center=[35., 75.])
            details = dict(
                prediction=prediction, prediction_roi=[0, 0, 160, 160],
                mode='detection' if index == 2 else 'roi', roi_tracking=index == 3,
                detector_ran=index == 2, detector_calls=int(index == 2),
                motion_computed=index == 2, motion_valid=True if index == 2 else None,
                motion_scope='full' if index == 2 else None,
                stage_timings=dict(motion_seconds=0., detector_seconds=0., tracking_seconds=.0001),
                processing_seconds=.0001, detection_windows=[[0, 0, 160, 160]] if index == 2 else [])
            result = TrackResult('LOST') if coasting else TrackResult(
                'LOCKED', (20., 70., 30., 80.), 1, 'detector' if index == 2 else 'template')
            return ROIObservation(index, frame, result, details)

    monkeypatch.setattr(yolo_detector, 'YOLOMGDetector', Detector)
    monkeypatch.setattr(roi_cli, 'ZoomROIPipeline', Pipeline)
    return observed


@pytest.mark.parametrize('headless', [True, False])
def test_prediction_wiring_counts_and_no_video_skips_rendering(tmp_path, monkeypatch, headless):
    source, output = tmp_path/'input.mp4', tmp_path/'out'
    make_video(source)
    observed = install_fake_pipeline(monkeypatch)
    renderer_type = trajectory.TrajectoryRenderer
    rendered = []

    class Renderer(renderer_type):
        def __init__(self, **kwargs):
            assert not headless, 'no-video must not create a renderer'
            super().__init__(**kwargs)

        def render(self, *args, **kwargs):
            rendered.append(args[2])
            return super().render(*args, **kwargs)

    monkeypatch.setattr(trajectory, 'TrajectoryRenderer', Renderer)
    arguments = ['--pipeline', 'zoom', '--motion-prediction', '--input', str(source),
                 '--output', str(output), '--prediction-max-gap', '25', '--prediction-gate', '12',
                 '--prediction-process-noise', '.5', '--prediction-max-roi', '700']
    if headless:
        arguments.append('--no-video')
    assert roi_cli.main(arguments) == 0
    tracker = observed['tracker']
    assert isinstance(tracker, predictive_tracking.PredictiveTargetTracker)
    assert tracker.prediction_max_gap == 25 and tracker.prediction_gate == 12
    assert observed['pipeline_options']['prediction_max_roi'] == 700  # May exceed motion context.
    rows, metadata = read_tracking_run(output, strict_finite=True)
    assert metadata['motion_prediction'] == dict(enabled=True, max_gap_frames=25,
            gate_threshold=12., process_noise=.5, max_roi=700, velocity_units='pixels/frame', trajectory_points=60)
    assert metadata['locked_frames'] == 2 and metadata['predicted_only_frames'] == 6
    assert all(r['bbox'] is None and r['state'] == 'LOST' for r in rows[2:])
    assert len(rendered) == (0 if headless else 8)
    assert metadata['video_frames_written'] == len(rendered)
    assert {p.name for p in output.iterdir()} == ({'tracks.json'} if headless else {'tracks.json', 'tracked_zoom.mp4'})


def test_prediction_disabled_preserves_original_tracker_and_pipeline_options(tmp_path, monkeypatch):
    source, output = tmp_path/'input.mp4', tmp_path/'out'
    make_video(source)
    observed = install_fake_pipeline(monkeypatch)
    roi_cli.main(['--pipeline', 'zoom', '--input', str(source), '--output', str(output), '--no-video'])
    assert type(observed['tracker']) is SingleTargetTracker
    assert 'prediction_max_roi' not in observed['pipeline_options']
    _, metadata = read_tracking_run(output)
    assert 'motion_prediction' not in metadata and 'predicted_only_frames' not in metadata


@pytest.mark.parametrize('options', [
    ['--pipeline', 'template'], ['--prediction-max-gap', '0'],
    ['--prediction-max-gap', '9', '--zoom-detect-every', '10'],
    ['--prediction-gate', 'nan'], ['--prediction-gate', '0'],
    ['--prediction-process-noise', 'inf'], ['--prediction-process-noise', '-1'],
    ['--prediction-max-roi', '319'],
])
def test_bad_prediction_configuration_rejected_before_model_load(tmp_path, monkeypatch, options):
    source = tmp_path/'input.mp4'
    source.write_bytes(b'validation runs before decode')

    def unexpected_load(**kwargs):
        pytest.fail('Model loading must not start with invalid options')

    monkeypatch.setattr(yolo_detector, 'YOLOMGDetector', unexpected_load)
    with pytest.raises(SystemExit):
        roi_cli.main(['--pipeline', 'zoom', '--motion-prediction', '--input', str(source),
                      '--output', str(tmp_path/'out'), *options])
