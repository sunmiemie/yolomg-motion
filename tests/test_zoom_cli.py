"""Exercise zoom selection, actual-call accounting and artifact failure handling."""
import cv2
import numpy as np
import pytest

from yolomg_motion import roi_cli, yolo_detector
from yolomg_motion.roi_pipeline import ROIObservation
from yolomg_motion.run_artifacts import read_tracking_run
from yolomg_motion.tracking import TrackResult


def make_video(path, count=12):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 30., (160, 96))
    assert writer.isOpened()
    for index in range(count):
        frame = np.full((96, 160, 3), 35 + index * 5, np.uint8)
        frame[30:38, 25+index:37+index] = 220
        writer.write(frame)
    writer.release()


def install_pipeline(monkeypatch, fail_at=None):
    observed = {}

    class FakeDetector:
        def __init__(self, **options):
            observed['detector'] = options

    class FakeZoomPipeline:
        def __init__(self, detector, extractor, tracker, **options):
            observed.update(options, full_extractor=extractor, tracker=tracker)
            self.frame_shape = None
            self.received = 0

        def push(self, frame):
            self.frame_shape = frame.shape
            self.received += 1
            if self.received < 5:
                return None
            index = self.received - 3
            if index == fail_at:
                raise RuntimeError('zoom test inference failure')
            calls = 2 if index == 2 else (1 if index == 4 else 0)
            scope = 'full' if calls == 2 else ('local' if calls else None)
            details = dict(
                mode='detection' if calls else 'roi', roi_tracking=not bool(calls),
                detector_ran=bool(calls), detector_calls=calls, motion_computed=bool(calls),
                motion_valid=True if calls else None, motion_scope=scope,
                network_input_shapes=[[1, 3, 640, 640], None][:calls],
                detection_windows=[[0, 0, 96, 96], [64, 0, 160, 96]][:calls],
                stage_timings=dict(motion_seconds=.001 if calls else 0.,
                                   detector_seconds=.002*calls, tracking_seconds=.0001,
                                   proposal_seconds=.0001 if scope == 'full' else 0.),
                processing_seconds=.001 + .002*calls)
            return ROIObservation(index, frame, TrackResult('LOCKED', (30., 30., 42., 38.),
                                  1, 'detector' if calls else 'template',
                                  detector_confidence=.9 if calls else None,
                                  appearance_score=None if calls else .95), details)

    monkeypatch.setattr(yolo_detector, 'YOLOMGDetector', FakeDetector)
    monkeypatch.setattr(roi_cli, 'ZoomROIPipeline', FakeZoomPipeline)
    return observed


@pytest.mark.parametrize('headless', [False, True])
def test_zoom_default_selection_multiple_calls_and_artifacts(tmp_path, monkeypatch, headless):
    source, output = tmp_path/'input.mp4', tmp_path/'result'
    make_video(source)
    observed = install_pipeline(monkeypatch)
    args = ['--pipeline', 'zoom', '--input', str(source), '--output', str(output)]
    if headless:
        args.append('--no-video')
    assert roi_cli.main(args) == 0
    assert observed['detector']['image_size'] == 640
    assert observed['detect_every'] == 10
    assert observed['roi_size'] == 320 and observed['motion_context'] == 640
    assert observed['max_rois'] == 2 and observed['scan_every'] == 3
    assert observed['motion_threshold'] == 8.
    local = observed['local_extractor'].config
    assert (local.alignment_width, local.alignment_height, local.grid_step, local.grid_step_y) == (320, 320, 24, 24)
    assert local.blur_size == 5 and local.min_inliers == 10
    assert local.open_iterations == local.close_iterations == 0
    assert observed['tracker'].roi_only and observed['tracker'].max_detector_gap is None
    assert observed['tracker'].confirmations == 2
    rows, metadata = read_tracking_run(output, strict_finite=True)
    assert metadata['pipeline'] == 'roi_zoom_detection'
    assert metadata['detector_calls'] == 3 and metadata['detector_frames'] == 2
    assert metadata['detector_attempt_frames'] == 2 and metadata['max_detector_calls_per_frame'] == 2
    assert metadata['motion_calls'] == 2
    assert metadata['local_motion_calls'] == metadata['full_motion_calls'] == 1
    assert metadata['network_input_shapes'] == [[1, 3, 640, 640]]
    assert metadata['stage_timings']['proposal_seconds'] == pytest.approx(.0001)
    assert [r['frame_index'] for r in rows] == list(range(2, 10))
    assert all(row['timestamp_seconds'] == pytest.approx(row['frame_index']/30) for row in rows)
    assert metadata['completed'] and not metadata['partial']
    assert {file.name for file in output.iterdir()} == ({'tracks.json'} if headless else {'tracks.json', 'tracked_zoom.mp4'})
    if not headless:
        cap = cv2.VideoCapture(str(output/'tracked_zoom.mp4'))
        decoded = 0
        while cap.read()[0]:
            decoded += 1
        cap.release()
        assert decoded == metadata['video_frames_written'] == len(rows)
    with pytest.raises(SystemExit):
        roi_cli.main(args)


def test_zoom_explicit_parameters_and_failed_partial(tmp_path, monkeypatch):
    source, output = tmp_path/'input.mp4', tmp_path/'result'
    make_video(source)
    observed = install_pipeline(monkeypatch, fail_at=6)
    with pytest.raises(RuntimeError, match='zoom test inference failure'):
        roi_cli.main(['--pipeline', 'zoom', '--input', str(source), '--output', str(output),
                      '--no-video', '--image-size', '512', '--roi-size', '256',
                      '--motion-context', '512', '--max-rois', '1', '--zoom-detect-every', '15',
                      '--motion-threshold', '4.5', '--scan-every', '4'])
    assert observed['detector']['image_size'] == 512
    assert observed['roi_size'] == 256 and observed['motion_context'] == 512
    assert observed['max_rois'] == 1 and observed['detect_every'] == 15
    assert observed['scan_every'] == 4 and observed['motion_threshold'] == 4.5
    rows, metadata = read_tracking_run(output, strict_finite=True)
    assert [r['frame_index'] for r in rows] == [2, 3, 4, 5]
    assert metadata['partial'] and not metadata['completed']
    assert metadata['status'] == 'failed' and metadata['error']['source_frame_index'] == 6
    assert metadata['detector_calls'] == 3


@pytest.mark.parametrize('extra', [
    ['--roi-size', '0'], ['--motion-context', '319'], ['--max-rois', '0'],
    ['--zoom-detect-every', '0'], ['--motion-threshold', 'nan'], ['--scan-every', '0'],
])
def test_invalid_zoom_budget_or_size_is_rejected_before_model_load(tmp_path, monkeypatch, extra):
    source = tmp_path/'input.mp4'
    source.write_bytes(b'test validation before decoding')

    def unexpected_load(**kwargs):
        pytest.fail('Invalid options must be rejected before loading the detector')

    monkeypatch.setattr(yolo_detector, 'YOLOMGDetector', unexpected_load)
    with pytest.raises(SystemExit):
        roi_cli.main(['--pipeline', 'zoom', '--input', str(source), '--output', str(tmp_path/'out'), *extra])
