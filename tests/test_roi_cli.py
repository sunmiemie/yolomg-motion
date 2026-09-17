"""Exercise genuine MFEM, ROI matcher, centered output and both artifact modes."""
import cv2
import numpy as np
import pytest

from yolomg_motion import roi_cli, yolo_detector
from yolomg_motion.run_artifacts import read_tracking_run
from tools.audit_tracking import audit_run


def make_video(path, count=30):
    rng = np.random.default_rng(46)
    background = rng.integers(20, 130, (360, 640, 3), dtype=np.uint8)
    background = cv2.GaussianBlur(background, (3, 3), 0)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 60., (640, 360))
    assert writer.isOpened()
    for index in range(count):
        image = background.copy()
        image[140:146, 100+index:112+index] = 255
        image[300:320, 30:50] = 15+index*6
        writer.write(image)
    writer.release()


class BrightDetector:
    def __init__(self, **kwargs):
        pass

    def detect(self, frame, motion):
        assert motion.shape == frame.shape[:2] and motion.dtype == np.float32
        ys, xs = np.nonzero(frame[:, :, 0] > 230)
        return [yolo_detector.Detection((float(xs.min()), float(ys.min()), float(xs.max()+1), float(ys.max()+1)), .9, 0)]


@pytest.mark.parametrize('headless', [False, True])
def test_roi_artifacts_motion_skipping_and_center_timestamps(tmp_path, monkeypatch, headless):
    source, output = tmp_path/'source.mp4', tmp_path/'result'
    make_video(source)
    monkeypatch.setattr(yolo_detector, 'YOLOMGDetector', BrightDetector)
    arguments = ['--input', str(source), '--output', str(output)]
    if headless:
        arguments.append('--no-video')
    assert roi_cli.main(arguments) == 0
    rows, metadata = read_tracking_run(output, strict_finite=True)
    assert [r['frame_index'] for r in rows] == list(range(2, 28))
    assert metadata['detector_calls'] == metadata['motion_calls'] == 2
    assert metadata['input_dimensions'] == [640, 360]
    assert metadata['timing']['roi_compute']['samples'] == 22
    assert metadata['lookahead_seconds'] == pytest.approx(2/60)
    assert all(r['timestamp_seconds'] == pytest.approx(r['frame_index']/60) for r in rows)
    assert all(not r['motion_computed'] for r in rows if r['roi_tracking'])
    assert {p.name for p in output.iterdir()} == ({'tracks.json'} if headless else {'tracks.json', 'tracked_zoom.mp4'})
    audit = audit_run(output)
    assert audit['passed'], audit['errors']
    assert audit['motion_skipped_frames'] == 24
    if not headless:
        source_cap = cv2.VideoCapture(str(source))
        result_cap = cv2.VideoCapture(str(output/'tracked_zoom.mp4'))
        source_frames = []
        while True:
            ok, image = source_cap.read()
            if not ok:
                break
            source_frames.append(image)
        for r in rows:
            ok, image = result_cap.read()
            assert ok
            expected = source_frames[r['frame_index']][304:316, 34:46].mean()
            assert abs(image[304:316, 34:46].mean()-expected) < 7
        assert not result_cap.read()[0]
        source_cap.release()
        result_cap.release()
    with pytest.raises(SystemExit):
        roi_cli.main(arguments)  # a result is never silently overwritten


def test_failure_retains_explicit_partial_json_and_stops(tmp_path, monkeypatch):
    source, output = tmp_path/'source.mp4', tmp_path/'result'
    make_video(source)
    class FailingDetector(BrightDetector):
        def detect(self, frame, motion):
            raise RuntimeError('test inference failure')
    monkeypatch.setattr(yolo_detector, 'YOLOMGDetector', FailingDetector)
    with pytest.raises(RuntimeError, match='test inference failure'):
        roi_cli.main(['--input', str(source), '--output', str(output), '--no-video'])
    rows, metadata = read_tracking_run(output)
    assert not rows and metadata['partial'] and metadata['status'] == 'failed'
    assert metadata['error']['source_frame_index'] == 2
