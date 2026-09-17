"""Offline video integration: real MFEM/tracker, only the neural net is stubbed.

The synthetic marker makes source/center frame alignment observable in the
encoded output, while a tiny bright moving target drives automatic acquisition.
No model weights, vendor repository, Torch, or downloads are required.
"""
import json

import cv2
import numpy as np
import pytest

from yolomg_motion import run_artifacts, track_cli, yolo_detector
from yolomg_motion.run_artifacts import read_tracking_run, write_tracking_run
from yolomg_motion.tracking import TrackResult


WIDTH, HEIGHT, FPS = 640, 360, 30.0
MARKER = (slice(304, 316), slice(24, 40))


def _make_video(path, frame_count=14):
    rng = np.random.default_rng(1803)
    gray = rng.integers(35, 85, (HEIGHT, WIDTH), dtype=np.uint8)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # Large background structure supports the actual default LK grid after
    # resizing to its 1920 x 1080 estimation resolution.
    for _ in range(200):
        x, y = int(rng.integers(12, WIDTH - 12)), int(rng.integers(12, HEIGHT - 12))
        cv2.circle(gray, (x, y), int(rng.integers(3, 9)), int(rng.integers(60, 160)), -1)
    background = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), FPS, (WIDTH, HEIGHT))
    assert writer.isOpened(), 'OpenCV MP4 writer is required by the video adapter'
    try:
        for index in range(frame_count):
            frame = background.copy()
            frame[140:143, 100 + 2 * index:110 + 2 * index] = 255
            frame[300:322, 20:44] = 25 + index * 10
            writer.write(frame)
    finally:
        writer.release()


def _decode(path):
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened(), f'Cannot decode {path}'
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    return fps, frames


def test_real_motion_and_tracker_produce_decodable_center_aligned_outputs(tmp_path, monkeypatch):
    video = tmp_path / 'synthetic.mp4'
    output = tmp_path / 'tracking'
    _make_video(video)
    source_fps, source_frames = _decode(video)
    model_instances = []

    class BrightTargetDetector:
        def __init__(self, **kwargs):
            self.arguments = kwargs
            self.calls = []
            model_instances.append(self)

        def detect(self, frame, motion):
            # These are genuine MFEM outputs, not externally fabricated maps.
            assert motion.dtype == np.float32
            assert motion.shape == frame.shape[:2]
            assert np.isfinite(motion).all()
            assert 0 <= float(motion.min()) <= float(motion.max()) <= 255
            assert np.any(motion > 0)
            self.calls.append(float(frame[MARKER].mean()))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ys, xs = np.nonzero(gray > 205)
            assert len(xs), 'The encoded synthetic target must remain visible'
            bbox = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
            return [yolo_detector.Detection(bbox, 0.9, 0)]

    monkeypatch.setattr(yolo_detector, 'YOLOMGDetector', BrightTargetDetector)
    arguments = ['--input', str(video), '--output', str(output),
                 '--max-frames', '12', '--detect-every', '3', '--snapshot-every', '3']
    assert track_cli.main(arguments) == 0

    payload = json.loads((output / 'tracks.json').read_text(encoding='utf-8'))
    assert payload['schema_version'] == 2
    assert set(payload) == {'schema_version', 'metadata', 'frames'}
    rows, summary = read_tracking_run(output, strict_finite=True)
    assert {path.name for path in output.iterdir()} == {'tracked_zoom.mp4', 'tracks.json'}
    assert summary['status'] == 'complete'
    assert summary['completed'] and not summary['partial']
    assert summary['error'] is None
    assert summary['video_frames_written'] == 8
    # 12 input frames and k=2 yield centers 2..9, with no fabricated boundaries.
    assert [row['frame_index'] for row in rows] == list(range(2, 10))
    assert summary['output_frames'] == len(rows) == 8
    assert summary['first_lock_frame'] == 5
    assert summary['last_lock_frame'] == 9
    assert summary['locked_frames'] == 5
    assert summary['detector_calls'] == 3
    assert summary['detector_backend'] == 'cpu'
    assert summary['motion_backend'] == 'cpu'
    assert summary['tracking_backend'] == 'cpu'
    assert summary['stage_timings']['motion_seconds'] > 0
    assert len(model_instances) == 1
    assert [row['frame_index'] for row in rows if row['detector_ran']] == [2, 5, 8]
    assert all(row['motion_valid'] for row in rows)
    assert all(row['timestamp_seconds'] == pytest.approx(row['frame_index'] / FPS) for row in rows)
    assert rows[0]['state'] == 'CONFIRMING'
    assert all(row['track_id'] == 1 for row in rows if row['state'] == 'LOCKED')
    assert any(row['source'] == 'template' for row in rows)
    fps, frames = _decode(output / 'tracked_zoom.mp4')
    assert fps == pytest.approx(source_fps)
    assert len(frames) == len(rows)
    assert all(frame.shape == (HEIGHT, WIDTH, 3) for frame in frames)
    # A source-frame marker outside the overlays proves that the output
    # pixels actually correspond to the declared center-frame indices.
    for frame, row in zip(frames, rows):
        expected = source_frames[row['frame_index']][MARKER].mean()
        assert frame[MARKER].mean() == pytest.approx(expected, abs=6)
    # Prove that the one encoded video contains the zoom presentation, rather
    # than silently retaining the old non-zoom video under a different name.
    locked_index = next(i for i, row in enumerate(rows) if row['state'] == 'LOCKED')
    row = rows[locked_index]
    result = TrackResult(**{key: row[key] for key in TrackResult.__dataclass_fields__})
    original = source_frames[row['frame_index']]
    expected_zoom = track_cli.draw_result(original, result, row['frame_index'], fps, zoom=True)
    unzoomed = track_cli.draw_result(original, result, row['frame_index'], fps)
    decoded = frames[locked_index].astype(np.float32)
    zoom_error = np.abs(decoded-expected_zoom.astype(np.float32)).mean()
    unzoomed_error = np.abs(decoded-unzoomed.astype(np.float32)).mean()
    assert zoom_error < 8
    assert unzoomed_error > zoom_error + 4

    expected_markers = [source_frames[index][MARKER].mean() for index in (2, 5, 8)]
    np.testing.assert_allclose(model_instances[0].calls, expected_markers, atol=1)

    saved_artifact = (output / 'tracks.json').read_bytes()
    with pytest.raises(SystemExit) as error:
        track_cli.main(arguments)
    assert error.value.code == 2
    assert len(model_instances) == 1, 'Refuse overwriting before constructing another model'
    assert (output / 'tracks.json').read_bytes() == saved_artifact

    # Reusing actual recorded detector observations must leave timing/source
    # provenance explicit and reproduce the same tracker observations.
    replay_output = tmp_path / 'replay'
    replay_args = ['--input', str(video), '--output', str(replay_output),
                   '--max-frames', '12', '--detection-cache', str(output)]
    assert track_cli.main(replay_args) == 0
    assert len(model_instances) == 1, 'Detection replay must not construct a model'
    replay_rows, replay_summary = read_tracking_run(replay_output)
    assert {path.name for path in replay_output.iterdir()} == {'tracked_zoom.mp4', 'tracks.json'}
    for actual, expected in zip(replay_rows, rows):
        assert actual['state'] == expected['state']
        assert actual['bbox'] == expected['bbox']
        assert actual['detections'] == expected['detections']
    assert len(replay_rows) == len(rows)
    assert replay_summary['detector_seconds'] == 0
    assert replay_summary['cached_detector_seconds'] == summary['detector_seconds']
    assert replay_summary['detection_cache'] == str(output.resolve())
    assert replay_summary['detector_runtime'] == dict(mode='replay', inference_executed=False)

    # Existing v1 caches remain usable, including Windows UTF-8 BOM files.
    legacy = tmp_path / 'legacy'
    legacy.mkdir()
    (legacy / 'tracks.json').write_text(json.dumps(rows), encoding='utf-8-sig')
    (legacy / 'summary.json').write_text(json.dumps(summary), encoding='utf-8-sig')
    legacy_replay = tmp_path / 'legacy_replay'
    assert track_cli.main(['--input', str(video), '--output', str(legacy_replay),
                           '--max-frames', '12', '--detection-cache', str(legacy)]) == 0
    legacy_rows, legacy_metadata = read_tracking_run(legacy_replay)
    assert legacy_rows == rows
    assert legacy_metadata['detection_cache'] == str(legacy.resolve())
    assert {path.name for path in legacy_replay.iterdir()} == {'tracked_zoom.mp4', 'tracks.json'}
    assert len(model_instances) == 1

    with pytest.raises(SystemExit) as mismatch:
        track_cli.main(['--input', str(video), '--output', str(tmp_path / 'mismatch'),
                        '--detection-cache', str(output), '--confidence', '0.11'])
    assert mismatch.value.code == 2
    assert not (tmp_path / 'mismatch').exists()

    # Saved CPU observations cannot be relabeled as a GPU computation. Check
    # this before importing a GPU runtime or creating any output artifacts.
    for backend_args in (['--device', 'directml'], ['--motion-backend', 'opencl']):
        backend_output = tmp_path / ('mismatched_' + backend_args[-1])
        with pytest.raises(SystemExit) as backend_mismatch:
            track_cli.main(['--input', str(video), '--output', str(backend_output),
                            '--detection-cache', str(output)] + backend_args)
        assert backend_mismatch.value.code == 2
        assert not backend_output.exists()


def test_processing_failure_preserves_partial_v2_trajectory_and_decodable_video(tmp_path, monkeypatch):
    video = tmp_path / 'synthetic.mp4'
    _make_video(video, frame_count=8)
    output = tmp_path / 'interrupted'

    class FailingDetector:
        def __init__(self, **kwargs):
            self.calls = 0

        def detect(self, frame, motion):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError('synthetic detector failure')
            ys, xs = np.nonzero(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) > 205)
            return [yolo_detector.Detection((float(xs.min()), float(ys.min()),
                    float(xs.max()+1), float(ys.max()+1)), .9, 0)]

    monkeypatch.setattr(yolo_detector, 'YOLOMGDetector', FailingDetector)
    with pytest.raises(RuntimeError, match='synthetic detector failure'):
        track_cli.main(['--input', str(video), '--output', str(output)])
    rows, metadata = read_tracking_run(output, strict_finite=True)
    assert [row['frame_index'] for row in rows] == [2, 3, 4]
    assert metadata['status'] == 'failed'
    assert metadata['partial'] and not metadata['completed']
    assert metadata['error'] == dict(type='RuntimeError', message='synthetic detector failure', source_frame_index=5)
    assert metadata['last_recorded_source_frame'] == 4
    assert metadata['output_frames'] == metadata['video_frames_written'] == 3
    assert len(_decode(output/'tracked_zoom.mp4')[1]) == 3
    assert {path.name for path in output.iterdir()} == {'tracked_zoom.mp4', 'tracks.json'}


@pytest.mark.parametrize('legacy', [False, True])
def test_artifact_reader_accepts_bom_and_preserves_all_fields(tmp_path, legacy):
    rows = [dict(frame_index=2, bbox=[1.5, 2, 3, 4], custom={'unmodified': [1, None, '无人机']})]
    metadata = dict(input='example.mp4', custom={'value': 3.5})
    if legacy:
        (tmp_path/'tracks.json').write_text(json.dumps(rows), encoding='utf-8-sig')
        (tmp_path/'summary.json').write_text(json.dumps(metadata), encoding='utf-8-sig')
    else:
        write_tracking_run(tmp_path, rows, metadata)
        contents = (tmp_path/'tracks.json').read_text(encoding='utf-8')
        (tmp_path/'tracks.json').write_text(contents, encoding='utf-8-sig')
    assert read_tracking_run(tmp_path, strict_finite=True) == (rows, metadata)


@pytest.mark.parametrize('number', ['NaN', 'Infinity', '-Infinity', '1e400'])
def test_artifact_strict_reader_rejects_nonfinite_numbers(tmp_path, number):
    (tmp_path/'tracks.json').write_text(
        '{"schema_version":2,"metadata":{},"frames":[{"confidence":'+number+'}]}', encoding='utf-8')
    with pytest.raises(ValueError, match='Non-finite'):
        read_tracking_run(tmp_path, strict_finite=True)


def test_atomic_write_preserves_previous_artifact_when_serialization_or_replace_fails(tmp_path, monkeypatch):
    original_rows, original_metadata = [dict(frame_index=2)], dict(input='source.mp4')
    write_tracking_run(tmp_path, original_rows, original_metadata)
    saved = (tmp_path/'tracks.json').read_bytes()
    with pytest.raises(ValueError):
        write_tracking_run(tmp_path, [dict(confidence=float('nan'))], {})
    assert (tmp_path/'tracks.json').read_bytes() == saved

    def reject_replace(source, target):
        raise OSError('simulated replace failure')

    monkeypatch.setattr(run_artifacts.os, 'replace', reject_replace)
    with pytest.raises(OSError, match='simulated replace failure'):
        write_tracking_run(tmp_path, [dict(frame_index=3)], {})
    assert (tmp_path/'tracks.json').read_bytes() == saved
    assert {path.name for path in tmp_path.iterdir()} == {'tracks.json'}


def test_cached_center_frame_limit_obeys_same_future_frame_requirement(tmp_path):
    video = tmp_path / 'synthetic.mp4'
    _make_video(video)
    cache = tmp_path / 'cache'
    (cache / 'maps').mkdir(parents=True)
    (cache / 'summary.json').write_text(json.dumps({
        'input': str(video.resolve()), 'first_center_index': 2,
        'last_center_index': 11, 'config': {'frame_step': 2},
    }), encoding='utf-8')
    for index in range(2, 12):
        np.savez_compressed(cache / 'maps' / f'{index:06d}.npz',
                            center_index=np.int64(index), success=np.bool_(True),
                            difference=np.zeros((HEIGHT, WIDTH), np.float32))
    cap = cv2.VideoCapture(str(video))
    try:
        centers = list(track_cli.cached_centers(cap, cache, video, max_frames=8))
    finally:
        cap.release()
    # Eight input frames permit centers 2..5: later centers would consume
    # future frames outside the requested input range, despite existing cache.
    assert [item[0] for item in centers] == [2, 3, 4, 5]
