"""Compact/legacy artifact adapters and explicitly requested diagnostics."""
import json
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
import pytest

from tools.audit_tracking import audit_run, main as audit_main


PROJECT = Path(__file__).resolve().parents[1]


def _video(path, count=3):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 30, (64, 48))
    assert writer.isOpened(), 'The test requires the MP4 writer used by the tracking CLI'
    try:
        for index in range(count):
            writer.write(np.full((48, 64, 3), index*30, np.uint8))
    finally:
        writer.release()


def _run(tmp_path, layout='v2'):
    run = tmp_path/'run'
    run.mkdir()
    source = tmp_path/'phantom_test.mp4'
    _video(source)
    _video(run/'tracked_zoom.mp4')
    box = [10, 12, 20, 22]
    rows = [dict(frame_index=index, timestamp_seconds=index/30, state='LOCKED',
                 bbox=box, track_id=1, source='detector', detector_confidence=.8,
                 appearance_score=None, detector_ran=True, motion_valid=True,
                 detections=[dict(bbox=box, confidence=.8)]) for index in range(3)]
    metadata = dict(input=str(source), input_fps=30, output_frames=3, locked_frames=3,
                    detector_calls=3, first_lock_frame=0, last_lock_frame=2,
                    states={'LOCKED': 3}, sources={'detector': 3})
    if layout == 'v2':
        payload = dict(schema_version=2, metadata=metadata, frames=rows)
    else:
        payload = rows
        (run/'summary.json').write_text(json.dumps(metadata), encoding='utf-8')
    (run/'tracks.json').write_text(json.dumps(payload), encoding='utf-8')
    return run


def test_audit_v2_direct_script_prints_only_and_imports_outside_project(tmp_path):
    run = _run(tmp_path)
    before = {p.name: p.read_bytes() for p in run.iterdir()}
    completed = subprocess.run([sys.executable, '-B', str(PROJECT/'tools/audit_tracking.py'),
                                '--run', str(run)], cwd=tmp_path, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report['passed']
    assert report['json_frames'] == 3
    assert set(report['videos']) == {'tracked_zoom.mp4'}
    assert report['videos']['tracked_zoom.mp4']['decoded_frames'] == 3
    assert {p.name: p.read_bytes() for p in run.iterdir()} == before


def test_audit_report_requires_explicit_path(tmp_path, capsys):
    run = _run(tmp_path)
    report_path = tmp_path/'diagnostics'/'verification.json'
    assert audit_main(['--run', str(run), '--report', str(report_path)]) == 0
    assert json.loads(report_path.read_text(encoding='utf-8'))['passed']
    assert json.loads(capsys.readouterr().out)['passed']
    assert {p.name for p in run.iterdir()} == {'tracks.json', 'tracked_zoom.mp4'}


def test_legacy_metadata_supported_and_present_clean_video_checked(tmp_path):
    run = _run(tmp_path, layout='legacy')
    assert audit_run(run)['passed']
    _video(run/'tracked.mp4', count=2)
    report = audit_run(run)
    assert not report['passed']
    assert set(report['videos']) == {'tracked.mp4', 'tracked_zoom.mp4'}
    assert any('tracked.mp4: decoded frame count' in error for error in report['errors'])


def test_zoom_video_is_required_even_if_legacy_clean_video_exists(tmp_path):
    run = _run(tmp_path)
    (run/'tracked_zoom.mp4').rename(run/'tracked.mp4')
    report = audit_run(run)
    assert not report['passed']
    assert any('tracked_zoom.mp4: cannot open video' in error for error in report['errors'])


@pytest.mark.parametrize('layout', ['v2', 'legacy'])
@pytest.mark.parametrize('invalid_number', ['NaN', '1e400'])
def test_audit_rejects_nonfinite_metadata_in_both_formats(tmp_path, layout, invalid_number):
    run = _run(tmp_path, layout=layout)
    path = run/('tracks.json' if layout == 'v2' else 'summary.json')
    payload = json.loads(path.read_text(encoding='utf-8'))
    metadata = payload['metadata'] if layout == 'v2' else payload
    metadata['nested'] = {'bad': '__NONFINITE__'}
    path.write_text(json.dumps(payload).replace('"__NONFINITE__"', invalid_number), encoding='utf-8')
    report = audit_run(run)
    assert not report['passed']
    assert any('Non-finite' in error for error in report['errors'])
    # The diagnostic result must still be valid JSON even when the input is not.
    json.dumps(report, allow_nan=False)
    assert not (run/'verification.json').exists()


def _headless_zoom_run(tmp_path):
    run = tmp_path/'zoom_run'
    run.mkdir()
    box = [10, 12, 20, 22]
    rows = []
    for index in range(5):
        calls = {0: 2, 3: 1}.get(index, 0)
        rows.append(dict(
            frame_index=index, timestamp_seconds=index/30, state='LOCKED',
            bbox=box, track_id=1, source='detector' if calls else 'roi',
            detector_confidence=.8, appearance_score=None,
            detector_ran=bool(calls), detector_calls=calls,
            motion_computed=bool(calls), motion_valid=True if calls else None,
            motion_scope=('full' if index == 0 else 'local') if calls else None,
            detection_windows=[[0, 0, 32, 32], [32, 16, 64, 48]][:calls],
            network_input_shapes=[[1, 3, 640, 640] for _ in range(calls)],
            detections=[dict(bbox=box, confidence=.8)] if calls else None))
    metadata = dict(pipeline='roi_zoom_detection', video_enabled=False, video_frames_written=0,
                    input_dimensions=[64, 48], input_fps=30, output_frames=len(rows),
                    detector_calls=3, detector_frames=2, detect_every=3, max_rois=2)
    payload = dict(schema_version=2, metadata=metadata, frames=rows)
    (run/'tracks.json').write_text(json.dumps(payload), encoding='utf-8')
    return run, payload


def test_headless_zoom_audits_actual_network_calls_separately_from_detector_frames(tmp_path):
    run, payload = _headless_zoom_run(tmp_path)
    report = audit_run(run)
    assert report['passed'], report['errors']
    assert report['videos'] == {}
    assert report['detector_calls'] == 3
    assert report['detector_frames'] == 2
    assert report['motion_skipped_frames'] == 3
    assert {p.name for p in run.iterdir()} == {'tracks.json'}
    # A custom adapter may omit its tensor shape; one entry per call is retained.
    payload['frames'][0]['network_input_shapes'][0] = None
    del payload['metadata']['max_rois']  # The documented default budget is two.
    (run/'tracks.json').write_text(json.dumps(payload), encoding='utf-8')
    assert audit_run(run)['passed']


@pytest.mark.parametrize(('change', 'expected'), [
    ('total_calls', 'detector_calls disagrees'),
    ('total_frames', 'detector_frames disagrees'),
    ('boolean_calls', 'detector_calls must be a nonnegative integer'),
    ('negative_calls', 'detector_calls must be a nonnegative integer'),
    ('ran_flag', 'detector_ran disagrees'),
    ('budget', 'exceeds max_rois budget'),
    ('invalid_budget', 'max_rois must be a positive integer'),
    ('window_count', 'detection_windows count disagrees'),
    ('window_bounds', 'detection window 0: bbox is outside video bounds'),
    ('window_coordinate', 'requires four integer xyxy coordinates'),
    ('shape_count', 'network_input_shapes count disagrees'),
    ('shape_dimension', 'four positive NCHW dimensions'),
    ('invalid_motion', 'network calls require computed, valid motion input'),
    ('skipped_motion', 'network calls require computed, valid motion input'),
    ('skipped_scope', 'skipped motion requires null motion_scope'),
    ('computed_scope', 'computed motion requires full or local motion_scope'),
    ('interval', 'motion attempt violates detect_every interval'),
    ('invalid_interval', 'detect_every must be a positive integer'),
])
def test_headless_zoom_rejects_tampered_counts_windows_and_budget(tmp_path, change, expected):
    run, payload = _headless_zoom_run(tmp_path)
    metadata, rows = payload['metadata'], payload['frames']
    first = rows[0]
    if change == 'total_calls':
        metadata['detector_calls'] = 2
    elif change == 'total_frames':
        metadata['detector_frames'] = 3
    elif change == 'boolean_calls':
        first['detector_calls'] = True
    elif change == 'negative_calls':
        first['detector_calls'] = -1
    elif change == 'ran_flag':
        first['detector_ran'] = False
    elif change == 'budget':
        metadata['max_rois'] = 1
    elif change == 'invalid_budget':
        metadata['max_rois'] = True
    elif change == 'window_count':
        first['detection_windows'].pop()
    elif change == 'window_bounds':
        first['detection_windows'][0][2] = 65
    elif change == 'window_coordinate':
        first['detection_windows'][0][0] = .5
    elif change == 'shape_count':
        first['network_input_shapes'].pop()
    elif change == 'shape_dimension':
        first['network_input_shapes'][0][2] = 0
    elif change == 'invalid_motion':
        first['motion_valid'] = False
    elif change == 'skipped_motion':
        first.update(motion_computed=False, motion_valid=None, motion_scope=None)
    elif change == 'skipped_scope':
        rows[1]['motion_scope'] = 'full'
    elif change == 'computed_scope':
        first['motion_scope'] = None
    elif change == 'interval':
        metadata['detect_every'] = 4
    elif change == 'invalid_interval':
        metadata['detect_every'] = 0
    (run/'tracks.json').write_text(json.dumps(payload), encoding='utf-8')
    report = audit_run(run)
    assert not report['passed']
    assert any(expected in error for error in report['errors']), report['errors']


def test_zoom_failed_motion_attempt_still_consumes_interval_budget(tmp_path):
    run, payload = _headless_zoom_run(tmp_path)
    # Failed preprocessing at frame one must count as an attempt even with no inference.
    payload['frames'][1].update(motion_computed=True, motion_valid=False, motion_scope='full')
    (run/'tracks.json').write_text(json.dumps(payload), encoding='utf-8')
    report = audit_run(run)
    assert not report['passed']
    interval_errors = [error for error in report['errors'] if 'violates detect_every interval' in error]
    assert len(interval_errors) == 2  # Frames 0 -> 1 and 1 -> 3 both violate a three-frame gap.


def _prediction_zoom_run(tmp_path):
    run, payload = _headless_zoom_run(tmp_path)
    payload['metadata'].update(
        motion_prediction=dict(enabled=True, max_gap_frames=20, gate_threshold=16.,
                               process_noise=.25, max_roi=32),
        predicted_only_frames=2, locked_frames=3)
    for index, row in enumerate(payload['frames']):
        row.update(prediction=None, prediction_roi=None)
        if index > 2:
            continue
        coasting = index > 0
        if coasting:
            row.update(state='LOST' if index == 1 else 'SEARCHING', bbox=None,
                       track_id=None, source='none', detector_confidence=None)
        # Extrapolation can leave the image, including negative coordinates.
        row['prediction'] = dict(bbox=[65., -10., 75., 0.], center=[70., -5.],
                                 velocity=[2., -.5], std_xy=[2., 3.],
                                 covariance=[[4., 1.], [1., 9.]], age_frames=index,
                                 coasting=coasting, horizon_frames=3, future_center=[76., -6.5])
        row['prediction_roi'] = [32, 0, 64, 32]
    (run/'tracks.json').write_text(json.dumps(payload), encoding='utf-8')
    return run, payload


def test_prediction_hints_can_leave_image_without_becoming_observed_locks(tmp_path):
    run, payload = _prediction_zoom_run(tmp_path)
    report = audit_run(run)
    assert report['passed'], report['errors']
    assert report['predicted_only_frames'] == 2
    assert report['locked_frames'] == 3
    assert report['unlocked_frames'] == 2
    assert report['lock_fraction'] == .6
    # Prediction ROIs remain optional, and image clipping can make them smaller.
    del payload['frames'][0]['prediction_roi']
    payload['frames'][1]['prediction_roi'] = [48, 32, 64, 48]
    (run/'tracks.json').write_text(json.dumps(payload), encoding='utf-8')
    assert audit_run(run)['passed']


@pytest.mark.parametrize(('change', 'expected'), [
    ('wrong_count', 'predicted_only_frames disagrees'),
    ('boolean_count', 'predicted_only_frames disagrees'),
    ('counted_as_locks', 'locked_frames disagrees'),
    ('missing_prediction', 'requires prediction, which may be null'),
    ('not_object', 'prediction must be null or an object'),
    ('degenerate_box', 'bbox must have positive width and height'),
    ('bad_center', 'center must contain two finite numbers'),
    ('bad_velocity', 'velocity must contain two finite numbers'),
    ('negative_std', 'std_xy must be nonnegative'),
    ('bad_covariance_shape', 'covariance must be a finite 2x2 matrix'),
    ('asymmetric_covariance', 'covariance must be symmetric'),
    ('indefinite_covariance', 'covariance must be positive semidefinite'),
    ('old_prediction', 'age_frames exceeds max_gap_frames'),
    ('boolean_age', 'age_frames must be a nonnegative integer'),
    ('negative_horizon', 'horizon_frames must be a nonnegative integer'),
    ('missing_future', 'future_center must contain two finite numbers'),
    ('non_boolean_coasting', 'coasting must be boolean'),
    ('coasting_state', 'coasting must agree with LOST/SEARCHING state'),
    ('coasting_bbox', 'coasting requires null observed bbox and source none'),
    ('coasting_source', 'coasting requires null observed bbox and source none'),
    ('roi_outside_image', 'prediction_roi: bbox is outside video bounds'),
    ('roi_over_budget', 'prediction_roi exceeds max_roi budget'),
    ('roi_without_prediction', 'prediction_roi requires an active prediction'),
    ('disabled_metadata', 'prediction requires enabled motion_prediction metadata'),
    ('bad_max_gap', 'motion_prediction.max_gap_frames must be a positive integer'),
    ('bad_max_roi', 'motion_prediction.max_roi must be a positive integer'),
])
def test_prediction_audit_rejects_malformed_hints_and_observation_claims(tmp_path, change, expected):
    run, payload = _prediction_zoom_run(tmp_path)
    metadata, rows = payload['metadata'], payload['frames']
    row = rows[1]
    prediction = row['prediction']
    if change == 'wrong_count':
        metadata['predicted_only_frames'] = 3
    elif change == 'boolean_count':
        metadata['predicted_only_frames'] = True
    elif change == 'counted_as_locks':
        metadata['locked_frames'] = 5
    elif change == 'missing_prediction':
        del row['prediction']
    elif change == 'not_object':
        row['prediction'] = []
    elif change == 'degenerate_box':
        prediction['bbox'][2] = prediction['bbox'][0]
    elif change == 'bad_center':
        prediction['center'] = [1., 2., 3.]
    elif change == 'bad_velocity':
        prediction['velocity'] = [True, 1.]
    elif change == 'negative_std':
        prediction['std_xy'] = [-1., 2.]
    elif change == 'bad_covariance_shape':
        prediction['covariance'] = [[1., 2.], [3.]]
    elif change == 'asymmetric_covariance':
        prediction['covariance'] = [[4., 1.], [2., 9.]]
    elif change == 'indefinite_covariance':
        prediction['covariance'] = [[1., 2.], [2., 1.]]
    elif change == 'old_prediction':
        prediction['age_frames'] = 21
    elif change == 'boolean_age':
        prediction['age_frames'] = True
    elif change == 'negative_horizon':
        prediction['horizon_frames'] = -1
    elif change == 'missing_future':
        del prediction['future_center']
    elif change == 'non_boolean_coasting':
        prediction['coasting'] = 1
    elif change == 'coasting_state':
        prediction['coasting'] = False
    elif change == 'coasting_bbox':
        row['bbox'] = [10, 12, 20, 22]
    elif change == 'coasting_source':
        row['source'] = 'roi'
    elif change == 'roi_outside_image':
        row['prediction_roi'] = [32, 0, 65, 32]
    elif change == 'roi_over_budget':
        row['prediction_roi'] = [0, 0, 64, 48]
    elif change == 'roi_without_prediction':
        row['prediction'] = None
    elif change == 'disabled_metadata':
        metadata['motion_prediction']['enabled'] = False
    elif change == 'bad_max_gap':
        metadata['motion_prediction']['max_gap_frames'] = True
    elif change == 'bad_max_roi':
        metadata['motion_prediction']['max_roi'] = 0
    (run/'tracks.json').write_text(json.dumps(payload), encoding='utf-8')
    report = audit_run(run)
    assert not report['passed']
    assert any(expected in error for error in report['errors']), report['errors']


def test_prediction_audit_rejects_nonfinite_covariance_without_nonfinite_report(tmp_path):
    run, payload = _prediction_zoom_run(tmp_path)
    payload['frames'][1]['prediction']['covariance'][0][0] = float('inf')
    (run/'tracks.json').write_text(json.dumps(payload), encoding='utf-8')
    report = audit_run(run)
    assert not report['passed']
    assert any('Non-finite' in error for error in report['errors'])
    json.dumps(report, allow_nan=False)


def test_gt_comparison_v2_reads_one_json_and_only_explicitly_generates_outputs(tmp_path):
    run = _run(tmp_path)
    archive_path = tmp_path/'annotations.zip'
    xml = '''<annotation><size><width>64</width><height>48</height></size>
    <object><name>Drone</name><difficult>0</difficult><bndbox>
    <xmin>11</xmin><ymin>13</ymin><xmax>21</xmax><ymax>23</ymax>
    </bndbox></object></annotation>'''
    with ZipFile(archive_path, 'w') as archive:
        for frame in range(1, 4):
            archive.writestr(f'annotations/phantom_test/phantom_test_{frame:04d}.xml', xml)
    before = {p.name: p.read_bytes() for p in run.iterdir()}
    completed = subprocess.run([sys.executable, '-B', str(PROJECT/'tools/compare_ard100_ground_truth.py'),
                                '--run', str(run), '--annotations', str(archive_path),
                                '--sequence', 'phantom_test', '--alignment-verified'],
                               cwd=tmp_path, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    report = json.loads((run/'ground_truth_comparison/gt_evaluation.json').read_text(encoding='utf-8'))
    assert report['metrics']['locked_correct_frames'] == 3
    assert (report['metrics']['detector_tp'], report['metrics']['detector_fp'], report['metrics']['detector_fn']) == (3, 0, 0)
    assert report['metrics']['mean_iou_gt_frames_including_unlocked_zero'] == 1
    assert not (run/'summary.json').exists()
    assert {p.name for p in run.iterdir()} == {'tracks.json', 'tracked_zoom.mp4', 'ground_truth_comparison'}
    assert all((run/name).read_bytes() == value for name, value in before.items())
