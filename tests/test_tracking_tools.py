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


@pytest.mark.parametrize('update,expected', [
    ({'state': 'UNKNOWN'}, 'unknown tracker state'),
    ({'source': 'predicted'}, 'unknown tracking source'),
    ({'state': 'LOST'}, 'inactive tracker state'),
    ({'source': 'none'}, 'observation source'),
    ({'frame_index': 10**1000}, 'frame_index'),
])
def test_audit_rejects_impossible_tracker_records(tmp_path, update, expected):
    run = _run(tmp_path)
    path = run / 'tracks.json'
    payload = json.loads(path.read_text())
    payload['frames'][0].update(update)
    path.write_text(json.dumps(payload))
    report = audit_run(run)
    assert not report['passed']
    assert any(expected in error for error in report['errors'])


def test_complete_video_cannot_hide_interrupted_run_status(tmp_path):
    run = _run(tmp_path)
    path = run / 'tracks.json'
    payload = json.loads(path.read_text())
    payload['metadata']['status'] = 'interrupted'
    path.write_text(json.dumps(payload))
    assert any('incomplete' in error for error in audit_run(run)['errors'])
