import csv

import pytest

from tools.export_tracking_csv import export_csv, render_csv
from yolomg_motion.run_artifacts import write_tracking_run


ROW = dict(frame_index=2, timestamp_seconds=0.1, state='LOST', bbox=None,
           track_id=1, source='none', detector_confidence=None,
           appearance_score=.2, detector_ran=False, motion_valid=True)


def test_csv_roundtrip_keeps_missing_observations_distinct(tmp_path):
    run = tmp_path / '运行'
    write_tracking_run(run, [ROW], {})
    before = (run / 'tracks.json').read_bytes()
    output = tmp_path / '导出.csv'
    assert export_csv(run, output) == 1
    with output.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]['x1'] == rows[0]['detector_confidence'] == ''
    assert rows[0]['detector_ran'] == '0' and rows[0]['motion_valid'] == '1'
    assert (run / 'tracks.json').read_bytes() == before
    with pytest.raises(FileExistsError):
        export_csv(run, output)


def test_invalid_csv_input_never_creates_output(tmp_path):
    run = tmp_path / 'run'
    write_tracking_run(run, [ROW | {'source': '=formula'}], {})
    output = tmp_path / 'new' / 'result.csv'
    with pytest.raises(ValueError):
        export_csv(run, output)
    assert not output.parent.exists()


def test_duplicate_frames_are_not_silently_exported():
    with pytest.raises(ValueError, match='indices'):
        render_csv([ROW, ROW])
