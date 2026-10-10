import pytest

from tools.compare_tracking_runs import compare_runs
from yolomg_motion.run_artifacts import write_tracking_run


def test_comparison_aligns_by_frame_not_position(tmp_path):
    left, right = tmp_path / 'left', tmp_path / 'right'
    source = str(tmp_path / 'source.mp4')
    write_tracking_run(left, [dict(frame_index=2, state='LOST'), dict(frame_index=3, state='LOCKED')],
                       dict(input=source, detect_every=3))
    write_tracking_run(right, [dict(frame_index=3, state='LOST'), dict(frame_index=4, state='LOCKED')],
                       dict(input=source, detect_every=1))
    report = compare_runs(left, right)
    assert report['shared_frames'] == 1 and report['changed_frames'] == 1
    assert report['left_only_frames'] == [2] and report['right_only_frames'] == [4]
    assert report['changes'] == [dict(frame_index=3, fields=['state'])]
    assert report['settings_differences']['detect_every'] == dict(left=3, right=1)
    assert {p.name for p in left.iterdir()} == {'tracks.json'}


def test_comparison_refuses_different_sources_and_duplicate_indices(tmp_path):
    left, right = tmp_path / 'left', tmp_path / 'right'
    write_tracking_run(left, [], dict(input='one.mp4'))
    write_tracking_run(right, [], dict(input='two.mp4'))
    with pytest.raises(ValueError, match='different source'):
        compare_runs(left, right)
    write_tracking_run(right, [dict(frame_index=2)] * 2, dict(input='one.mp4'))
    with pytest.raises(ValueError, match='unique'):
        compare_runs(left, right)
