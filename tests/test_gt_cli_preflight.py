import pytest

from tools.compare_ard100_ground_truth import main
from yolomg_motion.run_artifacts import write_tracking_run


@pytest.mark.parametrize('index', [None, True, 2.5, -1, [2]])
def test_malformed_frame_indices_do_not_create_gt_output(tmp_path, index):
    write_tracking_run(tmp_path, [dict(frame_index=index)], dict(input='demo.mp4', output_frames=1))
    with pytest.raises(SystemExit) as error:
        main(['--run', str(tmp_path), '--annotations', 'missing.zip', '--sequence', 'demo', '--preview-only'])
    assert error.value.code == 2
    assert not (tmp_path / 'ground_truth_comparison').exists()


@pytest.mark.parametrize('args', [['--snapshot-every', '0'], ['--annotation-offset', 'nan']])
def test_invalid_render_options_fail_before_reading_inputs(tmp_path, args):
    with pytest.raises(SystemExit) as error:
        main(['--run', str(tmp_path), '--annotations', 'missing.zip', '--sequence', 'demo', '--preview-only'] + args)
    assert error.value.code == 2
    assert not list(tmp_path.iterdir())
