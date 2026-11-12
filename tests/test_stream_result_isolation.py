import numpy as np
import pytest

from yolomg_motion.core import MotionConfig, MotionExtractor


@pytest.mark.parametrize('frame_step', [1, 2])
@pytest.mark.parametrize('grayscale', [False, True], ids=['bgr', 'gray'])
@pytest.mark.parametrize('initial_failure', [False, True], ids=['valid-result', 'invalid-result'])
def test_editing_returned_gray_image_does_not_change_later_stream_results(
        frame_step, grayscale, initial_failure):
    config = MotionConfig(
        frame_step=frame_step, blur_size=3, grid_step=8, grid_step_y=8,
        alignment_width=64, alignment_height=64, min_inliers=4,
        open_iterations=0, close_iterations=0,
    )
    shape = (64, 64) if grayscale else (64, 64, 3)
    frame = np.random.default_rng(19).integers(0, 256, shape, dtype=np.uint8)
    original = frame.copy()
    blank = np.zeros_like(frame)
    control, edited = MotionExtractor(config), MotionExtractor(config)

    for index in range(2 * frame_step + 1):
        # A blank first frame gives one genuine LK failure while retaining a
        # textured center image that later becomes a previous source frame.
        source = blank if initial_failure and index == 0 else frame
        expected = control.push(source)
        actual = edited.push(source)
        if index < 2 * frame_step:
            assert expected is None and actual is None

    assert expected.success is (not initial_failure)
    assert actual.success == expected.success
    assert actual.center_index == expected.center_index == frame_step
    np.testing.assert_array_equal(actual.center_gray, expected.center_gray)
    assert actual.center_gray.any()
    actual.center_gray.fill(0)
    np.testing.assert_array_equal(frame, original)
    np.testing.assert_array_equal(blank, np.zeros_like(frame))

    # The edited center is reused as an earlier frame after k more pushes.
    # Check public results from the real LK path, not just memory ownership.
    for offset in range(1, frame_step + 2):
        expected = control.push(frame)
        actual = edited.push(frame)
        np.testing.assert_array_equal(frame, original)
        assert expected.success
        assert actual.success == expected.success, actual.previous_alignment.reason
        assert actual.center_index == expected.center_index == frame_step + offset
        for field in ('current', 'center_gray', 'difference', 'motion',
                      'unaligned_difference', 'valid_mask'):
            np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field),
                                          err_msg=f'{field} changed after editing an earlier result')
