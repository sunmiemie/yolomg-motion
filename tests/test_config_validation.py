"""Reject invalid public settings before they reach NumPy/OpenCV."""
import numpy as np
import pytest

from yolomg_motion.core import MotionConfig


@pytest.mark.parametrize('field', [
    'frame_step', 'blur_size', 'grid_step', 'grid_step_y', 'alignment_width',
    'alignment_height', 'lk_window', 'lk_levels', 'lk_iterations',
    'min_inliers', 'morph_kernel', 'open_iterations', 'close_iterations',
])
@pytest.mark.parametrize('value', [True, 3.5, '3', float('nan'), float('inf')])
def test_motion_integer_settings_reject_non_integers(field, value):
    with pytest.raises(ValueError, match=field):
        MotionConfig(**{field: value})


@pytest.mark.parametrize('field', [
    'lk_epsilon', 'ransac_threshold', 'min_inlier_ratio', 'fb_threshold', 'min_overlap',
])
@pytest.mark.parametrize('value', [True, '0.5', float('nan'), float('inf')])
def test_motion_real_settings_must_be_finite_numbers(field, value):
    with pytest.raises(ValueError, match=field):
        MotionConfig(**{field: value})


def test_numpy_numeric_settings_remain_accepted():
    config = MotionConfig(frame_step=np.int64(2), fb_threshold=np.float32(1.5))
    assert config.frame_step == 2
    assert config.fb_threshold == 1.5
