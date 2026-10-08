import numpy as np
import pytest

from yolomg_motion.core import MotionExtractor, three_frame_difference


@pytest.mark.parametrize('value', [[], np.empty((0, 4)), np.full((2, 2), np.nan),
    np.full((2, 2), -1), np.full((2, 2), 256), np.ones((2, 2), complex),
    np.ones((2, 2), bool)])
def test_public_difference_rejects_invalid_intensity_arrays(value):
    with pytest.raises(ValueError):
        three_frame_difference(value, np.zeros((2, 2)), np.zeros((2, 2)))


def test_float_inputs_keep_absolute_intensity_units():
    image = np.full((2, 2), 127.5, np.float64)
    result = three_frame_difference(image, np.zeros((2, 2)), np.full((2, 2), 255))
    assert result.dtype == np.float32
    np.testing.assert_equal(result, 127.5)


@pytest.mark.parametrize('config', [False, 0, {}, {'frame_step': 2}])
def test_invalid_configuration_is_not_replaced_by_defaults(config):
    with pytest.raises(TypeError, match='MotionConfig'):
        MotionExtractor(config)
