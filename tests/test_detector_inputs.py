import numpy as np
import pytest

from yolomg_motion.yolo_detector import YOLOMGDetector


@pytest.mark.parametrize('frame,motion', [
    (None, np.zeros((16, 16), np.float32)),
    (np.empty((0, 16, 3), np.uint8), np.empty((0, 16), np.float32)),
    (np.zeros((16, 16, 3), np.uint8), None),
    (np.zeros((16, 16, 3), np.uint8), np.full((16, 16), '1')),
    (np.zeros((16, 16, 3), np.uint8), np.zeros((16, 16), np.complex64)),
    (np.zeros((16, 16, 3), np.uint8), np.zeros((16, 16), bool)),
])
def test_invalid_inputs_rejected_before_model_preprocessing(frame, motion):
    # Validation can run without importing Torch or downloading weights.
    detector = YOLOMGDetector.__new__(YOLOMGDetector)
    with pytest.raises(ValueError, match='frame_bgr|Motion map'):
        detector.detect(frame, motion)
