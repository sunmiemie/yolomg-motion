import numpy as np
import pytest

from yolomg_motion.yolo_detector import _decode_detections


def test_invalid_scores_labels_and_boxes_never_reach_tracker():
    valid = [0, 0, 20, 10, .8, 0]
    rows = [valid, [0, 0, 20, 10, 1.1, 0], [0, 0, 20, 10, .8, .5],
            [-1, 0, 20, 10, .8, 0], [0, 0, 21, 10, .8, 0],
            [0, 0, 20, 10, .8, -1], [0, 0, 20, 10, np.nan, 0]]
    detections = _decode_detections(np.array(rows), (10, 20))
    assert len(detections) == 1
    assert detections[0].bbox_xyxy == (0., 0., 20., 10.)
    assert detections[0].class_id == 0


@pytest.mark.parametrize('values', [[], np.zeros(6), np.zeros((2, 5)), np.full((1, 6), 'x')])
def test_malformed_model_output_reports_adapter_error(values):
    with pytest.raises(ValueError, match='N x 6'):
        _decode_detections(values, (10, 20))


def test_empty_detection_batch_is_valid():
    assert _decode_detections(np.zeros((0, 6), np.float32), (10, 20)) == []
