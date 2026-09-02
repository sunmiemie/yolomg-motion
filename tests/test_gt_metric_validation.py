import pytest

from tools.compare_ard100_ground_truth import box_iou, evaluate_rows, match_detections


@pytest.mark.parametrize('box', [None, [1, 2], [0, 0, 0, 4], [0, 0, float('nan'), 5],
                                 [[0, 0, 4, 4]], [0, 0, float('inf'), 5]])
def test_invalid_boxes_cannot_be_scored(box):
    with pytest.raises(ValueError, match='xyxy'):
        box_iou(box, [0, 0, 4, 4])


def test_large_finite_boxes_do_not_produce_nan():
    assert box_iou([0, 0, 1e200, 1e200], [0, 0, 1e200, 1e200]) == 1
    assert box_iou([0, 0, 1e-200, 1e-200], [0, 0, 1e-200, 1e-200]) == 1


@pytest.mark.parametrize('confidence,iou', [(float('nan'), .5), (.1, 0), (.1, 2), (-.1, .5)])
def test_invalid_thresholds_rejected_even_without_rows(confidence, iou):
    with pytest.raises(ValueError):
        evaluate_rows([], {}, confidence, iou)


def test_equal_iou_uses_first_annotation_deterministically():
    result = match_detections([dict(bbox=[0, 0, 4, 4], confidence=.9)], [[0, 0, 4, 4]] * 2)
    assert result['matches'][0]['gt_index'] == 0
    with pytest.raises(ValueError, match='confidence'):
        match_detections([dict(bbox=[0, 0, 4, 4], confidence=float('nan'))], [])
