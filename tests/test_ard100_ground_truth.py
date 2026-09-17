"""Ground-truth coordinate mapping and fixed-threshold one-to-one metrics."""
from zipfile import ZipFile

import pytest

from tools.compare_ard100_ground_truth import box_iou, evaluate_rows, match_detections, read_annotation


def _annotation(boxes):
    return dict(member='synthetic.xml', width=1920, height=1080,
                boxes=boxes, difficult_drone_objects_ignored=0)


def test_voc_one_based_filename_and_official_minus_one_offset(tmp_path):
    xml = '''<annotation><size><width>1920</width><height>1080</height></size>
    <object><name>Drone</name><difficult>0</difficult><bndbox>
    <xmin>821</xmin><ymin>281</ymin><xmax>840</xmax><ymax>292</ymax>
    </bndbox></object>
    <object><name>Drone</name><difficult>1</difficult><bndbox>
    <xmin>2</xmin><ymin>3</ymin><xmax>8</xmax><ymax>9</ymax>
    </bndbox></object></annotation>'''
    path = tmp_path/'annotations.zip'
    with ZipFile(path, 'w') as archive:
        archive.writestr('annotations/phantom02/phantom02_0001.xml', xml)
    with ZipFile(path) as archive:
        parsed = read_annotation(archive, 'phantom02', frame_index=0)
        assert parsed['boxes'] == [(820, 280, 839, 291)]
        assert parsed['difficult_drone_objects_ignored'] == 1
        assert read_annotation(archive, 'phantom02', frame_index=0, offset=0)['boxes'] == [(821, 281, 840, 292)]
        assert read_annotation(archive, 'phantom02', frame_index=1) is None


def test_one_to_one_matching_penalizes_duplicate_and_filters_low_confidence():
    ground_truth = [(10, 10, 20, 20), (50, 50, 60, 60)]
    candidates = [dict(bbox=(10, 10, 20, 20), confidence=.9),
                  dict(bbox=(10, 10, 20, 20), confidence=.8),
                  dict(bbox=(50, 50, 60, 60), confidence=.09),
                  dict(bbox=(80, 80, 90, 90), confidence=.7)]
    result = match_detections(candidates, ground_truth)
    assert (result['tp'], result['fp'], result['fn']) == (1, 2, 1)
    assert result['eligible_predictions'] == 3
    assert box_iou((10, 10, 20, 20), (10, 10, 20, 20)) == 1
    # Continuous xyxy convention: touching edges have zero area intersection.
    assert box_iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0
    assert match_detections([dict(bbox=(0, 0, 5, 10), confidence=.1)], [(0, 0, 10, 10)])['tp'] == 1


def test_missing_annotations_excluded_and_unlocked_gt_counts_zero_iou():
    gt = (10, 10, 20, 20)
    wrong = (100, 100, 110, 110)
    rows = [
        dict(frame_index=0, state='LOCKED', bbox=gt, detector_ran=True,
             detections=[dict(bbox=gt, confidence=.9)]),
        dict(frame_index=1, state='LOST', bbox=None, detector_ran=False, detections=None),
        dict(frame_index=2, state='LOCKED', bbox=wrong, detector_ran=True,
             detections=[dict(bbox=wrong, confidence=.9)]),
        # No XML: even an apparently false prediction is not silently scored.
        dict(frame_index=3, state='LOCKED', bbox=wrong, detector_ran=True,
             detections=[dict(bbox=wrong, confidence=.9)]),
        dict(frame_index=4, state='LOCKED', bbox=wrong, detector_ran=True,
             detections=[dict(bbox=wrong, confidence=.9)]),
    ]
    annotations = {0: _annotation([gt]), 1: _annotation([gt]),
                   2: _annotation([gt]), 3: None, 4: _annotation([])}
    metrics, details = evaluate_rows(rows, annotations)
    assert metrics['missing_annotations'] == 1
    assert metrics['gt_positive_frames'] == 3
    assert metrics['gt_negative_frames'] == 1
    assert metrics['locked_correct_frames'] == 1
    assert metrics['locked_wrong_frames'] == 2
    assert metrics['locked_on_negative_frames'] == 1
    assert metrics['missed_tracking_frames'] == 1
    assert metrics['gt_frames_without_correct_lock'] == 2
    assert metrics['mean_iou_gt_frames_including_unlocked_zero'] == pytest.approx(1/3)
    assert metrics['detector_evaluated_frames'] == 3
    assert (metrics['detector_tp'], metrics['detector_fp'], metrics['detector_fn']) == (1, 2, 1)
    assert metrics['detector_precision'] == pytest.approx(1/3)
    assert metrics['detector_recall'] == .5
    assert details[3]['evaluation_status'] == 'missing_annotation'
