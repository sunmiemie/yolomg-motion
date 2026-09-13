from tools.compare_ard100_ground_truth import evaluate_rows, match_detections


def test_difficult_overlap_does_not_count_as_false_positive():
    box = [10, 10, 20, 20]
    predictions = [dict(bbox=box, confidence=.8), dict(bbox=[40, 40, 50, 50], confidence=.9)]
    result = match_detections(predictions, [], ignored_boxes=[box])
    assert (result['tp'], result['fp'], result['fn'], result['ignored_predictions']) == (0, 1, 0, 1)
    assert match_detections(predictions[:1], [box], ignored_boxes=[box])['tp'] == 1


def test_difficult_only_frame_is_distinct_from_true_background():
    box = [10, 10, 20, 20]
    row = dict(frame_index=0, state='LOCKED', bbox=box, detector_ran=True,
               detections=[dict(bbox=box, confidence=.9)])
    annotation = dict(member='example.xml', boxes=[], ignored_boxes=[box],
                      difficult_drone_objects_ignored=1)
    counts, details = evaluate_rows([row], {0: annotation})
    assert counts['ignored_only_frames'] == counts['locked_ignored_frames'] == 1
    assert counts['gt_negative_frames'] == counts['locked_wrong_frames'] == counts['detector_fp'] == 0
    assert counts['detector_ignored_predictions'] == 1
    assert counts['detector_precision'] is None
    assert details[0]['ignored_locked_frame'] is True
