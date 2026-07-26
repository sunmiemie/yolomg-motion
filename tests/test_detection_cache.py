import copy

import pytest

from yolomg_motion.track_cli import validate_detection_cache


ROW = dict(frame_index=2, detector_ran=True, motion_valid=True,
           detections=[dict(bbox=[0, 0, 4, 4], confidence=0.9)])
META = dict(input='video.mp4', weights='model.pt', output_frames=1)


@pytest.mark.parametrize('field,value', [
    ('frame_index', True), ('frame_index', 3), ('detector_ran', 1),
    ('motion_valid', False), ('detections', None), ('detections', [None]),
    ('detections', [dict(bbox=[0, 0, 0, 4], confidence=0.5)]),
    ('detections', [dict(bbox=[0, 0, 4, 4], confidence=float('nan'))]),
])
def test_bad_cache_observations_fail_before_replay(field, value):
    row = copy.deepcopy(ROW)
    row[field] = value
    with pytest.raises(ValueError, match='Detection cache'):
        validate_detection_cache([row], META)


@pytest.mark.parametrize('update', [{'completed': False}, {'input': None}, {'output_frames': 3}])
def test_cache_metadata_is_checked(update):
    with pytest.raises(ValueError, match='Detection cache'):
        validate_detection_cache([ROW], META | update)


def test_empty_and_skipped_observations_are_distinct():
    validate_detection_cache([ROW | {'detections': []}], META)
    validate_detection_cache([ROW | {'detector_ran': False, 'detections': None}], META)
    with pytest.raises(ValueError, match='without detector'):
        validate_detection_cache([ROW | {'detector_ran': False}], META)
