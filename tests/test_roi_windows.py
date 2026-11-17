"""Crop geometry and bounded proposals without a model or ground truth."""
from types import SimpleNamespace

import numpy as np
import pytest

from yolomg_motion.roi_windows import contrast_windows, crop_detect, motion_windows, square_window
from yolomg_motion.tracking import Detection


@pytest.mark.parametrize('center,side,shape,expected', [
    ((50, 60), 20, (100, 200), (40, 50, 60, 70)),
    ((0, 0), 20, (100, 200, 3), (0, 0, 20, 20)),
    ((199, 99), 20, (100, 200), (180, 80, 200, 100)),
    ((500, -100), 20, (100, 200), (180, 0, 200, 20)),
    ((50, 40), 320, (100, 200), (0, 0, 100, 100)),
    ((0, 20), 320, (40, 1), (0, 19, 1, 20)),
    ((0, 0), 1, (1, 1), (0, 0, 1, 1)),
])
def test_square_geometry(center, side, shape, expected):
    assert square_window(center, side, shape) == expected


@pytest.mark.parametrize('center,side,shape', [
    ((np.nan, 1), 10, (50, 50)), ((1, np.inf), 10, (50, 50)),
    ((1,), 10, (50, 50)), ((1, 2, 3), 10, (50, 50)),
    ((1, 2), 0, (50, 50)), ((1, 2), True, (50, 50)),
    ((1, 2), 2.5, (50, 50)), ((1, 2), 10, (0, 50)),
    ((1, 2), 10, (50,)), ((1, 2), 10, (50.5, 50)),
])
def test_square_rejects_invalid_inputs(center, side, shape):
    with pytest.raises(ValueError):
        square_window(center, side, shape)


def test_motion_proposals_ignore_giant_background_and_respect_budget():
    motion = np.zeros((180, 400), np.float32)
    motion[1:45, 1:45] = 255  # large moving background, not a small target
    motion[80:83, 100:103] = 110
    motion[90:93, 280:283] = 80
    motion[130:133, 180:183] = 50
    mask = np.ones(motion.shape, bool)
    windows = motion_windows(motion, mask, side=40, max_windows=2)
    assert len(windows) == 2
    assert windows[0][0] <= 101 < windows[0][2]
    assert windows[0][1] <= 81 < windows[0][3]
    assert windows[1][0] <= 281 < windows[1][2]
    assert motion_windows(motion, mask, max_windows=0) == []


def test_nearby_proposals_share_one_window_and_invalid_region_is_ignored():
    motion = np.zeros((100, 220), np.float32)
    motion[30:32, 30:32] = 80
    motion[38:40, 40:42] = 90
    motion[60:62, 170:172] = 255
    mask = np.ones(motion.shape, np.uint8)
    mask[:, 150:] = 0
    windows = motion_windows(motion, mask, side=64, max_windows=5)
    assert len(windows) == 1
    assert windows[0][0] <= 31 < windows[0][2]
    assert windows[0][0] <= 41 < windows[0][2]


def test_local_prominence_downweights_a_motion_clutter_region():
    motion = np.zeros((100, 220), np.float32)
    motion[24:38, 24:38] = 7
    motion[30:32, 30:32] = 20
    motion[60:62, 170:172] = 18
    windows = motion_windows(motion, np.ones_like(motion, bool), side=30, max_windows=1)
    assert windows[0][0] <= 171 < windows[0][2]


@pytest.mark.parametrize('shape', [(1, 1), (1, 80), (80, 1), (30, 40)])
def test_empty_motion_never_fabricates_window(shape):
    motion = np.zeros(shape, np.float32)
    assert motion_windows(motion, np.ones(shape, bool), threshold=0) == []


def test_single_pixel_frame_can_propose_when_min_area_allows_it():
    motion = np.full((1, 1), 20, np.float32)
    assert motion_windows(motion, np.ones((1, 1), bool), min_area=1) == [(0, 0, 1, 1)]


@pytest.mark.parametrize('kwargs', [
    {'threshold': np.nan}, {'threshold': -1}, {'threshold': 256},
    {'threshold': True}, {'max_windows': -1}, {'max_windows': 1.5},
    {'min_area': 0}, {'max_area': 1, 'min_area': 2}, {'side': 0},
])
def test_motion_invalid_options(kwargs):
    motion = np.zeros((20, 20), np.float32)
    with pytest.raises(ValueError):
        motion_windows(motion, np.ones_like(motion, bool), **kwargs)


@pytest.mark.parametrize('bad_value', [np.nan, np.inf, -1, 256])
def test_motion_invalid_values_rejected(bad_value):
    motion = np.zeros((20, 20), np.float32)
    motion[5, 5] = bad_value
    with pytest.raises(ValueError):
        motion_windows(motion, np.ones_like(motion, bool))


def test_motion_invalid_mask_and_dtype_rejected():
    motion = np.zeros((20, 20), np.float32)
    for mask in [np.ones((19, 20), bool), np.full((20, 20), np.nan)]:
        with pytest.raises(ValueError):
            motion_windows(motion, mask)
    with pytest.raises(ValueError):
        motion_windows(motion.astype(np.uint8), np.ones_like(motion, bool))


class DetectorStub:
    def __init__(self, detections):
        self.detections = detections
        self.calls = []

    def detect(self, image, motion):
        self.calls.append((image.copy(), motion.copy()))
        return self.detections


def test_crop_preserves_bgr_and_raw_motion_and_restores_original_coordinates():
    frame = np.arange(60 * 80 * 3, dtype=np.uint8).reshape((60, 80, 3))
    motion = np.arange(60 * 80, dtype=np.float32).reshape((60, 80)) % 30 + .25
    detector = DetectorStub([SimpleNamespace(bbox_xyxy=(2, 3, 7, 9), confidence=.8)])
    result = crop_detect(detector, frame, motion, (10, 20, 30, 40))
    assert result == [Detection((12, 23, 17, 29), .8)]
    assert len(detector.calls) == 1
    image_crop, motion_crop = detector.calls[0]
    np.testing.assert_array_equal(image_crop, frame[20:40, 10:30])
    np.testing.assert_array_equal(motion_crop, motion[20:40, 10:30])
    assert motion_crop.dtype == np.float32


def test_crop_filters_invalid_and_artificial_boundary_detections():
    bad_boxes = [
        (np.nan, 2, 5, 6), (2, 2, np.inf, 6), (5, 2, 2, 6),
        (-1, 2, 5, 6), (2, 2, 21, 6), (2, 2, 2.5, 6),
        (0, 2, 5, 6), (1, 2, 5, 6), (2, 0, 5, 6),
        (2, 2, 19, 6), (2, 2, 5, 20), (1, 2, 3),
    ]
    detections = [Detection(box, .8) for box in bad_boxes]
    detections += [Detection((2, 2, 5, 6), conf) for conf in [np.nan, np.inf, -1, 0, 1.1]]
    detections += [SimpleNamespace(), Detection((2, 2, 5, 6), .9)]
    detector = DetectorStub(detections)
    frame = np.zeros((60, 80, 3), np.uint8)
    result = crop_detect(detector, frame, np.zeros((60, 80), np.float32), (10, 20, 30, 40))
    assert result == [Detection((12, 22, 15, 26), .9)]


def test_true_frame_edges_and_single_pixel_frame_are_allowed():
    frame = np.zeros((1, 1, 3), np.uint8)
    detector = DetectorStub([Detection((0, 0, 1, 1), .8)])
    assert crop_detect(detector, frame, np.zeros((1, 1), np.float32), (0, 0, 1, 1)) == detector.detections


@pytest.mark.parametrize('window', [
    (0, 0, 0, 5), (-1, 0, 10, 10), (0, 0, 11, 10),
    (0, 0, 10, 11), (0, 0, 10.0, 10), (0, 0, 10), None,
])
def test_bad_crops_never_call_detector(window):
    detector = DetectorStub([])
    with pytest.raises(ValueError):
        crop_detect(detector, np.zeros((10, 10, 3), np.uint8),
                    np.zeros((10, 10), np.float32), window)
    assert detector.calls == []


def test_invalid_image_or_motion_never_calls_detector():
    detector = DetectorStub([])
    frame = np.zeros((10, 10, 3), np.uint8)
    motion = np.zeros((10, 10), np.float32)
    cases = [(frame[:, :, 0], motion), (frame.astype(np.float32), motion),
             (frame, motion[:9]), (frame, np.full((10, 10), np.nan, np.float32))]
    for image, difference in cases:
        with pytest.raises(ValueError):
            crop_detect(detector, image, difference, (0, 0, 10, 10))
    assert detector.calls == []


def test_contrast_finds_dark_and_light_targets_without_motion_and_preserves_pixels():
    image = np.full((240, 480, 3), 120, np.uint8)
    image[90:94, 90:98] = 20
    image[110:114, 350:358] = 230
    before = image.copy()
    valid = np.ones(image.shape[:2], bool)
    windows = contrast_windows(image, valid, side=64)
    assert len(windows) == 2
    for cx, cy in ((94, 92), (354, 112)):
        assert any(x1 <= cx < x2 and y1 <= cy < y2 for x1, y1, x2, y2 in windows)
    np.testing.assert_array_equal(image, before)
    valid[:, :240] = False
    windows = contrast_windows(image, valid, side=64)
    assert len(windows) == 1 and windows[0][0] > 240
    assert contrast_windows(image, valid, max_windows=0) == []


def test_contrast_coordinates_return_to_native_resolution():
    image = np.full((1080, 1920, 3), 160, np.uint8)
    image[600:606, 1400:1412] = 30
    windows = contrast_windows(image, np.ones(image.shape[:2], bool), side=160)
    assert len(windows) == 1
    x1, y1, x2, y2 = windows[0]
    assert x1 <= 1406 < x2 and y1 <= 603 < y2
    assert x2 - x1 == y2 - y1 == 160


def test_contrast_does_not_fabricate_target_in_flat_image_or_invalid_region():
    image = np.full((160, 240, 3), 90, np.uint8)
    valid = np.ones(image.shape[:2], bool)
    assert contrast_windows(image, valid) == []
    image[70:74, 100:108] = 230
    assert contrast_windows(image, np.zeros_like(valid)) == []
    with pytest.raises(ValueError, match='valid_mask'):
        contrast_windows(image, valid[:100])
    with pytest.raises(ValueError, match='frame'):
        contrast_windows(image.astype(np.float32), valid)


def test_contrast_prefers_isolated_target_over_similar_texture():
    image = np.full((240, 480, 3), 120, np.uint8)
    rng = np.random.default_rng(8)
    image[30:210, 20:200] = rng.integers(30, 230, (180, 180, 1), dtype=np.uint8)
    image[110:114, 350:358] = 230
    window = contrast_windows(image, np.ones(image.shape[:2], bool), side=64, max_windows=1)[0]
    assert window[0] <= 354 < window[2] and window[1] <= 112 < window[3]
