"""Bounded, model-independent proposals and original-resolution crop detection.

Windows use half-open original-frame coordinates. The detector owns letterbox
and resizing; cropping here preserves the original BGR pixels and the absolute
0..255 motion signal, with no display gain or intermediate resize.
"""
from numbers import Integral, Real

import cv2
import numpy as np

from .tracking import Detection


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _shape(frame_shape):
    try:
        height, width = frame_shape[:2]
    except (TypeError, ValueError, IndexError) as error:
        raise ValueError('frame_shape must contain positive integer height and width') from error
    return _integer(height, 'height'), _integer(width, 'width')


def _motion_map(motion):
    if not isinstance(motion, np.ndarray) or motion.ndim != 2 or motion.dtype != np.float32:
        raise ValueError('motion must be an H x W float32 raw difference map')
    _shape(motion.shape)
    if not np.isfinite(motion).all() or np.any(motion < 0) or np.any(motion > 255):
        raise ValueError('motion must contain finite raw intensities in 0..255')


def square_window(center_xy, side, frame_shape):
    """Return an integer square, translated at edges to preserve its size.

    ``side`` is a positive integer, capped by the shorter frame dimension.
    Finite centers outside the frame are allowed and yield an edge window.
    One-pixel-wide/high inputs therefore yield a valid 1 x 1 window.
    """
    height, width = _shape(frame_shape)
    side = min(_integer(side, 'side'), height, width)
    try:
        center = np.asarray(center_xy, dtype=np.float64)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError('center_xy must contain two finite coordinates') from error
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError('center_xy must contain two finite coordinates')
    left = int(np.clip(np.floor(center[0] - side / 2), 0, width - side))
    top = int(np.clip(np.floor(center[1] - side / 2), 0, height - side))
    return left, top, left + side, top + side


def _window_iou(first, second):
    overlap = max(0, min(first[2], second[2]) - max(first[0], second[0]))
    overlap *= max(0, min(first[3], second[3]) - max(first[1], second[1]))
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return overlap / (first_area + second_area - overlap)


def motion_windows(motion, valid_mask, side=320, max_windows=2,
                   threshold=8.0, min_area=2, max_area=400):
    """Propose at most ``max_windows`` square crops from raw motion components.

    Large connected background regions are excluded by area. Component mean
    intensity preselects a bounded shortlist; local foreground/background
    contrast then ranks it. Windows with IoU >= 0.35 are suppressed. This is a
    proposal heuristic, not target classification, and empty motion yields no
    proposals. ``max_windows=0`` disables proposals without inventing a crop.
    """
    _motion_map(motion)
    _integer(side, 'side')
    max_windows = _integer(max_windows, 'max_windows', 0)
    min_area = _integer(min_area, 'min_area')
    max_area = _integer(max_area, 'max_area')
    if max_area < min_area:
        raise ValueError('max_area must be >= min_area')
    if isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, Real):
        raise ValueError('threshold must be finite and in 0..255')
    if not np.isfinite(threshold) or not 0 <= threshold <= 255:
        raise ValueError('threshold must be finite and in 0..255')
    if (not isinstance(valid_mask, np.ndarray) or valid_mask.shape != motion.shape
            or valid_mask.dtype.kind not in 'buif' or not np.isfinite(valid_mask).all()):
        raise ValueError('valid_mask must be a finite numeric/bool array matching motion')
    if max_windows == 0:
        return []
    valid = valid_mask.astype(bool, copy=False)
    foreground = (motion > threshold) & valid
    count, labels, stats, centers = cv2.connectedComponentsWithStats(
        foreground.astype(np.uint8), connectivity=8)
    areas = stats[:, cv2.CC_STAT_AREA]
    candidates = np.flatnonzero((areas >= min_area) & (areas <= max_area))
    candidates = candidates[candidates != 0]
    if not len(candidates):
        return []

    # Exclude background before bincount to avoid converting the entire
    # full-resolution frame to temporary int64/float64 arrays on sparse motion.
    sums = np.bincount(labels[foreground], weights=motion[foreground], minlength=count)
    means = sums[candidates] / areas[candidates]
    # Only these small neighborhoods need Python-level background statistics,
    # even if noisy frames have thousands of tiny connected components.
    strength = means * np.sqrt(np.sqrt(areas[candidates]))
    shortlist = np.argsort(-strength, kind='stable')[:max(64, max_windows * 32)]
    height, width = motion.shape
    ranked = []
    for position in shortlist:
        label = int(candidates[position])
        x, y, w, h, area = (int(value) for value in stats[label])
        margin = max(4, int(np.ceil(np.sqrt(area))))
        x1, y1 = max(0, x - margin), max(0, y - margin)
        x2, y2 = min(width, x + w + margin), min(height, y + h + margin)
        background = valid[y1:y2, x1:x2] & (labels[y1:y2, x1:x2] != label)
        local = motion[y1:y2, x1:x2]
        background_mean = float(local[background].mean()) if background.any() else 0.0
        prominence = max(0.0, float(means[position]) - background_mean)
        score = prominence * float(np.sqrt(np.sqrt(area)))
        ranked.append((score, float(means[position]), label))
    ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
    windows = []
    for score, _, label in ranked:
        if score <= 0:
            continue
        window = square_window(centers[label], side, motion.shape)
        if any(_window_iou(window, previous) >= 0.35 for previous in windows):
            continue
        windows.append(window)
        if len(windows) == max_windows:
            break
    return windows


def crop_detect(detector, frame, motion, window):
    """Detect one synchronized native-resolution crop and restore coordinates.

    ``detector.detect`` returns crop coordinates after its own resizing and
    letterbox reversal. Both existing ``bbox_xyxy`` detections and tracking
    ``bbox`` detections are accepted. Invalid detections and boxes within one
    pixel of an artificial crop boundary are discarded; boundaries coincident
    with the real image edge remain eligible. The caller can widen a failed
    crop instead of accepting a truncated box as a confirmed target.
    """
    if (not isinstance(frame, np.ndarray) or frame.ndim != 3
            or frame.shape[2] != 3 or frame.dtype != np.uint8):
        raise ValueError('frame must be a uint8 H x W x 3 BGR image')
    height, width = _shape(frame.shape)
    _motion_map(motion)
    if motion.shape != frame.shape[:2]:
        raise ValueError('frame and motion must have the same spatial dimensions')
    try:
        if len(window) != 4:
            raise ValueError('window must contain four integer coordinates')
        x1, y1, x2, y2 = (_integer(value, 'window coordinate', 0) for value in window)
    except TypeError as error:
        raise ValueError('window must contain four integer coordinates') from error
    if not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height:
        raise ValueError('window must be nonempty and within the frame')
    image_crop = frame[y1:y2, x1:x2]
    motion_crop = motion[y1:y2, x1:x2]
    detections = detector.detect(image_crop, motion_crop)
    result = []
    crop_width, crop_height = x2 - x1, y2 - y1
    for detection in detections:
        try:
            bbox = getattr(detection, 'bbox_xyxy', None)
            bbox = detection.bbox if bbox is None else bbox
            box = np.asarray(bbox, dtype=np.float64)
            confidence = float(detection.confidence)
        except (AttributeError, TypeError, ValueError, OverflowError):
            continue
        if box.shape != (4,) or not np.isfinite(box).all():
            continue
        if not np.isfinite(confidence) or not 0 < confidence <= 1:
            continue
        left, top, right, bottom = box
        if not (0 <= left < right <= crop_width and 0 <= top < bottom <= crop_height):
            continue
        if right - left < 1 or bottom - top < 1:
            continue
        if ((x1 > 0 and left <= 1) or (y1 > 0 and top <= 1)
                or (x2 < width and right >= crop_width - 1)
                or (y2 < height and bottom >= crop_height - 1)):
            continue
        result.append(Detection((float(left + x1), float(top + y1),
                                 float(right + x1), float(bottom + y1)), confidence))
    return result
