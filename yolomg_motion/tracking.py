"""Conservative single-target tracking between intermittent detector updates.

Only detector observations can create or confirm an identity. Local normalized
cross-correlation can bridge short detector gaps, but cannot extend a track
beyond ``max_detector_gap`` source frames. This is a tracking utility, not a
drone classifier and not a component claimed by the YOLOMG paper.
"""
from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
from typing import List, Optional, Tuple

import cv2
import numpy as np


Box = Tuple[float, float, float, float]


@dataclass(frozen=True)
class Detection:
    bbox: Box
    confidence: float


@dataclass(frozen=True)
class TrackResult:
    state: str
    bbox: Optional[Box] = None
    track_id: Optional[int] = None
    source: str = 'none'
    detector_confidence: Optional[float] = None
    appearance_score: Optional[float] = None


def _center(box):
    return np.array([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5])


def _size(box):
    return np.array([box[2] - box[0], box[3] - box[1]])


class SingleTargetTracker:
    """Automatically acquire one target and preserve its local identity.

    ``detections=None`` means no detector ran on this frame; ``[]`` means a
    detector ran but reported no eligible target. Two consecutive *detector
    runs* with spatially consistent detections confirm a track. An empty or
    inconsistent detector run breaks an unconfirmed sequence.

    Detector confidence is returned only for an accepted current-frame
    detection. A template result instead carries ``appearance_score``; these
    are different quantities. LOST results never contain a predicted box.

    ``max_detector_gap`` is measured in the caller's source frame indices,
    not update call count. The tracking patch can be larger than a tiny box;
    returned boxes always preserve the detector's target size.
    """

    def __init__(self, confirmations=2, max_detector_gap=15,
                 search_radius=32, min_confidence=0.2,
                 appearance_threshold=0.65, max_size_ratio=2.0,
                 min_template_width=24, min_template_height=16,
                 min_template_std=2.0, backend='cpu'):
        from .opencl_motion import backend_information
        self.backend = backend
        self.backend_info = backend_information(backend)
        integer_settings = dict(confirmations=confirmations, max_detector_gap=max_detector_gap,
                                search_radius=search_radius, min_template_width=min_template_width,
                                min_template_height=min_template_height)
        for name, value in integer_settings.items():
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f'{name} must be an integer')
        real_settings = dict(min_confidence=min_confidence, appearance_threshold=appearance_threshold,
                             max_size_ratio=max_size_ratio, min_template_std=min_template_std)
        for name, value in real_settings.items():
            if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
                raise ValueError(f'{name} must be a finite number')
        if confirmations < 1 or max_detector_gap < 1:
            raise ValueError('confirmations and max_detector_gap must be positive')
        if search_radius < 1 or min_template_width < 2 or min_template_height < 2:
            raise ValueError('Search radius and template dimensions must be positive')
        if not 0 <= min_confidence <= 1 or not 0 < appearance_threshold <= 1:
            raise ValueError('Confidence thresholds must be in their valid ranges')
        if max_size_ratio < 1 or min_template_std <= 0:
            raise ValueError('Invalid size-ratio or template-variance threshold')
        self.confirmations = int(confirmations)
        self.max_detector_gap = int(max_detector_gap)
        self.search_radius = int(search_radius)
        self.min_confidence = float(min_confidence)
        self.appearance_threshold = float(appearance_threshold)
        self.max_size_ratio = float(max_size_ratio)
        self.min_template_width = int(min_template_width)
        self.min_template_height = int(min_template_height)
        self.min_template_std = float(min_template_std)
        self.reset()

    def reset(self):
        """Start a new sequence; its first confirmed identity will be 1."""
        self._state = 'SEARCHING'
        self._bbox = None
        self._track_id = None
        self._next_id = 1
        self._confirmation_count = 0
        self._last_detection_index = None
        self._last_box_index = None
        self._last_frame_index = None
        self._frame_shape = None
        self._velocity = np.zeros(2, dtype=np.float64)
        self._template = None
        self._template_gpu = None
        self._template_offset = None

    def _valid_detections(self, detections, width, height):
        valid = []
        for detection in detections:
            try:
                box = np.asarray(detection.bbox, dtype=np.float64)
                confidence = float(detection.confidence)
            except (ValueError, TypeError, AttributeError):
                continue
            if box.shape != (4,) or not np.isfinite(box).all():
                continue
            if not np.isfinite(confidence) or not self.min_confidence <= confidence <= 1:
                continue
            x1, y1, x2, y2 = box
            x1, x2 = np.clip([x1, x2], 0, width)
            y1, y2 = np.clip([y1, y2], 0, height)
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            valid.append(Detection((float(x1), float(y1), float(x2), float(y2)), confidence))
        return valid

    def _predicted_box(self, frame_index):
        elapsed = frame_index - self._last_box_index
        shift = self._velocity * elapsed
        return tuple(np.asarray(self._bbox) + np.tile(shift, 2))

    def _associate(self, detections, predicted):
        size = _size(predicted)
        radius = max(float(self.search_radius), float(np.linalg.norm(size) * 0.5))
        candidates = []
        for detection in detections:
            ratios = _size(detection.bbox) / size
            if np.any(ratios > self.max_size_ratio) or np.any(ratios < 1 / self.max_size_ratio):
                continue
            distance = float(np.linalg.norm(_center(detection.bbox) - _center(predicted)))
            if distance > radius:
                continue
            # Locality dominates confidence; a distant high score cannot steal
            # an existing identity from a nearby lower-confidence detection.
            cost = distance / radius + 0.25 * float(np.abs(np.log(ratios)).sum())
            cost -= 0.05 * detection.confidence
            candidates.append((cost, detection))
        return min(candidates, key=lambda item: item[0])[1] if candidates else None

    def _set_box(self, box, frame_index):
        if self._bbox is not None and self._last_box_index is not None:
            elapsed = frame_index - self._last_box_index
            if elapsed > 0:
                measured = (_center(box) - _center(self._bbox)) / elapsed
                self._velocity = 0.5 * self._velocity + 0.5 * measured
        self._bbox = tuple(float(value) for value in box)
        self._last_box_index = frame_index

    def _set_template(self, gray):
        height, width = self._frame_shape[:2] if self.backend == 'opencl' else gray.shape
        box_width, box_height = _size(self._bbox)
        patch_width = min(width, max(self.min_template_width, int(np.ceil(box_width)) + 12))
        patch_height = min(height, max(self.min_template_height, int(np.ceil(box_height)) + 12))
        center = _center(self._bbox)
        left = int(np.clip(np.floor(center[0] - patch_width / 2), 0, width - patch_width))
        top = int(np.clip(np.floor(center[1] - patch_height / 2), 0, height - patch_height))
        if self.backend == 'opencl':
            patch = cv2.UMat(gray, (left, top, patch_width, patch_height))
            template = patch.get()
        else:
            template = gray[top:top + patch_height, left:left + patch_width].copy()
        if float(template.std()) < self.min_template_std:
            self._template = None
            self._template_gpu = None
            self._template_offset = None
            return
        self._template = template
        if self.backend == 'opencl':
            self._template_gpu = cv2.UMat(template)
        self._template_offset = center - [left, top]

    def _match_template(self, gray, predicted):
        if self._template is None:
            return None, None
        height, width = self._frame_shape[:2] if self.backend == 'opencl' else gray.shape
        th, tw = self._template.shape
        expected = _center(predicted) - self._template_offset
        radius = self.search_radius
        left = max(0, int(np.floor(expected[0] - radius)))
        top = max(0, int(np.floor(expected[1] - radius)))
        right = min(width, int(np.ceil(expected[0] + tw + radius)))
        bottom = min(height, int(np.ceil(expected[1] + th + radius)))
        if right - left < tw or bottom - top < th:
            return None, None
        if self.backend == 'opencl':
            search = cv2.UMat(gray, (left, top, right-left, bottom-top))
            scores = cv2.matchTemplate(search, self._template_gpu, cv2.TM_CCOEFF_NORMED).get()
        else:
            search = gray[top:bottom, left:right]
            scores = cv2.matchTemplate(search, self._template, cv2.TM_CCOEFF_NORMED)
        scores[~np.isfinite(scores)] = -1
        _, score, _, location = cv2.minMaxLoc(scores)
        if score < self.appearance_threshold:
            return None, float(score)
        matched = (cv2.UMat(search, (location[0], location[1], tw, th)).get()
                   if self.backend == 'opencl' else search[location[1]:location[1] + th, location[0]:location[0] + tw])
        if float(matched.std()) < self.min_template_std:
            return None, float(score)
        center = np.array([left + location[0], top + location[1]]) + self._template_offset
        half_size = _size(self._bbox) / 2
        box = (center[0] - half_size[0], center[1] - half_size[1],
               center[0] + half_size[0], center[1] + half_size[1])
        if box[0] < 0 or box[1] < 0 or box[2] > width or box[3] > height:
            return None, float(score)
        return box, float(score)

    def _lose(self, appearance_score=None):
        self._state = 'LOST'
        self._bbox = None
        self._template = None
        self._template_gpu = None
        self._template_offset = None
        self._confirmation_count = 0
        self._velocity[:] = 0
        return TrackResult('LOST', track_id=self._track_id, appearance_score=appearance_score)

    def _start_candidate(self, gray, detection, frame_index):
        self._bbox = None
        self._template = None
        self._template_gpu = None
        self._template_offset = None
        self._velocity[:] = 0
        self._track_id = None
        self._confirmation_count = 0
        self._state = 'CONFIRMING'
        return self._accept_detection(gray, detection, frame_index)

    def _accept_detection(self, gray, detection, frame_index):
        self._set_box(detection.bbox, frame_index)
        self._last_detection_index = frame_index
        self._set_template(gray)
        if self._state != 'LOCKED':
            self._confirmation_count += 1
            if self._confirmation_count >= self.confirmations:
                self._state = 'LOCKED'
                self._track_id = self._next_id
                self._next_id += 1
        return TrackResult(self._state, self._bbox, self._track_id,
                           'detector', detection.confidence)

    def update(self, frame: np.ndarray, detections: Optional[List[Detection]],
               frame_index: int) -> TrackResult:
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
            raise ValueError('Expected uint8 BGR frame')
        if frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 16:
            raise ValueError('Expected BGR frame of at least 16x16 pixels')
        if isinstance(frame_index, bool) or not isinstance(frame_index, (int, np.integer)) or frame_index < 0:
            raise ValueError('frame_index must be a nonnegative integer')
        if self._last_frame_index is not None and frame_index <= self._last_frame_index:
            raise ValueError('frame_index must strictly increase; call reset for a new sequence')
        if self._frame_shape is not None and frame.shape != self._frame_shape:
            raise ValueError('Frame shape changed; call reset for a new sequence')
        self._frame_shape = frame.shape
        self._last_frame_index = int(frame_index)
        if self.backend == 'opencl':
            from .opencl_motion import ensure_opencl_enabled
            ensure_opencl_enabled()
            gray = cv2.cvtColor(cv2.UMat(frame), cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eligible = None if detections is None else self._valid_detections(
            detections, frame.shape[1], frame.shape[0])

        if self._state in ('SEARCHING', 'LOST'):
            if eligible:
                return self._start_candidate(gray, max(eligible, key=lambda item: item.confidence), frame_index)
            return TrackResult(self._state, track_id=self._track_id)

        # A time gap invalidates the old identity even if a new detection is
        # now present. Reacquisition must earn fresh confirmations.
        if frame_index - self._last_detection_index > self.max_detector_gap:
            if eligible:
                self._lose()
                return self._start_candidate(gray, max(eligible, key=lambda item: item.confidence), frame_index)
            return self._lose()

        predicted = self._predicted_box(frame_index)
        if eligible is not None:
            associated = self._associate(eligible, predicted)
            if associated is not None:
                return self._accept_detection(gray, associated, frame_index)
            if self._state == 'CONFIRMING':
                if eligible:
                    return self._start_candidate(gray, max(eligible, key=lambda item: item.confidence), frame_index)
                self._bbox = None
                self._template = None
                self._template_gpu = None
                self._confirmation_count = 0
                self._state = 'SEARCHING'
                return TrackResult('SEARCHING')

        box, score = self._match_template(gray, predicted)
        if box is None:
            return self._lose(score)
        self._set_box(box, frame_index)
        # Do not refresh the template on its own predictions: only a matched
        # detector observation is allowed to change the appearance reference.
        return TrackResult(self._state, self._bbox, self._track_id,
                           'template', appearance_score=score)
