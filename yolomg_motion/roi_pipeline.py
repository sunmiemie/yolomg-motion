"""Lazy full-frame acquisition followed by local-only appearance tracking.

The centered k=2 detector input retains two future frames (and their latency).
Raw frames are buffered; expensive motion preprocessing runs only on detector
attempts. Synchronous acquisition/reacquisition can stall; this is not a hard
real-time scheduler or an RKNN detector implementation.
"""
from collections import deque
from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np

from .core import _validate_frame
from .tracking import Detection, SingleTargetTracker, TrackResult


@dataclass
class ROIObservation:
    frame_index: int
    frame: np.ndarray
    result: TrackResult
    details: dict


class ROIPipeline:
    def __init__(self, detector, extractor, tracker=None, detect_every=3, verify_every=0):
        if detect_every < 1 or verify_every < 0:
            raise ValueError('detect_every must be positive and verify_every nonnegative')
        if extractor.config.frame_step != 2:
            raise ValueError('ROI acquisition currently requires centered k=2 motion')
        self.detector, self.extractor = detector, extractor
        self.tracker = tracker or SingleTargetTracker(max_detector_gap=None, roi_only=True,
                                                      min_template_width=8, min_template_height=8, template_padding=2)
        if not self.tracker.roi_only or self.tracker.max_detector_gap is not None:
            raise ValueError('ROI pipeline requires roi_only=True and max_detector_gap=None')
        self.detect_every, self.verify_every = int(detect_every), int(verify_every)
        self._frames = deque(maxlen=5)
        self._received = 0
        self._shape = None
        self._last_attempt = None
        self._force_detection = False

    @property
    def frame_shape(self):
        return self._shape

    def push(self, frame):
        # The caller must not mutate buffered arrays after handing them over.
        _validate_frame(frame)
        if frame.ndim != 3:
            raise ValueError('ROI pipeline requires BGR input')
        if self._shape is not None and frame.shape != self._shape:
            raise ValueError('Frame shape changed; construct a new ROI pipeline')
        self._shape = frame.shape
        self._frames.append(frame)
        self._received += 1
        if len(self._frames) < 5:
            return None
        index = self._received - 3
        current = self._frames[2]
        started = perf_counter()
        locked = self.tracker.state == 'LOCKED'
        verify = bool(locked and self.verify_every and
                      index - self.tracker.last_detection_index >= self.verify_every)
        attempt = bool(verify or (not locked and (self._force_detection or
                       self._last_attempt is None or index - self._last_attempt >= self.detect_every)))
        reason = 'verification' if verify else ('reacquisition' if self.tracker.state == 'LOST' else 'acquisition')
        proposals, motion_valid = None, None
        motion_seconds, detector_seconds = 0., 0.
        ran_detector = False
        self._force_detection = False
        if attempt:
            self._last_attempt = index
            tick = perf_counter()
            motion = self.extractor.process_triplet(self._frames[0], current, self._frames[4])
            motion_seconds = perf_counter() - tick
            motion_valid = bool(motion.success)
            if motion_valid:
                tick = perf_counter()
                raw = self.detector.detect(current, motion.difference)
                detector_seconds = perf_counter() - tick
                proposals = [Detection(tuple(d.bbox_xyxy), float(d.confidence)) for d in raw]
                ran_detector = True
        tick = perf_counter()
        result = self.tracker.update(current, proposals, index)
        if verify and result.source != 'detector':
            result = self.tracker.invalidate()
        if locked and result.state == 'LOST':
            self._force_detection = True
        tracking_seconds = perf_counter() - tick
        details = dict(mode='detection' if attempt else 'roi',
                       roi_tracking=bool(locked and not attempt),
                       detection_reason=reason if attempt else None,
                       detector_ran=ran_detector, motion_computed=attempt,
                       motion_valid=motion_valid,
                       detections=[asdict(d) for d in proposals] if proposals is not None else None,
                       search_roi=self.tracker.last_search_roi,
                       stage_timings=dict(motion_seconds=motion_seconds, detector_seconds=detector_seconds,
                                          tracking_seconds=tracking_seconds),
                       processing_seconds=perf_counter() - started)
        return ROIObservation(index, current, result, details)


def timing_statistics(seconds, target_fps=60.):
    values = np.asarray(seconds, dtype=float)
    if not len(values):
        return dict(samples=0, mean_ms=None, p50_ms=None, p95_ms=None, p99_ms=None,
                    max_ms=None, budget_ms=1000. / target_fps, over_budget_frames=0)
    return dict(samples=len(values), mean_ms=float(values.mean()*1000),
                p50_ms=float(np.percentile(values, 50)*1000),
                p95_ms=float(np.percentile(values, 95)*1000),
                p99_ms=float(np.percentile(values, 99)*1000), max_ms=float(values.max()*1000),
                budget_ms=1000. / target_fps, over_budget_frames=int((values > 1/target_fps).sum()))
