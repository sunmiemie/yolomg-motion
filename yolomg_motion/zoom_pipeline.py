"""Bounded motion proposals -> original-resolution crops -> local YOLOMG.

The detector receives matching RGB/motion crops and performs its own resize to
the configured network input. No ground truth or hand-picked coordinates enter
acquisition. Camera compensation uses a larger context than the detector crop.
Inference is synchronous and budgeted by source-frame interval, not real-time.
"""
from collections import deque
from dataclasses import asdict
from time import perf_counter

import numpy as np

from .core import _validate_frame
from .roi_pipeline import ROIObservation
from .roi_windows import _window_iou, crop_detect, motion_windows, square_window
from .tracking import Detection, SingleTargetTracker


def _center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


class ZoomROIPipeline:
    def __init__(self, detector, extractor, tracker=None, *, local_extractor=None,
                 detect_every=10, roi_size=320, motion_context=640, max_rois=2,
                 motion_threshold=8.0, scan_every=3, prediction_max_roi=None):
        for name, value in (('detect_every', detect_every), ('roi_size', roi_size),
                            ('motion_context', motion_context), ('max_rois', max_rois),
                            ('scan_every', scan_every)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if roi_size < 16 or motion_context < roi_size:
            raise ValueError('Require motion_context >= roi_size >= 16')
        if not np.isfinite(motion_threshold) or not 0 < motion_threshold <= 255:
            raise ValueError('motion_threshold must lie in (0, 255]')
        self.detector, self.extractor = detector, extractor
        self.local_extractor = local_extractor or extractor
        if any(e.config.frame_step != 2 for e in (extractor, self.local_extractor)):
            raise ValueError('ROI zoom requires centered frame_step=2')
        self.tracker = tracker or SingleTargetTracker(
            max_detector_gap=None, roi_only=True, min_confidence=.1,
            min_template_width=8, min_template_height=8, template_padding=2)
        if not self.tracker.roi_only or self.tracker.max_detector_gap is not None:
            raise ValueError('ROI zoom requires roi_only and max_detector_gap=None')
        self.detect_every, self.roi_size = int(detect_every), int(roi_size)
        self.motion_context, self.max_rois = int(motion_context), int(max_rois)
        self.motion_threshold, self.scan_every = float(motion_threshold), int(scan_every)
        self.prediction_enabled = callable(getattr(self.tracker, 'prediction_details', None))
        if self.prediction_enabled and self.tracker.prediction_max_gap < detect_every:
            raise ValueError('prediction_max_gap must be >= detect_every to permit detector confirmation')
        if prediction_max_roi is None:
            prediction_max_roi = max(roi_size, 480)
        if (isinstance(prediction_max_roi, bool) or
                not isinstance(prediction_max_roi, (int, np.integer)) or prediction_max_roi < roi_size):
            raise ValueError('prediction_max_roi must be an integer >= roi_size')
        self.prediction_max_roi = int(prediction_max_roi)
        self._frames = deque(maxlen=5)
        self._received, self._shape, self._last_attempt = 0, None, None
        self._scan_cursor, self._search_attempts = 0, 0
        self._recovery_center, self._recovery_attempts = None, 0

    @property
    def frame_shape(self):
        return self._shape

    def _scan_window(self):
        """Round-robin overlapping tiles, one budget slot; catches weak motion."""
        height, width = self._shape[:2]
        side = min(self.roi_size, height, width)
        step = max(1, int(side * .75))
        xs = list(range(0, width - side + 1, step))
        ys = list(range(0, height - side + 1, step))
        if xs[-1] != width - side:
            xs.append(width - side)
        if ys[-1] != height - side:
            ys.append(height - side)
        position = self._scan_cursor % (len(xs) * len(ys))
        self._scan_cursor += 1
        x, y = xs[position % len(xs)], ys[position // len(xs)]
        return (x, y, x + side, y + side)

    def _search_windows(self, motion):
        self._search_attempts += 1
        windows = motion_windows(motion.difference, motion.valid_mask,
                                 side=self.roi_size, max_windows=self.max_rois,
                                 threshold=self.motion_threshold)
        proposal_count = len(windows)
        # Reserve a bounded scan slot periodically, even with strong distractors.
        if not windows or self._search_attempts % self.scan_every == 0:
            scan = self._scan_window()
            # Preserve the scan's progress while avoiding a second inference
            # on nearly the same crop; spend remaining slots on other regions.
            windows = [w for w in windows if _window_iou(w, scan) < .8][:self.max_rois - 1]
            windows.append(scan)
        return windows, proposal_count

    def push(self, frame):
        _validate_frame(frame)
        if frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 16:
            raise ValueError('Expected BGR frame of at least 16x16 pixels')
        if self._shape is not None and frame.shape != self._shape:
            raise ValueError('Frame shape changed; construct a new pipeline')
        self._shape = frame.shape
        self._frames.append(frame)
        self._received += 1
        if len(self._frames) < 5:
            return None
        index, current = self._received - 3, self._frames[2]
        started = perf_counter()
        before = self.tracker.state
        prediction_started = perf_counter()
        prediction = self.tracker.prediction_details(index, horizon_frames=0) if self.prediction_enabled else None
        predicted = (tuple(prediction['bbox']) if prediction is not None else None) if self.prediction_enabled else self.tracker.predicted_box(index)
        prediction_roi = None
        if prediction is not None:
            side = self.roi_size
            if prediction['coasting']:
                # Keep the nominal magnification while observed. During loss,
                # grow only the crop, in bounded 32px steps, with uncertainty.
                side += int(np.ceil(4 * max(prediction['std_xy']) / 32)) * 32
            side = min(side, self.prediction_max_roi)
            prediction_roi = square_window(_center(predicted), side, self._shape)
        prediction_seconds = perf_counter() - prediction_started
        attempt = self._last_attempt is None or index - self._last_attempt >= self.detect_every
        local = predicted is not None or (
            not self.prediction_enabled and self._recovery_center is not None and self._recovery_attempts < 2)
        reason = None
        scope, motion_valid, context = None, None, None
        proposals, windows, input_shapes = None, [], []
        proposal_count = calls = 0
        motion_seconds = detector_seconds = proposal_seconds = 0.
        if attempt:
            self._last_attempt = index
            if local:
                center = _center(predicted) if predicted is not None else self._recovery_center
                side = self.roi_size if predicted is not None else int(
                    self.roi_size * (1.5 + .5 * self._recovery_attempts))
                window = prediction_roi or square_window(center, side, self._shape)
                context = square_window(_center(window), max(self.motion_context, window[2]-window[0]), self._shape)
                x, y, right, bottom = context
                triplet = [self._frames[i][y:bottom, x:right] for i in (0, 2, 4)]
                scope = 'local'
                reason = 'roi_verification' if predicted is not None else 'expanded_reacquisition'
                if prediction is not None and prediction['coasting']:
                    reason = 'predicted_reacquisition'
                if predicted is None:
                    self._recovery_attempts += 1
                tick = perf_counter()
                motion = self.local_extractor.process_triplet(*triplet)
                motion_seconds = perf_counter() - tick
                windows = [window]
            else:
                scope, reason = 'full', 'motion_search'
                tick = perf_counter()
                motion = self.extractor.process_triplet(self._frames[0], current, self._frames[4])
                motion_seconds = perf_counter() - tick
            motion_valid = bool(motion.success)
            if motion_valid:
                if not local:
                    tick = perf_counter()
                    windows, proposal_count = self._search_windows(motion)
                    proposal_seconds = perf_counter() - tick
                proposals = []
                for window in windows[:self.max_rois]:
                    tick = perf_counter()
                    if local:
                        x, y, right, bottom = context
                        relative = (window[0]-x, window[1]-y, window[2]-x, window[3]-y)
                        found = crop_detect(self.detector, current[y:bottom, x:right],
                                            motion.difference, relative)
                        found = [Detection((d.bbox[0]+x, d.bbox[1]+y,
                                            d.bbox[2]+x, d.bbox[3]+y), d.confidence) for d in found]
                        # The context is itself a crop, not the physical frame.
                        # If window == context, crop_detect cannot distinguish
                        # its artificial edges from true camera-image edges.
                        height, width = self._shape[:2]
                        found = [d for d in found if not (
                            (window[0] > 0 and d.bbox[0] <= window[0] + 1) or
                            (window[1] > 0 and d.bbox[1] <= window[1] + 1) or
                            (window[2] < width and d.bbox[2] >= window[2] - 1) or
                            (window[3] < height and d.bbox[3] >= window[3] - 1))]
                    else:
                        found = crop_detect(self.detector, current, motion.difference, window)
                    detector_seconds += perf_counter() - tick
                    calls += 1
                    shape = getattr(self.detector, 'last_input_shape', None)
                    input_shapes.append(list(shape) if shape is not None else None)
                    proposals.extend(found)
                # Different overlapping windows can detect the same object.
                proposals = self._deduplicate(proposals)
            else:
                windows = []
        tick = perf_counter()
        tracker_proposals = proposals
        if self.prediction_enabled and attempt and predicted is not None and not calls:
            # Failed camera registration is not a new visual observation.
            # End the visual lock while retaining only the bounded search hint.
            tracker_proposals = []
        result = self.tracker.update(current, tracker_proposals, index)
        # A local YOLO verification must actually associate a detector box.
        # A high template score alone cannot extend a stale lock indefinitely.
        if attempt and predicted is not None and result.source != 'detector':
            result = self.tracker.invalidate()
        if result.source == 'detector' or self.prediction_enabled:
            self._recovery_center, self._recovery_attempts = None, 0
        elif before in ('LOCKED', 'CONFIRMING') and result.state in ('LOST', 'SEARCHING'):
            self._recovery_center = _center(predicted) if predicted is not None else None
            self._recovery_attempts = 0
        if self.prediction_enabled:
            prediction = self.tracker.prediction_details(index, horizon_frames=self.detect_every)
            # This proposed ROI is the pre-observation window actually used to
            # guide this frame; it is null after the hard prediction lifetime.
            if prediction is None:
                prediction_roi = None
        tracking_seconds = perf_counter() - tick + prediction_seconds
        details = dict(
            mode='detection' if attempt else 'roi',
            roi_tracking=bool(before == 'LOCKED' and not attempt),
            detection_reason=reason, detector_ran=bool(calls), detector_calls=calls,
            motion_computed=bool(attempt), motion_valid=motion_valid, motion_scope=scope,
            motion_context_window=context, detection_windows=[list(w) for w in windows],
            network_input_shapes=input_shapes, proposal_count=proposal_count,
            detections=[asdict(d) for d in proposals] if calls else None,
            search_roi=self.tracker.last_search_roi,
            stage_timings=dict(motion_seconds=motion_seconds, detector_seconds=detector_seconds,
                               tracking_seconds=tracking_seconds, proposal_seconds=proposal_seconds),
            processing_seconds=perf_counter() - started)
        if self.prediction_enabled:
            details.update(prediction=prediction,
                           prediction_roi=list(prediction_roi) if prediction_roi is not None else None)
        return ROIObservation(index, current, result, details)

    @staticmethod
    def _deduplicate(detections):
        kept = []
        for detection in sorted(detections, key=lambda d: d.confidence, reverse=True):
            box = np.asarray(detection.bbox)
            duplicate = False
            for other in kept:
                b = np.asarray(other.bbox)
                intersection = np.maximum(0, np.minimum(box[2:], b[2:]) - np.maximum(box[:2], b[:2])).prod()
                union = (box[2:] - box[:2]).prod() + (b[2:] - b[:2]).prod() - intersection
                if union > 0 and intersection / union > .5:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(detection)
        return kept
