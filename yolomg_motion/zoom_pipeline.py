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
import cv2

from .core import _validate_frame
from .roi_pipeline import ROIObservation
from .roi_windows import _window_iou, contrast_windows, crop_detect, motion_windows, square_window
from .tracking import Detection, SingleTargetTracker


def _center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


class ZoomROIPipeline:
    def __init__(self, detector, extractor, tracker=None, *, local_extractor=None,
                 detect_every=10, roi_size=320, motion_context=640, max_rois=2,
                 motion_threshold=8.0, scan_every=3, prediction_max_roi=None,
                 recent_search_attempts=2, contrast_search=False, scale_retry=False,
                 adaptive_roi=False):
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
        if (isinstance(recent_search_attempts, bool) or
                not isinstance(recent_search_attempts, (int, np.integer)) or
                not 0 <= recent_search_attempts <= 2):
            raise ValueError('recent_search_attempts must be an integer from 0 to 2')
        self.recent_search_attempts = int(recent_search_attempts)
        self.contrast_search = bool(contrast_search)
        self.scale_retry = bool(scale_retry)
        self.adaptive_roi = bool(adaptive_roi)
        self._target_sizes = deque(maxlen=3)
        self._adaptive_side = self.roi_size
        self._frames = deque(maxlen=5)
        self._received, self._shape, self._last_attempt = 0, None, None
        self._scan_cursor, self._search_attempts = 0, 0
        self._recovery_center, self._recovery_attempts = None, 0
        self._recent_box, self._recent_detection_index = None, None

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
        x, y = xs[position % len(xs)], ys[position // len(xs)]
        return (x, y, x + side, y + side)

    def _search_windows(self, motion, recent_window=None, frame=None):
        self._search_attempts += 1
        windows = motion_windows(motion.difference, motion.valid_mask,
                                 side=self.roi_size, max_windows=self.max_rois,
                                 threshold=self.motion_threshold)
        proposal_count = len(windows)
        scan = None
        periodic_scan = self._search_attempts % self.scan_every == 0
        # Reserve a bounded scan slot periodically, even with strong distractors.
        if not windows or periodic_scan:
            scan = self._scan_window()
            # Preserve the scan's progress while avoiding a second inference
            # on nearly the same crop; spend remaining slots on other regions.
            windows = [w for w in windows if _window_iou(w, scan) < .8][:self.max_rois - 1]
            windows.append(scan)
        if recent_window is not None:
            # Reuse one existing budget slot. Keep a global proposal/scan in
            # the other slot; never exceed the configured call budget.
            windows = [recent_window] + [w for w in windows
                                         if _window_iou(w, recent_window) < .8][:self.max_rois - 1]
        contrast = contrast_windows(frame, motion.valid_mask, side=self.roi_size,
                                    max_windows=1) if self.contrast_search else []
        # Reserve one of the existing slots for a compact appearance candidate.
        # With one slot, periodic scans still progress instead of repeatedly
        # looking at the same static distractor. The model alone classifies it.
        self._contrast_window = None
        if contrast:
            # Scheduled scans must keep the strongest original motion proposal
            # in the other slot. Appearance search complements those proposals
            # between scans, rather than removing established acquisition paths.
            priority = (([scan] if periodic_scan else []) +
                        ([recent_window] if recent_window is not None else []) +
                        (windows[:1] if periodic_scan else []) + contrast + windows)
            windows = []
            for window in priority:
                if not any(_window_iou(window, other) >= .8 for other in windows):
                    windows.append(window)
                if len(windows) == self.max_rois:
                    break
            if contrast[0] in windows:
                self._contrast_window = contrast[0]
        # A proposed tile must not advance the scan if a higher-priority
        # candidate displaced it. Otherwise some grid cells are never visited.
        if scan is not None and scan in windows:
            self._scan_cursor += 1
        return windows, proposal_count

    def _smooth_retry_background(self, frame, center, predicted):
        """Spend the optional scale pass only on small targets on smooth backgrounds."""
        if predicted is not None and max(predicted[2]-predicted[0], predicted[3]-predicted[1]) > 32:
            return False
        x, y, right, bottom = square_window(center, 64, frame.shape)
        gray = cv2.cvtColor(frame[y:bottom, x:right], cv2.COLOR_BGR2GRAY)
        mask = np.ones(gray.shape, dtype=np.uint8)
        cx, cy = round(center[0]-x), round(center[1]-y)
        mask[max(0, cy-16):min(gray.shape[0], cy+16),
             max(0, cx-16):min(gray.shape[1], cx+16)] = 0
        if np.count_nonzero(mask) < 32:
            return False
        _, std = cv2.meanStdDev(gray, mask=mask)
        return float(std[0, 0]) <= 12.

    def _update_adaptive_size(self, result, before):
        """Only regular detector measurements can set the next crop scale."""
        if not self.adaptive_roi:
            return
        if before in ('LOST', 'SEARCHING') and result.source == 'detector':
            self._target_sizes.clear()
            self._adaptive_side = self.roi_size
        if (result.source != 'detector' or result.bbox is None or
                getattr(self.tracker, 'low_confidence_used', False)):
            return
        self._target_sizes.append(max(result.bbox[2]-result.bbox[0], result.bbox[3]-result.bbox[1]))
        if result.state != 'LOCKED' or len(self._target_sizes) < 2:
            return
        network_side = getattr(self.detector, 'image_size', 640)
        # Aim for roughly 32 network pixels along the target's longer side.
        # A median and a bounded 32px change prevent noisy boxes from pumping
        # the scale. Never replace initial/global search with a tiny crop.
        minimum = min(self.roi_size, max(16, int(np.ceil(self.roi_size*.7/16))*16))
        desired = float(np.clip(np.median(self._target_sizes)*network_side/32, minimum, self.roi_size))
        desired = np.clip(desired, self._adaptive_side-32, self._adaptive_side+32)
        self._adaptive_side = min(self.roi_size, max(minimum, int(np.ceil(desired/16))*16))

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
        recent_window = None
        if (self.prediction_enabled and predicted is None and self.recent_search_attempts and
                self._recent_detection_index is not None):
            deadline = (self._recent_detection_index + self.tracker.prediction_max_gap +
                        self.recent_search_attempts * self.detect_every)
            if index <= deadline:
                recent_window = square_window(_center(self._recent_box), self.roi_size, self._shape)
        prediction_roi = None
        adaptive_used = False
        if prediction is not None:
            side = self.roi_size
            if (self.adaptive_roi and before == 'LOCKED' and len(self._target_sizes) >= 2 and
                    self._smooth_retry_background(current, _center(predicted), predicted)):
                side = self._adaptive_side
                adaptive_used = side < self.roi_size
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
        scale_retry_used = False
        self._contrast_window = None
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
                    windows, proposal_count = self._search_windows(motion, recent_window, current)
                    if recent_window is not None and recent_window in windows:
                        reason = 'recent_target_search'
                    proposal_seconds = perf_counter() - tick
                proposals = []
                while calls < min(len(windows), self.max_rois):
                    window = windows[calls]
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
                    if scale_retry_used:
                        # A second-scale rescue must remain near the current
                        # visual search center. A roomy Kalman gate alone can
                        # join two unrelated weak detections in the same crop.
                        found = [d for d in found if np.linalg.norm(
                            np.asarray(_center(d.bbox)) - np.asarray(center)) <= self.tracker.search_radius]
                    detector_seconds += perf_counter() - tick
                    calls += 1
                    shape = getattr(self.detector, 'last_input_shape', None)
                    input_shapes.append(list(shape) if shape is not None else None)
                    proposals.extend(found)
                    regular_found = any(d.confidence >= self.tracker.min_confidence for d in found)
                    if (self.scale_retry and local and calls == 1 and not regular_found and self.max_rois > 1
                            and self._smooth_retry_background(current, center, predicted)):
                        # Only use an existing spare slot after an empty local
                        # detection. Both passes reuse the same registered
                        # motion context and fixed-size network session.
                        retry_side = (self.roi_size if adaptive_used else
                                      max(16, int((window[2] - window[0]) * .7) // 16 * 16))
                        retry = square_window(center, retry_side, self._shape)
                        if retry != window:
                            windows.append(retry)
                            scale_retry_used = True
                # Different overlapping windows can detect the same object.
                proposals = self._deduplicate(proposals)
            else:
                windows = []
        tick = perf_counter()
        tracker_proposals = proposals
        if self.prediction_enabled and attempt and predicted is not None and not calls:
            # Failed camera registration is not a new visual observation.
            # End the visual lock while retaining only the bounded search hint.
            self.tracker.invalidate()
            tracker_proposals = []
        result = self.tracker.update(current, tracker_proposals, index)
        # Only the predictive tracker can authorize a bounded visual grace.
        if (attempt and predicted is not None and result.source != 'detector' and
                not getattr(self.tracker, 'verification_pending', False)):
            result = self.tracker.invalidate()
        weak_used = bool(getattr(self.tracker, 'low_confidence_used', False))
        self._update_adaptive_size(result, before)
        if self.prediction_enabled and result.source == 'detector' and not weak_used:
            if result.state == 'LOCKED':
                self._recent_box, self._recent_detection_index = result.bbox, index
            else:
                self._recent_box, self._recent_detection_index = None, None
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
            contrast_window=list(self._contrast_window) if self._contrast_window is not None else None,
            scale_retry_used=scale_retry_used,
            adaptive_roi_used=bool(attempt and local and adaptive_used and calls),
            low_confidence_used=weak_used,
            detections=[asdict(d) for d in proposals] if calls else None,
            search_roi=self.tracker.last_search_roi,
            stage_timings=dict(motion_seconds=motion_seconds, detector_seconds=detector_seconds,
                               tracking_seconds=tracking_seconds, proposal_seconds=proposal_seconds),
            processing_seconds=perf_counter() - started)
        if self.prediction_enabled:
            details.update(prediction=prediction,
                           verification_pending=self.tracker.verification_pending,
                           verification_misses=self.tracker.verification_misses,
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
