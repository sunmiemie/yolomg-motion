"""Bounded Kalman search hints; predictions never become observed track boxes."""
from copy import deepcopy
from numbers import Real

import cv2
import numpy as np

from .motion_prediction import MotionPredictor
from .tracking import Detection, SingleTargetTracker, TrackResult, _center, _size


class PredictiveTargetTracker(SingleTargetTracker):
    """A small CPU filter augments the existing appearance tracker.

    Detector observations set a hard prediction lifetime in source frames.
    Template observations have weaker weight and cannot extend that lifetime.
    A confirmed track may bridge one empty verification with a strong,
    detector-consistent template observation. This cannot extend the detector
    lifetime. Reacquisition after loss still requires detector confirmations.
    Optional low-confidence rescue requires a mature existing lock and both
    context and target-patch appearance support. Its observed box is reported,
    but the strong detector reference and lifetime are never refreshed.
    """
    def __init__(self, *args, prediction_max_gap=20, prediction_gate=16.,
                 process_noise=.25, verification_grace=1, low_confidence=None, **kwargs):
        if (isinstance(prediction_max_gap, bool) or
                not isinstance(prediction_max_gap, (int, np.integer)) or prediction_max_gap < 1):
            raise ValueError('prediction_max_gap must be a positive integer')
        self.prediction_max_gap = int(prediction_max_gap)
        if (isinstance(verification_grace, bool) or
                not isinstance(verification_grace, (int, np.integer)) or
                verification_grace not in (0, 1)):
            raise ValueError('verification_grace must be 0 or 1')
        self.verification_grace = int(verification_grace)
        self.predictor = MotionPredictor(process_noise=process_noise, gate_threshold=prediction_gate)
        # A separate four-state anchor receives detectors only. Repeated weak
        # template matches must not make the gate reject a correcting detector.
        self.detector_predictor = MotionPredictor(process_noise=process_noise, gate_threshold=prediction_gate)
        self.prediction_gate = float(prediction_gate)
        super().__init__(*args, **kwargs)
        if low_confidence is not None and (
                isinstance(low_confidence, (bool, np.bool_)) or
                not isinstance(low_confidence, Real) or
                not np.isfinite(low_confidence) or
                not .04 <= low_confidence < self.min_confidence):
            raise ValueError('low_confidence must be at least .04 and below min_confidence')
        self.low_confidence = None if low_confidence is None else float(low_confidence)

    def reset(self):
        super().reset()
        self.predictor.reset()
        self.detector_predictor.reset()
        self._last_motion_detection_index = None
        self._measurement_source = 'template'
        self._resume_motion = False
        self.verification_misses = 0
        self._consecutive_detector_hits = 0
        self._recent_detection_confidences = ()
        self.low_confidence_used = False
        self._weak_rescue_used = False
        self._weak_detection = None
        self._weak_appearance_score = None

    def _valid_detections(self, detections, width, height):
        valid = super()._valid_detections(detections, width, height)
        # Only the already image-checked, current-frame object can use this
        # exception. Never lower the acquisition threshold, even temporarily.
        if self._weak_detection is not None and any(
                item is self._weak_detection for item in detections):
            valid.append(self._weak_detection)
        return valid

    @property
    def verification_pending(self):
        return self.state == 'LOCKED' and self.verification_misses > 0

    def _lose(self, appearance_score=None):
        self.verification_misses = 0
        self._consecutive_detector_hits = 0
        self._recent_detection_confidences = ()
        return super()._lose(appearance_score)

    def prediction_details(self, frame_index, horizon_frames=10):
        if (isinstance(frame_index, bool) or not isinstance(frame_index, (int, np.integer))
                or frame_index < 0):
            raise ValueError('frame_index must be a nonnegative integer')
        if (isinstance(horizon_frames, bool) or not isinstance(horizon_frames, (int, np.integer))
                or horizon_frames < 0):
            raise ValueError('horizon_frames must be a nonnegative integer')
        if (not self.predictor.active or self._last_motion_detection_index is None or
                frame_index - self._last_motion_detection_index > self.prediction_max_gap):
            return None
        prediction = self.predictor.predict(frame_index)
        age = int(frame_index - self._last_motion_detection_index)
        horizon_frames = min(horizon_frames, self.prediction_max_gap - age)
        future = self.predictor.predict(frame_index + horizon_frames) if horizon_frames else prediction
        return dict(prediction, age_frames=age,
                    coasting=self.state in ('LOST', 'SEARCHING'),
                    horizon_frames=int(horizon_frames), future_center=future['center'])

    def predicted_box(self, frame_index):
        if (isinstance(frame_index, bool) or not isinstance(frame_index, (int, np.integer))
                or frame_index < 0):
            raise ValueError('frame_index must be a nonnegative integer')
        prediction = self.prediction_details(frame_index, horizon_frames=0)
        return tuple(prediction['bbox']) if prediction is not None else None

    def _predicted_box(self, frame_index):
        prediction = self.predicted_box(frame_index)
        return prediction if prediction is not None else super()._predicted_box(frame_index)

    def _associate(self, detections, predicted):
        index = self._last_frame_index
        hint = self.detector_predictor.predict(index)
        if hint is None:
            return super()._associate(detections, predicted)
        predicted = hint['bbox']
        size = _size(predicted)
        # Statistical gates grow with uncertainty but never accept an arbitrary
        # distant object merely because a long gap inflated the covariance.
        radius = min(96., max(float(self.search_radius), 3 * max(hint['std_xy'])))
        candidates = []
        for detection in detections:
            ratios = _size(detection.bbox) / size
            if np.any(ratios > self.max_size_ratio) or np.any(ratios < 1 / self.max_size_ratio):
                continue
            distance = float(np.linalg.norm(_center(detection.bbox) - _center(predicted)))
            innovation = self.detector_predictor.gate_distance(detection.bbox, index, source='detector')
            if distance > radius or innovation > self.prediction_gate:
                continue
            cost = innovation / self.prediction_gate + distance / radius
            cost += .25 * float(np.abs(np.log(ratios)).sum()) - .05 * detection.confidence
            candidates.append((cost, detection))
        return min(candidates, key=lambda item: item[0])[1] if candidates else None

    def _start_candidate(self, gray, detection, frame_index):
        self._consecutive_detector_hits = 0
        self._recent_detection_confidences = ()
        if not self._resume_motion:
            self.predictor.reset()
            self.detector_predictor.reset()
        return super()._start_candidate(gray, detection, frame_index)

    def _accept_detection(self, gray, detection, frame_index):
        if detection is self._weak_detection:
            # A real YOLO observation with weak evidence gets template-level
            # KF weight. It cannot refresh appearance, detector geometry or
            # either detector-age timestamp.
            self._set_box(detection.bbox, frame_index)
            self.low_confidence_used = True
            self._weak_rescue_used = True
            self.verification_misses = 1
            self._consecutive_detector_hits = 0
            self._recent_detection_confidences = ()
            return TrackResult(self.state, self._bbox, self._track_id,
                               'detector', detection.confidence, self._weak_appearance_score)
        self._measurement_source = 'detector'
        try:
            result = super()._accept_detection(gray, detection, frame_index)
        finally:
            self._measurement_source = 'template'
        self._last_motion_detection_index = int(frame_index)
        self.verification_misses = 0
        self._weak_rescue_used = False
        self._consecutive_detector_hits += 1
        self._recent_detection_confidences = (
            self._recent_detection_confidences + (detection.confidence,))[-2:]
        return result

    def _weak_target_score(self, frame, detection, reference_size):
        """Check the target itself as well as its larger context template.

        A narrow two-pixel border supplies contrast for a solid dark/light
        object. Background texture elsewhere in the reference cannot supply
        this score when the object at the candidate location is absent.
        """
        half = np.ceil(reference_size * .5).astype(int) + 2
        offset = np.asarray(self._template_offset)
        left, top = np.maximum(0, np.floor(offset - half).astype(int))
        right, bottom = np.minimum(
            self._template.shape[::-1], np.ceil(offset + half).astype(int))
        reference = self._template[top:bottom, left:right]
        if min(reference.shape) < 3 or float(reference.std()) < self.min_template_std:
            return None
        center = _center(detection.bbox)
        scale = _size(detection.bbox) / reference_size
        x1, y1 = np.floor(center + (np.array([left, top]) - offset) * scale).astype(int)
        x2, y2 = np.ceil(center + (np.array([right, bottom]) - offset) * scale).astype(int)
        if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
            return None
        observed = self._gray_patch(frame[y1:y2, x1:x2])
        observed = cv2.resize(observed, (reference.shape[1], reference.shape[0]),
                              interpolation=cv2.INTER_LINEAR)
        if float(observed.std()) < self.min_template_std:
            return None
        score = float(cv2.matchTemplate(observed, reference, cv2.TM_CCOEFF_NORMED)[0, 0])
        return score if np.isfinite(score) else None

    def _associate_weak(self, frame, detections, frame_index):
        if (self.low_confidence is None or self.state != 'LOCKED' or
                self._weak_rescue_used or self._consecutive_detector_hits < 3 or
                self._template is None):
            return None, None
        if (self.max_detector_gap is not None and
                frame_index - self._last_detection_index > self.max_detector_gap):
            return None, None
        anchor = self.detector_predictor.predict(frame_index)
        if anchor is None:
            return None, None
        size = _size(anchor['bbox'])
        radius = min(float(self.search_radius), max(4., 1.5 * max(size)))
        candidates = []
        for detection in detections:
            try:
                bbox = np.array(detection.bbox, dtype=np.float64, copy=True)
                confidence = float(detection.confidence)
            except (ValueError, TypeError, AttributeError):
                continue
            if (bbox.shape != (4,) or not np.isfinite(bbox).all() or
                    not self.low_confidence <= confidence < self.min_confidence):
                continue
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, frame.shape[1])
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, frame.shape[0])
            ratios = _size(bbox) / size
            if (np.any(_size(bbox) < 1) or np.any(ratios < 1 / 1.5) or
                    np.any(ratios > 1.5)):
                continue
            distance = float(np.linalg.norm(_center(bbox) - _center(anchor['bbox'])))
            if (distance > radius or self.detector_predictor.gate_distance(
                    bbox, frame_index, source='detector') > min(4., self.prediction_gate)):
                continue
            candidates.append((distance, Detection(tuple(float(v) for v in bbox), confidence)))
        if not candidates:
            return None, None
        if self.backend == 'opencl':
            from .opencl_motion import ensure_opencl_enabled
            ensure_opencl_enabled()
            gray = cv2.cvtColor(cv2.UMat(frame), cv2.COLOR_BGR2GRAY)
        else:
            gray = frame if self.roi_only else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        matched, score = self._match_template(gray, anchor['bbox'])
        threshold = max(.8, self.appearance_threshold)
        if matched is None or score is None or score < threshold:
            return None, score
        match_radius = min(8., max(2., .5 * min(size)))
        for _, detection in sorted(candidates, key=lambda item: item[0]):
            if np.linalg.norm(_center(detection.bbox) - _center(matched)) > match_radius:
                continue
            target_score = self._weak_target_score(frame, detection, size)
            if target_score is not None and target_score >= threshold:
                return detection, min(score, target_score)
        return None, score

    def _set_box(self, box, frame_index):
        if self._measurement_source == 'detector':
            if not self.detector_predictor.active:
                self.detector_predictor.initialize(box, frame_index)
            else:
                self.detector_predictor.correct(box, frame_index, source='detector')
            # Restore detector-grounded uncertainty and velocity at each valid
            # correction. All arrays are tiny, and this happens only on YOLO.
            self.predictor = deepcopy(self.detector_predictor)
        else:
            self.predictor.correct(box, frame_index, source=self._measurement_source)
        super()._set_box(box, frame_index)

    def _match_template(self, gray, predicted):
        box, score = super()._match_template(gray, predicted)
        if box is not None and self.predictor.gate_distance(
                box, self._last_frame_index, source='template') > self.prediction_gate:
            return None, score
        if box is not None and self.verification_pending:
            # Apply on EVERY grace frame, against the detector-only anchor.
            # Template matches cannot gradually move their own acceptance gate.
            anchor = self.detector_predictor.predict(self._last_frame_index)
            radius = min(float(self.search_radius), max(4., 2 * max(_size(anchor['bbox']))))
            distance = float(np.linalg.norm(_center(box) - _center(anchor['bbox'])))
            if (score < max(.8, self.appearance_threshold) or distance > radius or
                    self.detector_predictor.gate_distance(
                        box, self._last_frame_index, source='template') > self.prediction_gate):
                return None, score
        return box, score

    def update(self, frame, detections, frame_index):
        # Validate before any lifecycle mutation. Base update performs the full
        # image/backend checks and records the source index exactly once.
        if (isinstance(frame_index, bool) or not isinstance(frame_index, (int, np.integer))
                or frame_index < 0):
            raise ValueError('frame_index must be a nonnegative integer')
        if self._last_frame_index is not None and frame_index <= self._last_frame_index:
            raise ValueError('frame_index must strictly increase; call reset for a new sequence')
        if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or
                frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 16):
            raise ValueError('Expected uint8 BGR frame of at least 16x16 pixels')
        if self._frame_shape is not None and frame.shape != self._frame_shape:
            raise ValueError('Frame shape changed; call reset for a new sequence')
        self.low_confidence_used = False
        prediction = self.predicted_box(frame_index)
        if prediction is None and self.predictor.active:
            self._lose()
            self.predictor.reset()
            self.detector_predictor.reset()
            self._last_motion_detection_index = None

        self._resume_motion = False
        if prediction is not None and detections is not None:
            eligible = self._valid_detections(detections, frame.shape[1], frame.shape[0])
            # _associate uses the current source index before base update starts.
            previous_index = self._last_frame_index
            self._last_frame_index = int(frame_index)
            try:
                associated = self._associate(eligible, prediction)
                # A conflicting ordinary-confidence detection is evidence of
                # ambiguity, so weak observations must not override it.
                if associated is None and not eligible:
                    weak, score = self._associate_weak(frame, detections, frame_index)
                    if weak is not None:
                        associated = self._weak_detection = weak
                        self._weak_appearance_score = score
            finally:
                self._last_frame_index = previous_index
            if associated is None:
                if (not eligible and self.state == 'LOCKED' and
                        self._consecutive_detector_hits >= 3 and
                        min(self._recent_detection_confidences) >= max(.3, self.min_confidence) and
                        self.verification_misses < self.verification_grace):
                    self.verification_misses += 1
                    # Low-confidence/new identities must not gain persistence
                    # solely because their background patch matches well.
                    # Earn three fresh detector hits before another grace.
                    self._consecutive_detector_hits = 0
                    # Base update must still find an image observation. No
                    # predicted box or timestamp is promoted to a detection.
                    detections = None
                else:
                    self._lose()
                    detections = []
            else:
                detections = [associated]
                self._resume_motion = self.state in ('LOST', 'SEARCHING')
        try:
            return super().update(frame, detections, frame_index)
        finally:
            self._resume_motion = False
            self._weak_detection = None
            self._weak_appearance_score = None
