"""Bounded Kalman search hints; predictions never become observed track boxes."""
from copy import deepcopy

import numpy as np

from .motion_prediction import MotionPredictor
from .tracking import SingleTargetTracker, _center, _size


class PredictiveTargetTracker(SingleTargetTracker):
    """A small CPU filter augments the existing appearance tracker.

    Detector observations set a hard prediction lifetime in source frames.
    Template observations have weaker weight and cannot extend that lifetime.
    A failed verification loses the visual lock immediately, but preserves a
    bounded search hint. Reacquisition still requires detector confirmations.
    """
    def __init__(self, *args, prediction_max_gap=20, prediction_gate=16.,
                 process_noise=.25, **kwargs):
        if (isinstance(prediction_max_gap, bool) or
                not isinstance(prediction_max_gap, (int, np.integer)) or prediction_max_gap < 1):
            raise ValueError('prediction_max_gap must be a positive integer')
        self.prediction_max_gap = int(prediction_max_gap)
        self.predictor = MotionPredictor(process_noise=process_noise, gate_threshold=prediction_gate)
        # A separate four-state anchor receives detectors only. Repeated weak
        # template matches must not make the gate reject a correcting detector.
        self.detector_predictor = MotionPredictor(process_noise=process_noise, gate_threshold=prediction_gate)
        self.prediction_gate = float(prediction_gate)
        super().__init__(*args, **kwargs)

    def reset(self):
        super().reset()
        self.predictor.reset()
        self.detector_predictor.reset()
        self._last_motion_detection_index = None
        self._measurement_source = 'template'
        self._resume_motion = False

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
        if not self._resume_motion:
            self.predictor.reset()
            self.detector_predictor.reset()
        return super()._start_candidate(gray, detection, frame_index)

    def _accept_detection(self, gray, detection, frame_index):
        self._measurement_source = 'detector'
        try:
            result = super()._accept_detection(gray, detection, frame_index)
        finally:
            self._measurement_source = 'template'
        self._last_motion_detection_index = int(frame_index)
        return result

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
            finally:
                self._last_frame_index = previous_index
            if associated is None:
                # No template fallback on explicit failed detector verification:
                # it must not feed a background match into the motion state.
                self._lose()
                detections = []
            else:
                detections = [associated]
                self._resume_motion = self.state in ('LOST', 'SEARCHING')
        try:
            return super().update(frame, detections, frame_index)
        finally:
            self._resume_motion = False
