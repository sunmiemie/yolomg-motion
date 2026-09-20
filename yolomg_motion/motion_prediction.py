"""A constant-velocity Kalman model for source-frame target coordinates.

The state has only four numbers: center x/y and velocity x/y. No image-sized
arrays are created. Predictions are search proposals, not target observations.
The owner must explicitly initialize or reset the model when identity changes.
"""
from numbers import Integral, Real

import numpy as np


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f'{name} must be a finite positive number')
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f'{name} must be a finite positive number') from error
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f'{name} must be a finite positive number')
    return result


def _frame_index(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
        raise ValueError('frame_index must be a nonnegative integer source-frame index')
    return int(value)


def _box_geometry(box):
    try:
        values = np.asarray(box, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError('box must contain four finite xyxy coordinates') from error
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError('box must contain four finite xyxy coordinates')
    with np.errstate(over='ignore', invalid='ignore'):
        size = values[2:] - values[:2]
        center = values[:2] * .5 + values[2:] * .5
    if not np.isfinite(size).all() or np.any(size <= 0):
        raise ValueError('box must have finite positive width and height')
    return center, size


class MotionPredictor:
    """Predict a detected object's center with weak template corrections.

    ``detector_noise`` and ``template_noise`` are isotropic position variances
    in pixels squared, not standard deviations. ``process_noise`` is continuous
    white-acceleration intensity in pixels squared per source-frame cubed. Its
    exact integral is used for arbitrary source-frame gaps, so predicting one
    large time interval has the same process uncertainty as splitting it.

    Gate distances are squared Mahalanobis distances using the innovation
    covariance (predicted position covariance plus observation noise).

    ``initialize`` is explicit: inactive ``predict`` returns None, inactive
    ``gate_distance`` returns infinity, and inactive ``correct`` returns False.
    Predictions and gates never advance internal time. Accepted corrections
    must use a strictly later frame than the previous accepted observation;
    same-frame prediction/gating is allowed. Rejected outliers change nothing.
    Only initialization and detector corrections can change target dimensions.
    """

    def __init__(self, process_noise=.25, detector_noise=4., template_noise=36.,
                 gate_threshold=16.):
        self.process_noise = _positive(process_noise, 'process_noise')
        self.detector_noise = _positive(detector_noise, 'detector_noise')
        self.template_noise = _positive(template_noise, 'template_noise')
        self.gate_threshold = _positive(gate_threshold, 'gate_threshold')
        self.reset()

    @property
    def active(self):
        return self._state is not None

    def reset(self):
        self._state = None
        self._covariance = None
        self._size = None
        self._last_frame = None

    def initialize(self, box, frame_index):
        center, size = _box_geometry(box)
        index = _frame_index(frame_index)
        self._state = np.array([center[0], center[1], 0., 0.], dtype=np.float64)
        self._covariance = np.diag([16., 16., 25., 25.])
        self._size = size.copy()
        self._last_frame = index

    def _observation_noise(self, source):
        if source == 'detector':
            return self.detector_noise
        if source == 'template':
            return self.template_noise
        raise ValueError("source must be 'detector' or 'template'")

    def _prediction(self, index):
        if index < self._last_frame:
            raise ValueError('Cannot predict before the last accepted observation')
        try:
            dt = np.float64(index - self._last_frame)
        except (OverflowError, ValueError) as error:
            raise ValueError('Source-frame interval is too large for a finite prediction') from error
        transition = np.eye(4)
        transition[0, 2] = transition[1, 3] = dt
        with np.errstate(over='ignore', invalid='ignore'):
            q_position = self.process_noise * dt ** 3 / 3.
            q_cross = self.process_noise * dt ** 2 / 2.
            q_velocity = self.process_noise * dt
            process = np.array([
                [q_position, 0., q_cross, 0.],
                [0., q_position, 0., q_cross],
                [q_cross, 0., q_velocity, 0.],
                [0., q_cross, 0., q_velocity],
            ])
            state = transition @ self._state
            covariance = transition @ self._covariance @ transition.T + process
            covariance = (covariance + covariance.T) * .5
        if not np.isfinite(state).all() or not np.isfinite(covariance).all():
            raise ValueError('Source-frame interval or noise produces non-finite prediction')
        return state, covariance

    def predict(self, frame_index):
        """Return a detached search proposal without advancing the filter."""
        index = _frame_index(frame_index)
        if not self.active:
            return None
        state, covariance = self._prediction(index)
        with np.errstate(over='ignore', invalid='ignore'):
            box = np.concatenate((state[:2] - self._size * .5,
                                  state[:2] + self._size * .5))
        if not np.isfinite(box).all():
            raise ValueError('Predicted box is non-finite')
        position_covariance = covariance[:2, :2]
        return dict(bbox=box.tolist(), center=state[:2].tolist(),
                    velocity=state[2:].tolist(), covariance=position_covariance.tolist(),
                    std_xy=np.sqrt(np.maximum(0., np.diag(position_covariance))).tolist())

    @staticmethod
    def _innovation_distance(center, state, covariance, noise):
        with np.errstate(over='ignore', invalid='ignore'):
            residual = center - state[:2]
            innovation = covariance[:2, :2] + np.eye(2) * noise
        if not np.isfinite(residual).all() or not np.isfinite(innovation).all():
            return float('inf'), residual, innovation
        try:
            with np.errstate(over='ignore', invalid='ignore'):
                distance = float(residual @ np.linalg.solve(innovation, residual))
        except np.linalg.LinAlgError:
            return float('inf'), residual, innovation
        if not np.isfinite(distance):
            distance = float('inf')
        return max(0., distance), residual, innovation

    def gate_distance(self, box, frame_index, source='detector'):
        """Squared innovation distance; infinity when inactive or unresolvable."""
        center, _ = _box_geometry(box)
        index = _frame_index(frame_index)
        noise = self._observation_noise(source)
        if not self.active:
            return float('inf')
        state, covariance = self._prediction(index)
        return self._innovation_distance(center, state, covariance, noise)[0]

    def correct(self, box, frame_index, source='detector'):
        """Absorb one gated later-frame observation; return whether accepted."""
        center, size = _box_geometry(box)
        index = _frame_index(frame_index)
        noise = self._observation_noise(source)
        if not self.active:
            return False
        if index <= self._last_frame:
            raise ValueError('Corrections must follow the last accepted source frame')
        state, covariance = self._prediction(index)
        distance, residual, innovation = self._innovation_distance(center, state, covariance, noise)
        if distance > self.gate_threshold:
            return False
        # K = P H^T S^-1, with H selecting the two position coordinates.
        gain = np.linalg.solve(innovation, covariance[:, :2].T).T
        corrected_state = state + gain @ residual
        identity_minus_kh = np.eye(4)
        identity_minus_kh[:, :2] -= gain
        # Joseph form maintains symmetry/PSD under repeated weak corrections.
        corrected_covariance = (identity_minus_kh @ covariance @ identity_minus_kh.T
                                + noise * (gain @ gain.T))
        corrected_covariance = (corrected_covariance + corrected_covariance.T) * .5
        if not np.isfinite(corrected_state).all() or not np.isfinite(corrected_covariance).all():
            raise ValueError('Correction produced non-finite state or covariance')
        self._state = corrected_state
        self._covariance = corrected_covariance
        self._last_frame = index
        if source == 'detector':
            self._size = size.copy()
        return True
