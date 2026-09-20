"""Bounded observed trails and explicitly separate motion-prediction overlays."""
from collections import deque

import cv2
import numpy as np


class TrajectoryRenderer:
    """Render without mutating input frames, tracking results or prediction data.

    Only detector/template observations from LOCKED results enter the trail.
    Coasting predictions can be drawn while LOST but never extend that trail.
    """

    OBSERVED = (60, 235, 100)
    TEMPLATE = (0, 210, 255)
    PREDICTED = (230, 120, 255)
    WINDOW = (255, 205, 50)

    def __init__(self, max_points=60):
        if isinstance(max_points, bool) or not isinstance(max_points, int) or not 1 <= max_points <= 60:
            raise ValueError('max_points must be an integer between 1 and 60')
        self.max_points = max_points
        self._points = deque(maxlen=max_points)
        self._identity = None
        self._last_index = None
        self._shape = None

    @property
    def points(self):
        """Immutable snapshot of (source frame, center, source) trail entries."""
        return tuple(self._points)

    def reset(self):
        self._points.clear()
        self._identity = None
        self._last_index = None
        self._shape = None

    @staticmethod
    def _point(value, shape):
        if value is None:
            return None
        values = np.asarray(value, dtype=float)
        if values.shape != (2,) or not np.isfinite(values).all():
            return None
        h, w = shape[:2]
        return tuple(int(v) for v in np.rint(np.clip(values, (0, 0), (w-1, h-1))))

    @classmethod
    def _rectangle(cls, canvas, box, color, dashed=False):
        if box is None:
            return
        values = np.asarray(box, dtype=float)
        if values.shape != (4,) or not np.isfinite(values).all():
            return
        x1, y1 = cls._point(values[:2], canvas.shape)
        x2, y2 = cls._point(values[2:], canvas.shape)
        if x2 <= x1 or y2 <= y1:
            return
        if dashed:
            corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            for index in range(4):
                cls._dashed_line(canvas, corners[index], corners[(index+1) % 4], color)
        else:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

    @staticmethod
    def _dashed_line(canvas, start, end, color, arrow=False):
        start, end = np.asarray(start, float), np.asarray(end, float)
        distance = float(np.linalg.norm(end-start))
        if distance < 1:
            return
        direction = (end-start)/distance
        for step in np.arange(0, distance, 12.):
            p1 = tuple(np.rint(start+direction*step).astype(int))
            p2 = tuple(np.rint(start+direction*min(step+7., distance)).astype(int))
            cv2.line(canvas, p1, p2, color, 1, cv2.LINE_AA)
        if arrow:
            tip = tuple(np.rint(end).astype(int))
            perpendicular = np.array((-direction[1], direction[0]))
            for side in (-1, 1):
                tail = tuple(np.rint(end-direction*8+perpendicular*side*4).astype(int))
                cv2.line(canvas, tail, tip, color, 1, cv2.LINE_AA)

    def render(self, frame, result, frame_index, details=None, fps=None):
        if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or
                frame.ndim != 3 or frame.shape[2] != 3):
            raise ValueError('Expected uint8 BGR frame')
        if isinstance(frame_index, bool) or not isinstance(frame_index, (int, np.integer)) or frame_index < 0:
            raise ValueError('frame_index must be a nonnegative integer')
        canvas = frame.copy()
        details = details or {}
        if (self._shape is not None and frame.shape != self._shape or
                self._last_index is not None and frame_index != self._last_index + 1):
            self.reset()
        self._shape, self._last_index = frame.shape, int(frame_index)
        observed = result.state == 'LOCKED' and result.bbox is not None and result.source in ('detector', 'template')
        if observed:
            if (result.track_id != self._identity or
                    self._points and frame_index != self._points[-1][0] + 1):
                self._points.clear()
            self._identity = result.track_id
            x1, y1, x2, y2 = result.bbox
            center = self._point(((x1+x2)/2., (y1+y2)/2.), frame.shape)
            if center is not None:
                self._points.append((int(frame_index), center, result.source))
        elif not details.get('prediction'):
            self._points.clear()
            self._identity = None
        for window in details.get('detection_windows', []):
            self._rectangle(canvas, window, self.WINDOW)
        self._rectangle(canvas, details.get('prediction_roi'), self.PREDICTED, dashed=True)
        points = list(self._points)
        for previous, current in zip(points, points[1:]):
            color = self.OBSERVED if current[2] == 'detector' else self.TEMPLATE
            cv2.line(canvas, previous[1], current[1], color, 2, cv2.LINE_AA)
        for _, point, source in points:
            cv2.circle(canvas, point, 2 if source == 'detector' else 1,
                       self.OBSERVED if source == 'detector' else self.TEMPLATE, -1, cv2.LINE_AA)
        prediction = details.get('prediction') or {}
        center = self._point(prediction.get('center'), frame.shape)
        future = self._point(prediction.get('future_center'), frame.shape)
        if center is not None:
            if result.state == 'LOST':
                self._rectangle(canvas, prediction.get('bbox'), self.PREDICTED, dashed=True)
                if points:
                    self._dashed_line(canvas, points[-1][1], center, self.PREDICTED)
            if future is not None:
                self._dashed_line(canvas, center, future, self.PREDICTED, arrow=True)
        speed = ''
        velocity = np.asarray(prediction.get('velocity', []), dtype=float)
        if velocity.shape == (2,) and np.isfinite(velocity).all():
            speed = f'v=({velocity[0]:.1f}, {velocity[1]:.1f}) px/frame'
        status = 'PREDICTED ONLY - target LOST' if prediction and result.state == 'LOST' else 'MOTION PREDICTION'
        legend = [
            (status + ('  |  ' + speed if speed else ''), self.PREDICTED),
            ('Solid trail: observed positions (green YOLO / yellow template)', (230, 230, 230)),
            ('Dashed: prediction only   Cyan box: actual detection ROI', (230, 230, 230)),
        ]
        h, w = canvas.shape[:2]
        top = max(0, h-68)
        cv2.rectangle(canvas, (0, top), (min(w-1, 650), h-1), (22, 22, 22), -1)
        scale = min(.43, max(.22, w/1500.))
        for number, (text, color) in enumerate(legend):
            cv2.putText(canvas, text, (8, top+17+20*number), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, color, 1, cv2.LINE_AA)
        return canvas
