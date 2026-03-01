"""Independent implementation of YOLOMG section III-A (arXiv:2503.07115v1).

All homographies map SOURCE coordinates to CENTER coordinates. Difference maps
are float32 in [0, 255]. A failed registration produces an explicitly invalid
result, never an unregistered difference masquerading as a valid motion map.
"""
from collections import deque
from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class MotionConfig:
    frame_step: int = 2
    blur_size: int = 11
    grid_step: int = 64
    grid_step_y: int = 48
    alignment_width: int = 1920
    alignment_height: int = 1080
    lk_window: int = 15
    lk_levels: int = 3
    lk_iterations: int = 30
    lk_epsilon: float = 0.003
    ransac_threshold: float = 3.0
    min_inliers: int = 15
    min_inlier_ratio: float = 0.25
    fb_threshold: float = 1.5
    min_overlap: float = 0.4
    morph_kernel: int = 3
    open_iterations: int = 1
    close_iterations: int = 1

    def __post_init__(self):
        # Dataclass annotations do not validate JSON or public Python inputs.
        # Check types before ranges so malformed settings fail here, rather
        # than later in deque construction or an opaque OpenCV overload error.
        for key in ('frame_step', 'blur_size', 'grid_step', 'grid_step_y',
                    'alignment_width', 'alignment_height', 'lk_window',
                    'lk_levels', 'lk_iterations', 'min_inliers', 'morph_kernel',
                    'open_iterations', 'close_iterations'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f'{key} must be an integer')
        for key in ('lk_epsilon', 'ransac_threshold', 'min_inlier_ratio',
                    'fb_threshold', 'min_overlap'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
                raise ValueError(f'{key} must be a finite number')
        for key in ('frame_step', 'grid_step', 'grid_step_y', 'alignment_width',
                    'alignment_height', 'lk_iterations'):
            if getattr(self, key) < 1:
                raise ValueError(f'{key} must be positive')
        for key in ('blur_size', 'lk_window', 'morph_kernel'):
            value = getattr(self, key)
            if value < 1 or value % 2 == 0:
                raise ValueError(f'{key} must be a positive odd integer')
        if self.min_inliers < 4:
            raise ValueError('A homography requires at least four inliers')
        if not 0 < self.min_inlier_ratio <= 1 or not 0 < self.min_overlap <= 1:
            raise ValueError('Ratios must be in (0, 1]')
        if min(self.lk_levels, self.open_iterations, self.close_iterations, self.fb_threshold) < 0:
            raise ValueError('Levels, iteration counts, and FB threshold must be nonnegative')
        if self.ransac_threshold <= 0 or self.lk_epsilon <= 0:
            raise ValueError('RANSAC threshold and LK epsilon must be positive')


@dataclass
class AlignmentResult:
    success: bool
    homography: Optional[np.ndarray]
    aligned: np.ndarray
    valid_mask: np.ndarray
    tracked_points: int = 0
    inlier_count: int = 0
    inlier_ratio: float = 0.0
    reason: str = ''


@dataclass
class MotionResult:
    success: bool
    current: np.ndarray
    center_gray: np.ndarray
    difference: np.ndarray
    motion: np.ndarray
    unaligned_difference: np.ndarray
    valid_mask: np.ndarray
    previous_alignment: AlignmentResult
    next_alignment: AlignmentResult
    center_index: Optional[int] = None


def _validate_frame(frame):
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
        raise ValueError('Expected an uint8 NumPy frame')
    if frame.ndim not in (2, 3) or (frame.ndim == 3 and frame.shape[2] != 3):
        raise ValueError('Expected HxW grayscale or HxWx3 BGR input')
    if min(frame.shape[:2]) < 16:
        raise ValueError('Frames must be at least 16x16 pixels')


def three_frame_difference(center, aligned_previous, aligned_next):
    """Paper Eq. (1), with floating-point arithmetic BEFORE subtraction/addition."""
    if center.shape != aligned_previous.shape or center.shape != aligned_next.shape:
        raise ValueError('The three grayscale images must have identical shapes')
    if center.ndim != 2:
        raise ValueError('three_frame_difference expects grayscale images')
    c = center.astype(np.float32)
    return (np.abs(c - aligned_previous.astype(np.float32)) +
            np.abs(c - aligned_next.astype(np.float32))) * np.float32(0.5)


class MotionExtractor:
    """Use process_triplet() offline or push() for a delayed live stream.

    push() returns the CENTER frame after k future frames have arrived. It stores
    at most 2*k+1 frames and emits no fabricated first/last boundary frames.
    """

    def __init__(self, config=None, backend='cpu'):
        from .opencl_motion import OpenCLMotionBackend, backend_information
        self.config = config or MotionConfig()
        self.backend = backend
        self._opencl = OpenCLMotionBackend() if backend == 'opencl' else None
        self.backend_info = self._opencl.info if self._opencl else backend_information(backend)
        self.reset()

    def reset(self):
        self._frames = deque(maxlen=2*self.config.frame_step+1)
        self._received = 0
        self._shape = None

    def _prepare(self, frame):
        _validate_frame(frame)
        c = self.config
        if self._opencl:
            return self._opencl.prepare(frame, c.blur_size)
        smooth = cv2.GaussianBlur(frame, (c.blur_size, c.blur_size), 0)
        return cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY) if smooth.ndim == 3 else smooth

    def _align(self, source, center, shape=None):
        c = self.config
        h, w = shape if shape is not None else center.shape

        def failed(reason, tracked=0, inliers=0):
            blank = np.zeros((h, w), np.uint8)
            aligned = cv2.UMat(blank) if self._opencl else blank
            mask = cv2.UMat(blank) if self._opencl else blank.astype(bool)
            return AlignmentResult(False, None, aligned, mask, tracked,
                                   inliers, inliers/max(tracked, 1), reason)

        # Preserve official estimation resolution, but explicitly transform H
        # back to native coordinates before warping the original image.
        aw, ah = c.alignment_width, c.alignment_height
        if (w, h) == (aw, ah):
            small_source, small_center = source, center
        else:
            small_source = cv2.resize(source, (aw, ah), interpolation=cv2.INTER_CUBIC)
            small_center = cv2.resize(center, (aw, ah), interpolation=cv2.INTER_CUBIC)
        nx = max(0, int(aw/c.grid_step-1))
        ny = max(0, int(ah/c.grid_step_y-1))
        if nx*ny < c.min_inliers:
            return failed('insufficient_grid_points')
        xx, yy = np.meshgrid((np.arange(nx)+0.5)*c.grid_step,
                             (np.arange(ny)+0.5)*c.grid_step_y)
        p0 = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float32).reshape(-1, 1, 2)
        lk = dict(winSize=(c.lk_window, c.lk_window), maxLevel=c.lk_levels,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                            c.lk_iterations, c.lk_epsilon))
        flow = self._opencl.lk if self._opencl else lambda a, b, p, **kwargs: cv2.calcOpticalFlowPyrLK(a, b, p, None, **kwargs)
        p1, status, _ = flow(small_source, small_center, p0, **lk)
        if p1 is None or status is None:
            return failed('lk_no_tracks')
        dst = p1.reshape(-1, 2)
        good = (status.ravel() == 1) & np.isfinite(dst).all(axis=1)
        good &= (dst[:, 0] >= 0) & (dst[:, 0] < aw-1) & (dst[:, 1] >= 0) & (dst[:, 1] < ah-1)
        src = p0.reshape(-1, 2)[good]
        dst = dst[good]
        if c.fb_threshold > 0 and len(src):
            back, bs, _ = flow(small_center, small_source, dst.reshape(-1, 1, 2), **lk)
            if back is None or bs is None:
                return failed('lk_backward_failed')
            error = np.linalg.norm(back.reshape(-1, 2)-src, axis=1)
            keep = (bs.ravel() == 1) & np.isfinite(error) & (error <= c.fb_threshold)
            src, dst = src[keep], dst[keep]
        tracked = len(src)
        if tracked < c.min_inliers:
            return failed('insufficient_tracks', tracked)
        H_est, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, c.ransac_threshold)
        if H_est is None or inlier_mask is None or not np.isfinite(H_est).all():
            return failed('homography_estimation_failed', tracked)
        count = int(inlier_mask.sum())
        if count < c.min_inliers or count/tracked < c.min_inlier_ratio:
            return failed('insufficient_ransac_inliers', tracked, count)
        S = np.diag([aw/w, ah/h, 1.0])
        H = np.linalg.inv(S) @ H_est @ S
        if abs(H[2, 2]) < 1e-10 or abs(np.linalg.det(H)) < 1e-10:
            return failed('degenerate_homography', tracked, count)
        H /= H[2, 2]
        corners = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], np.float32).reshape(-1, 1, 2)
        denom = (H[2, :2] @ corners.reshape(-1, 2).T) + H[2, 2]
        if np.min(denom) <= 0:
            return failed('homography_crosses_horizon', tracked, count)
        projected = cv2.perspectiveTransform(corners, H).astype(np.float32)
        area = abs(cv2.contourArea(projected))
        if not np.isfinite(projected).all() or not cv2.isContourConvex(projected) or not 0.1*w*h < area < 10*w*h:
            return failed('implausible_homography', tracked, count)
        aligned = cv2.warpPerspective(source, H, (w, h), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        support_image = self._opencl.support((h, w)) if self._opencl else np.ones((h, w), np.float32)
        support = cv2.warpPerspective(support_image, H, (w, h),
                                      flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        valid = cv2.compare(support, 0.999, cv2.CMP_GE) if self._opencl else (support >= 0.999).astype(np.uint8)
        # Remove interpolation / blur boundary support; otherwise black warp
        # borders would look like strong moving foreground.
        radius = c.blur_size//2 + 1
        valid = cv2.erode(valid, np.ones((2*radius+1, 2*radius+1), np.uint8),
                          borderType=cv2.BORDER_CONSTANT, borderValue=0)
        if not self._opencl:
            valid = valid.astype(bool)
        overlap = cv2.mean(valid)[0]/255.0 if self._opencl else valid.mean()
        if overlap < c.min_overlap:
            return failed('insufficient_overlap', tracked, count)
        return AlignmentResult(True, H, aligned, valid, tracked, count, count/tracked, '')

    def _process(self, previous, current, following, center_original):
        if self._opencl:
            shape = center_original.shape[:2]
            a = self._align(previous, current, shape)
            b = self._align(following, current, shape)
            return self._opencl.finish(previous, current, following, center_original, a, b, self.config)
        if previous.shape != current.shape or following.shape != current.shape:
            raise ValueError('All frames must have identical sizes')
        a = self._align(previous, current)
        b = self._align(following, current)
        unaligned = three_frame_difference(current, previous, following)
        valid = a.valid_mask & b.valid_mask
        success = a.success and b.success and bool(valid.any())
        difference = np.zeros(current.shape, np.float32)
        motion = difference.copy()
        if success:
            difference = three_frame_difference(current, a.aligned, b.aligned)
            difference[~valid] = 0
            motion = difference.copy()
            c = self.config
            kernel = np.ones((c.morph_kernel, c.morph_kernel), np.uint8)
            if c.open_iterations:
                motion = cv2.morphologyEx(motion, cv2.MORPH_OPEN, kernel, iterations=c.open_iterations)
            if c.close_iterations:
                motion = cv2.morphologyEx(motion, cv2.MORPH_CLOSE, kernel, iterations=c.close_iterations)
            motion[~valid] = 0
        else:
            valid[:] = False
        return MotionResult(success, center_original.copy(), current, difference, motion,
                            unaligned, valid, a, b)

    def process_triplet(self, previous, current, following):
        for frame in (previous, current, following):
            _validate_frame(frame)
        if previous.shape != current.shape or following.shape != current.shape:
            raise ValueError('All frames must have identical sizes')
        frames = [self._prepare(f) for f in (previous, current, following)]
        return self._process(*frames, center_original=current)

    def push(self, frame):
        _validate_frame(frame)
        if self._shape is not None and frame.shape != self._shape:
            raise ValueError('Frame shape changed; reset() before starting a new sequence')
        self._shape = frame.shape
        self._frames.append((frame.copy(), self._prepare(frame)))
        self._received += 1
        k = self.config.frame_step
        if len(self._frames) < 2*k+1:
            return None
        result = self._process(self._frames[0][1], self._frames[k][1],
                               self._frames[-1][1], self._frames[k][0])
        result.center_index = self._received-k-1
        return result
