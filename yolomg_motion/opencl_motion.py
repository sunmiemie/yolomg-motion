"""OpenCV Transparent API helpers with explicit OpenCL GPU availability checks.

UMat requests OpenCL acceleration; OpenCV may execute individual unsupported
operations on CPU. Device availability is verified, not per-kernel execution.
No CUDA capability or completely GPU-resident pipeline is implied.
"""
import cv2
import numpy as np


FALLBACK_NOTE = ('OpenCV UMat requests OpenCL GPU execution, but individual operations '
                 'may fall back to CPU internally. RANSAC/homography fitting, small '
                 'matrix checks, and state logic run on CPU; public results are downloaded '
                 'as NumPy arrays. CPU/OpenCL numerical results can differ.')


def backend_information(backend):
    if backend not in ('cpu', 'opencl'):
        raise ValueError("backend must be 'cpu' or 'opencl'")
    if backend == 'cpu':
        return dict(backend='cpu', device='CPU', opencl_enabled=False)
    if not cv2.ocl.haveOpenCL():
        raise RuntimeError('OpenCL was requested, but this OpenCV runtime has no OpenCL platform')
    cv2.ocl.setUseOpenCL(True)
    if not cv2.ocl.useOpenCL():
        raise RuntimeError('OpenCL was requested, but OpenCV could not enable it')
    device = cv2.ocl.Device_getDefault()
    if not device.available() or not (device.type() & cv2.ocl.Device_TYPE_GPU):
        raise RuntimeError('OpenCL was requested, but the selected OpenCV device is not an available GPU')
    return dict(backend='opencl', opencl_enabled=True, device=device.name(),
                vendor=device.vendorName(), version=device.version(),
                driver_version=device.driverVersion(), device_type='GPU',
                execution_guarantee='UMat acceleration requested; individual kernel execution is not profiled',
                note=FALLBACK_NOTE)


def ensure_opencl_enabled():
    if not cv2.ocl.useOpenCL():
        raise RuntimeError('OpenCL was disabled after backend initialization; refusing a silent CPU-only run')


def as_numpy(value):
    return value.get() if isinstance(value, cv2.UMat) else value


def float_difference(center, previous, following):
    """Return UMat float32 mean absolute difference, converting BEFORE arithmetic."""
    c = cv2.multiply(center, 1.0, dtype=cv2.CV_32F)
    p = cv2.multiply(previous, 1.0, dtype=cv2.CV_32F)
    n = cv2.multiply(following, 1.0, dtype=cv2.CV_32F)
    return cv2.addWeighted(cv2.absdiff(c, p), .5, cv2.absdiff(c, n), .5, 0,
                           dtype=cv2.CV_32F)


class OpenCLMotionBackend:
    def __init__(self):
        self.info = backend_information('opencl')
        self.info['opencv_version'] = cv2.__version__
        self.info['lk_parameter_note'] = (
            'OpenCL LK is a separate implementation, not CPU parameter-exact. '
            'In verified OpenCV 4.13.0 source, the GPU kernel uses a fixed '
            'per-axis convergence threshold of 0.01 and a determinant check; '
            'it does not consume criteria.epsilon or minEigThreshold. '
            'Use the CPU motion backend when CPU parameter parity is required.')
        self._support_shape = None
        self._support = None

    def prepare(self, frame, blur_size):
        ensure_opencl_enabled()
        smooth = cv2.GaussianBlur(cv2.UMat(frame), (blur_size, blur_size), 0)
        return cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else smooth

    def lk(self, source, center, points, **settings):
        result, status, error = cv2.calcOpticalFlowPyrLK(source, center, cv2.UMat(points), None, **settings)
        return as_numpy(result), as_numpy(status), as_numpy(error)

    def support(self, shape):
        if self._support_shape != shape:
            self._support = cv2.UMat(np.ones(shape, np.float32))
            self._support_shape = shape
        return self._support

    def finish(self, previous, current, following, original, first, second, config):
        from .core import MotionResult
        unaligned = float_difference(current, previous, following)
        valid = cv2.bitwise_and(first.valid_mask, second.valid_mask)
        success = first.success and second.success and cv2.countNonZero(valid) > 0
        if success:
            difference = float_difference(current, first.aligned, second.aligned)
            difference = cv2.bitwise_and(difference, difference, mask=valid)
            motion = difference
            kernel = np.ones((config.morph_kernel, config.morph_kernel), np.uint8)
            if config.open_iterations:
                motion = cv2.morphologyEx(motion, cv2.MORPH_OPEN, kernel, iterations=config.open_iterations)
            if config.close_iterations:
                motion = cv2.morphologyEx(motion, cv2.MORPH_CLOSE, kernel, iterations=config.close_iterations)
            motion = cv2.bitwise_and(motion, motion, mask=valid)
            difference_array, motion_array = difference.get(), motion.get()
            valid_array = valid.get().astype(bool)
        else:
            difference_array = np.zeros(original.shape[:2], np.float32)
            motion_array = difference_array.copy()
            valid_array = np.zeros(original.shape[:2], bool)
        # Retain images on the device through all pixel operations. Downloads
        # occur here for the existing public NumPy result contract.
        for alignment in (first, second):
            alignment.aligned = as_numpy(alignment.aligned)
            alignment.valid_mask = as_numpy(alignment.valid_mask).astype(bool)
        return MotionResult(success, original.copy(), current.get(), difference_array,
                            motion_array, unaligned.get(), valid_array, first, second)
