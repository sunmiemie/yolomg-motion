"""GPU-optional geometry, numeric safety and tracker checks on synthetic data."""
import cv2
import numpy as np
import pytest

from yolomg_motion.core import MotionExtractor, three_frame_difference
from yolomg_motion.opencl_motion import backend_information, float_difference
from yolomg_motion.tracking import Detection, SingleTargetTracker
from test_motion import _config, _project, _triplet
from test_tracking import _box, _frame


@pytest.fixture
def opencl_gpu():
    try:
        return backend_information('opencl')
    except RuntimeError as error:
        pytest.skip(str(error))


def test_opencl_request_fails_explicitly_without_a_runtime(monkeypatch):
    monkeypatch.setattr(cv2.ocl, 'haveOpenCL', lambda: False)
    with pytest.raises(RuntimeError, match='no OpenCL platform'):
        MotionExtractor(backend='opencl')
    with pytest.raises(RuntimeError, match='no OpenCL platform'):
        SingleTargetTracker(backend='opencl')
    assert MotionExtractor().backend_info['backend'] == 'cpu'


def test_unknown_backend_is_not_silently_accepted():
    with pytest.raises(ValueError, match='backend'):
        MotionExtractor(backend='cuda')
    with pytest.raises(ValueError, match='backend'):
        SingleTargetTracker(backend='cuda')


def test_opencl_difference_preserves_float_before_both_subtraction_and_sum(opencl_gpu):
    center = np.tile(np.array([[255, 0, 100, 10]], np.uint8), (32, 16))
    previous = np.tile(np.array([[0, 255, 120, 200]], np.uint8), (32, 16))
    following = np.tile(np.array([[0, 255, 160, 30]], np.uint8), (32, 16))
    actual = float_difference(cv2.UMat(center), cv2.UMat(previous), cv2.UMat(following)).get()
    expected = three_frame_difference(center, previous, following)
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected, atol=1e-5)
    assert actual[0, 0] == 255
    assert actual[0, 3] == 105


@pytest.mark.parametrize('perspective', [False, True])
def test_opencl_registration_suppresses_camera_motion_and_retains_target(opencl_gpu, perspective):
    if perspective:
        previous_h = np.array([[1.003,.008,5],[-.006,.998,-3],[.000018,-.000012,1]],float)
        following_h = np.array([[.998,-.007,-5],[.005,1.002,3],[-.000014,.000017,1]],float)
    else:
        previous_h = np.array([[1,0,7],[0,1,-5],[0,0,1]],float)
        following_h = np.array([[1,0,-6],[0,1,4],[0,0,1]],float)
    previous, current, following, positions = _triplet(previous_h, following_h)
    config = _config()  # Deliberately changes estimation scale and aspect ratio.
    cpu = MotionExtractor(config).process_triplet(previous, current, following)
    extractor = MotionExtractor(config, backend='opencl')
    accelerated = extractor.process_triplet(previous, current, following)
    assert accelerated.success, (accelerated.previous_alignment.reason, accelerated.next_alignment.reason)
    assert cpu.success
    assert extractor.backend_info['device_type'] == 'GPU'
    assert extractor.backend_info['device']
    assert 'may fall back to CPU' in extractor.backend_info['note']
    for image in (accelerated.difference, accelerated.motion, accelerated.unaligned_difference):
        assert isinstance(image, np.ndarray) and image.dtype == np.float32
        assert np.isfinite(image).all() and image.min() >= 0 and image.max() <= 255
    probes = np.array([[50,50],[420,50],[420,270],[50,270],[240,120]],np.float32)
    for alignment, expected in ((accelerated.previous_alignment, previous_h),
                                (accelerated.next_alignment, following_h)):
        assert np.max(np.linalg.norm(_project(probes, alignment.homography)-_project(probes, expected),axis=1)) < 2
    common = cpu.valid_mask & accelerated.valid_mask
    assert common.mean() > .7
    # LK and interpolation may differ; this tests useful image-level agreement,
    # not identical keypoints or an accidental bit-for-bit backend requirement.
    assert np.abs(cpu.difference[common]-accelerated.difference[common]).mean() < 2
    background = common.copy()
    for x,y in positions.astype(int):
        background[y-22:y+23,x-22:x+23] = False
    assert accelerated.difference[background].mean() < accelerated.unaligned_difference[background].mean() * .4
    x,y = positions[1].astype(int)
    assert accelerated.motion[y-4:y+5,x-4:x+5].mean() > 30
    assert not accelerated.motion[~accelerated.valid_mask].any()


def test_opencl_textureless_failure_and_mismatched_inputs_are_explicit(opencl_gpu):
    extractor = MotionExtractor(_config(), backend='opencl')
    images = [np.full((320,480,3),v,np.uint8) for v in (20,80,140)]
    result = extractor.process_triplet(*images)
    assert not result.success
    assert result.previous_alignment.reason and result.next_alignment.reason
    assert not result.motion.any() and not result.difference.any()
    assert not result.valid_mask.any()
    with pytest.raises(ValueError, match='identical sizes'):
        extractor.process_triplet(images[0],images[1],images[2][:200])


def test_opencl_template_tracker_matches_cpu_and_obeys_loss_limits(opencl_gpu):
    trackers = [SingleTargetTracker(max_detector_gap=4, backend=backend) for backend in ('cpu','opencl')]
    for index in range(7):
        detection = [Detection(_box(index),.8)] if index < 2 else None
        results = [tracker.update(_frame(_box(index)),detection,index) for tracker in trackers]
        assert results[0].state == results[1].state
        if results[0].bbox is not None:
            np.testing.assert_allclose(results[0].bbox,results[1].bbox,atol=1)
        if results[1].source == 'template':
            assert results[1].appearance_score >= .65
    assert results[1].state == 'LOST' and results[1].bbox is None
    tracker = SingleTargetTracker(backend='opencl')
    for index in (0,1):
        tracker.update(_frame(_box(index)),[Detection(_box(index),.8)],index)
    lost = tracker.update(_frame(),[],2)
    assert lost.state == 'LOST' and lost.bbox is None


def test_opencl_stream_preserves_center_frames_and_checks_runtime(opencl_gpu, monkeypatch):
    extractor = MotionExtractor(_config(), backend='opencl')
    frame = _triplet(np.eye(3),np.eye(3))[1]
    for _ in range(4):
        assert extractor.push(frame) is None
    result = extractor.push(frame)
    assert result.center_index == 2
    np.testing.assert_array_equal(result.current,frame)
    monkeypatch.setattr(cv2.ocl,'useOpenCL',lambda:False)
    with pytest.raises(RuntimeError,match='disabled after backend initialization'):
        extractor.push(frame)
