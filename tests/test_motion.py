"""Synthetic checks of the motion pipeline's geometry and temporal semantics.

The scene is generated independently of the implementation.  Its background
camera transforms are known, while a bright target moves in scene coordinates.
No external videos, weights, or network access are needed.
"""

import cv2
import numpy as np
import pytest

from yolomg_motion.core import MotionConfig, MotionExtractor, three_frame_difference


HEIGHT, WIDTH = 320, 480


def _textured_scene():
    rng = np.random.default_rng(20260911)
    noise = rng.normal(100.0, 38.0, (HEIGHT, WIDTH)).clip(20, 180).astype(np.uint8)
    scene = cv2.GaussianBlur(noise, (3, 3), 0)
    # Structures at different scales give the LK pyramid useful local texture.
    for _ in range(180):
        x = int(rng.integers(12, WIDTH - 12))
        y = int(rng.integers(12, HEIGHT - 12))
        radius = int(rng.integers(2, 8))
        color = int(rng.integers(30, 175))
        cv2.circle(scene, (x, y), radius, color, -1, lineType=cv2.LINE_AA)
    return cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)


def _project(points, homography):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(points, homography).reshape(-1, 2)


def _paint_target(image, position):
    x, y = np.rint(position).astype(int)
    cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (255, 255, 255), -1)


def _triplet(previous_to_current, next_to_current):
    background = _textured_scene()
    previous = cv2.warpPerspective(
        background, np.linalg.inv(previous_to_current), (WIDTH, HEIGHT)
    )
    following = cv2.warpPerspective(
        background, np.linalg.inv(next_to_current), (WIDTH, HEIGHT)
    )
    current = background.copy()
    # These coordinates describe the independently moving target after camera
    # compensation, not simply the apparent displacement due to camera motion.
    target_positions = np.array([[220, 180], [250, 164], [280, 150]], np.float32)
    _paint_target(previous, _project([target_positions[0]], np.linalg.inv(previous_to_current))[0])
    _paint_target(current, target_positions[1])
    _paint_target(following, _project([target_positions[2]], np.linalg.inv(next_to_current))[0])
    return previous, current, following, target_positions


def _config():
    return MotionConfig(
        frame_step=2,
        blur_size=11,
        grid_step=20,
        grid_step_y=20,
        lk_window=21,
        lk_levels=3,
        ransac_threshold=3.0,
        min_inliers=15,
        min_inlier_ratio=0.25,
        fb_threshold=1.5,
        # Deliberately use a different aspect ratio from the 480 x 320 input.
        # The resulting homographies must still use original-image coordinates.
        alignment_width=640,
        alignment_height=480,
        morph_kernel=3,
        open_iterations=1,
        close_iterations=1,
    )


def test_three_frame_difference_uses_float_before_subtraction_and_averaging():
    center = np.array([[255, 0, 100], [10, 250, 80]], dtype=np.uint8)
    previous = np.array([[0, 255, 120], [200, 0, 80]], dtype=np.uint8)
    following = np.array([[0, 255, 160], [30, 240, 20]], dtype=np.uint8)

    difference = three_frame_difference(center, previous, following)

    assert difference.dtype == np.float32
    # Explicit values catch both uint8 subtraction wraparound and addition
    # overflow (255 + 255 must average to 255, not 127).
    expected = np.array([[255, 255, 40], [105, 130, 30]], dtype=np.float32)
    np.testing.assert_array_equal(difference, expected)


@pytest.mark.parametrize(
    "previous_to_current,next_to_current",
    [
        (
            np.array([[1, 0, 7], [0, 1, -5], [0, 0, 1]], dtype=np.float64),
            np.array([[1, 0, -6], [0, 1, 4], [0, 0, 1]], dtype=np.float64),
        ),
        (
            np.array(
                [[1.003, 0.008, 5], [-0.006, 0.998, -3], [0.000018, -0.000012, 1]],
                dtype=np.float64,
            ),
            np.array(
                [[0.998, -0.007, -5], [0.005, 1.002, 3], [-0.000014, 0.000017, 1]],
                dtype=np.float64,
            ),
        ),
    ],
    ids=["camera_translation", "camera_perspective"],
)
def test_camera_compensation_preserves_independently_moving_small_target(
    previous_to_current, next_to_current
):
    cfg = _config()
    previous, current, following, target_positions = _triplet(
        previous_to_current, next_to_current
    )
    result = MotionExtractor(cfg).process_triplet(previous, current, following)

    assert result.success, (
        result.previous_alignment.reason,
        result.next_alignment.reason,
    )
    assert result.difference.shape == (HEIGHT, WIDTH)
    assert result.motion.shape == (HEIGHT, WIDTH)
    assert result.valid_mask.dtype == np.bool_
    assert result.difference.dtype == result.motion.dtype == np.float32

    probes = np.array([[50, 50], [420, 50], [420, 270], [50, 270], [240, 120]], np.float32)
    for alignment, expected_h in [
        (result.previous_alignment, previous_to_current),
        (result.next_alignment, next_to_current),
    ]:
        assert alignment.success
        assert alignment.inlier_count >= cfg.min_inliers
        assert alignment.inlier_ratio >= cfg.min_inlier_ratio
        # The direction is previous/next -> current. An inverse matrix or a
        # matrix in the resized estimation frame fails this coordinate check.
        projected = _project(probes, alignment.homography)
        expected = _project(probes, expected_h)
        error = np.linalg.norm(projected - expected, axis=1)
        assert error.max() < 1.5, f"Incorrect camera alignment: errors={error}"

    background_mask = result.valid_mask.copy()
    background_mask[:36, :] = False
    background_mask[-36:, :] = False
    background_mask[:, :36] = False
    background_mask[:, -36:] = False
    for x, y in target_positions.astype(int):
        background_mask[y - 22:y + 23, x - 22:x + 23] = False
    assert background_mask.sum() > HEIGHT * WIDTH * 0.45

    # Compare against the same blur with no camera compensation. The target
    # areas and warp boundaries are excluded so they cannot bias this metric.
    grays = [
        cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (cfg.blur_size, cfg.blur_size), 0)
        for frame in (previous, current, following)
    ]
    before = (
        cv2.absdiff(grays[1], grays[0]).astype(np.float32)
        + cv2.absdiff(grays[1], grays[2]).astype(np.float32)
    ) / 2.0
    background_before = float(before[background_mask].mean())
    background_after = float(result.difference[background_mask].mean())
    assert background_before > 2.0, "Synthetic camera motion must be visibly nontrivial"
    assert background_after < background_before * 0.35, (
        f"Insufficient background suppression: {background_before:.3f} -> {background_after:.3f}"
    )

    x, y = target_positions[1].astype(int)
    target_patch = result.motion[y - 4:y + 5, x - 4:x + 5]
    target_signal = float(target_patch.mean())
    assert result.valid_mask[y - 4:y + 5, x - 4:x + 5].all()
    assert target_signal > 35.0, f"Small target was erased: signal={target_signal:.3f}"
    assert target_signal > float(result.motion[background_mask].mean()) * 8.0


def test_textureless_alignment_failure_does_not_fabricate_motion():
    # Even a large global intensity change must not be passed off as a valid
    # motion result when no background correspondences can be estimated.
    frames = [np.full((HEIGHT, WIDTH, 3), value, dtype=np.uint8) for value in (20, 80, 140)]
    result = MotionExtractor(_config()).process_triplet(*frames)

    assert not result.success
    assert not result.previous_alignment.success
    assert not result.next_alignment.success
    assert result.previous_alignment.reason
    assert result.next_alignment.reason
    assert not result.valid_mask.any()
    assert not np.count_nonzero(result.motion)
    assert np.isfinite(result.difference).all()
    assert np.isfinite(result.motion).all()


def test_stream_waits_for_future_frames_and_returns_original_center_frame():
    cfg = _config()
    extractor = MotionExtractor(cfg)
    background = _textured_scene()
    frames = []
    for index in range(6):
        frame = background.copy()
        frame[0, 0] = (index, index + 40, index + 100)
        frames.append(frame)

    # For k=2, t=2 cannot be computed until t+2=4 has arrived.
    for frame in frames[:4]:
        assert extractor.push(frame) is None
    first = extractor.push(frames[4])
    assert first is not None
    assert first.center_index == 2
    np.testing.assert_array_equal(first.current, frames[2])

    second = extractor.push(frames[5])
    assert second is not None
    assert second.center_index == 3
    np.testing.assert_array_equal(second.current, frames[3])

    extractor.reset()
    for frame in frames[:4]:
        assert extractor.push(frame) is None
    restarted = extractor.push(frames[4])
    assert restarted is not None
    assert restarted.center_index == 2
    np.testing.assert_array_equal(restarted.current, frames[2])
