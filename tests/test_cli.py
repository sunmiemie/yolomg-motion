"""Exercise the real video/CLI adapter and its saved artifacts end to end."""

import hashlib
import json

import cv2
import numpy as np
import pytest

from yolomg_motion.cli import main


def _write_camera_motion_video(path):
    width, height = 320, 240
    rng = np.random.default_rng(718)
    noise = rng.integers(30, 190, (height, width), dtype=np.uint8)
    background = cv2.GaussianBlur(noise, (3, 3), 0)
    for _ in range(100):
        point = (int(rng.integers(12, width - 12)), int(rng.integers(12, height - 12)))
        cv2.circle(background, point, int(rng.integers(3, 8)), int(rng.integers(25, 180)), -1)
    background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (width, height))
    assert writer.isOpened(), "The local OpenCV runtime must be able to create the input AVI"
    try:
        for index in range(9):
            camera_transform = np.array(
                [[1, 0, 2 * (index - 4)], [0, 1, index - 4]], dtype=np.float32
            )
            frame = cv2.warpAffine(background, camera_transform, (width, height))
            # The target moves independently of the translated background.
            x, y = 110 + 5 * index, 125 - 2 * index
            cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (255, 255, 255), -1)
            writer.write(frame)
    finally:
        writer.release()
    return width, height


def _artifact_hashes(directory):
    return {
        path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.rglob("*")
        if path.is_file()
    }


def test_cli_saves_centered_motion_artifacts_and_refuses_overwrite(tmp_path, capsys):
    source = tmp_path / "camera_motion.avi"
    output = tmp_path / "results"
    width, height = _write_camera_motion_video(source)
    config = tmp_path / "motion_config.json"
    config.write_text(
        json.dumps({"grid_step": 20, "grid_step_y": 20, "lk_window": 21, "min_inliers": 15}),
        encoding="utf-8",
    )
    arguments = [
        "--input", str(source),
        "--output", str(output),
        "--config", str(config),
        "--frame-step", "2",
        "--alignment-width", str(width),
        "--alignment-height", str(height),
        "--snapshot-every", "1",
        "--save-maps",
        "--save-video",
    ]

    assert main(arguments) == 0
    capsys.readouterr()

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    diagnostics = json.loads((output / "diagnostics.json").read_text(encoding="utf-8"))
    center_indices = list(range(2, 7))
    assert summary["input_frames_read"] == 9
    assert summary["output_frames"] == 5
    assert summary["valid_frames"] == 5
    assert summary["failed_frames"] == 0
    assert summary["first_center_index"] == 2
    assert summary["last_center_index"] == 6
    assert summary["video_fps"] == pytest.approx(30.0)
    assert summary["lookahead_seconds"] == pytest.approx(2 / 30.0)
    assert summary["config"]["alignment_width"] == width
    assert summary["config"]["alignment_height"] == height
    assert summary["mean_aligned_difference"] < summary["mean_unaligned_difference"]
    assert summary["processing_seconds"] > 0
    assert [row["center_index"] for row in diagnostics] == center_indices
    for row in diagnostics:
        assert row["success"]
        assert row["timestamp_seconds"] == pytest.approx(row["center_index"] / 30.0)
        assert 0.5 < row["valid_fraction"] <= 1.0
        assert row["processing_ms"] >= 0
        for direction in ("previous", "next"):
            assert row[direction]["success"]
            assert np.asarray(row[direction]["homography"]).shape == (3, 3)
            assert row[direction]["inlier_count"] >= 15

    map_paths = sorted((output / "maps").glob("*.npz"))
    assert [path.stem for path in map_paths] == [f"{index:06d}" for index in center_indices]
    for index, path in zip(center_indices, map_paths):
        with np.load(path, allow_pickle=False) as saved:
            assert saved["center_index"].dtype == np.int64
            assert saved["center_index"].shape == ()
            assert int(saved["center_index"]) == index
            assert saved["success"].dtype == np.bool_
            assert bool(saved["success"])
            assert saved["valid_mask"].dtype == np.bool_
            assert saved["valid_mask"].shape == (height, width)
            for key in ("difference", "motion"):
                assert saved[key].dtype == np.float32
                assert saved[key].shape == (height, width)
                assert np.isfinite(saved[key]).all()
                assert 0.0 <= float(saved[key].min()) <= float(saved[key].max()) <= 255.0
            assert np.count_nonzero(saved["motion"]) > 0
            map_png = cv2.imread(str(path.with_suffix(".png")), cv2.IMREAD_GRAYSCALE)
            assert map_png is not None
            # The saved map must preserve its absolute scale rather than using
            # the visualization gain or per-frame contrast normalization.
            np.testing.assert_array_equal(
                map_png, np.rint(saved["motion"]).clip(0, 255).astype(np.uint8)
            )

    snapshots = sorted(output.glob("comparison_*.png"))
    assert [path.stem for path in snapshots] == [f"comparison_{index:06d}" for index in center_indices]
    panel_shape = (2 * (height + 28), width * 3, 3)
    for path in snapshots:
        image = cv2.imread(str(path))
        assert image is not None
        assert image.shape == panel_shape
        assert float(image.std()) > 1.0

    # Decode every frame: the mere presence of a file or a container's declared
    # frame count does not establish that the saved video is usable.
    capture = cv2.VideoCapture(str(output / "comparison.mp4"))
    assert capture.isOpened()
    decoded_count = 0
    try:
        assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(30.0)
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            assert frame.shape == panel_shape
            decoded_count += 1
    finally:
        capture.release()
    assert decoded_count == 5

    # Reusing an output directory must fail before any existing artifact is
    # replaced, truncated, removed, or supplemented.
    originals = _artifact_hashes(output)
    with pytest.raises(SystemExit) as rejected:
        main(arguments)
    assert rejected.value.code == 2
    assert "Output directory is not empty" in capsys.readouterr().err
    assert _artifact_hashes(output) == originals
