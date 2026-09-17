"""Synthetic ROI-only CPU timing at a requested resolution; no model or video I/O.

This measures local tracking after a supplied detection, not automatic detector
accuracy or end-to-end camera throughput. Frame generation is outside timing.
Prints JSON to stdout and creates no artifacts. Can also run on a board.
"""
import argparse
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from yolomg_motion.roi_pipeline import timing_statistics
from yolomg_motion.tracking import Detection, SingleTargetTracker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--width', type=int, default=2560)
    parser.add_argument('--height', type=int, default=1440)
    parser.add_argument('--frames', type=int, default=600)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if min(args.width, args.height) < 256 or args.frames < 20 or args.threads < 1:
        parser.error('Require dimensions >=256, frames >=20 and positive thread count')
    cv2.setNumThreads(args.threads)
    tracker = SingleTargetTracker(confirmations=1, max_detector_gap=None, roi_only=True,
                                  min_template_width=8, min_template_height=8, template_padding=2)
    rng = np.random.default_rng(109)
    background = rng.integers(30, 65, (args.height, args.width, 3), dtype=np.uint8)
    patch = rng.integers(90, 255, (12, 18, 3), dtype=np.uint8)
    samples, errors, areas = [], [], []
    incorrect = 0
    for index in range(args.frames + 30):
        # A smooth displacement stays in a small search window; not an aerial accuracy benchmark.
        x = args.width//2 + int(45*np.sin(index*.035))
        y = args.height//2 + int(25*np.sin(index*.021))
        image = background.copy()
        image[y:y+12, x:x+18] = patch
        detections = [Detection((x, y, x+18, y+12), .9)] if index == 0 else None
        tick = perf_counter()
        result = tracker.update(image, detections, index)
        elapsed = perf_counter() - tick
        if index >= 30:
            samples.append(elapsed)
            if result.state != 'LOCKED':
                incorrect += 1
            else:
                errors.append(float(np.max(np.abs(np.asarray(result.bbox)-[x, y, x+18, y+12]))))
            if tracker.last_search_roi:
                x1, y1, x2, y2 = tracker.last_search_roi
                areas.append((x2-x1)*(y2-y1))
    report = dict(scope='synthetic ROI tracker only; detector, MFEM, camera, decode, frame generation and output excluded',
                  platform=platform.platform(), processor=platform.processor(), opencv=cv2.__version__,
                  dimensions=[args.width, args.height], threads=args.threads, warmup_frames=30,
                  timing=timing_statistics(samples), mean_compute_fps=len(samples)/sum(samples),
                  unlocked_frames=incorrect, max_box_coordinate_error_pixels=max(errors) if errors else None,
                  max_search_area_pixels=max(areas) if areas else None,
                  board_60fps_verified=False)
    print(json.dumps(report, indent=2))
    return 0 if incorrect == 0 and errors and max(errors) <= 2 else 2


if __name__ == '__main__':
    raise SystemExit(main())
