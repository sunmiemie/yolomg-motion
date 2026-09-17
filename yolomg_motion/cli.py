"""Offline video adapter; the core has no GUI or video-file dependency."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from .core import MotionConfig, MotionExtractor


def write_image(path, data):
    ok, encoded = cv2.imencode('.png', data)
    if not ok:
        raise RuntimeError(f'Cannot encode image: {path}')
    encoded.tofile(str(path))


def preview(result, gain=6.0):
    """Use the SAME fixed gain for both differences; no per-image normalization."""
    h, w = result.center_gray.shape
    tw = min(480, w)
    th = max(1, round(h*tw/w))
    def tile(frame, title, heat=False):
        if heat:
            frame = cv2.applyColorMap(np.clip(frame*gain, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        elif frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        resized = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((th+28, tw, 3), np.uint8)
        canvas[28:] = resized
        cv2.putText(canvas, title, (7, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (240, 240, 240), 1, cv2.LINE_AA)
        return canvas
    raw = result.unaligned_difference.copy()
    raw[~result.valid_mask] = 0
    top = np.hstack([tile(result.current, f'Center frame {result.center_index}'),
                     tile(result.previous_alignment.aligned, 'Previous aligned to center'),
                     tile(result.next_alignment.aligned, 'Next aligned to center')])
    bottom = np.hstack([tile(raw, f'Unaligned difference x{gain:g}', True),
                        tile(result.difference, f'Aligned difference x{gain:g}', True),
                        tile(result.motion, f'Morphology result x{gain:g}', True)])
    panel = np.vstack([top, bottom])
    if not result.success:
        reason = result.previous_alignment.reason or result.next_alignment.reason or 'no_common_valid_area'
        cv2.putText(panel, 'INVALID REGISTRATION: '+reason, (12, panel.shape[0]-14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
    return panel


def alignment_stats(a):
    return dict(success=a.success, reason=a.reason, tracked_points=a.tracked_points,
                inlier_count=a.inlier_count, inlier_ratio=a.inlier_ratio,
                homography=a.homography.tolist() if a.homography is not None else None)


def main(argv=None):
    parser = argparse.ArgumentParser(description='YOLOMG sparse LK alignment + centered three-frame difference')
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--frame-step', type=int, default=2)
    parser.add_argument('--max-frames', type=int, default=0, help='Maximum INPUT frames, 0 means all')
    parser.add_argument('--start-frame', type=int, default=0, help='Zero-based first input frame')
    parser.add_argument('--save-video', action='store_true')
    parser.add_argument('--save-maps', action='store_true', help='Save float32 NPZ maps and unscaled uint8 PNG')
    parser.add_argument('--snapshot-every', type=int, default=30)
    parser.add_argument('--preview-gain', type=float, default=6.0)
    parser.add_argument('--no-morphology', action='store_true')
    parser.add_argument('--fb-threshold', type=float, default=1.5, help='0 disables forward/backward filtering')
    parser.add_argument('--alignment-width', type=int, default=1920)
    parser.add_argument('--alignment-height', type=int, default=1080)
    parser.add_argument('--config', type=Path, help='Optional JSON overrides of MotionConfig')
    args = parser.parse_args(argv)
    if not args.input.is_file():
        parser.error(f'Input video not found: {args.input}')
    if args.max_frames < 0 or args.start_frame < 0 or args.snapshot_every < 1 or args.preview_gain <= 0:
        parser.error('Invalid frame limits, snapshot interval, or preview gain')
    settings = dict(frame_step=args.frame_step, fb_threshold=args.fb_threshold,
                    alignment_width=args.alignment_width, alignment_height=args.alignment_height)
    if args.config:
        settings.update(json.loads(args.config.read_text(encoding='utf-8-sig')))
    if args.no_morphology:
        settings.update(open_iterations=0, close_iterations=0)
    try:
        cfg = MotionConfig(**settings)
    except (ValueError, TypeError) as error:
        parser.error(str(error))
    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        parser.error(f'OpenCV cannot open video: {args.input}')
    if args.output.exists() and any(args.output.iterdir()):
        cap.release()
        parser.error('Output directory is not empty; choose a new directory')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.save_maps:
        (args.output/'maps').mkdir()
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        cap.release()
        parser.error('Video has no valid FPS metadata')
    # Sequentially discard the prefix so center_index is not affected by
    # codec-specific approximate seeking to keyframes.
    for _ in range(args.start_frame):
        if not cap.grab():
            cap.release()
            parser.error('start-frame is past the end of the video')
    extractor = MotionExtractor(cfg)
    rows, writer = [], None
    read_count, processing_seconds = 0, 0.0
    began = perf_counter()
    try:
        while not args.max_frames or read_count < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            read_count += 1
            t0 = perf_counter()
            result = extractor.push(frame)
            dt = perf_counter()-t0
            processing_seconds += dt
            if result is None:
                continue
            result.center_index += args.start_frame
            valid = result.valid_mask
            row = dict(center_index=result.center_index, timestamp_seconds=result.center_index/fps,
                       success=result.success, processing_ms=dt*1000,
                       valid_fraction=float(valid.mean()),
                       unaligned_mean=float(result.unaligned_difference[valid].mean()) if valid.any() else None,
                       aligned_mean=float(result.difference[valid].mean()) if valid.any() else None,
                       previous=alignment_stats(result.previous_alignment), next=alignment_stats(result.next_alignment))
            rows.append(row)
            make_snapshot = (len(rows)-1) % args.snapshot_every == 0
            if args.save_video or make_snapshot:
                panel = preview(result, args.preview_gain)
                if args.save_video:
                    if writer is None:
                        writer = cv2.VideoWriter(str(args.output/'comparison.mp4'),
                                                 cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                                 (panel.shape[1], panel.shape[0]))
                        if not writer.isOpened():
                            raise RuntimeError('MP4 video writer could not be opened')
                    writer.write(panel)
                if make_snapshot:
                    write_image(args.output/f'comparison_{result.center_index:06d}.png', panel)
            if args.save_maps:
                stem = args.output/'maps'/f'{result.center_index:06d}'
                np.savez_compressed(str(stem)+'.npz', difference=result.difference,
                                    motion=result.motion, valid_mask=result.valid_mask,
                                    center_index=np.int64(result.center_index), success=np.bool_(result.success))
                write_image(str(stem)+'.png', np.rint(result.motion).clip(0, 255).astype(np.uint8))
            if len(rows) % 30 == 0:
                print(f'Processed {len(rows)} centers; current source frame {result.center_index}', flush=True)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    valid_rows = [r for r in rows if r['success']]
    before = float(np.mean([r['unaligned_mean'] for r in valid_rows])) if valid_rows else None
    after = float(np.mean([r['aligned_mean'] for r in valid_rows])) if valid_rows else None
    summary = dict(input=str(args.input.resolve()), config=asdict(cfg), video_fps=fps,
                   input_frames_read=read_count, output_frames=len(rows), valid_frames=len(valid_rows),
                   failed_frames=len(rows)-len(valid_rows),
                   first_center_index=rows[0]['center_index'] if rows else None,
                   last_center_index=rows[-1]['center_index'] if rows else None,
                   lookahead_seconds=cfg.frame_step/fps,
                   processing_seconds=processing_seconds, wall_seconds=perf_counter()-began,
                   processing_fps=len(rows)/processing_seconds if processing_seconds else 0,
                   mean_unaligned_difference=before, mean_aligned_difference=after,
                   difference_reduction=1-after/before if before and after is not None else None,
                   preview_gain=args.preview_gain,
                   metric_note='Means use the same valid area and include moving targets; these are not detection AP or accuracy.')
    (args.output/'diagnostics.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding='utf-8')
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if rows and valid_rows else 2


if __name__ == '__main__':
    raise SystemExit(main())
