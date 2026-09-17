"""Detect to acquire a drone, then track only a local ROI until visual lock fails."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from .core import MotionConfig, MotionExtractor
from .roi_pipeline import ROIPipeline, timing_statistics
from .run_artifacts import write_tracking_run
from .track_cli import draw_result
from .tracking import SingleTargetTracker


def main(argv=None):
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repo', type=Path, default=project/'vendor'/'YOLOMG')
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda', 'directml'), default='cpu')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--motion-backend', choices=('cpu', 'opencl'), default='cpu')
    parser.add_argument('--image-size', type=int, default=1280)
    parser.add_argument('--confidence', type=float, default=.10)
    parser.add_argument('--lock-confidence', type=float, default=.10)
    parser.add_argument('--confirmations', type=int, default=2)
    parser.add_argument('--detect-every', type=int, default=3,
                        help='Detection attempt interval while searching/confirming')
    parser.add_argument('--verify-every', type=int, default=0,
                        help='Locked-frame detector verification interval; 0 means only reacquire after visual loss')
    parser.add_argument('--search-radius', type=int, default=32)
    parser.add_argument('--appearance-threshold', type=float, default=.65)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--target-fps', type=float, default=60., help='Timing budget only; does not change video FPS')
    parser.add_argument('--no-video', action='store_true', help='Write only tracks.json for headless timing')
    args = parser.parse_args(argv)
    if not args.input.is_file():
        parser.error(f'Input not found: {args.input}')
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error('Output directory must be new or empty')
    if (min(args.detect_every, args.threads, args.confirmations, args.search_radius) < 1 or
            min(args.max_frames, args.gpu_id, args.verify_every) < 0 or args.image_size < 32 or
            not np.isfinite(args.target_fps) or args.target_fps <= 0 or
            not 0 < args.confidence <= args.lock_confidence < 1 or
            not 0 < args.appearance_threshold <= 1):
        parser.error('Invalid interval, size, confidence, frame limit or timing budget')
    cv2.setNumThreads(args.threads)
    weights = args.weights or args.repo/'runs/train/ARD100_mask32-1280_uavs/weights/best.pt'
    options = dict(repo_dir=args.repo, weights=weights, image_size=args.image_size,
                   conf_threshold=args.confidence, threads=args.threads)
    setup_started = perf_counter()
    if args.device == 'directml':
        from .directml_detector import DirectMLYOLOMGDetector
        detector = DirectMLYOLOMGDetector(**options, device_id=args.gpu_id)
    else:
        from .yolo_detector import YOLOMGDetector
        detector = YOLOMGDetector(**options, device=f'cuda:{args.gpu_id}' if args.device == 'cuda' else 'cpu')
    extractor = MotionExtractor(MotionConfig(open_iterations=0, close_iterations=0), backend=args.motion_backend)
    tracker = SingleTargetTracker(confirmations=args.confirmations, max_detector_gap=None,
                                  min_confidence=args.lock_confidence, search_radius=args.search_radius,
                                  appearance_threshold=args.appearance_threshold, roi_only=True,
                                  min_template_width=8, min_template_height=8, template_padding=2)
    pipeline = ROIPipeline(detector, extractor, tracker, args.detect_every, args.verify_every)
    setup_seconds = perf_counter() - setup_started
    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        cap.release()
        parser.error('Cannot open video')
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        cap.release()
        parser.error('Input has no valid FPS')
    args.output.mkdir(parents=True, exist_ok=True)
    rows, writer = [], None
    read_count, written, current_index = 0, 0, None
    status, error = 'complete', None
    decode_seconds, output_seconds = 0., 0.
    began = perf_counter()
    try:
        while not args.max_frames or read_count < args.max_frames:
            frame_started = perf_counter()
            ok, frame = cap.read()
            read_seconds = perf_counter() - frame_started
            decode_seconds += read_seconds
            if not ok:
                break
            read_count += 1
            current_index = read_count - 3 if read_count >= 5 else None
            observation = pipeline.push(frame)
            if observation is None:
                continue
            row = asdict(observation.result)
            row.update(observation.details, frame_index=observation.frame_index,
                       timestamp_seconds=observation.frame_index/fps)
            tick = perf_counter()
            if not args.no_video:
                canvas = draw_result(observation.frame, observation.result, observation.frame_index, fps, zoom=True)
                if writer is None:
                    writer = cv2.VideoWriter(str(args.output/'tracked_zoom.mp4'),
                                             cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                             (canvas.shape[1], canvas.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError('Video writer could not open tracked_zoom.mp4')
                writer.write(canvas)
                written += 1
            elapsed_output = perf_counter() - tick
            output_seconds += elapsed_output
            row['decode_seconds'] = read_seconds
            row['visualization_and_encoding_seconds'] = elapsed_output
            row['loop_seconds'] = perf_counter() - frame_started
            rows.append(row)
            if len(rows) % 120 == 0:
                print(f'Frame {observation.frame_index}: {observation.result.state}; '
                      f'detector calls {sum(r["detector_ran"] for r in rows)}', flush=True)
    except BaseException as caught:
        status = 'interrupted' if isinstance(caught, KeyboardInterrupt) else 'failed'
        error = dict(type=type(caught).__name__, message=str(caught), source_frame_index=current_index)
        raise
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        wall = perf_counter() - began
        locked = [r for r in rows if r['state'] == 'LOCKED' and r['bbox'] is not None]
        roi = [r for r in rows if r['roi_tracking']]
        reacquisition = [r for r in rows if r['mode'] == 'detection']
        stage = {key: sum(r['stage_timings'][key] for r in rows)
                 for key in ('motion_seconds', 'detector_seconds', 'tracking_seconds')}
        metadata = dict(pipeline='detect_then_roi', input=str(args.input.resolve()), weights=str(weights.resolve()),
                        model_input_size=args.image_size, confidence_threshold=args.confidence,
                        lock_confidence_threshold=args.lock_confidence, confirmations=args.confirmations,
                        detect_every=args.detect_every, verify_every=args.verify_every, max_detector_gap=None,
                        search_radius=args.search_radius, appearance_threshold=args.appearance_threshold,
                        template_config=dict(min_width=8, min_height=8, padding=2),
                        detector_backend=args.device, motion_backend=args.motion_backend, tracking_backend='cpu',
                        detector_runtime=getattr(detector, 'backend_info', dict(backend=args.device)),
                        motion_runtime=extractor.backend_info, tracking_runtime=tracker.backend_info,
                        input_fps=fps, input_frames_read=read_count, output_frames=len(rows),
                        input_dimensions=list(pipeline.frame_shape[1::-1]) if pipeline.frame_shape else None,
                        locked_frames=len(locked), first_lock_frame=locked[0]['frame_index'] if locked else None,
                        last_lock_frame=locked[-1]['frame_index'] if locked else None,
                        last_recorded_source_frame=rows[-1]['frame_index'] if rows else None,
                        states=dict(Counter(r['state'] for r in rows)), sources=dict(Counter(r['source'] for r in rows)),
                        detector_calls=sum(r['detector_ran'] for r in rows),
                        motion_calls=sum(r['motion_computed'] for r in rows), roi_frames=len(roi),
                        detector_seconds=stage['detector_seconds'], stage_timings=stage,
                        decode_seconds=decode_seconds, visualization_and_encoding_seconds=output_seconds,
                        setup_seconds=setup_seconds, wall_seconds=wall,
                        processing_fps=len(rows)/wall if wall else None,
                        target_fps=args.target_fps, lookahead_frames=2, lookahead_seconds=2/fps,
                        timing=dict(roi_compute=timing_statistics([r['processing_seconds'] for r in roi], args.target_fps),
                                    roi_with_decode_and_output=timing_statistics([r['loop_seconds'] for r in roi], args.target_fps),
                                    acquisition_compute=timing_statistics([r['processing_seconds'] for r in reacquisition], args.target_fps),
                                    all_with_decode_and_output=timing_statistics([r['loop_seconds'] for r in rows], args.target_fps)),
                        video_enabled=not args.no_video, video_frames_written=written,
                        status=status, completed=status == 'complete', partial=status != 'complete', error=error,
                        note='Centered k=2 omits first/last two frames and adds lookahead latency. '
                             'Locked ROI frames skip full-image preprocessing, motion and detection. '
                             'Acquisition is synchronous and can stall. ROI-only compute FPS is not end-to-end board FPS. '
                             'No automatic template refresh; shape change, occlusion and background drift can cause loss or wrong lock. '
                             'motion_valid=null means motion was deliberately skipped, not failed. '
                             'Timing excludes final JSON serialization; wall includes decode, optional draw/encode and writer flush.')
        write_tracking_run(args.output, rows, metadata)
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0 if locked else 2


if __name__ == '__main__':
    raise SystemExit(main())
