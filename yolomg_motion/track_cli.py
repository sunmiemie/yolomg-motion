"""Run a real YOLOMG detector and a bounded single-target tracker on video."""
import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from .core import MotionConfig, MotionExtractor
from .run_artifacts import read_tracking_run, write_tracking_run
from .tracking import Detection, SingleTargetTracker


def draw_result(frame, result, frame_index, fps, zoom=False):
    canvas = frame.copy()
    h, w = canvas.shape[:2]
    yellow = (0, 230, 255)
    state_color = yellow if result.state == 'LOCKED' else (210, 210, 210)
    cv2.rectangle(canvas, (12, 12), (430, 68), (25, 25, 25), -1)
    text = f'UAV #{result.track_id}  {result.state}' if result.track_id else f'UAV  {result.state}'
    cv2.putText(canvas, text, (23, 36), cv2.FONT_HERSHEY_SIMPLEX, .62, state_color, 1, cv2.LINE_AA)
    cv2.putText(canvas, f'{frame_index / fps:.2f}s  |  {result.source.upper()}',
                (23, 57), cv2.FONT_HERSHEY_SIMPLEX, .43, (220, 220, 220), 1, cv2.LINE_AA)
    if result.bbox is None or result.state != 'LOCKED':
        return canvas
    x1, y1, x2, y2 = map(float, result.bbox)
    cx, cy = (x1+x2)/2, (y1+y2)/2
    # Display padding helps a 10x3-pixel object remain visible. Saved boxes are
    # the actual measured coordinates, without this presentation-only padding.
    dw, dh = max(18, x2-x1+6), max(12, y2-y1+6)
    lo = (max(0, round(cx-dw/2)), max(0, round(cy-dh/2)))
    hi = (min(w-1, round(cx+dw/2)), min(h-1, round(cy+dh/2)))
    cv2.rectangle(canvas, lo, hi, yellow, 1, cv2.LINE_AA)
    cv2.putText(canvas, f'UAV {result.track_id}', (lo[0], max(14, lo[1]-6)),
                cv2.FONT_HERSHEY_SIMPLEX, .43, yellow, 1, cv2.LINE_AA)
    if zoom and w >= 640 and h >= 360:
        cw, ch = min(96, w), min(54, h)
        left = int(np.clip(round(cx-cw/2), 0, w-cw))
        top = int(np.clip(round(cy-ch/2), 0, h-ch))
        crop = frame[top:top+ch, left:left+cw]
        scale = min(4, (w-40)//cw, (h-100)//ch)
        patch = cv2.resize(crop, (cw*scale, ch*scale), interpolation=cv2.INTER_NEAREST)
        p1 = (round((x1-left)*scale)-2, round((y1-top)*scale)-2)
        p2 = (round((x2-left)*scale)+2, round((y2-top)*scale)+2)
        cv2.rectangle(patch, p1, p2, yellow, 1, cv2.LINE_AA)
        ph, pw = patch.shape[:2]
        ox, oy = w-pw-20, 52
        cv2.rectangle(canvas, (ox-3, oy-28), (ox+pw+3, oy+ph+3), (25, 25, 25), -1)
        canvas[oy:oy+ph, ox:ox+pw] = patch
        cv2.putText(canvas, f'TARGET ZOOM {scale}x', (ox+8, oy-9),
                    cv2.FONT_HERSHEY_SIMPLEX, .48, yellow, 1, cv2.LINE_AA)
    return canvas


def cached_centers(cap, cache, input_path, max_frames):
    summary = json.loads((cache/'summary.json').read_text(encoding='utf-8'))
    if Path(summary['input']).resolve() != input_path.resolve():
        raise ValueError('Motion cache belongs to a different input video')
    if summary.get('first_center_index') != 2 or summary.get('config', {}).get('frame_step') != 2:
        raise ValueError('This adapter requires a complete k=2 motion cache starting at frame 2')
    end = summary['last_center_index']
    if max_frames:
        end = min(end, max_frames-3)
    index = 0
    while not max_frames or index < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if 2 <= index <= end:
            path = cache/'maps'/f'{index:06d}.npz'
            with np.load(path, allow_pickle=False) as data:
                if int(data['center_index']) != index or data['difference'].shape != frame.shape[:2]:
                    raise ValueError(f'Motion cache index/shape mismatch: {path}')
                # Official training uses the unthresholded three-frame residual.
                yield index, frame, data['difference'].copy(), bool(data['success'])
        index += 1


def calculated_centers(cap, max_frames, extractor=None, timings=None):
    extractor = extractor or MotionExtractor(MotionConfig())
    read = 0
    while not max_frames or read < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        read += 1
        started = perf_counter()
        result = extractor.push(frame)
        if timings is not None:
            timings['motion_seconds'] += perf_counter() - started
        if result is not None:
            yield result.center_index, result.current, result.difference, result.success


def main(argv=None):
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repo', type=Path, default=project/'vendor'/'YOLOMG')
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--motion-cache', type=Path)
    parser.add_argument('--detection-cache', type=Path,
                        help='Replay a prior run directory with verified matching model settings')
    parser.add_argument('--image-size', type=int, default=1280)
    parser.add_argument('--confidence', type=float, default=.10)
    parser.add_argument('--lock-confidence', type=float, default=.20,
                        help='Minimum detector confidence eligible for target acquisition/association')
    parser.add_argument('--detect-every', type=int, default=3)
    parser.add_argument('--max-detector-gap', type=int, default=15,
                        help='Maximum source frames without an associated model detection')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', choices=('cpu', 'cuda', 'directml'), default='cpu',
                        help='Neural-network backend; DirectML supports compatible Windows AMD/Intel/NVIDIA GPUs')
    parser.add_argument('--gpu-id', type=int, default=0, help='CUDA device index or DirectML DXGI adapter index')
    parser.add_argument('--motion-backend', choices=('cpu', 'opencl'), default='cpu',
                        help='Image processing backend for sparse flow alignment and frame differences')
    parser.add_argument('--tracking-backend', choices=('cpu', 'opencl'), default='cpu',
                        help='Image processing backend for grayscale/template matching; track state remains on CPU')
    parser.add_argument('--max-frames', type=int, default=0)
    # Accept old scripts without producing any of the retired snapshot files.
    parser.add_argument('--snapshot-every', type=int, default=60, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not args.input.is_file():
        parser.error(f'Input not found: {args.input}')
    if args.output.exists():
        if not args.output.is_dir():
            parser.error('Output must be a directory; choose a new directory')
        if any(args.output.iterdir()):
            parser.error('Output directory is not empty; choose a new directory')
    if args.image_size < 32:
        parser.error('image-size must be at least 32')
    if min(args.detect_every, args.threads, args.snapshot_every, args.max_detector_gap) < 1 or min(args.max_frames, args.gpu_id) < 0:
        parser.error('Invalid frame interval, threads, or frame limit')
    if args.motion_cache and args.motion_backend != 'cpu':
        parser.error('A saved motion cache does not run OpenCL; use --motion-backend cpu or omit --motion-cache')
    if not 0 < args.confidence <= args.lock_confidence < 1:
        parser.error('Require 0 < confidence <= lock-confidence < 1')
    weights = args.weights or args.repo/'runs/train/ARD100_mask32-1280_uavs/weights/best.pt'
    detector, cached_detections, cached_summary = None, None, None
    if args.detection_cache:
        cached_rows, cached_summary = read_tracking_run(args.detection_cache, strict_finite=True)
        if (Path(cached_summary['input']).resolve() != args.input.resolve()
                or Path(cached_summary['weights']).resolve() != weights.resolve()
                or cached_summary['model_input_size'] != args.image_size
                or cached_summary['confidence_threshold'] != args.confidence
                or cached_summary['detect_every'] != args.detect_every
                or cached_summary.get('detector_backend', 'cpu') != args.device
                or cached_summary.get('motion_backend', 'cpu') != args.motion_backend):
            parser.error('Detection cache input, model, or inference settings do not match')
        cached_detections = {r['frame_index']: r for r in cached_rows}
        if len(cached_detections) != len(cached_rows):
            parser.error('Detection cache contains duplicate frame indices')
    else:
        options = dict(repo_dir=args.repo, weights=weights, image_size=args.image_size,
                       conf_threshold=args.confidence, threads=args.threads)
        if args.device == 'directml':
            from .directml_detector import DirectMLYOLOMGDetector
            detector = DirectMLYOLOMGDetector(**options, device_id=args.gpu_id)
        else:
            from .yolo_detector import YOLOMGDetector
            if args.device == 'cuda':
                options['device'] = f'cuda:{args.gpu_id}'
            detector = YOLOMGDetector(**options)
    tracker = SingleTargetTracker(min_confidence=args.lock_confidence, max_detector_gap=args.max_detector_gap,
                                  backend=args.tracking_backend)
    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        parser.error('Cannot open video')
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        cap.release()
        parser.error('Video has no valid FPS')
    try:
        extractor = None if args.motion_cache else MotionExtractor(MotionConfig(), backend=args.motion_backend)
    except BaseException:
        cap.release()
        raise
    args.output.mkdir(parents=True, exist_ok=False) if not args.output.exists() else None
    rows, writer = [], None
    video_frames_written, processing_error = 0, None
    status, current_index = 'complete', None
    timings = dict(motion_seconds=0.0, tracking_seconds=0.0, visualization_and_encoding_seconds=0.0)
    detector_seconds, began = 0., perf_counter()
    source = cached_centers(cap, args.motion_cache, args.input, args.max_frames) if args.motion_cache else calculated_centers(cap, args.max_frames, extractor, timings)
    try:
        for index, frame, motion, motion_valid in source:
            current_index = index
            # Failure to align is not a valid zero-motion observation. Let the
            # bounded appearance tracker bridge it instead of classifying it.
            run_detector = len(rows) % args.detect_every == 0 and motion_valid
            proposals = None
            if cached_detections is not None:
                cached = cached_detections.get(index)
                if cached is None or cached['detector_ran'] != run_detector or cached['motion_valid'] != bool(motion_valid):
                    raise ValueError(f'Detection cache missing frame or observation mismatch: {index}')
                if run_detector:
                    proposals = [Detection(tuple(d['bbox']), float(d['confidence'])) for d in cached['detections']]
            elif run_detector:
                start = perf_counter()
                raw = detector.detect(frame, motion)
                detector_seconds += perf_counter()-start
                proposals = [Detection(tuple(d.bbox_xyxy), float(d.confidence)) for d in raw]
            started = perf_counter()
            result = tracker.update(frame, proposals, index)
            timings['tracking_seconds'] += perf_counter() - started
            row = asdict(result)
            row.update(frame_index=index, timestamp_seconds=index/fps, detector_ran=run_detector,
                       motion_valid=bool(motion_valid), detections=[asdict(d) for d in proposals] if proposals is not None else None)
            # Enforce a portable JSON schema even if a tracker returns numpy scalars.
            row = json.loads(json.dumps(row, allow_nan=False,
                                       default=lambda x: x.item() if hasattr(x, 'item') else list(x)))
            rows.append(row)
            started = perf_counter()
            enlarged = draw_result(frame, result, index, fps, zoom=True)
            if writer is None:
                writer = cv2.VideoWriter(str(args.output/'tracked_zoom.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                         (frame.shape[1], frame.shape[0]))
                if not writer.isOpened():
                    raise RuntimeError('Video writer could not open tracked_zoom.mp4')
            writer.write(enlarged)
            video_frames_written += 1
            timings['visualization_and_encoding_seconds'] += perf_counter() - started
            if len(rows) % 30 == 0:
                print(f'Frame {index}: {result.state}, detector calls {sum(r["detector_ran"] for r in rows)}', flush=True)
    except BaseException as error:
        status = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        processing_error = dict(type=type(error).__name__, message=str(error),
                                source_frame_index=current_index)
        raise
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        locked = [r for r in rows if r['state'] == 'LOCKED' and r['bbox'] is not None]
        summary = dict(input=str(args.input.resolve()), weights=str(weights.resolve()),
                   motion_input='aligned centered three-frame mean residual, k=2, no morphology',
                   model_input_size=args.image_size, confidence_threshold=args.confidence,
                   lock_confidence_threshold=args.lock_confidence,
                   detect_every=args.detect_every, max_detector_gap=tracker.max_detector_gap if hasattr(tracker, 'max_detector_gap') else 15,
                   detection_cache=str(args.detection_cache.resolve()) if args.detection_cache else None,
                   cached_detector_seconds=cached_summary.get('detector_seconds') if cached_summary else None,
                   detector_backend=args.device, motion_backend=args.motion_backend,
                   tracking_backend=args.tracking_backend, gpu_id=args.gpu_id if args.device != 'cpu' else None,
                   detector_runtime=(dict(mode='replay', inference_executed=False) if cached_summary is not None else
                                     getattr(detector, 'backend_info', dict(backend=args.device, inference_executed=any(r['detector_ran'] for r in rows)))),
                   motion_runtime=(dict(mode='cache', computation_executed=False) if extractor is None else
                                   getattr(extractor, 'backend_info', dict(backend=args.motion_backend))),
                   tracking_runtime=getattr(tracker, 'backend_info', dict(backend=args.tracking_backend)),
                   stage_timings=timings,
                   input_fps=fps, output_frames=len(rows), locked_frames=len(locked),
                   first_lock_frame=locked[0]['frame_index'] if locked else None,
                   last_lock_frame=locked[-1]['frame_index'] if locked else None,
                   states=dict(Counter(r['state'] for r in rows)),
                   sources=dict(Counter(r['source'] for r in rows)),
                   detector_calls=sum(r['detector_ran'] for r in rows), detector_seconds=detector_seconds,
                   wall_seconds=perf_counter()-began,
                   status=status, completed=status == 'complete', partial=status != 'complete',
                   error=processing_error, video_frames_written=video_frames_written,
                   last_recorded_source_frame=rows[-1]['frame_index'] if rows else None,
                   note='Automatically initialized. Lock coverage is tracker status, not measured detection accuracy. No manual ROI or video-specific target coordinates. Display boxes include padding; tracks.json contains unpadded boxes. No audio; centered k=2 processing omits first/last 2 input frames.')
        write_tracking_run(args.output, rows, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if locked else 2


if __name__ == '__main__':
    raise SystemExit(main())
