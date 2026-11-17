"""Run legacy template tracking or opt-in original-resolution ROI zoom detection."""
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
from .zoom_pipeline import ZoomROIPipeline


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
    parser.add_argument('--pipeline', choices=('template', 'zoom'), default='template',
                        help='template: existing full-frame acquisition; zoom: detect enlarged original-frame crops')
    parser.add_argument('--image-size', type=int,
                        help='Network input size: default 1280 for template, 640 for zoom')
    parser.add_argument('--roi-size', type=int, default=320,
                        help='Zoom detection window side in original-frame pixels')
    parser.add_argument('--motion-context', type=int, default=640,
                        help='Local motion registration context side; must contain the detection ROI')
    parser.add_argument('--max-rois', type=int, default=2,
                        help='Maximum network calls per scheduled zoom detection frame')
    parser.add_argument('--zoom-detect-every', type=int, default=10,
                        help='Minimum source-frame interval between zoom detection attempts, including lost frames')
    parser.add_argument('--motion-threshold', type=float, default=8.,
                        help='Motion proposal threshold in grayscale difference units')
    parser.add_argument('--scan-every', type=int, default=3,
                        help='Use a scan window every N zoom search attempts to cover weak-motion targets')
    parser.add_argument('--motion-prediction', action='store_true',
                        help='Opt-in motion prediction for bounded zoom search; does not confirm detections')
    parser.add_argument('--strict-verification', action='store_true',
                        help='Reproduce the original prediction policy: no visual grace or recent-target search')
    parser.add_argument('--contrast-search', action='store_true',
                        help='Use one existing zoom search slot for compact light/dark image candidates')
    parser.add_argument('--scale-retry', action='store_true',
                        help='Try an alternate local crop scale for small targets on smooth backgrounds')
    parser.add_argument('--adaptive-roi', action='store_true',
                        help='Choose a bounded primary crop scale from recent confirmed target sizes')
    parser.add_argument('--low-confidence', type=float,
                        help='Optional lower score floor for tightly checked existing-target support only')
    parser.add_argument('--prediction-max-gap', type=int, default=20,
                        help='Maximum source frames to retain an unobserved prediction')
    parser.add_argument('--prediction-gate', type=float, default=16.,
                        help='Motion tracker association gate')
    parser.add_argument('--prediction-process-noise', type=float, default=.25,
                        help='Motion prediction process noise; larger values permit more acceleration')
    parser.add_argument('--prediction-max-roi', type=int, default=480,
                        help='Maximum prediction search ROI side in source pixels; at least roi-size')
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
    if args.image_size is None:
        args.image_size = 640 if args.pipeline == 'zoom' else 1280
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
    if (min(args.roi_size, args.motion_context) < 32 or args.motion_context < args.roi_size or
            min(args.max_rois, args.zoom_detect_every, args.scan_every) < 1 or
            not np.isfinite(args.motion_threshold) or not 0 < args.motion_threshold <= 255):
        parser.error('Invalid zoom ROI, context, network-call budget, interval or motion threshold')
    if args.strict_verification and not args.motion_prediction:
        parser.error('--strict-verification requires --motion-prediction')
    if (args.contrast_search or args.scale_retry) and args.pipeline != 'zoom':
        parser.error('--contrast-search and --scale-retry require --pipeline zoom')
    if (args.adaptive_roi or args.low_confidence is not None) and not args.motion_prediction:
        parser.error('--adaptive-roi and --low-confidence require --motion-prediction')
    if args.low_confidence is not None and not .04 <= args.low_confidence < args.confidence:
        parser.error('--low-confidence must be at least .04 and below --confidence')
    if args.motion_prediction:
        if args.pipeline != 'zoom':
            parser.error('--motion-prediction requires --pipeline zoom')
        if args.prediction_max_gap < args.zoom_detect_every:
            parser.error('--prediction-max-gap must be >= --zoom-detect-every to permit detector confirmation')
        if (args.prediction_max_gap < 1 or args.prediction_max_roi < args.roi_size or
                not np.isfinite(args.prediction_gate) or args.prediction_gate <= 0 or
                not np.isfinite(args.prediction_process_noise) or args.prediction_process_noise <= 0):
            parser.error('Invalid prediction gap, association gate, process noise or maximum ROI')
    cv2.setNumThreads(args.threads)
    weights = args.weights or args.repo/'runs/train/ARD100_mask32-1280_uavs/weights/best.pt'
    detector_floor = args.confidence if args.low_confidence is None else args.low_confidence
    options = dict(repo_dir=args.repo, weights=weights, image_size=args.image_size,
                   conf_threshold=detector_floor, threads=args.threads)
    setup_started = perf_counter()
    if args.device == 'directml':
        from .directml_detector import DirectMLYOLOMGDetector
        detector = DirectMLYOLOMGDetector(**options, device_id=args.gpu_id)
    else:
        from .yolo_detector import YOLOMGDetector
        detector = YOLOMGDetector(**options, device=f'cuda:{args.gpu_id}' if args.device == 'cuda' else 'cpu')
    extractor = MotionExtractor(MotionConfig(open_iterations=0, close_iterations=0), backend=args.motion_backend)
    tracker_options = dict(confirmations=args.confirmations, max_detector_gap=None,
                           min_confidence=args.lock_confidence, search_radius=args.search_radius,
                           appearance_threshold=args.appearance_threshold, roi_only=True,
                           min_template_width=8, min_template_height=8, template_padding=2)
    if args.motion_prediction:
        from .predictive_tracking import PredictiveTargetTracker
        tracker = PredictiveTargetTracker(**tracker_options, prediction_max_gap=args.prediction_max_gap,
                                          prediction_gate=args.prediction_gate,
                                          process_noise=args.prediction_process_noise,
                                          verification_grace=0 if args.strict_verification else 1,
                                          low_confidence=args.low_confidence)
    else:
        tracker = SingleTargetTracker(**tracker_options)
    local_extractor = None
    if args.pipeline == 'zoom':
        local_extractor = MotionExtractor(
            MotionConfig(alignment_width=320, alignment_height=320, grid_step=24, grid_step_y=24,
                         min_inliers=10, blur_size=5, open_iterations=0, close_iterations=0),
            backend=args.motion_backend)
        prediction_options = dict(prediction_max_roi=args.prediction_max_roi,
                                  recent_search_attempts=0 if args.strict_verification else 2) if args.motion_prediction else {}
        pipeline = ZoomROIPipeline(detector, extractor, tracker, local_extractor=local_extractor,
                                   detect_every=args.zoom_detect_every, roi_size=args.roi_size,
                                   motion_context=args.motion_context, max_rois=args.max_rois,
                                   motion_threshold=args.motion_threshold, scan_every=args.scan_every,
                                   contrast_search=args.contrast_search,
                                   scale_retry=args.scale_retry,
                                   adaptive_roi=args.adaptive_roi,
                                   **prediction_options)
    else:
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
    trajectory = None
    if args.motion_prediction and not args.no_video:
        from .trajectory import TrajectoryRenderer
        trajectory = TrajectoryRenderer(max_points=60)
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
                canvas = draw_result(observation.frame, observation.result, observation.frame_index, fps,
                                     zoom=True, prediction=observation.details.get('prediction'))
                if trajectory is not None:
                    canvas = trajectory.render(canvas, observation.result, observation.frame_index,
                                               observation.details, fps=fps)
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
                      f'detector calls {sum(r.get("detector_calls", int(r["detector_ran"])) for r in rows)}', flush=True)
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
        metadata = dict(pipeline='roi_zoom_detection' if args.pipeline == 'zoom' else 'detect_then_roi',
                        input=str(args.input.resolve()), weights=str(weights.resolve()),
                        model_input_size=args.image_size, confidence_threshold=detector_floor,
                        lock_confidence_threshold=args.lock_confidence, confirmations=args.confirmations,
                        detect_every=args.zoom_detect_every if args.pipeline == 'zoom' else args.detect_every,
                        verify_every=None if args.pipeline == 'zoom' else args.verify_every, max_detector_gap=None,
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
                        detector_frames=sum(bool(r['detector_ran']) for r in rows),
                        detector_calls=sum(r.get('detector_calls', int(r['detector_ran'])) for r in rows),
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
        if args.pipeline == 'zoom':
            metadata.update(
                roi_size=args.roi_size, motion_context=args.motion_context, max_rois=args.max_rois,
                zoom_detect_every=args.zoom_detect_every, motion_threshold=args.motion_threshold,
                scan_every=args.scan_every, local_motion_runtime=local_extractor.backend_info,
                full_motion_config=asdict(extractor.config), local_motion_config=asdict(local_extractor.config),
                full_motion_calls=sum(r.get('motion_scope') == 'full' for r in rows),
                local_motion_calls=sum(r.get('motion_scope') == 'local' for r in rows),
                detector_attempt_frames=sum(r['mode'] == 'detection' for r in rows),
                max_detector_calls_per_frame=max((r.get('detector_calls', 0) for r in rows), default=0),
                network_input_shapes=[list(shape) for shape in sorted({
                    tuple(shape) for r in rows for shape in r.get('network_input_shapes', []) if shape})],
                note='Centered k=2 omits first/last two frames and adds lookahead latency. '
                     'Zoom crops come from original-resolution RGB and corresponding registered motion. '
                     'Detection attempts obey the source-frame interval and per-frame ROI budget; '
                     'intermediate frames use local template tracking. '
                     'Searching uses full-frame motion; tracked locations use local motion context. '
                     'motion_valid=null means deliberately skipped, not failed. '
                     'Interpolation does not recover missing image detail; weak motion, background clutter '
                     'and unmatched training scales can still cause missed detections or wrong locks. '
                     'Inference is synchronous; desktop throughput is not RKNN or board performance. '
                     'Timing excludes final JSON serialization; wall includes decode, optional draw/encode and writer flush.')
            stage['proposal_seconds'] = sum(r['stage_timings'].get('proposal_seconds', 0.) for r in rows)
        if args.pipeline == 'zoom':
            metadata['contrast_search'] = args.contrast_search
            metadata['scale_retry'] = args.scale_retry
            metadata['scale_retry_frames'] = sum(bool(r.get('scale_retry_used')) for r in rows)
            metadata['adaptive_roi'] = args.adaptive_roi
            metadata['adaptive_roi_frames'] = sum(bool(r.get('adaptive_roi_used')) for r in rows)
            metadata['low_confidence_floor'] = args.low_confidence
            metadata['low_confidence_frames'] = sum(bool(r.get('low_confidence_used')) for r in rows)
            metadata['regular_confidence_threshold'] = args.confidence
            metadata['adaptive_roi_policy'] = dict(target_network_pixels=32, size_history=3,
                min_roi_ratio=.7, side_alignment=16, max_change_per_detection=32,
                regular_window_fallback=args.scale_retry)
            metadata['scale_retry_policy'] = dict(
                scale=.7, side_alignment=16, min_side=16, background_side=64,
                excluded_center_side=32, max_target_side=32, max_background_std=12.,
                min_background_samples=32, max_center_distance=args.search_radius,
                adaptive_fallback='nominal_roi')
        if args.motion_prediction:
            metadata.update(
                motion_prediction=dict(enabled=True, max_gap_frames=args.prediction_max_gap,
                                       gate_threshold=args.prediction_gate, process_noise=args.prediction_process_noise,
                                       max_roi=args.prediction_max_roi, velocity_units='pixels/frame',
                                       trajectory_points=60, trajectory_history_visible=False,
                                       prediction_overlay_visible=False,
                                       verification_grace=tracker.verification_grace,
                                       grace_appearance_threshold=max(.8, args.appearance_threshold),
                                       grace_min_detector_hits=3,
                                       grace_min_recent_confidence=max(.3, args.lock_confidence),
                                       recent_search_attempts=0 if args.strict_verification else 2),
                verification_pending_frames=sum(bool(r.get('verification_pending')) for r in rows),
                predicted_only_frames=sum(
                    bool((r.get('prediction') or {}).get('coasting'))
                    for r in rows))
            metadata['note'] += (' Motion predictions guide bounded ROI search; prediction overlays are hidden by default. '
                                 'Predicted-only positions are not observed boxes or locked target coordinates.')
        write_tracking_run(args.output, rows, metadata)
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0 if locked else 2


if __name__ == '__main__':
    raise SystemExit(main())
