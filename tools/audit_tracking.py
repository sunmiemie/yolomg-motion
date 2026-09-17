"""Audit tracking artifacts structurally; this does not measure detection accuracy.

Usage: python tools/audit_tracking.py --run path/to/tracking_run
The zoom video, plus an optional legacy clean video, is decoded completely.
Results are printed only unless --report explicitly requests a JSON file.
No model or source video is required.
"""
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yolomg_motion.run_artifacts import read_tracking_run


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def decode_video(path, errors, warnings):
    """Count actual successfully decoded frames, not CAP_PROP_FRAME_COUNT."""
    report = dict(path=str(path.resolve()), decoded_frames=0, width=None, height=None,
                  fps=None, metadata_frames=None)
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            errors.append(f'{path.name}: cannot open video')
            return report
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        metadata_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        report['fps'] = fps if math.isfinite(fps) else None
        report['metadata_frames'] = metadata_count if math.isfinite(metadata_count) else None
        if not math.isfinite(fps) or fps <= 0:
            errors.append(f'{path.name}: invalid FPS metadata')
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or frame.size == 0:
                errors.append(f'{path.name}: decoder returned an empty frame')
                break
            height, width = frame.shape[:2]
            if report['decoded_frames'] == 0:
                report.update(width=width, height=height)
            elif (width, height) != (report['width'], report['height']):
                errors.append(f'{path.name}: frame size changed at output frame {report["decoded_frames"]}')
            report['decoded_frames'] += 1
        if not report['decoded_frames']:
            errors.append(f'{path.name}: no decodable frames')
        if math.isfinite(metadata_count) and abs(metadata_count-report['decoded_frames']) > 0.5:
            warnings.append(f'{path.name}: metadata frame count differs from actual decode count')
    finally:
        capture.release()
    return report


def _bbox_problem(box, width, height):
    if not isinstance(box, list) or len(box) != 4 or not all(_number(v) for v in box):
        return 'bbox must be four finite xyxy numbers'
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return 'bbox must have positive width and height'
    # xyxy uses continuous image coordinates; x2==width/y2==height is valid.
    if min(x1, y1) < 0 or (width is not None and x2 > width) or (height is not None and y2 > height):
        return 'bbox is outside video bounds'
    return None


def _intervals(rows, predicate, fps):
    spans = []
    start = None
    for position in range(len(rows)+1):
        active = position < len(rows) and predicate(rows[position])
        if active and start is None:
            start = position
        elif not active and start is not None:
            first, last = rows[start], rows[position-1]
            spans.append(dict(start_frame=first.get('frame_index'), end_frame=last.get('frame_index'),
                              frame_count=position-start, start_seconds=first.get('timestamp_seconds'),
                              end_seconds=last.get('timestamp_seconds'),
                              duration_seconds=(position-start)/fps if fps else None))
            start = None
    return spans


def audit_run(run_dir):
    run_dir = Path(run_dir)
    errors, warnings = [], []
    try:
        tracks, summary = read_tracking_run(run_dir, strict_finite=True)
    except (OSError, ValueError) as exc:
        errors.append(f'Tracking JSON: {exc}')
        tracks, summary = [], {}
    roi_pipeline = summary.get('pipeline') == 'detect_then_roi'
    headless = roi_pipeline and summary.get('video_enabled') is False
    video_names = [] if headless else ['tracked_zoom.mp4']
    if (run_dir/'tracked.mp4').exists():
        video_names.append('tracked.mp4')
    videos = {name: decode_video(run_dir/name, errors, warnings)
              for name in video_names}
    if headless:
        dimensions = summary.get('input_dimensions')
        if not isinstance(dimensions, list) or len(dimensions) != 2 or not all(_integer(v) and v > 0 for v in dimensions):
            errors.append('Run metadata: headless ROI run requires valid input_dimensions')
            width, height = None, None
        else:
            width, height = dimensions
        if summary.get('video_frames_written') != 0:
            errors.append('Run metadata: headless run cannot claim written video frames')
    else:
        primary = videos['tracked_zoom.mp4']
        width, height = primary['width'], primary['height']
    fps = summary.get('input_fps')
    if not _number(fps) or fps <= 0:
        errors.append('Run metadata: input_fps must be finite and positive')
        fps = None
    if not tracks:
        errors.append('tracks.json: no output frames')
    if not _integer(summary.get('output_frames')) or summary['output_frames'] != len(tracks):
        errors.append('Run metadata: output_frames does not equal tracking frame count')
    for name, video in videos.items():
        if video['decoded_frames'] != len(tracks):
            errors.append(f'{name}: decoded frame count does not equal tracks.json length')
        if (video['width'], video['height']) != (width, height):
            errors.append(f'{name}: dimensions differ from tracked_zoom.mp4')
        if fps and video['fps'] and abs(video['fps']-fps) > max(.001, fps*.0001):
            errors.append(f'{name}: FPS differs from run metadata input_fps')
    required = {'frame_index', 'timestamp_seconds', 'state', 'bbox', 'track_id', 'source',
                'detector_confidence', 'appearance_score', 'detector_ran', 'motion_valid', 'detections'}
    rows, previous_index = [], None
    previous_locked_id = None
    seen_ids, last_episode_id, episodes = set(), None, []
    states, sources, locked_sources = Counter(), Counter(), Counter()
    locked_frames = detector_calls = motion_invalid_frames = motion_skipped_frames = 0
    for position, raw in enumerate(tracks):
        label = f'tracks.json row {position}'
        if not isinstance(raw, dict):
            errors.append(f'{label}: expected an object')
            rows.append({})
            previous_locked_id = None
            continue
        rows.append(raw)
        missing = sorted(required-set(raw))
        if missing:
            errors.append(f'{label}: missing fields {missing}')
        index = raw.get('frame_index')
        if not _integer(index) or index < 0:
            errors.append(f'{label}: frame_index must be a nonnegative integer')
        else:
            if previous_index is not None and index != previous_index+1:
                errors.append(f'{label}: frame_index is not consecutive')
            previous_index = index
        timestamp = raw.get('timestamp_seconds')
        if not _number(timestamp) or timestamp < 0:
            errors.append(f'{label}: invalid timestamp_seconds')
        elif fps and _integer(index) and abs(timestamp-index/fps) > 1e-4:
            errors.append(f'{label}: timestamp_seconds disagrees with frame_index/input_fps')
        state, source = raw.get('state'), raw.get('source')
        if not isinstance(state, str) or not state:
            errors.append(f'{label}: state must be a nonempty string')
        else:
            states[state] += 1
        if not isinstance(source, str) or not source:
            errors.append(f'{label}: source must be a nonempty string')
        else:
            sources[source] += 1
        box, track_id = raw.get('bbox'), raw.get('track_id')
        box_valid = box is not None
        if box is not None:
            problem = _bbox_problem(box, width, height)
            if problem:
                errors.append(f'{label}: {problem}')
                box_valid = False
        if track_id is not None and (not _integer(track_id) or track_id < 0):
            errors.append(f'{label}: track_id must be null or a nonnegative integer')
        locked = state == 'LOCKED' and box_valid
        if state == 'LOCKED' and not box_valid:
            errors.append(f'{label}: LOCKED requires a valid bbox')
        if locked:
            locked_frames += 1
            if isinstance(source, str):
                locked_sources[source] += 1
            if not _integer(track_id) or track_id < 1:
                errors.append(f'{label}: LOCKED requires a positive integer track_id')
                previous_locked_id = None
            else:
                if previous_locked_id is not None and track_id != previous_locked_id:
                    errors.append(f'{label}: track_id changed during uninterrupted LOCKED frames')
                if previous_locked_id is None or previous_locked_id != track_id:
                    if track_id in seen_ids and last_episode_id != track_id:
                        errors.append(f'{label}: a previously replaced track_id was reused')
                    seen_ids.add(track_id)
                    last_episode_id = track_id
                    episodes.append(dict(track_id=track_id, start_frame=index, end_frame=index, frame_count=1))
                else:
                    episodes[-1]['end_frame'] = index
                    episodes[-1]['frame_count'] += 1
                previous_locked_id = track_id
        else:
            previous_locked_id = None
        for key in ('detector_confidence', 'appearance_score'):
            value = raw.get(key)
            if value is not None and not _number(value):
                errors.append(f'{label}: {key} must be null or a finite number')
        value = raw.get('detector_confidence')
        if _number(value) and not 0 <= value <= 1:
            errors.append(f'{label}: detector_confidence must lie in [0, 1]')
        ran, valid = raw.get('detector_ran'), raw.get('motion_valid')
        if not isinstance(ran, bool):
            errors.append(f'{label}: detector_ran must be boolean')
        skipped = roi_pipeline and raw.get('motion_computed') is False
        if roi_pipeline and not isinstance(raw.get('motion_computed'), bool):
            errors.append(f'{label}: motion_computed must be boolean')
        if skipped:
            motion_skipped_frames += 1
            if valid is not None or ran is not False:
                errors.append(f'{label}: skipped motion requires null motion_valid and no detector execution')
        elif not isinstance(valid, bool):
            errors.append(f'{label}: motion_valid must be boolean for computed motion')
        detector_calls += ran is True
        motion_invalid_frames += valid is False
        if ran is True and valid is not True:
            errors.append(f'{label}: detector ran without valid motion input')
        detections = raw.get('detections')
        if ran is True:
            if not isinstance(detections, list):
                errors.append(f'{label}: detector_ran requires a detections array')
            else:
                for number, detection in enumerate(detections):
                    prefix = f'{label} detection {number}'
                    if not isinstance(detection, dict):
                        errors.append(f'{prefix}: expected an object')
                        continue
                    problem = _bbox_problem(detection.get('bbox'), width, height)
                    if problem:
                        errors.append(f'{prefix}: {problem}')
                    confidence = detection.get('confidence')
                    if not _number(confidence) or not 0 <= confidence <= 1:
                        errors.append(f'{prefix}: confidence must lie in [0, 1]')
        elif detections is not None:
            errors.append(f'{label}: detections must be null when detector did not run')
    for key, observed in (('locked_frames', locked_frames), ('detector_calls', detector_calls)):
        if key in summary and (not _integer(summary[key]) or summary[key] != observed):
            errors.append(f'Run metadata: {key} disagrees with tracking frames')
    for key, observed in (('states', dict(states)), ('sources', dict(sources))):
        if key in summary and summary[key] != observed:
            errors.append(f'Run metadata: {key} disagrees with tracking frames')
    is_unlocked = lambda row: row.get('state') != 'LOCKED' or row.get('bbox') is None
    unlocked = _intervals(rows, is_unlocked, fps)
    locked_indices = [row.get('frame_index') for row in rows if not is_unlocked(row)]
    for key, observed in (('first_lock_frame', locked_indices[0] if locked_indices else None),
                          ('last_lock_frame', locked_indices[-1] if locked_indices else None)):
        if key in summary and summary[key] != observed:
            errors.append(f'Run metadata: {key} disagrees with tracking frames')
    if not locked_frames:
        warnings.append('No LOCKED frames: structural verification cannot establish target acquisition')
    return dict(schema_version=1, run=str(run_dir.resolve()), passed=not errors,
                errors=errors, warnings=warnings, videos=videos, json_frames=len(tracks),
                first_frame_index=rows[0].get('frame_index') if rows else None,
                last_frame_index=rows[-1].get('frame_index') if rows else None,
                states=dict(states), sources=dict(sources), locked_sources=dict(locked_sources),
                locked_frames=locked_frames, unlocked_frames=len(tracks)-locked_frames,
                lock_fraction=locked_frames/len(tracks) if tracks else 0.,
                detector_calls=detector_calls, motion_invalid_frames=motion_invalid_frames,
                motion_skipped_frames=motion_skipped_frames,
                track_ids=sorted(seen_ids), track_episodes=episodes, unlocked_intervals=unlocked,
                longest_unlocked_frames=max((span['frame_count'] for span in unlocked), default=0),
                note='Structural and complete-video-decode verification only. Lock coverage is not detection accuracy. '
                     'Unlocked intervals include initial acquisition and final loss. This audit does not verify '
                     'object identity, displayed pixels, or display padding against image content.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='Directory containing tracks.json and tracked_zoom.mp4; legacy runs are supported')
    parser.add_argument('--report', type=Path, help='Explicit path for an optional diagnostic JSON report; default is stdout only')
    args = parser.parse_args(argv)
    if not args.run.is_dir():
        parser.error(f'Run directory not found: {args.run}')
    if args.report and args.report.resolve() in {
            (args.run/name).resolve() for name in ('tracks.json', 'summary.json', 'tracked.mp4', 'tracked_zoom.mp4')}:
        parser.error('--report must not replace a tracking artifact')
    report = audit_run(args.run)
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(serialized, encoding='utf-8')
    print(serialized)
    return 0 if report['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
