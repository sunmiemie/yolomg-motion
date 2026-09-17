"""Evaluate an existing track run against ARD100 annotations, never feed GT back.

First run --preview-only and visually compare the green annotation with the
source video. After that inspection, pass --alignment-verified for evaluation.
All artifacts are written under RUN/ground_truth_comparison.
"""
import argparse
import json
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yolomg_motion.run_artifacts import read_tracking_run


def box_iou(first, second):
    a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    intersection = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    area_a = np.maximum(0, a[2:] - a[:2]).prod()
    area_b = np.maximum(0, b[2:] - b[:2]).prod()
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0 else 0.0


def read_annotation(archive, sequence, frame_index, offset=-1.0):
    """Map zero-based source index to one-based VOC name; absence is None."""
    member = f'annotations/{sequence}/{sequence}_{frame_index + 1:04d}.xml'
    try:
        payload = archive.read(member)
    except KeyError:
        return None
    root = ET.fromstring(payload)
    width, height = int(root.findtext('size/width')), int(root.findtext('size/height'))
    boxes, ignored = [], 0
    for item in root.findall('object'):
        if item.findtext('name', '').strip().lower() != 'drone':
            continue
        if int(item.findtext('difficult', '0')) != 0:
            ignored += 1
            continue
        box = tuple(float(item.findtext('bndbox/' + key)) + offset
                    for key in ('xmin', 'ymin', 'xmax', 'ymax'))
        if not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(f'Invalid Drone box in {member}: {box}')
        boxes.append(box)
    return dict(member=member, width=width, height=height,
                boxes=boxes, difficult_drone_objects_ignored=ignored)


def match_detections(detections, ground_truth, confidence=0.1, iou_threshold=0.5):
    """Confidence-ordered greedy one-to-one matches at one operating point."""
    eligible = [d for d in (detections or []) if float(d['confidence']) >= confidence]
    eligible.sort(key=lambda d: float(d['confidence']), reverse=True)
    unmatched = set(range(len(ground_truth)))
    matches, false_positives = [], 0
    for detection in eligible:
        overlaps = [(box_iou(detection['bbox'], ground_truth[index]), index)
                    for index in unmatched]
        best_iou, best_index = max(overlaps, default=(0.0, None))
        if best_index is not None and best_iou >= iou_threshold:
            unmatched.remove(best_index)
            matches.append(dict(gt_index=best_index, iou=best_iou,
                                confidence=float(detection['confidence'])))
        else:
            false_positives += 1
    return dict(tp=len(matches), fp=false_positives, fn=len(unmatched),
                eligible_predictions=len(eligible), matches=matches)


def evaluate_rows(rows, annotations, confidence=0.1, iou_threshold=0.5):
    """Exclude missing XML from all GT-based denominators and detection counts."""
    counts = dict(run_frames=len(rows), annotated_frames=0, missing_annotations=0,
                  gt_positive_frames=0, gt_negative_frames=0, gt_objects=0,
                  difficult_drone_objects_ignored=0, locked_correct_frames=0,
                  locked_wrong_frames=0, locked_wrong_on_gt_frames=0,
                  locked_on_negative_frames=0, missed_tracking_frames=0,
                  gt_frames_without_correct_lock=0, detector_evaluated_frames=0,
                  detector_gt_objects=0, detector_tp=0, detector_fp=0, detector_fn=0)
    per_frame, iou_sum = [], 0.0
    for row in rows:
        index = row['frame_index']
        annotation = annotations.get(index)
        if annotation is None:
            counts['missing_annotations'] += 1
            per_frame.append(dict(frame_index=index, evaluation_status='missing_annotation'))
            continue
        boxes = annotation['boxes']
        counts['annotated_frames'] += 1
        counts['gt_positive_frames' if boxes else 'gt_negative_frames'] += 1
        counts['gt_objects'] += len(boxes)
        counts['difficult_drone_objects_ignored'] += annotation['difficult_drone_objects_ignored']
        prediction = row.get('bbox') if row.get('state') == 'LOCKED' else None
        best_iou = max((box_iou(prediction, gt) for gt in boxes), default=0.0) if prediction is not None else 0.0
        correct = prediction is not None and bool(boxes) and best_iou >= iou_threshold
        if prediction is not None:
            counts['locked_correct_frames' if correct else 'locked_wrong_frames'] += 1
            if not correct:
                counts['locked_wrong_on_gt_frames' if boxes else 'locked_on_negative_frames'] += 1
        elif boxes:
            counts['missed_tracking_frames'] += 1
        if boxes:
            iou_sum += best_iou
            counts['gt_frames_without_correct_lock'] += int(not correct)
        record = dict(frame_index=index, annotation_member=annotation['member'],
                      evaluation_status='evaluated', gt_boxes=boxes,
                      locked_bbox=prediction, best_locked_iou=best_iou,
                      correct_locked_frame=bool(correct))
        if row.get('detector_ran'):
            if row.get('detections') is None:
                raise ValueError(f'detector_ran is true but detections is null at frame {index}')
            detection = match_detections(row['detections'], boxes, confidence, iou_threshold)
            counts['detector_evaluated_frames'] += 1
            counts['detector_gt_objects'] += len(boxes)
            for key in ('tp', 'fp', 'fn'):
                counts['detector_' + key] += detection[key]
            record['detection'] = detection
        per_frame.append(record)
    counts['mean_iou_gt_frames_including_unlocked_zero'] = (
        iou_sum / counts['gt_positive_frames'] if counts['gt_positive_frames'] else None)
    tp, fp, fn = (counts['detector_' + key] for key in ('tp', 'fp', 'fn'))
    counts['detector_precision'] = tp / (tp + fp) if tp + fp else None
    counts['detector_recall'] = tp / (tp + fn) if tp + fn else None
    counts['correct_lock_fraction_of_gt_frames'] = (
        counts['locked_correct_frames'] / counts['gt_positive_frames'] if counts['gt_positive_frames'] else None)
    return counts, per_frame


def draw_comparison(frame, row, annotation):
    """Ground-truth-centered crop is for display, not target acquisition."""
    canvas = frame.copy()
    h, w = frame.shape[:2]
    green, yellow = (40, 230, 40), (0, 230, 255)
    boxes = annotation['boxes'] if annotation is not None else []
    prediction = row.get('bbox') if row.get('state') == 'LOCKED' else None

    def rectangle(image, box, color, origin=(0, 0), scale=1):
        x1, y1, x2, y2 = box
        first = (round((x1-origin[0])*scale), round((y1-origin[1])*scale))
        second = (round((x2-origin[0])*scale), round((y2-origin[1])*scale))
        cv2.rectangle(image, first, second, color, 2, cv2.LINE_AA)

    for box in boxes:
        rectangle(canvas, box, green)
    if prediction is not None:
        rectangle(canvas, prediction, yellow)
    cv2.rectangle(canvas, (10, 10), (min(w-10, 980), 112), (20, 20, 20), -1)
    lines = [f'SOURCE FRAME {row["frame_index"]} | STATE {row.get("state", "unknown")}',
             'GREEN: official GT  |  YELLOW: current LOCKED prediction',
             'GT is used for evaluation/display ONLY; never fed to detector/tracker']
    if annotation is None:
        lines[0] += ' | ANNOTATION MISSING: excluded'
    for y, line in zip((34, 63, 92), lines):
        cv2.putText(canvas, line, (22, y), cv2.FONT_HERSHEY_SIMPLEX,
                    .63 if w >= 1280 else .43, (240, 240, 240), 1, cv2.LINE_AA)
    focus = boxes[0] if boxes else prediction
    if focus is not None and w >= 640 and h >= 360:
        cx, cy = (focus[0]+focus[2])/2, (focus[1]+focus[3])/2
        cw, ch = min(160, w), min(100, h)
        left, top = int(np.clip(round(cx-cw/2), 0, w-cw)), int(np.clip(round(cy-ch/2), 0, h-ch))
        scale = min(3, (w-40)//cw, (h-160)//ch)
        if scale >= 1:
            crop = cv2.resize(frame[top:top+ch, left:left+cw], (cw*scale, ch*scale), interpolation=cv2.INTER_NEAREST)
            for box in boxes:
                rectangle(crop, box, green, (left, top), scale)
            if prediction is not None:
                rectangle(crop, prediction, yellow, (left, top), scale)
            ph, pw = crop.shape[:2]
            ox, oy = w-pw-20, 152
            canvas[oy:oy+ph, ox:ox+pw] = crop
            title = 'GT-CENTERED ZOOM: DISPLAY ONLY' if boxes else 'PREDICTION-CENTERED ZOOM'
            cv2.rectangle(canvas, (ox, oy-30), (ox+pw, oy), (20, 20, 20), -1)
            cv2.putText(canvas, title, (ox+5, oy-10), cv2.FONT_HERSHEY_SIMPLEX, .48, (240, 240, 240), 1, cv2.LINE_AA)
    return canvas


def write_png(path, image):
    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        raise RuntimeError(f'Cannot encode PNG: {path}')
    encoded.tofile(str(path))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--sequence', required=True)
    parser.add_argument('--annotation-offset', type=float, default=-1.0)
    parser.add_argument('--snapshot-every', type=int, default=120)
    parser.add_argument('--preview-only', action='store_true')
    parser.add_argument('--alignment-verified', action='store_true',
                        help='Record that the preview was visually checked against the source video')
    args = parser.parse_args(argv)
    if not args.preview_only and not args.alignment_verified:
        parser.error('First use --preview-only, inspect GT alignment, then pass --alignment-verified')
    if args.snapshot_every < 1 or not np.isfinite(args.annotation_offset):
        parser.error('Invalid snapshot interval or annotation offset')
    try:
        rows, run_summary = read_tracking_run(args.run, strict_finite=True)
    except (OSError, ValueError) as exc:
        parser.error(f'Cannot read tracking run: {exc}')
    indices = [row['frame_index'] for row in rows]
    if not indices or indices != sorted(set(indices)) or indices[0] < 0:
        parser.error('tracks.json needs nonempty, unique, increasing source frame indices')
    source = Path(run_summary['input'])
    if source.stem != args.sequence:
        parser.error('Sequence must match the source video filename; do not evaluate with another video\'s labels')
    if run_summary.get('output_frames') != len(rows):
        parser.error('Run metadata output_frames does not match tracking frame count')
    output = args.run/'ground_truth_comparison'
    output.mkdir(parents=True, exist_ok=True)
    if not args.preview_only and any((output/name).exists() for name in ('comparison.mp4', 'gt_evaluation.json')):
        parser.error('Ground-truth evaluation output already exists; choose a fresh run copy')
    with ZipFile(args.annotations) as archive:
        annotations = {index: read_annotation(archive, args.sequence, index, args.annotation_offset)
                       for index in indices}
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        parser.error(f'Cannot open source video: {source}')
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        cap.release()
        parser.error('Source video needs valid FPS')
    mapping = {row['frame_index']: row for row in rows}
    writer, visited, index = None, 0, 0
    try:
        while index <= indices[-1]:
            ok, frame = cap.read()
            if not ok:
                break
            if index in mapping:
                row, annotation = mapping[index], annotations[index]
                if annotation is not None and frame.shape[:2] != (annotation['height'], annotation['width']):
                    raise ValueError(f'Frame {index}: video {frame.shape[:2]} does not match XML dimensions {(annotation["height"],annotation["width"])}')
                if args.preview_only:
                    if annotation is not None and annotation['boxes']:
                        path = output/f'gt_preview_{index:06d}.png'
                        write_png(path, draw_comparison(frame, row, annotation))
                        print(json.dumps(dict(preview=str(path), source=str(source),
                            frame_index=index, annotation_member=annotation['member'],
                            annotation_offset=args.annotation_offset, gt_boxes=annotation['boxes'],
                            instruction='Visually check green GT against the source target before evaluation.'), indent=2))
                        return 0
                else:
                    panel = draw_comparison(frame, row, annotation)
                    if writer is None:
                        h,w = panel.shape[:2]
                        writer = cv2.VideoWriter(str(output/'comparison.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))
                        if not writer.isOpened():
                            raise RuntimeError('Cannot open comparison MP4 writer')
                    writer.write(panel)
                    if visited % args.snapshot_every == 0:
                        write_png(output/f'comparison_{index:06d}.png', panel)
                visited += 1
            index += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if args.preview_only:
        parser.error('No annotated Drone frame was available for visual verification')
    if visited != len(rows):
        raise RuntimeError(f'Video ended before all run frames: decoded {visited}/{len(rows)} evaluated positions')
    metrics, frame_evaluations = evaluate_rows(rows, annotations)
    report = dict(sequence=args.sequence, source_video=str(source.resolve()),
        run=str(args.run.resolve()), annotation_zip=str(args.annotations.resolve()),
        annotation_offset=args.annotation_offset, annotation_frame_mapping='XML number = source frame_index + 1',
        coordinate_convention='Subtract/add annotation_offset to all four VOC coordinates; continuous xyxy IoU, no inclusive +1.',
        alignment_visually_verified=bool(args.alignment_verified), video_fps=fps,
        prediction_source='Existing tracks.json only; GT never used to select, initialize, or update predictions.',
        iou_threshold=0.5, detection_confidence_threshold=0.1,
        matching='Confidence-ordered greedy one-to-one IoU matching on detector_ran frames only.',
        difficult_policy='Only Drone objects with difficult=0 are scored; ignored object count is reported.',
        missing_annotation_policy='Missing XML excluded from GT metrics and detection TP/FP/FN; missing count reported.',
        tracking_definitions='Correct: LOCKED with IoU>=0.5 against any eligible GT. Wrong: LOCKED without a match. Missed: eligible GT but no LOCKED box; wrong and missed are disjoint. Mean IoU uses all GT-positive frames, with unlocked frames scored zero.',
        limitations='Fixed-threshold precision/recall, NOT AP. This evaluates only the recorded run frames of one sequence, NOT the full 35-sequence test benchmark. GT-centered zoom is for display only.',
        metrics=metrics, per_frame=frame_evaluations)
    (output/'gt_evaluation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(output=str(output), metrics=metrics), ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
