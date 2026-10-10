"""Compare recorded tracker behavior by source frame without rerunning a model."""
import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yolomg_motion.run_artifacts import read_tracking_run


def _index(rows):
    result = {}
    for row in rows:
        index = row.get('frame_index')
        if type(index) is not int or index < 0 or index in result:
            raise ValueError('Run comparison requires unique nonnegative source frame indices')
        result[index] = row
    return result


def compare_runs(left_path, right_path):
    left_rows, left_metadata = read_tracking_run(left_path, strict_finite=True)
    right_rows, right_metadata = read_tracking_run(right_path, strict_finite=True)
    for metadata in (left_metadata, right_metadata):
        if not isinstance(metadata.get('input'), str) or not metadata['input']:
            raise ValueError('Both runs must identify their source video')
    if Path(left_metadata['input']).resolve() != Path(right_metadata['input']).resolve():
        raise ValueError('Runs identify different source videos')
    left, right = _index(left_rows), _index(right_rows)
    shared = sorted(left.keys() & right.keys())
    fields = ('state', 'source', 'track_id', 'bbox', 'detector_ran', 'motion_valid', 'detections')
    changes = []
    counts = dict.fromkeys(fields, 0)
    for index in shared:
        changed = [key for key in fields if left[index].get(key) != right[index].get(key)]
        if changed:
            changes.append(dict(frame_index=index, fields=changed))
            for key in changed:
                counts[key] += 1
    settings = ('weights', 'model_input_size', 'confidence_threshold', 'detect_every',
                'detector_backend', 'motion_backend', 'tracking_backend', 'lock_confidence_threshold',
                'max_detector_gap', 'input_fps')
    differences = {key: dict(left=left_metadata.get(key), right=right_metadata.get(key))
                   for key in settings if left_metadata.get(key) != right_metadata.get(key)}
    return dict(left=str(Path(left_path).resolve()), right=str(Path(right_path).resolve()),
                shared_frames=len(shared), left_only_frames=sorted(left.keys() - right.keys()),
                right_only_frames=sorted(right.keys() - left.keys()),
                changed_frames=len(changes), changed_field_counts=counts, changes=changes,
                settings_differences=differences,
                source_identity='Matched declared source paths; source file bytes were not verified.',
                note='This is an exact comparison of recorded behavior, not an accuracy ranking. '
                     'Identity numbers may be local to each run; differences do not prove identity errors.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--left', type=Path, required=True)
    parser.add_argument('--right', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = compare_runs(args.left, args.right)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
