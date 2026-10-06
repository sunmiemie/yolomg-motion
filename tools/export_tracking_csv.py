"""Export recorded tracking observations to a spreadsheet-friendly CSV."""
import argparse
import csv
import io
import math
import os
from pathlib import Path
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yolomg_motion.run_artifacts import read_tracking_run


FIELDS = ('frame_index', 'timestamp_seconds', 'state', 'track_id', 'source',
          'x1', 'y1', 'x2', 'y2', 'detector_confidence', 'appearance_score',
          'detector_ran', 'motion_valid')


def render_csv(rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=FIELDS)
    writer.writeheader()
    previous = -1
    for row in rows:
        index = row.get('frame_index')
        if type(index) is not int or index <= previous:
            raise ValueError('CSV export requires increasing nonnegative frame indices')
        previous = index
        if row.get('state') not in ('SEARCHING', 'CONFIRMING', 'LOCKED', 'LOST'):
            raise ValueError('CSV export encountered an unknown tracker state')
        if row.get('source') not in ('none', 'detector', 'template'):
            raise ValueError('CSV export encountered an unknown tracking source')
        for key in ('timestamp_seconds', 'detector_confidence', 'appearance_score'):
            value = row.get(key)
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
                raise ValueError(f'CSV export requires numeric {key}')
        if row.get('track_id') is not None and (type(row['track_id']) is not int or row['track_id'] < 0):
            raise ValueError('CSV export requires a numeric track identity')
        box = row.get('bbox')
        if box is not None and (not isinstance(box, list) or len(box) != 4
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in box)):
            raise ValueError('CSV export requires four finite bbox coordinates')
        record = {key: row.get(key) for key in FIELDS if key not in ('x1', 'y1', 'x2', 'y2')}
        record.update(zip(('x1', 'y1', 'x2', 'y2'), box if box is not None else ('',) * 4))
        for key in ('detector_ran', 'motion_valid'):
            if type(row.get(key)) is not bool:
                raise ValueError(f'CSV export requires boolean {key}')
            record[key] = int(row[key])
        writer.writerow(record)
    return stream.getvalue()


def export_csv(run, output):
    rows, _ = read_tracking_run(run, strict_finite=True)
    serialized = render_csv(rows)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix='.tracking-csv-', suffix='.tmp',
                                         encoding='utf-8-sig', mode='w', newline='', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return len(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        count = export_csv(args.run, args.output)
    except (OSError, ValueError, OverflowError) as error:
        parser.error(str(error))
    print(f'Exported {count} recorded frames to {args.output}; these are observations, not accuracy scores.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
