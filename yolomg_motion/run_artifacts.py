"""Read old/new tracking runs and atomically write the compact v2 artifact."""
import json
import math
import os
from pathlib import Path
import tempfile


def _check_finite(value, location='run'):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f'Non-finite numeric value at {location}')
    if isinstance(value, dict):
        for key, item in value.items():
            _check_finite(item, f'{location}.{key}')
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_finite(item, f'{location}[{index}]')


def _validate_parts(rows, metadata):
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError('Tracking frames must be a list of objects')
    if not isinstance(metadata, dict):
        raise ValueError('Tracking metadata must be an object')


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key in tracking artifact: {key}')
        result[key] = value
    return result


def _read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'), object_pairs_hook=_unique_object)


def read_tracking_run(run_dir, *, strict_finite=False):
    """Return ``(rows, metadata)`` for v2 or legacy tracks+summary runs.

    UTF-8 BOMs are accepted in both formats. ``strict_finite=True`` additionally
    rejects NaN/Infinity anywhere, including overflowing JSON numeric literals.
    Schema and container shape are always checked; no frame data are rewritten.
    """
    run_dir = Path(run_dir)
    payload = _read_json(run_dir/'tracks.json')
    if isinstance(payload, list):
        rows = payload
        metadata = _read_json(run_dir/'summary.json')
    elif isinstance(payload, dict) and payload.get('schema_version') == 2:
        rows, metadata = payload.get('frames'), payload.get('metadata')
    else:
        raise ValueError('Unsupported tracks.json format; expected schema_version 2 or legacy frame array')
    _validate_parts(rows, metadata)
    if strict_finite:
        _check_finite(rows, 'frames')
        _check_finite(metadata, 'metadata')
    return rows, metadata


def _json_default(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    if hasattr(value, 'item'):
        return value.item()
    raise TypeError(f'Cannot serialize {type(value).__name__}')


def write_tracking_run(run_dir, rows, metadata):
    """Atomically replace tracks.json with one finite, UTF-8 v2 document.

    Serialize before touching the destination. A failed serialization or file
    replacement leaves an existing artifact intact and removes its temporary
    file. No summary.json or other permanent artifact is produced.
    """
    _validate_parts(rows, metadata)
    payload = dict(schema_version=2, metadata=metadata, frames=rows)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2,
                            allow_nan=False, default=_json_default)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n',
                                         prefix='.tracks.', suffix='.tmp',
                                         dir=str(run_dir), delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(run_dir/'tracks.json'))
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
