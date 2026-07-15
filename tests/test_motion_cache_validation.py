import json

import numpy as np
import pytest

from yolomg_motion.track_cli import cached_centers


class Capture:
    def __init__(self):
        self.count = 0

    def read(self):
        self.count += 1
        return (True, np.zeros((16, 16, 3), np.uint8)) if self.count <= 5 else (False, None)


def make_cache(tmp_path, **overrides):
    (tmp_path / 'maps').mkdir()
    video = tmp_path / 'input.mp4'
    summary = {'input': str(video), 'first_center_index': 2,
               'last_center_index': 2, 'config': {'frame_step': 2}}
    (tmp_path / 'summary.json').write_text(json.dumps(summary), encoding='utf-8-sig')
    fields = dict(center_index=2, difference=np.zeros((16, 16), np.float32), success=True)
    fields.update(overrides)
    np.savez(tmp_path / 'maps/000002.npz', **fields)
    return video


def test_windows_bom_cache_can_be_replayed(tmp_path):
    video = make_cache(tmp_path)
    rows = list(cached_centers(Capture(), tmp_path, video, max_frames=5))
    assert len(rows) == 1 and rows[0][0] == 2 and rows[0][3] is True


@pytest.mark.parametrize('fields', [
    {'center_index': 2.5}, {'center_index': [2]},
    {'difference': np.full((16, 16), np.nan)},
    {'difference': np.full((16, 16), 256)},
    {'difference': np.ones((16, 16), np.complex64)},
    {'success': [True]}, {'success': 'false'},
])
def test_invalid_cache_values_are_not_silently_coerced(tmp_path, fields):
    video = make_cache(tmp_path, **fields)
    with pytest.raises(ValueError, match='cache'):
        list(cached_centers(Capture(), tmp_path, video, max_frames=5))


@pytest.mark.parametrize('overrides', [
    {'config': None}, {'last_center_index': 2.5}, {'last_center_index': None},
    {'completed': False}, {'partial': True}, {'output_frames': 8},
])
def test_invalid_or_partial_cache_metadata_is_rejected(tmp_path, overrides):
    video = make_cache(tmp_path)
    path = tmp_path / 'summary.json'
    summary = json.loads(path.read_text(encoding='utf-8-sig'))
    summary.update(overrides)
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='cache'):
        list(cached_centers(Capture(), tmp_path, video, max_frames=5))


def test_truncated_source_cannot_complete_cached_replay(tmp_path):
    video = make_cache(tmp_path)
    cap = Capture()
    cap.count = 1  # Only four source frames remain, but center 2 needs frame 4.
    with pytest.raises(ValueError, match='future-frame'):
        list(cached_centers(cap, tmp_path, video, max_frames=5))


def test_short_requested_prefix_still_has_no_fabricated_centers(tmp_path):
    video = make_cache(tmp_path)
    assert list(cached_centers(Capture(), tmp_path, video, max_frames=3)) == []
