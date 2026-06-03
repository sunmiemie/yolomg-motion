import pytest

from yolomg_motion.run_artifacts import read_tracking_run


@pytest.mark.parametrize('content', [
    '{"schema_version":2,"metadata":{},"frames":[{"frame_index":2,"frame_index":3}]}',
    '{"schema_version":2,"metadata":{"input":"a","input":"b"},"frames":[]}',
])
def test_duplicate_fields_cannot_silently_change_observations(tmp_path, content):
    (tmp_path / 'tracks.json').write_text(content, encoding='utf-8')
    with pytest.raises(ValueError, match='Duplicate JSON key'):
        read_tracking_run(tmp_path)


def test_duplicate_legacy_metadata_rejected(tmp_path):
    (tmp_path / 'tracks.json').write_text('[]')
    (tmp_path / 'summary.json').write_text('{"input_fps":30,"input_fps":60}')
    with pytest.raises(ValueError, match='input_fps'):
        read_tracking_run(tmp_path)
