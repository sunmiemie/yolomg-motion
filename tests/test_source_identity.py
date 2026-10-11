import pytest

from yolomg_motion.run_artifacts import source_file_identity, validate_source_identity


def test_replaced_video_at_same_path_invalidates_new_cache(tmp_path):
    source = tmp_path / 'source.mp4'
    source.write_bytes(b'original')
    metadata = dict(source_file_identity=source_file_identity(source))
    validate_source_identity(metadata, source)
    source.write_bytes(b'replacement video content')
    with pytest.raises(ValueError, match='changed'):
        validate_source_identity(metadata, source)


def test_legacy_cache_does_not_claim_a_source_identity(tmp_path):
    validate_source_identity({}, tmp_path / 'legacy-missing-video')


@pytest.mark.parametrize('identity', [[], {}, {'size_bytes': True, 'mtime_ns': 2}])
def test_malformed_identity_cannot_bypass_freshness_check(tmp_path, identity):
    with pytest.raises(ValueError, match='malformed'):
        validate_source_identity(dict(source_file_identity=identity), tmp_path / 'missing')
