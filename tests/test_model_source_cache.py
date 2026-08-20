from yolomg_motion.directml_detector import _vendor_source_digest


def test_cache_fingerprint_covers_nested_utilities_and_file_names(tmp_path):
    (tmp_path / 'models').mkdir()
    (tmp_path / 'utils' / 'nested').mkdir(parents=True)
    (tmp_path / 'models' / 'yolo.py').write_text('model = 1')
    utility = tmp_path / 'utils' / 'nested' / 'activation.py'
    utility.write_text('activation = 1')
    initial = _vendor_source_digest(tmp_path)
    utility.write_text('activation = 2')
    changed = _vendor_source_digest(tmp_path)
    assert changed != initial
    utility.rename(utility.with_name('other.py'))
    renamed = _vendor_source_digest(tmp_path)
    assert renamed != changed
    (tmp_path / 'README.md').write_text('unrelated documentation')
    assert _vendor_source_digest(tmp_path) == renamed


def test_fingerprint_is_independent_of_checkout_location(tmp_path):
    for directory in ('one', 'two'):
        root = tmp_path / directory / 'models'
        root.mkdir(parents=True)
        (root / 'yolo.py').write_text('same source')
    assert _vendor_source_digest(tmp_path / 'one') == _vendor_source_digest(tmp_path / 'two')
