import numpy as np
import pytest

from yolomg_motion.core import MotionConfig, MotionExtractor


def test_failed_preprocessing_does_not_start_a_sequence(monkeypatch):
    extractor = MotionExtractor(MotionConfig(frame_step=1))
    prepare = extractor._prepare
    monkeypatch.setattr(extractor, '_prepare', lambda frame: (_ for _ in ()).throw(RuntimeError('device failed')))
    with pytest.raises(RuntimeError, match='device failed'):
        extractor.push(np.zeros((32, 32, 3), np.uint8))
    assert extractor._received == 0 and not extractor._frames and extractor._shape is None
    monkeypatch.setattr(extractor, '_prepare', prepare)
    assert extractor.push(np.zeros((48, 48, 3), np.uint8)) is None
    assert extractor._received == 1
