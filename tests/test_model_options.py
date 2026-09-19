import pytest

from yolomg_motion.yolo_detector import YOLOMGDetector
from yolomg_motion.directml_detector import DirectMLYOLOMGDetector


@pytest.mark.parametrize('constructor', [YOLOMGDetector, DirectMLYOLOMGDetector])
@pytest.mark.parametrize('options', [dict(image_size=True), dict(image_size=128.5),
    dict(threads=0), dict(threads=1.5), dict(conf_threshold=float('nan')),
    dict(conf_threshold=True), dict(conf_threshold='0.5')])
def test_invalid_options_fail_before_optional_runtime_or_weight_loading(constructor, options):
    with pytest.raises(ValueError):
        constructor('missing-vendor', 'missing-checkpoint.pt', **options)
