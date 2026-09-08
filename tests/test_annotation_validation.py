import io
from zipfile import ZipFile

import pytest

from tools.compare_ard100_ground_truth import read_annotation


def archive_with(xml):
    buffer = io.BytesIO()
    with ZipFile(buffer, 'w') as archive:
        archive.writestr('annotations/demo/demo_0001.xml', xml)
    return ZipFile(buffer)


@pytest.mark.parametrize('xml', ['<annotation>', '<annotation/>',
    '<annotation><size><width>0</width><height>10</height></size></annotation>',
    '<annotation><size><width>10</width><height>10</height></size>'
    '<object><name>Drone</name><bndbox><xmin>1</xmin></bndbox></object></annotation>'])
def test_invalid_xml_reports_exact_member(xml):
    with archive_with(xml) as archive, pytest.raises(ValueError, match='demo_0001.xml'):
        read_annotation(archive, 'demo', 0)


@pytest.mark.parametrize('sequence,index,offset', [('demo', True, 0), ('../demo', 0, 0),
    ('demo', -1, 0), ('demo', 0, float('nan'))])
def test_invalid_mapping_is_rejected(sequence, index, offset):
    with archive_with('<annotation/>') as archive, pytest.raises(ValueError):
        read_annotation(archive, sequence, index, offset)
