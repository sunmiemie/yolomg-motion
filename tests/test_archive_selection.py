import io
from zipfile import ZipFile

import pytest

from tools.fetch_ard100_archive import plan_members


def make_archive(names):
    stream = io.BytesIO()
    with ZipFile(stream, 'w') as archive:
        for name in names:
            archive.writestr(name, b'example')
    return ZipFile(stream)


@pytest.mark.parametrize('name', ['../escape', '/root/file', 'C:/file', 'a\\b', 'a/../../b'])
def test_unsafe_members_rejected_before_writing(tmp_path, name):
    with make_archive(['valid', name]) as archive, pytest.raises(ValueError):
        plan_members(archive, tmp_path, ['valid', name])
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('names', [['a', 'a'], ['a', 'A'], ['a', 'a/file'], ['a', 'a.partial']])
def test_destination_collisions_rejected(tmp_path, names):
    with make_archive(set(names)) as archive, pytest.raises(ValueError):
        plan_members(archive, tmp_path, names)


def test_existing_later_file_prevents_earlier_download(tmp_path):
    (tmp_path / 'existing').write_bytes(b'preserve')
    with make_archive(['first', 'existing']) as archive, pytest.raises(FileExistsError):
        plan_members(archive, tmp_path, ['first', 'existing'])
    assert not (tmp_path / 'first').exists()


def test_valid_nested_members_are_planned_without_side_effects(tmp_path):
    with make_archive(['folder/video.mp4']) as archive:
        plan = plan_members(archive, tmp_path, ['folder/video.mp4'])
    assert plan[0][1] == tmp_path / 'folder/video.mp4'
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('name', ['dir/file:stream', 'dir/NUL.txt', 'dir/COM1', 'dir/trailing.',
                                 'dir/trailing ', 'dir/LPT².txt', 'dir/wild?card', 'dir/CONIN$'])
def test_windows_aliases_and_device_paths_are_rejected(tmp_path, name):
    with make_archive([name]) as archive, pytest.raises(ValueError, match='Windows'):
        plan_members(archive, tmp_path, [name])
    assert not list(tmp_path.iterdir())
