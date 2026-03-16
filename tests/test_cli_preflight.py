"""Invalid output arguments must fail before opening a video or model."""
import pytest

from yolomg_motion import cli, track_cli


@pytest.mark.parametrize('entrypoint', [cli.main, track_cli.main])
def test_output_file_is_reported_without_opening_video(tmp_path, monkeypatch, capsys, entrypoint):
    video = tmp_path / 'input.mp4'
    video.touch()
    output = tmp_path / 'result.txt'
    output.write_text('keep me')
    monkeypatch.setattr(cli.cv2, 'VideoCapture', lambda *a: pytest.fail('opened video before validation'))
    with pytest.raises(SystemExit) as exc:
        entrypoint(['--input', str(video), '--output', str(output)])
    assert exc.value.code == 2
    assert 'directory' in capsys.readouterr().err.lower()
    assert output.read_text() == 'keep me'


@pytest.mark.parametrize('gain', ['nan', 'inf', '-inf'])
def test_preview_gain_must_be_finite(tmp_path, monkeypatch, capsys, gain):
    video = tmp_path / 'input.mp4'
    video.touch()
    monkeypatch.setattr(cli.cv2, 'VideoCapture', lambda *a: pytest.fail('opened video before validation'))
    with pytest.raises(SystemExit) as exc:
        cli.main(['--input', str(video), '--output', str(tmp_path / 'out'), '--preview-gain=' + gain])
    assert exc.value.code == 2
    assert 'preview gain' in capsys.readouterr().err.lower()


def test_image_size_is_validated_before_model_or_cache_loading(tmp_path, capsys):
    video = tmp_path / 'input.mp4'
    video.touch()
    with pytest.raises(SystemExit) as exc:
        track_cli.main(['--input', str(video), '--output', str(tmp_path / 'out'),
                        '--detection-cache', str(tmp_path / 'absent'), '--image-size', '0'])
    assert exc.value.code == 2
    assert 'image-size' in capsys.readouterr().err
