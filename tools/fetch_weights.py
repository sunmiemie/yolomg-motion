"""Download the published checkpoint at a pinned revision and verify SHA-256."""
import hashlib
import os
from pathlib import Path
import tempfile
import urllib.request


REVISION = '090a74cd3ece15c66e857cc2e08e01cb103c4550'
RELATIVE = 'runs/train/ARD100_mask32-1280_uavs/weights/best.pt'
EXPECTED_SHA256 = 'fb28a4063f0e935ce419db47cf6ff26c443d12940eb727522a95644adba34bef'


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_weights(target):
    target = Path(target)
    if target.exists():
        if file_sha256(target) != EXPECTED_SHA256:
            raise RuntimeError('Existing checkpoint differs from the pinned weights; refusing to overwrite')
        print('Verified existing checkpoint:', target)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f'https://raw.githubusercontent.com/Irisky123/YOLOMG/{REVISION}/{RELATIVE}'
    temporary = None
    try:
        digest = hashlib.sha256()
        with tempfile.NamedTemporaryFile(dir=target.parent, suffix='.download', delete=False) as output:
            temporary = Path(output.name)
            with urllib.request.urlopen(url, timeout=60) as response:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                    output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        if digest.hexdigest() != EXPECTED_SHA256:
            raise RuntimeError('Downloaded checkpoint SHA-256 does not match the published source record')
        # Hard-link publication is atomic and refuses a target created by
        # another process while this download was in flight.
        os.link(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print('Downloaded and verified:', target)


def main():
    fetch_weights(Path(__file__).resolve().parents[1] / 'vendor' / 'YOLOMG' / RELATIVE)


if __name__ == '__main__':
    main()
