"""Download the published checkpoint at a pinned revision and verify SHA-256."""
import hashlib
from pathlib import Path
import tempfile
import urllib.request


REVISION = '090a74cd3ece15c66e857cc2e08e01cb103c4550'
RELATIVE = 'runs/train/ARD100_mask32-1280_uavs/weights/best.pt'
EXPECTED_SHA256 = 'fb28a4063f0e935ce419db47cf6ff26c443d12940eb727522a95644adba34bef'


def main():
    target = Path(__file__).resolve().parents[1] / 'vendor' / 'YOLOMG' / RELATIVE
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() != EXPECTED_SHA256:
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
        if digest.hexdigest() != EXPECTED_SHA256:
            raise RuntimeError('Downloaded checkpoint SHA-256 does not match the published source record')
        temporary.replace(target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print('Downloaded and verified:', target)


if __name__ == '__main__':
    main()
