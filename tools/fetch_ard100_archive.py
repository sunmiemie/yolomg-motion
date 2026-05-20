"""Inspect an ARD100 public mirror with HTTP ranges; download selected members."""
import argparse
import io
import json
import operator
from pathlib import Path
import zipfile

import requests


RECORD = 'https://zenodo.org/api/records/15870538'


class HTTPRangeFile(io.RawIOBase):
    def __init__(self, url, size):
        super().__init__()
        size = operator.index(size)
        if size < 0:
            raise ValueError('Archive size must be nonnegative')
        self.url, self.size, self.position = url, size, 0
        self.session = requests.Session()
        self._validator = None

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        self._checkClosed()
        return self.position

    def seek(self, offset, whence=0):
        self._checkClosed()
        offset, whence = operator.index(offset), operator.index(whence)
        if whence not in (0, 1, 2):
            raise ValueError('Invalid seek origin')
        position = offset if whence == 0 else self.position+offset if whence == 1 else self.size+offset
        if position < 0:
            raise ValueError('Cannot seek before the archive')
        self.position = position
        return position

    def read(self, size=-1):
        self._checkClosed()
        size = -1 if size is None else operator.index(size)
        size = min(self.size-self.position, size if size >= 0 else self.size-self.position)
        if size <= 0:
            return b''
        start, end = self.position, self.position+size-1
        headers = {'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'}
        if self._validator is not None:
            headers['If-Range'] = self._validator[1]
        with self.session.get(self.url, headers=headers,
                              stream=True, timeout=(20, 90)) as response:
            response.raise_for_status()
            expected = f'bytes {start}-{end}/{self.size}'
            if response.status_code != 206 or response.headers.get('Content-Range') != expected:
                raise IOError(f'Expected exact partial response {expected}, got {response.status_code} {response.headers.get("Content-Range")}')
            if response.headers.get('Content-Encoding', 'identity').lower() != 'identity':
                raise IOError('Partial archive response must not be content-encoded')
            etag = response.headers.get('ETag')
            validator = (('ETag', etag) if etag and not etag.startswith('W/') else
                         ('Last-Modified', response.headers['Last-Modified'])
                         if response.headers.get('Last-Modified') else None)
            if self._validator is not None and validator != self._validator:
                raise IOError('Remote archive changed between range reads')
            content = response.raw.read(size+1)
        if len(content) != size:
            raise IOError('Partial download was truncated')
        self.position += size
        self._validator = validator
        return content

    def close(self):
        self.session.close()
        super().close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--member', action='append', default=[])
    args = parser.parse_args()
    response = requests.get(RECORD, timeout=30)
    response.raise_for_status()
    record = response.json()
    metadata = next(item for item in record['files'] if item['key'] == 'ARD100.zip')
    args.output.mkdir(parents=True, exist_ok=True)
    with HTTPRangeFile(metadata['links']['self'], metadata['size']) as remote, zipfile.ZipFile(remote) as archive:
        members = [dict(name=i.filename, bytes=i.file_size, compressed_bytes=i.compress_size,
                        crc32=f'{i.CRC:08x}') for i in archive.infolist()]
        manifest = dict(record=RECORD, archive=metadata, source_note='Public Zenodo mirror uploaded by Yu-Hsi Chen; official author download is Baidu. Archive contents are checked against the official sequence split; not a claim of bytewise verification against Baidu.', members=members)
        (args.output/'archive_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'Archive contains {len(members)} entries; full metadata saved to archive_manifest.json', flush=True)
        for name in args.member:
            info = archive.getinfo(name)
            destination = (args.output/name).resolve()
            destination.relative_to(args.output.resolve())
            if info.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            if destination.exists():
                raise FileExistsError(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            partial = destination.with_name(destination.name+'.partial')
            if partial.exists():
                raise FileExistsError(partial)
            print(f'Downloading {name}: {info.file_size:,} bytes', flush=True)
            # zipfile verifies the selected member's CRC at EOF. A failed
            # download remains .partial and is never presented as complete.
            with archive.open(info) as source, partial.open('xb') as target:
                copied, reported = 0, 0
                while True:
                    chunk = source.read(8*1024*1024)
                    if not chunk:
                        break
                    target.write(chunk)
                    copied += len(chunk)
                    if copied-reported >= 32*1024*1024:
                        print(f'{name}: {copied:,}/{info.file_size:,} bytes', flush=True)
                        reported = copied
            partial.rename(destination)
            print(f'Verified member CRC and saved {destination}', flush=True)


if __name__ == '__main__':
    main()
