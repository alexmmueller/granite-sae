"""Verify tracked source files and, with --weights, downloaded SAE weights."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', action='store_true', help='require and hash all downloaded weight files')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    entries = [line.split('  ', 1) for line in (root / 'SHA256SUMS').read_text().splitlines()]
    for expected, relative in entries:
        if digest(root / relative) != expected:
            raise SystemExit(f'Checksum mismatch: {relative}')
    print(f'Verified {len(entries)} source/documentation files.')
    if args.weights:
        artifacts = json.loads((root / 'manifest.json').read_text())['artifacts']
        for item in artifacts:
            path = root / item['path']
            if not path.is_file():
                raise SystemExit(f'Missing weight: {item["path"]}; see docs/usage.md')
            if path.stat().st_size != item['bytes'] or digest(path) != item['sha256']:
                raise SystemExit(f'Weight mismatch: {item["path"]}')
            print(f'Verified weight: {item["path"]}')


if __name__ == '__main__':
    main()
