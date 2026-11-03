#!/usr/bin/env python3
"""Download the three official mini-ESM weights atomically before GPU work."""
import argparse
import concurrent.futures
import os
from pathlib import Path
import urllib.request

NAMES = ('protenix_mini_esm_v0.5.0', 'esm2_t36_3B_UR50D',
         'esm2_t36_3B_UR50D-contact-regression')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint-dir', required=True)
    args = p.parse_args()
    from protenix.web_service.dependency_url import URL
    root = Path(args.checkpoint_dir)
    root.mkdir(parents=True, exist_ok=True)
    def download(name):
        target = root / (name + '.pt')
        if target.exists():
            print(f'Present: {target} ({target.stat().st_size} bytes)', flush=True)
            return
        temporary = target.with_suffix('.pt.download')
        print(f'Downloading {URL[name]}', flush=True)
        request = urllib.request.Request(URL[name])
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as f:
            expected = response.headers.get('Content-Length')
            while chunk := response.read(8 * 1024 * 1024):
                f.write(chunk)
        if expected and temporary.stat().st_size != int(expected):
            raise RuntimeError(f'Incomplete download: {name}')
        os.replace(temporary, target)
        print(f'Ready: {target} ({target.stat().st_size} bytes)', flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(download, NAMES))

if __name__ == '__main__':
    main()
