"""Download the nine UTIAS MR.CLAM datasets by anonymous FTP and unpack them.

Each archive is unpacked into its own directory, data/raw/ds{N}/, with the
17 .dat files placed directly inside (the archives use MRCLAM_Dataset{N}/,
and MRSLAM_Dataset4/ for dataset 4, so the top folder is flattened).

Usage: python scripts/download_mrclam.py [--datasets 1-9] [--root data/raw]
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

URL = "ftp://asrl3.utias.utoronto.ca/MRCLAM/MRCLAM{n}.zip"
EXPECTED_FILES = 17


def parse_range(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def fetch(url: str, dest: Path, tries: int = 4) -> None:
    for attempt in range(1, tries + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp, open(dest, "wb") as fh:
                shutil.copyfileobj(resp, fh)
            return
        except Exception as exc:  # network errors are retried
            print(f"  attempt {attempt} failed: {exc}", file=sys.stderr)
            time.sleep(3 * attempt)
    raise RuntimeError(f"could not download {url}")


def unpack(zip_path: Path, out_dir: Path) -> int:
    """Extract only plain .dat members, flattened, and return the count."""
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = Path(info.filename).name
            if info.is_dir() or not name.endswith(".dat") or name.startswith("."):
                continue
            with zf.open(info) as src, open(out_dir / name, "wb") as dst:
                shutil.copyfileobj(src, dst)
            count += 1
    return count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="1-9")
    ap.add_argument("--root", default="data/raw")
    args = ap.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    for n in parse_range(args.datasets):
        out_dir = root / f"ds{n}"
        if out_dir.is_dir() and len(list(out_dir.glob("*.dat"))) == EXPECTED_FILES:
            print(f"dataset {n}: present")
            continue
        zip_path = root / f"MRCLAM{n}.zip"
        if not zip_path.exists():
            print(f"dataset {n}: downloading")
            fetch(URL.format(n=n), zip_path)
        count = unpack(zip_path, out_dir)
        status = "ok" if count == EXPECTED_FILES else f"UNEXPECTED file count {count}"
        print(f"dataset {n}: unpacked {count} files, {status}")


if __name__ == "__main__":
    main()
