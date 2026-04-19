from __future__ import annotations

import argparse
import importlib.util
import shutil
from pathlib import Path


DEFAULT_DATASET = "phhasian0710/yoochoose"


def find_single(search_root: Path, pattern: str) -> Path:
    matches = sorted(search_root.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"Could not find required file matching {pattern!r} in {search_root}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download YooChoose dataset artifacts using kagglehub.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Kaggle dataset handle, e.g. owner/dataset")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--copy-buys", action="store_true", help="Also copy yoochoose-buys.dat when available")
    args = parser.parse_args()

    if importlib.util.find_spec("kagglehub") is None:
        raise SystemExit(
            "kagglehub is not installed. Install dependencies first (e.g., `pip install -r requirements.txt`) and retry."
        )

    import kagglehub

    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    downloaded_path = Path(kagglehub.dataset_download(args.dataset))
    print(f"Downloaded dataset directory: {downloaded_path}")

    clicks_src = find_single(downloaded_path, "yoochoose-clicks.dat")
    clicks_dst = raw_dir / "yoochoose-clicks.dat"
    shutil.copy2(clicks_src, clicks_dst)
    print(f"Prepared: {clicks_dst}")

    if args.copy_buys:
        buys_src = find_single(downloaded_path, "yoochoose-buys.dat")
        buys_dst = raw_dir / "yoochoose-buys.dat"
        shutil.copy2(buys_src, buys_dst)
        print(f"Prepared: {buys_dst}")


if __name__ == "__main__":
    main()
