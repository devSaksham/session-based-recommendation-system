from __future__ import annotations

import argparse
import re
import shutil
import zipfile
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from urllib.request import urlopen


DEFAULT_DRIVE_URL = "https://drive.google.com/file/d/1c5s1ugm-6-xJvLpj5_ibjVkNsGRhJGzO/view?usp=drive_link"


def extract_file_id(google_drive_url: str) -> str:
    parsed = urlparse(google_drive_url)
    if "id" in parse_qs(parsed.query):
        return parse_qs(parsed.query)["id"][0]

    match = re.search(r"/d/([a-zA-Z0-9_-]+)", parsed.path)
    if match is None:
        raise ValueError(f"Could not extract Google Drive file id from URL: {google_drive_url}")
    return match.group(1)


def build_download_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def download_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as response, output_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def maybe_extract_zip(archive_path: Path, extract_dir: Path) -> Path:
    if zipfile.is_zipfile(archive_path):
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(extract_dir)
        return extract_dir
    return archive_path.parent


def find_required_files(search_root: Path) -> tuple[Path, Path]:
    clicks = sorted(search_root.rglob("yoochoose-clicks.dat"))
    buys = sorted(search_root.rglob("yoochoose-buys.dat"))
    if not clicks or not buys:
        raise FileNotFoundError(
            "Could not find yoochoose-clicks.dat and yoochoose-buys.dat after download. "
            "Please check the Drive file content."
        )
    return clicks[0], buys[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download YooChoose dataset artifacts from a Google Drive link.")
    parser.add_argument("--drive-url", default=DEFAULT_DRIVE_URL)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--archive-name", default="yoochoose_source.zip")
    parser.add_argument("--keep-archive", action="store_true")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    file_id = extract_file_id(args.drive_url)
    direct_url = build_download_url(file_id)

    downloads_dir = raw_dir / "downloads"
    archive_path = downloads_dir / args.archive_name
    print(f"Downloading from: {direct_url}")
    download_file(direct_url, archive_path)

    extracted_root = maybe_extract_zip(archive_path, downloads_dir / "extracted")
    clicks_path, buys_path = find_required_files(extracted_root)

    final_clicks = raw_dir / "yoochoose-clicks.dat"
    final_buys = raw_dir / "yoochoose-buys.dat"
    shutil.copy2(clicks_path, final_clicks)
    shutil.copy2(buys_path, final_buys)

    if not args.keep_archive and archive_path.exists():
        archive_path.unlink()

    print(f"Prepared: {final_clicks}")
    print(f"Prepared: {final_buys}")


if __name__ == "__main__":
    main()
