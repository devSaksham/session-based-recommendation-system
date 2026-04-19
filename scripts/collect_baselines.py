from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.utils.io import ensure_dir


def load_baselines() -> list[dict]:
    asset_path = Path("src/research/literature_baselines.json")
    with asset_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(rows: list[dict], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    headers = [
        "model_name",
        "reported_metric_name",
        "reported_value",
        "mrr_at_20",
        "dataset_variant",
        "citation_key",
    ]
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("| Model | Metric Name | Metric Value | MRR@20 | Dataset | Citation |\n")
        handle.write("| --- | --- | ---: | ---: | --- | --- |\n")
        for row in rows:
            handle.write(
                f"| {row['model_name']} | {row['reported_metric_name']} | {row['reported_value']} | "
                f"{row['mrr_at_20']} | {row['dataset_variant']} | {row['citation_key']} |\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export literature baselines to CSV, JSON, and Markdown.")
    parser.add_argument("--output", default="results/literature_baselines.csv")
    args = parser.parse_args()

    rows = load_baselines()
    csv_path = Path(args.output)
    json_path = csv_path.with_suffix(".json")
    markdown_path = csv_path.with_suffix(".md")

    write_csv(rows, csv_path)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    write_markdown(rows, markdown_path)
    print(f"Wrote {len(rows)} baseline rows.")


if __name__ == "__main__":
    main()
