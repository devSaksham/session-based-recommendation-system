from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from src.utils.io import ensure_dir


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def load_run_summaries(runs_root: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    if not runs_root.exists():
        return summaries

    for run_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        final_summary_path = run_dir / "final_summary.json"
        if not final_summary_path.exists():
            continue
        with final_summary_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["run_name"] = run_dir.name
        summaries.append(payload)
    return summaries


def summarize_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        training = summary.get("training_summary", {})
        test_metrics = summary.get("test_metrics", {})
        history = training.get("history", [])

        total_train_seconds = sum(_safe_float(epoch.get("train_duration_seconds")) or 0.0 for epoch in history)
        total_validation_seconds = sum(_safe_float(epoch.get("validation_duration_seconds")) or 0.0 for epoch in history)
        total_runtime_seconds = total_train_seconds + total_validation_seconds if history else None

        row = {
            "run_name": summary.get("run_name"),
            "recall_at_20": _safe_float(test_metrics.get("Recall@20")),
            "mrr_at_20": _safe_float(test_metrics.get("MRR@20")),
            "hitrate_at_20": _safe_float(test_metrics.get("HitRate@20")),
            "parameter_count": training.get("parameter_count"),
            "best_validation_mrr_at_20": _safe_float(training.get("best_validation_mrr_at_20")),
            "epochs_ran": len(history),
            "runtime_seconds": total_runtime_seconds,
        }
        rows.append(row)
    return rows


def write_markdown_table(rows: list[dict[str, Any]], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("| Run | Recall@20 | MRR@20 | Parameters | Runtime (s) | Epochs |\n")
        handle.write("| --- | ---: | ---: | ---: | ---: | ---: |\n")
        for row in rows:
            recall = "NA" if row["recall_at_20"] is None else f"{row['recall_at_20']:.4f}"
            mrr = "NA" if row["mrr_at_20"] is None else f"{row['mrr_at_20']:.4f}"
            runtime = "NA" if row["runtime_seconds"] is None else f"{row['runtime_seconds']:.2f}"
            handle.write(
                f"| {row['run_name']} | {recall} | {mrr} | {row['parameter_count']} | {runtime} | {row['epochs_ran']} |\n"
            )


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    fieldnames = [
        "run_name",
        "recall_at_20",
        "mrr_at_20",
        "hitrate_at_20",
        "parameter_count",
        "best_validation_mrr_at_20",
        "epochs_ran",
        "runtime_seconds",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble experiment summaries into machine-readable and Markdown tables.")
    parser.add_argument("--runs-root", default="results/runs")
    parser.add_argument("--output-prefix", default="results/generated/model_results")
    args = parser.parse_args()

    summaries = load_run_summaries(Path(args.runs_root))
    rows = summarize_rows(summaries)

    output_prefix = Path(args.output_prefix)
    markdown_path = output_prefix.with_suffix(".md")
    csv_path = output_prefix.with_suffix(".csv")
    json_path = output_prefix.with_suffix(".json")

    write_markdown_table(rows, markdown_path)
    write_csv(rows, csv_path)
    ensure_dir(json_path.parent)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)

    print(f"Wrote {len(rows)} run summaries to {markdown_path}, {csv_path}, and {json_path}.")


if __name__ == "__main__":
    main()
