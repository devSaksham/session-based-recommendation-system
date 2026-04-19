from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.utils.io import ensure_dir


def load_run_summaries(runs_root: Path) -> list[dict]:
    summaries = []
    if not runs_root.exists():
        return summaries

    for run_dir in sorted(runs_root.iterdir()):
        final_summary_path = run_dir / "final_summary.json"
        if not final_summary_path.exists():
            continue
        with final_summary_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["run_name"] = run_dir.name
        summaries.append(payload)
    return summaries


def write_markdown_table(summaries: list[dict], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("| Run | Recall@20 | MRR@20 | Parameters |\n")
        handle.write("| --- | ---: | ---: | ---: |\n")
        for summary in summaries:
            training = summary.get("training_summary", {})
            test_metrics = summary.get("test_metrics", {})
            handle.write(
                f"| {summary['run_name']} | {test_metrics.get('Recall@20', 'NA'):.4f} | "
                f"{test_metrics.get('MRR@20', 'NA'):.4f} | {training.get('parameter_count', 'NA')} |\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble experiment summaries into a Markdown table.")
    parser.add_argument("--runs-root", default="results/runs")
    parser.add_argument("--output", default="results/generated/model_results.auto.md")
    args = parser.parse_args()

    summaries = load_run_summaries(Path(args.runs_root))
    write_markdown_table(summaries, Path(args.output))
    print(f"Wrote {len(summaries)} run summaries.")


if __name__ == "__main__":
    main()
