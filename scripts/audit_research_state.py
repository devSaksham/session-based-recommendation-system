from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def audit_processed_data(processed_dir: Path) -> dict[str, Any]:
    expected = [
        "metadata.json",
        "train_events.parquet",
        "validation_events.parquet",
        "test_events.parquet",
        "train_examples.pkl",
        "validation_examples.pkl",
        "test_examples.pkl",
        "item_encoder.pkl",
        "item_decoder.pkl",
    ]
    report = {
        "processed_dir": str(processed_dir),
        "exists": processed_dir.exists(),
        "present_files": [],
        "missing_files": [],
        "metadata": None,
    }

    if not processed_dir.exists():
        report["missing_files"] = expected
        return report

    for name in expected:
        if (processed_dir / name).exists():
            report["present_files"].append(name)
        else:
            report["missing_files"].append(name)

    metadata_path = processed_dir / "metadata.json"
    if metadata_path.exists():
        report["metadata"] = load_json(metadata_path)

    return report


def audit_runs(runs_root: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "runs_root": str(runs_root),
        "exists": runs_root.exists(),
        "runs": [],
    }
    if not runs_root.exists():
        return report

    for run_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        final_summary_path = run_dir / "final_summary.json"
        resolved_config_path = run_dir / "resolved_config.json"
        runtime_path = run_dir / "runtime.json"
        training_summary_path = run_dir / "training_summary.json"
        test_metrics_path = run_dir / "test_metrics.json"

        run_report = {
            "run_name": run_dir.name,
            "has_final_summary": final_summary_path.exists(),
            "has_resolved_config": resolved_config_path.exists(),
            "has_runtime": runtime_path.exists(),
            "has_training_summary": training_summary_path.exists(),
            "has_test_metrics": test_metrics_path.exists(),
        }
        if final_summary_path.exists():
            run_report["final_summary"] = load_json(final_summary_path)
        report["runs"].append(run_report)

    return report


def check_url(url: str, timeout_sec: float = 8.0) -> str:
    request = Request(url, method="HEAD", headers={"User-Agent": "research-audit/1.0"})
    try:
        with urlopen(request, timeout=timeout_sec) as response:
            return f"ok:{response.status}"
    except HTTPError as exc:
        # Retry with GET for servers that do not allow HEAD.
        if exc.code in {403, 405}:
            try:
                get_request = Request(url, method="GET", headers={"User-Agent": "research-audit/1.0"})
                with urlopen(get_request, timeout=timeout_sec) as response:
                    return f"ok_get:{response.status}"
            except Exception as nested_exc:  # noqa: BLE001
                return f"error:{type(nested_exc).__name__}:{nested_exc}"
        return f"error:http:{exc.code}"
    except URLError as exc:
        return f"error:url:{exc.reason}"
    except Exception as exc:  # noqa: BLE001
        return f"error:{type(exc).__name__}:{exc}"


def audit_literature_baselines(path: Path, check_urls: bool = False) -> dict[str, Any]:
    rows = load_json(path)
    required_fields = {
        "model_name",
        "paper_title",
        "year",
        "dataset_variant",
        "reported_metric_name",
        "reported_value",
        "mrr_at_20",
        "citation_key",
        "source_url",
        "comparability",
    }

    missing_field_rows = []
    duplicate_keys = set()
    seen_keys = set()
    metric_names = set()

    for idx, row in enumerate(rows):
        missing = sorted(list(required_fields - set(row.keys())))
        if missing:
            missing_field_rows.append({"row_index": idx, "missing_fields": missing})

        key = (row.get("model_name"), row.get("citation_key"), row.get("dataset_variant"))
        if key in seen_keys:
            duplicate_keys.add(key)
        seen_keys.add(key)
        metric_names.add(row.get("reported_metric_name"))

    url_checks = []
    if check_urls:
        for row in rows:
            url_checks.append(
                {
                    "model_name": row.get("model_name"),
                    "source_url": row.get("source_url"),
                    "status": check_url(row.get("source_url", "")),
                }
            )

    return {
        "path": str(path),
        "num_rows": len(rows),
        "required_fields": sorted(list(required_fields)),
        "missing_field_rows": missing_field_rows,
        "duplicate_row_keys": [list(key) for key in sorted(duplicate_keys)],
        "reported_metric_names": sorted(name for name in metric_names if name is not None),
        "url_checks": url_checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit preprocessing artifacts, experiment runs, and literature-baseline integrity.")
    parser.add_argument("--data-config", default="configs/data/yoochoose_1_64.yaml")
    parser.add_argument("--runs-root", default="results/runs")
    parser.add_argument("--baselines", default="src/research/literature_baselines.json")
    parser.add_argument("--check-urls", action="store_true")
    parser.add_argument("--output", default="results/generated/research_state_audit.json")
    args = parser.parse_args()

    with Path(args.data_config).open("r", encoding="utf-8") as handle:
        import yaml

        data_config = yaml.safe_load(handle)

    processed_dir = Path(data_config["output"]["processed_dir"])
    report = {
        "processed_data": audit_processed_data(processed_dir),
        "runs": audit_runs(Path(args.runs_root)),
        "literature_baselines": audit_literature_baselines(Path(args.baselines), check_urls=args.check_urls),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
