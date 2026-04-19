from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import SessionDataset, collate_sessions
from src.models.session_rec import SessionRecModel
from src.training.trainer import evaluate, fit
from src.utils.config import load_config
from src.utils.io import dump_json, ensure_dir, load_json, timestamp_now
from src.utils.runtime import collect_runtime_info, resolve_device
from src.utils.seed import set_seed


REQUIRED_PROCESSED_FILES = [
    "metadata.json",
    "train_examples.pkl",
    "validation_examples.pkl",
    "test_examples.pkl",
]


def validate_processed_dir(processed_dir: Path) -> None:
    missing = [name for name in REQUIRED_PROCESSED_FILES if not (processed_dir / name).exists()]
    if missing:
        missing_text = ", ".join(missing)
        raise FileNotFoundError(
            f"Processed dataset is incomplete at {processed_dir}. Missing: {missing_text}. "
            "If preprocessed data was committed, ensure these files exist in the configured folder. "
            "Otherwise run: PYTHONPATH=. python scripts/preprocess.py --config configs/data/yoochoose_1_64.yaml"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a session-based recommendation model.")
    parser.add_argument("--data-config", default="configs/data/yoochoose_1_64.yaml")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--eval-config", default="configs/eval/default.yaml")
    parser.add_argument("--base-config", default="configs/base.yaml")
    args = parser.parse_args()

    config = load_config(args.base_config, args.data_config, args.model_config, args.eval_config)
    set_seed(int(config["project"]["seed"]), deterministic=bool(config["environment"]["deterministic"]))

    processed_dir = Path(config["output"]["processed_dir"])
    validate_processed_dir(processed_dir)
    metadata = load_json(processed_dir / "metadata.json")
    num_items = int(metadata["num_items"])

    run_name = f"{config['variant']}_{config['model']['name']}_{timestamp_now()}"
    run_dir = ensure_dir(Path(config["project"]["output_root"]) / run_name)
    checkpoint_dir = ensure_dir(Path(config["project"]["checkpoints_root"]) / run_name)

    dump_json(config, run_dir / "resolved_config.json")
    dump_json(collect_runtime_info(), run_dir / "runtime.json")

    train_dataset = SessionDataset(processed_dir / "train_examples.pkl")
    validation_dataset = SessionDataset(processed_dir / "validation_examples.pkl")
    test_dataset = SessionDataset(processed_dir / "test_examples.pkl")

    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["environment"].get("num_workers", 0))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_sessions)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_sessions)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_sessions)

    device = resolve_device()
    model = SessionRecModel(num_items=num_items, config=config)
    training_summary = fit(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        device=device,
        config=config,
        run_dir=run_dir,
        checkpoint_dir=checkpoint_dir,
        topk=int(config["metrics"]["topk"]),
    )

    test_metrics = evaluate(model=model, dataloader=test_loader, device=device, topk=int(config["metrics"]["topk"]))
    dump_json(test_metrics, run_dir / "test_metrics.json")
    dump_json({"training_summary": training_summary, "test_metrics": test_metrics}, run_dir / "final_summary.json")
    print(test_metrics)


if __name__ == "__main__":
    main()
