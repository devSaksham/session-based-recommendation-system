from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import SessionDataset, collate_sessions
from src.models.session_rec import SessionRecModel
from src.training.trainer import evaluate
from src.utils.config import load_config
from src.utils.io import dump_json, load_json
from src.utils.runtime import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained session recommendation checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-config", default="configs/data/yoochoose_1_64.yaml")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--eval-config", default="configs/eval/default.yaml")
    parser.add_argument("--base-config", default="configs/base.yaml")
    args = parser.parse_args()

    config = load_config(args.base_config, args.data_config, args.model_config, args.eval_config)
    processed_dir = Path(config["output"]["processed_dir"])
    metadata = load_json(processed_dir / "metadata.json")

    test_dataset = SessionDataset(processed_dir / "test_examples.pkl")
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["environment"].get("num_workers", 0)),
        collate_fn=collate_sessions,
    )

    device = resolve_device()
    model = SessionRecModel(num_items=int(metadata["num_items"]), config=config)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    metrics = evaluate(model=model, dataloader=test_loader, device=device, topk=int(config["metrics"]["topk"]))
    dump_json(metrics, Path(args.checkpoint).with_suffix(".metrics.json"))
    print(metrics)


if __name__ == "__main__":
    main()
