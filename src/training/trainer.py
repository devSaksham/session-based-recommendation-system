from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.metrics import recall_mrr_at_k
from src.utils.io import dump_json, ensure_dir


@dataclass
class EpochResult:
    loss: float
    metrics: Dict[str, float]
    duration_seconds: float


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    topk: int,
    grad_clip_norm: float | None = None,
    use_amp: bool = False,
    scaler: torch.amp.GradScaler | None = None,
) -> EpochResult:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_examples = 0
    metric_sums = {f"Recall@{topk}": 0.0, f"MRR@{topk}": 0.0, f"HitRate@{topk}": 0.0}
    start = time.time()

    for batch in tqdm(dataloader, disable=False, leave=False):
        batch = move_batch_to_device(batch, device)
        if is_training:
            optimizer.zero_grad(set_to_none=True)

        amp_enabled = use_amp and device.type == "cuda"
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(batch["sequences"], batch["lengths"])
            loss = criterion(outputs.logits, batch["targets"])

        if is_training:
            if scaler is not None and amp_enabled:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if grad_clip_norm is not None:
                if scaler is not None and amp_enabled:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            if scaler is not None and amp_enabled:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

        batch_size = batch["targets"].size(0)
        total_examples += batch_size
        total_loss += loss.item() * batch_size
        metrics = recall_mrr_at_k(outputs.logits.detach(), batch["targets"], k=topk)
        for key, value in metrics.items():
            metric_sums[key] += value * batch_size

    duration = time.time() - start
    averaged_metrics = {key: value / max(total_examples, 1) for key, value in metric_sums.items()}
    return EpochResult(
        loss=total_loss / max(total_examples, 1),
        metrics=averaged_metrics,
        duration_seconds=duration,
    )


def save_checkpoint(path: str | Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, metrics: Dict[str, float]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
        },
        target,
    )


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    config: Dict[str, object],
    run_dir: str | Path,
    checkpoint_dir: str | Path,
    topk: int = 20,
) -> Dict[str, object]:
    training_config = config["training"]
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config.get("weight_decay", 0.0)),
    )
    criterion = nn.CrossEntropyLoss()
    use_amp = bool(config.get("environment", {}).get("use_amp", False))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")

    best_metric = float("-inf")
    best_state = None
    patience = int(training_config.get("early_stopping_patience", 3))
    remaining_patience = patience

    history = []
    for epoch in range(1, int(training_config["epochs"]) + 1):
        train_result = run_epoch(
            model=model,
            dataloader=train_loader,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            topk=topk,
            grad_clip_norm=float(training_config.get("grad_clip_norm", 0.0)) or None,
            use_amp=use_amp,
            scaler=scaler,
        )
        validation_result = run_epoch(
            model=model,
            dataloader=validation_loader,
            device=device,
            criterion=criterion,
            optimizer=None,
            topk=topk,
            use_amp=use_amp,
        )

        epoch_summary = {
            "epoch": epoch,
            "train_loss": train_result.loss,
            "validation_loss": validation_result.loss,
            "train_metrics": train_result.metrics,
            "validation_metrics": validation_result.metrics,
            "train_duration_seconds": train_result.duration_seconds,
            "validation_duration_seconds": validation_result.duration_seconds,
        }
        history.append(epoch_summary)

        monitored_metric = validation_result.metrics[f"MRR@{topk}"]
        save_checkpoint(
            path=Path(checkpoint_dir) / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=validation_result.metrics,
        )

        if monitored_metric > best_metric:
            best_metric = monitored_metric
            remaining_patience = patience
            best_state = copy.deepcopy(model.state_dict())
            save_checkpoint(
                path=Path(checkpoint_dir) / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=validation_result.metrics,
            )
        else:
            remaining_patience -= 1
            if remaining_patience <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    summary = {
        "parameter_count": count_parameters(model),
        "best_validation_mrr_at_20": best_metric,
        "history": history,
    }
    dump_json(summary, Path(run_dir) / "training_summary.json")
    return summary


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    topk: int = 20,
) -> Dict[str, float]:
    model.eval()
    model.to(device)
    metric_sums = {f"Recall@{topk}": 0.0, f"MRR@{topk}": 0.0, f"HitRate@{topk}": 0.0}
    total_examples = 0

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        outputs = model(batch["sequences"], batch["lengths"])
        metrics = recall_mrr_at_k(outputs.logits, batch["targets"], k=topk)
        batch_size = batch["targets"].size(0)
        total_examples += batch_size
        for key, value in metrics.items():
            metric_sums[key] += value * batch_size

    return {key: value / max(total_examples, 1) for key, value in metric_sums.items()}
