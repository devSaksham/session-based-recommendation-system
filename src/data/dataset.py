from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset


class SessionDataset(Dataset):
    def __init__(self, examples_path: str | Path):
        with Path(examples_path).open("rb") as handle:
            payload = pickle.load(handle)
        self.sequences: List[List[int]] = payload["sequences"]
        self.targets: List[int] = payload["targets"]
        self.session_ids: List[int] = payload["session_ids"]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> Dict[str, object]:
        return {
            "sequence": self.sequences[index],
            "target": self.targets[index],
            "session_id": self.session_ids[index],
        }


def collate_sessions(batch: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
    lengths = torch.tensor([len(item["sequence"]) for item in batch], dtype=torch.long)
    max_length = int(lengths.max().item()) if len(batch) > 0 else 0

    padded = torch.zeros((len(batch), max_length), dtype=torch.long)
    targets = torch.tensor([item["target"] for item in batch], dtype=torch.long)
    session_ids = torch.tensor([item["session_id"] for item in batch], dtype=torch.long)

    for row, item in enumerate(batch):
        sequence = torch.tensor(item["sequence"], dtype=torch.long)
        padded[row, : sequence.size(0)] = sequence

    return {
        "sequences": padded,
        "lengths": lengths,
        "targets": targets,
        "session_ids": session_ids,
    }
