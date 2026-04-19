from __future__ import annotations

import torch

from src.evaluation.metrics import recall_mrr_at_k


def test_recall_mrr_at_k_hits_expected_items() -> None:
    logits = torch.tensor(
        [
            [0.0, 1.0, 5.0, 2.0],
            [0.0, 4.0, 3.0, 2.0],
        ]
    )
    targets = torch.tensor([2, 3])
    metrics = recall_mrr_at_k(logits, targets, k=2)

    assert metrics["Recall@2"] == 0.5
    assert metrics["MRR@2"] == 0.5
