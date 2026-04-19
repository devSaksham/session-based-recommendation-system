from __future__ import annotations

from typing import Dict

import torch


@torch.no_grad()
def recall_mrr_at_k(logits: torch.Tensor, targets: torch.Tensor, k: int = 20) -> Dict[str, float]:
    logits = logits.clone()
    logits[:, 0] = torch.finfo(logits.dtype).min
    effective_k = min(k, logits.size(1))
    topk = torch.topk(logits, k=effective_k, dim=1).indices
    hits = topk.eq(targets.unsqueeze(1))

    recall = hits.any(dim=1).float().mean().item()

    reciprocal_ranks = torch.zeros(targets.size(0), device=logits.device, dtype=torch.float32)
    hit_rows, hit_cols = hits.nonzero(as_tuple=True)
    reciprocal_ranks[hit_rows] = 1.0 / (hit_cols.float() + 1.0)
    mrr = reciprocal_ranks.mean().item()

    return {
        f"Recall@{k}": recall,
        f"MRR@{k}": mrr,
        f"HitRate@{k}": recall,
    }
