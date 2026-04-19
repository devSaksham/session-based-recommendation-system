from __future__ import annotations

import torch

from src.models.session_rec import SessionRecModel


def build_config(model_name: str) -> dict:
    config = {
        "model": {
            "name": model_name,
            "embedding_dim": 16,
            "hidden_dim": 16,
            "dropout": 0.1,
            "tie_embeddings": False,
            "mlp_hidden_dim": 16,
            "kan": {
                "hidden_dim": 16,
                "grid_size": 4,
                "min_value": -2.0,
                "max_value": 2.0,
            },
        }
    }
    return config


def test_all_model_variants_produce_logits() -> None:
    sequences = torch.tensor([[1, 2, 3], [4, 5, 0]])
    lengths = torch.tensor([3, 2])
    for model_name in ("gru_linear", "gru_mlp", "gru_kan"):
        model = SessionRecModel(num_items=20, config=build_config(model_name))
        output = model(sequences, lengths)
        assert output.logits.shape == (2, 21)
        assert output.representation.shape == (2, 16)
