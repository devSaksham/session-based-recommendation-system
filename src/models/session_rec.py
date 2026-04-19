from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn

from src.models.heads import KANHead, MLPHead


@dataclass
class ModelOutput:
    logits: torch.Tensor
    representation: torch.Tensor


class SessionRecModel(nn.Module):
    def __init__(self, num_items: int, config: Dict[str, object]) -> None:
        super().__init__()
        model_config = config["model"]
        name = model_config["name"]
        embedding_dim = int(model_config["embedding_dim"])
        hidden_dim = int(model_config["hidden_dim"])
        dropout = float(model_config.get("dropout", 0.0))
        tie_embeddings = bool(model_config.get("tie_embeddings", False))

        self.model_name = name
        self.embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.encoder = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

        if name == "gru_linear":
            self.head = nn.Linear(hidden_dim, hidden_dim)
            scorer_dim = hidden_dim
        elif name == "gru_mlp":
            self.head = MLPHead(
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                mlp_hidden_dim=int(model_config.get("mlp_hidden_dim", hidden_dim)),
                dropout=dropout,
            )
            scorer_dim = hidden_dim
        elif name == "gru_kan":
            kan_config = model_config["kan"]
            self.head = KANHead(
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                grid_size=int(kan_config.get("grid_size", 8)),
                min_value=float(kan_config.get("min_value", -2.0)),
                max_value=float(kan_config.get("max_value", 2.0)),
                dropout=dropout,
            )
            scorer_dim = hidden_dim
        else:
            raise ValueError(f"Unsupported model name: {name}")

        self.tie_embeddings = tie_embeddings and hidden_dim == embedding_dim
        if self.tie_embeddings:
            self.output_projection = None
        else:
            self.output_projection = nn.Linear(scorer_dim, num_items + 1)

    def encode(self, sequences: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(sequences)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.encoder(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        last_index = (lengths - 1).clamp_min(0)
        batch_index = torch.arange(output.size(0), device=output.device)
        session_state = output[batch_index, last_index]
        return self.dropout(session_state)

    def forward(self, sequences: torch.Tensor, lengths: torch.Tensor) -> ModelOutput:
        encoded = self.encode(sequences, lengths)
        representation = self.head(encoded)

        if self.tie_embeddings:
            logits = representation @ self.embedding.weight.t()
        else:
            logits = self.output_projection(representation)

        return ModelOutput(logits=logits, representation=representation)
