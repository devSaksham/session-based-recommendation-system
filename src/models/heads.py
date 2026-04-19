from __future__ import annotations

import torch
from torch import nn


class PiecewiseLinearKANLayer(nn.Module):
    """A lightweight KAN-style layer with learnable piecewise linear edge functions."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 8,
        min_value: float = -2.0,
        max_value: float = 2.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.min_value = min_value
        self.max_value = max_value

        self.basis = nn.Parameter(torch.randn(out_features, in_features, grid_size) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))
        grid = torch.linspace(min_value, max_value, grid_size)
        self.register_buffer("grid", grid)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        clipped = inputs.clamp(self.min_value, self.max_value)
        scaled = (clipped - self.min_value) / (self.max_value - self.min_value) * (self.grid_size - 1)
        left_index = torch.floor(scaled).long().clamp(max=self.grid_size - 2)
        right_index = left_index + 1
        right_weight = scaled - left_index.float()
        left_weight = 1.0 - right_weight

        basis = self.basis.unsqueeze(0).expand(inputs.size(0), -1, -1, -1)
        left_values = torch.gather(
            basis,
            dim=3,
            index=left_index.unsqueeze(1).unsqueeze(-1).expand(-1, self.out_features, -1, 1),
        ).squeeze(-1)
        right_values = torch.gather(
            basis,
            dim=3,
            index=right_index.unsqueeze(1).unsqueeze(-1).expand(-1, self.out_features, -1, 1),
        ).squeeze(-1)

        mixed = left_weight.unsqueeze(1) * left_values + right_weight.unsqueeze(1) * right_values
        return mixed.sum(dim=2) + self.bias


class KANHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        grid_size: int = 8,
        min_value: float = -2.0,
        max_value: float = 2.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layer_1 = PiecewiseLinearKANLayer(
            in_features=hidden_dim,
            out_features=hidden_dim,
            grid_size=grid_size,
            min_value=min_value,
            max_value=max_value,
        )
        self.layer_2 = PiecewiseLinearKANLayer(
            in_features=hidden_dim,
            out_features=output_dim,
            grid_size=grid_size,
            min_value=min_value,
            max_value=max_value,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(self.layer_1(inputs))
        hidden = self.dropout(hidden)
        return self.layer_2(hidden)


class MLPHead(nn.Module):
    def __init__(self, hidden_dim: int, output_dim: int, mlp_hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)
