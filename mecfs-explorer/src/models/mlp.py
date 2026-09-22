"""
A deliberately small MLP. With small-N tabular data (dozens to a few hundred
rows), a bigger network mostly buys you faster overfitting, not better
performance — so start here and only grow it if cross-validated performance
actually asks for it.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MECFSClassifier(nn.Module):
    def __init__(self, n_features: int, hidden_dims: tuple[int, ...] = (64, 16), dropout: float = 0.3):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = n_features
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))  # single logit; use BCEWithLogitsLoss
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
