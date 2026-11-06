"""
Архитектуры нейронных сетей для классификации акцентов (пункт 5 курсовой).

Входные данные: HuBERT hidden states размером (T, 768)
Задача: классификация на 6 акцентов

Модели:
- CNNAccentClassifier: 1D CNN по временной оси
- LSTMAccentClassifier: двунаправленный LSTM
- TransformerAccentClassifier: Transformer encoder
- MLPBaseline: простой MLP (уже есть в baseline, для сравнения)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MLPBaseline(nn.Module):
    """MLP baseline: mean-pool → 2-layer MLP."""

    def __init__(self, input_dim: int = 768, hidden_dims=(512, 256), n_classes: int = 6,
                 dropout: float = 0.3):
        super().__init__()
        layers = []
        in_d = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            in_d = h
        layers.append(nn.Linear(in_d, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) or (B, D)
        if x.dim() == 3:
            x = x.mean(dim=1)
        return self.net(x)


class CNNAccentClassifier(nn.Module):
    """
    1D CNN для классификации акцентов из последовательности HuBERT признаков.
    Архитектура: 3 слоя Conv1d с residual connections → global avg pooling → FC.
    """

    def __init__(self, input_dim: int = 768, n_filters: int = 256, n_classes: int = 6,
                 dropout: float = 0.3, kernel_sizes=(3, 5, 7)):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, n_filters)

        # Multi-scale convolution branches
        self.conv_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(n_filters, n_filters, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(n_filters),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(n_filters, n_filters, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(n_filters),
            )
            for k in kernel_sizes
        ])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Merge branches
        merged_dim = n_filters * len(kernel_sizes)
        self.merge = nn.Sequential(
            nn.Linear(merged_dim, n_filters),
            nn.BatchNorm1d(n_filters),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(n_filters, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = self.input_proj(x)  # (B, T, n_filters)
        x = x.permute(0, 2, 1)  # (B, n_filters, T)

        branch_outs = []
        for branch in self.conv_branches:
            b_out = self.relu(branch(x) + x)  # residual
            b_out = b_out.mean(dim=2)          # global avg pool → (B, n_filters)
            branch_outs.append(b_out)

        merged = torch.cat(branch_outs, dim=1)  # (B, n_filters * n_branches)
        out = self.merge(merged)
        return self.classifier(out)


class LSTMAccentClassifier(nn.Module):
    """
    Двунаправленный LSTM для классификации акцентов.
    Принимает последовательность HuBERT признаков (T, 768).
    """

    def __init__(self, input_dim: int = 768, hidden_size: int = 256, num_layers: int = 2,
                 n_classes: int = 6, dropout: float = 0.3, bidirectional: bool = True):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        lstm_out_dim = hidden_size * (2 if bidirectional else 1)
        self.attention = nn.Linear(lstm_out_dim, 1)  # attention over time
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = self.input_proj(x)                        # (B, T, hidden)
        out, _ = self.lstm(x)                         # (B, T, hidden*2)
        # Attention pooling
        attn_weights = torch.softmax(self.attention(out), dim=1)  # (B, T, 1)
        context = (out * attn_weights).sum(dim=1)     # (B, hidden*2)
        return self.classifier(self.dropout(context))


class TransformerAccentClassifier(nn.Module):
    """
    Transformer Encoder для классификации акцентов.
    Добавляет [CLS]-токен, применяет несколько слоёв self-attention.
    """

    def __init__(self, input_dim: int = 768, d_model: int = 256, nhead: int = 4,
                 num_layers: int = 2, n_classes: int = 6, dropout: float = 0.3,
                 dim_feedforward: int = 512, max_len: int = 1024):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = self.input_proj(x)                               # (B, T, d_model)
        cls = self.cls_token.expand(x.size(0), -1, -1)       # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)                       # (B, T+1, d_model)
        x = self.encoder(x)                                  # (B, T+1, d_model)
        cls_out = self.norm(x[:, 0, :])                      # CLS token output
        return self.classifier(self.dropout(cls_out))
