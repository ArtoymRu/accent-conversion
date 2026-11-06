"""
Мультимодальная модель: текст (TF-IDF) + аудио (HuBERT mean-pool) → классификация акцента.

Стратегии объединения (fusion):
- early:  конкатенация признаков до классификатора
- late:   независимые предсказания, взвешенная сумма
- cross:  cross-attention между текстовыми и аудио признаками

Пункт 6 курсовой: «Объединение модальностей»
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal


class EarlyFusionClassifier(nn.Module):
    """
    Early fusion: конкатенация аудио + текст признаков → MLP.
    Простейший способ объединения модальностей.
    """

    def __init__(self, audio_dim: int = 768, text_dim: int = 512,
                 hidden_dims=(512, 256), n_classes: int = 6, dropout: float = 0.3):
        super().__init__()
        # Projection heads for each modality
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout)
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout)
        )
        # Joint classifier
        fused_dim = 256 + 128
        layers = []
        in_d = fused_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            in_d = h
        layers.append(nn.Linear(in_d, n_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, audio: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        a = self.audio_proj(audio)
        t = self.text_proj(text)
        return self.classifier(torch.cat([a, t], dim=1))


class LateFusionClassifier(nn.Module):
    """
    Late fusion: отдельные классификаторы для каждой модальности,
    итоговый ответ — взвешенная сумма логитов.
    """

    def __init__(self, audio_dim: int = 768, text_dim: int = 512,
                 n_classes: int = 6, dropout: float = 0.3,
                 audio_weight: float = 0.7, text_weight: float = 0.3):
        super().__init__()
        self.audio_clf = nn.Sequential(
            nn.Linear(audio_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, n_classes)
        )
        self.text_clf = nn.Sequential(
            nn.Linear(text_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )
        self.audio_weight = audio_weight
        self.text_weight = text_weight

    def forward(self, audio: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        a_logits = self.audio_clf(audio)
        t_logits = self.text_clf(text)
        return self.audio_weight * a_logits + self.text_weight * t_logits

    def forward_separate(self, audio: torch.Tensor, text: torch.Tensor):
        return self.audio_clf(audio), self.text_clf(text)


class CrossAttentionFusionClassifier(nn.Module):
    """
    Cross-attention fusion: текстовые признаки используются как запрос (query)
    к аудио признакам — модель учится «фокусироваться» на акцентных паттернах.
    """

    def __init__(self, audio_dim: int = 768, text_dim: int = 512,
                 d_model: int = 256, nhead: int = 4,
                 n_classes: int = 6, dropout: float = 0.3):
        super().__init__()
        self.audio_proj = nn.Linear(audio_dim, d_model)
        self.text_proj = nn.Linear(text_dim, d_model)

        # Cross-attention: query=text, key/value=audio
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead,
            dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, n_classes)
        )

    def forward(self, audio: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        # audio: (B, audio_dim), text: (B, text_dim)
        a = self.audio_proj(audio).unsqueeze(1)  # (B, 1, d_model) — key/value
        t = self.text_proj(text).unsqueeze(1)    # (B, 1, d_model) — query

        attn_out, _ = self.cross_attn(query=t, key=a, value=a)  # (B, 1, d_model)
        attn_out = self.norm(attn_out.squeeze(1) + t.squeeze(1))  # residual

        # Concatenate attended text + audio features
        combined = torch.cat([attn_out, a.squeeze(1)], dim=1)   # (B, d_model*2)
        return self.classifier(self.dropout(combined))
