"""
Декодер «кадры HuBERT (50 Гц, 768) → лог-мел (≈86 Гц, 80)» для ресинтеза после OT.

Без явного условия на диктора: голос кодируется самими кадрами HuBERT. Поэтому если
OT сдвигает кадры в область другого диктора, декодер озвучит другой голос —
именно это и измеряется speaker embeddings на аудио.

Архитектура: Conv1d(k=5) → интерполяция по времени до длины мела → 3×Conv1d(k=5)+
GELU → BiGRU(2 слоя) → Linear(80). Мел нормируется по статистике обучающей выборки.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HubertToMel(nn.Module):
    def __init__(self, in_dim: int = 768, hidden: int = 512, gru_hidden: int = 384, n_mels: int = 80):
        super().__init__()
        self.in_conv = nn.Conv1d(in_dim, hidden, 5, padding=2)
        self.convs = nn.ModuleList([nn.Conv1d(hidden, hidden, 5, padding=2) for _ in range(3)])
        self.norms = nn.ModuleList([nn.GroupNorm(1, hidden) for _ in range(3)])
        self.gru = nn.GRU(hidden, gru_hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.out = nn.Linear(2 * gru_hidden, n_mels)
        self.register_buffer("mel_mean", torch.zeros(n_mels))
        self.register_buffer("mel_std", torch.ones(n_mels))

    def forward(self, feats: torch.Tensor, mel_len: int) -> torch.Tensor:
        """feats (B, T_h, D) → нормированный мел (B, mel_len, n_mels)."""
        x = F.gelu(self.in_conv(feats.transpose(1, 2)))                       # (B, H, T_h)
        x = F.interpolate(x, size=mel_len, mode="linear", align_corners=False)  # (B, H, T_m)
        for conv, norm in zip(self.convs, self.norms):
            x = x + F.gelu(norm(conv(x)))
        x, _ = self.gru(x.transpose(1, 2))
        return self.out(x)

    @torch.no_grad()
    def synthesize_mel(self, feats: torch.Tensor, mel_len: int) -> torch.Tensor:
        """feats (T_h, D) или (B, T_h, D) → лог-мел в масштабе HiFi-GAN (B, 80, T_m)."""
        if feats.dim() == 2:
            feats = feats.unsqueeze(0)
        y = self.forward(feats, mel_len) * self.mel_std + self.mel_mean
        return y.transpose(1, 2)


def mel_len_for(n_hubert_frames: int, ratio: float = 1.7222) -> int:
    """Длина мела по числу кадров HuBERT (50 Гц → 22050/256 ≈ 86.13 Гц)."""
    return max(1, int(round(n_hubert_frames * ratio)))
