"""AccentFlow model: AIV GRU encoder + GRU mel decoder.

Architecture
------------
tokens (B, T)       → Embedding(K=200, 64)        → (B, T, 64)
d_vec  (B, 256)     → broadcast                    → (B, T, 256)
aiv    (B, 24)      → AIVEncoder (GRU)  → (B, 64)  → broadcast → (B, T, 64)
                                                                      ↓ concat
                                                               (B, T, 384)
                    → Linear(384, 256) → ReLU
                    → GRU(256, 256, layers=2)
                    → Linear(256, 80)
                    → mel (B, T, 80)
"""
import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import cfg


class AIVEncoder(nn.Module):
    """Encode a raw 24-dim AIV feature vector to a 64-dim accent embedding.

    The GRU treats the feature vector as a 1-step sequence, which keeps the
    interface compatible with a future frame-level variant that passes a full
    sequence of per-frame acoustic features.
    """

    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(
            input_size=cfg.aiv_raw_dim,
            hidden_size=cfg.aiv_hidden_dim,
            num_layers=cfg.aiv_gru_layers,
            batch_first=True,
            dropout=0.1 if cfg.aiv_gru_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(cfg.aiv_hidden_dim, cfg.aiv_hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, raw_dim) or (B, T_seq, raw_dim)
        Returns:
            (B, aiv_hidden_dim)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)          # (B, 1, raw_dim)
        _, h = self.gru(x)              # h: (num_layers, B, hidden)
        return self.proj(h[-1])         # (B, hidden_dim)


class MelDecoder(nn.Module):
    """Decode unit tokens + speaker d-vector + accent AIV to mel spectrogram."""

    def __init__(self):
        super().__init__()
        self.token_embed = nn.Embedding(cfg.kmeans_k, cfg.token_embed_dim)

        # 64 (token) + 256 (d-vec) + 64 (aiv) = 384
        self.input_proj = nn.Linear(cfg.fusion_input_dim, cfg.fusion_hidden)
        self.gru = nn.GRU(
            input_size=cfg.fusion_hidden,
            hidden_size=cfg.fusion_hidden,
            num_layers=cfg.decoder_layers,
            batch_first=True,
            dropout=0.1 if cfg.decoder_layers > 1 else 0.0,
        )
        self.out_proj = nn.Linear(cfg.fusion_hidden, cfg.n_mels)

    def forward(
        self,
        tokens: torch.Tensor,   # (B, T)  int64
        d_vec:  torch.Tensor,   # (B, 256)
        aiv:    torch.Tensor,   # (B, 64)
    ) -> torch.Tensor:
        """Returns mel: (B, T, 80)."""
        emb = self.token_embed(tokens)              # (B, T, 64)
        T   = emb.size(1)

        d_exp = d_vec.unsqueeze(1).expand(-1, T, -1)   # (B, T, 256)
        a_exp = aiv.unsqueeze(1).expand(-1, T, -1)     # (B, T,  64)

        x = torch.cat([emb, d_exp, a_exp], dim=-1)     # (B, T, 384)
        x = torch.relu(self.input_proj(x))             # (B, T, 256)
        x, _ = self.gru(x)                             # (B, T, 256)
        return self.out_proj(x)                         # (B, T, 80)


class AccentFlowModel(nn.Module):
    """Full AccentFlow model: AIV encoder + mel decoder."""

    def __init__(self):
        super().__init__()
        self.aiv_encoder = AIVEncoder()
        self.decoder     = MelDecoder()

    def forward(
        self,
        tokens:       torch.Tensor,   # (B, T)
        d_vec:        torch.Tensor,   # (B, 256)
        aiv_features: torch.Tensor,   # (B, 24)
    ) -> torch.Tensor:
        """Returns predicted mel: (B, T, 80)."""
        aiv = self.aiv_encoder(aiv_features)    # (B, 64)
        return self.decoder(tokens, d_vec, aiv) # (B, T, 80)
