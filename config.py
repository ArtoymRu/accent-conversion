"""All hyperparameters in one place."""
from pathlib import Path

_BASE = Path(__file__).parent


class Config:
    # ── Paths ──────────────────────────────────────────────────────────────
    base_dir = _BASE
    checkpoint_dir = _BASE / "checkpoints"
    kmeans_path = str(_BASE / "checkpoints" / "kmeans.pkl")
    centroids_path = str(_BASE / "checkpoints" / "aiv_centroids.npy")

    # ── Audio ──────────────────────────────────────────────────────────────
    sample_rate: int = 16000

    # ── Content stream (HuBERT) ────────────────────────────────────────────
    hubert_model: str = "facebook/hubert-base-ls960"
    hubert_layer: int = 6   # 0 = embedding; 1–12 = transformer layers
    kmeans_k: int = 200

    # ── Speaker stream ─────────────────────────────────────────────────────
    dvec_dim: int = 256

    # ── Accent stream (AIV) ────────────────────────────────────────────────
    aiv_raw_dim: int = 24       # raw feature vector dimension
    aiv_hidden_dim: int = 64    # GRU hidden / output dimension
    aiv_gru_layers: int = 2

    # ── Mel spectrogram ────────────────────────────────────────────────────
    n_mels: int = 80
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024

    # ── Model ──────────────────────────────────────────────────────────────
    token_embed_dim: int = 64
    # fusion input = token_embed(64) + dvec(256) + aiv_hidden(64) = 384
    fusion_input_dim: int = 64 + 256 + 64
    fusion_hidden: int = 256
    decoder_layers: int = 2

    # ── Training ───────────────────────────────────────────────────────────
    learning_rate: float = 1e-3
    batch_size: int = 8
    num_epochs: int = 50

    # ── Inference ──────────────────────────────────────────────────────────
    lambda_default: float = 1.0     # accent transfer strength λ ∈ [0, 1]

    # ── Evaluation ─────────────────────────────────────────────────────────
    whisper_eval_model: str = "base"        # fast WER
    whisper_score_model: str = "large-v3"   # WhisperScore LID probs


cfg = Config()
