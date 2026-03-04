"""Content stream: HuBERT feature extraction + k-means quantization."""
import numpy as np
import torch
import librosa
import joblib
from pathlib import Path

from transformers import HubertModel, Wav2Vec2FeatureExtractor
from sklearn.cluster import KMeans

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import cfg


class ContentEncoder:
    """HuBERT layer-6 features → k-means unit tokens."""

    def __init__(self, kmeans_path: str | None = None):
        print("[ContentEncoder] Loading HuBERT...")
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(cfg.hubert_model)
        self.model = HubertModel.from_pretrained(cfg.hubert_model)
        self.model.eval()

        self.kmeans: KMeans | None = None
        if kmeans_path and Path(kmeans_path).exists():
            self.kmeans = joblib.load(kmeans_path)
            print(f"[ContentEncoder] Loaded k-means from {kmeans_path}")
        else:
            print("[ContentEncoder] No k-means — encode() will return random tokens.")

    def extract_features(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """HuBERT layer-6 hidden states. Returns (T, hidden_dim) float32 array."""
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        inputs = self.processor(
            audio, sampling_rate=16000, return_tensors="pt", padding=True
        )
        with torch.no_grad():
            outputs = self.model(
                inputs.input_values, output_hidden_states=True
            )
        # hidden_states: tuple of (embed + num_layers) tensors, each (1, T, dim)
        features = outputs.hidden_states[cfg.hubert_layer]   # (1, T, dim)
        return features.squeeze(0).numpy()                    # (T, dim)

    def quantize(self, features: np.ndarray) -> np.ndarray:
        """K-means assignment. Returns (T,) int array of token ids."""
        if self.kmeans is None:
            raise RuntimeError("k-means not loaded. Run scripts/extract_kmeans.py first.")
        return self.kmeans.predict(features).astype(np.int64)

    def encode(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Full pipeline: audio → unit token sequence (T,)."""
        features = self.extract_features(audio, sr)
        if self.kmeans is not None:
            return self.quantize(features)
        # Fallback: random tokens so the pipeline can run end-to-end before k-means is trained
        T = features.shape[0]
        return np.random.randint(0, cfg.kmeans_k, size=T, dtype=np.int64)
