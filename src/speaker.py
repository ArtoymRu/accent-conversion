"""Speaker stream: 256-dim d-vector extraction via resemblyzer (GE2E)."""
import numpy as np
import librosa
from pathlib import Path

from resemblyzer import VoiceEncoder, preprocess_wav


class SpeakerEncoder:
    """Thin wrapper around resemblyzer's GE2E-pretrained encoder."""

    def __init__(self, device: str = "cpu"):
        print("[SpeakerEncoder] Loading GE2E voice encoder...")
        self.encoder = VoiceEncoder(device=device)

    def encode(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Extract 256-dim d-vector from an audio waveform (float32 [-1,1])."""
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        wav = preprocess_wav(audio.astype(np.float32), source_sr=16000)
        return self.encoder.embed_utterance(wav)          # (256,) float32

    def encode_file(self, path: str | Path) -> np.ndarray:
        """Extract d-vector directly from a file path."""
        wav = preprocess_wav(Path(path))
        return self.encoder.embed_utterance(wav)          # (256,) float32
