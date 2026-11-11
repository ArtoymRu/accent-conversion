"""The channel: waveform -> frozen WavLM frames -> frozen HiFi-GAN -> waveform.

Both networks are the public kNN-VC checkpoints (Baas et al., 2023). The
vocoder was trained to invert exactly these frames. Nothing is trained here.
"""
import os

import numpy as np

SR = 16000
HOP_S = 0.02          # WavLM frame period
HUB = os.environ.get("KNNVC_REPO", "bshall/knn-vc")   # set KNNVC_REPO to a local clone to work offline
_SOURCE = "local" if os.path.isdir(HUB) else "github"


def load_audio(path, sr=SR) -> np.ndarray:
    """Mono float32 waveform at `sr`."""
    import soundfile as sf
    from scipy.signal import resample_poly
    wav, sr0 = sf.read(str(path), dtype="float32", always_2d=True)
    wav = wav.mean(1)
    if sr0 != sr:
        g = np.gcd(sr0, sr)
        wav = resample_poly(wav, sr // g, sr0 // g).astype(np.float32)
    return wav


def frame_labels(phones, n_frames, hop=HOP_S) -> np.ndarray:
    """Phone id of each frame, read at the frame centre. -1 is silence."""
    ends = np.array([e for _, e, _ in phones])
    ids = np.array([p for _, _, p in phones] + [-1])
    t = (np.arange(n_frames) + 0.5) * hop
    return ids[np.searchsorted(ends, t, side="right")]


class Encoder:
    """WavLM-Large. encoder(wav, layers) -> {layer: (T, 1024) float32}."""

    def __init__(self, device="cuda"):
        import torch
        self.torch, self.device = torch, device
        self.model = torch.hub.load(HUB, "wavlm_large", source=_SOURCE, trust_repo=True, device=device)

    def __call__(self, wav: np.ndarray, layers=(6,)) -> dict:
        torch = self.torch
        with torch.inference_mode():
            x = torch.from_numpy(wav)[None].to(self.device)
            (_, results), _ = self.model.extract_features(
                x, output_layer=max(layers), ret_layer_results=True)
            # results[l] is the output of transformer layer l, shaped (T, 1, d)
            return {l: results[l][0][:, 0].float().cpu().numpy() for l in layers}


class Vocoder:
    """HiFi-GAN on WavLM layer-6 frames. vocoder(frames (T, 1024)) -> waveform."""

    def __init__(self, device="cuda", prematched=True):
        import torch
        self.torch, self.device = torch, device
        self.model, _ = torch.hub.load(HUB, "hifigan_wavlm", source=_SOURCE, trust_repo=True,
                                       prematched=prematched, device=device)

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        torch = self.torch
        with torch.inference_mode():
            x = torch.from_numpy(np.asarray(frames, dtype=np.float32))[None].to(self.device)
            return self.model(x).squeeze().cpu().numpy()
