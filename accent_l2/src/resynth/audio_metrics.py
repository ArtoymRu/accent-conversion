"""
Метрики на аудио 16 кГц: WER (Whisper), speaker embeddings (ECAPA), вокодер HiFi-GAN.

Модели берутся из локальных копий:
  ASR       — openai/whisper-small (HF-кэш)
  speaker   — outputs/checkpoints/pretrained/ecapa (speechbrain/spkrec-ecapa-voxceleb)
  вокодер   — outputs/checkpoints/pretrained/hifigan-ljspeech (speechbrain/tts-hifigan-ljspeech)
"""

import os
from pathlib import Path
from typing import List

import numpy as np
import torch
import torchaudio.functional as AF

from src.baseline.metrics import normalize_text
from src.resynth.mel import MEL_SR
from src.resynth.sb_compat import hf_compat

_ROOT = Path(__file__).resolve().parents[2]
PRETRAINED = _ROOT / "outputs" / "checkpoints" / "pretrained"
SR = 16000


class Asr:
    """Whisper (transformers), английский, жадное декодирование."""

    def __init__(self, name: str = "openai/whisper-small", device: str = "cuda"):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.device = device
        self.proc = WhisperProcessor.from_pretrained(name)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            name, use_safetensors=True, torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device).eval()

    @torch.no_grad()
    def transcribe(self, waves: List[np.ndarray], batch: int = 8) -> List[str]:
        out = []
        for i in range(0, len(waves), batch):
            feats = self.proc.feature_extractor(
                [np.asarray(w, dtype=np.float32) for w in waves[i:i + batch]],
                sampling_rate=SR, return_tensors="pt",
            ).input_features.to(self.device, self.model.dtype)
            ids = self.model.generate(feats, language="en", task="transcribe", max_new_tokens=120, num_beams=1)
            out += self.proc.batch_decode(ids, skip_special_tokens=True)
        return [t.strip() for t in out]


def wer(hyps: List[str], refs: List[str]) -> float:
    """Корпусной WER после нормализации текста (нижний регистр, без пунктуации)."""
    from jiwer import wer as _wer

    h = [normalize_text(x) for x in hyps]
    r = [normalize_text(x) for x in refs]
    keep = [i for i, x in enumerate(r) if x]
    return float(_wer([r[i] for i in keep], [h[i] if h[i] else " " for i in keep]))


class SpeakerEcapa:
    """ECAPA-TDNN (VoxCeleb): L2-нормированные 192-мерные эмбеддинги."""

    def __init__(self, device: str = "cuda"):
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy

        d = str(PRETRAINED / "ecapa")
        self.device = device
        with hf_compat():
            self.model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb", savedir=d,
                run_opts={"device": device}, local_strategy=LocalStrategy.COPY)

    @torch.no_grad()
    def embed(self, waves: List[np.ndarray]) -> np.ndarray:
        out = []
        for w in waves:
            x = torch.as_tensor(np.asarray(w, dtype=np.float32)).unsqueeze(0).to(self.device)
            e = self.model.encode_batch(x).squeeze().float().cpu().numpy()
            out.append(e / max(np.linalg.norm(e), 1e-12))
        return np.stack(out)


class Vocoder:
    """HiFi-GAN: лог-мел (B, 80, T) в параметрах mel.py → волна 16 кГц (numpy)."""

    def __init__(self, device: str = "cuda"):
        from speechbrain.inference.vocoders import HIFIGAN
        from speechbrain.utils.fetching import LocalStrategy

        self.device = device
        with hf_compat():
            self.model = HIFIGAN.from_hparams(
                source="speechbrain/tts-hifigan-ljspeech", savedir=str(PRETRAINED / "hifigan-ljspeech"),
                run_opts={"device": device}, local_strategy=LocalStrategy.COPY)

    @torch.no_grad()
    def __call__(self, mel: torch.Tensor) -> np.ndarray:
        wav = self.model.decode_batch(mel.to(self.device)).squeeze(0).squeeze(0)   # (N22,)
        wav = AF.resample(wav.float().cpu(), MEL_SR, SR)
        return wav.numpy()


def to_16k(wav: torch.Tensor, sr: int) -> np.ndarray:
    return AF.resample(wav, sr, SR).numpy() if sr != SR else wav.numpy()


class HubertFeatures:
    """Кадры HuBERT-base слоя L — идентично src/extract_features.py (hidden_states[L], нормализация
    Wav2Vec2FeatureExtractor), чтобы признаки ресинтезированного аудио были сопоставимы с обучающими."""

    def __init__(self, layer: int = 6, name: str = "facebook/hubert-base-ls960", device: str = "cuda"):
        from transformers import HubertModel, Wav2Vec2FeatureExtractor

        self.layer, self.device = layer, device
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained(name)
        self.model = HubertModel.from_pretrained(name).to(device).eval()

    @torch.no_grad()
    def __call__(self, wav16k: np.ndarray) -> np.ndarray:
        x = self.fe(np.asarray(wav16k, dtype=np.float32), sampling_rate=SR, return_tensors="pt", padding=True)
        out = self.model(x.input_values.to(self.device), output_hidden_states=True)
        return out.hidden_states[self.layer].squeeze(0).float().cpu().numpy()
