"""
Мел-спектрограммы в параметрах предобученного HiFi-GAN (speechbrain/tts-hifigan-ljspeech):
22.05 кГц, n_fft=1024, hop=256, 80 мел-полос, 0–8 кГц, log-компрессия. Именно такой
мел ожидает вокодер; декодер учится предсказывать его из кадров HuBERT.

Пути данных: data/l2_arctic/<accent>/<SPEAKER>/{wav,transcript}/arctic_aXXXX.{wav,txt}
"""

from pathlib import Path
from typing import Optional

import soundfile as sf
import torch
import torchaudio.functional as AF

from src.data.accent_map import SPEAKER_TO_ACCENT

_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = _ROOT / "data" / "l2_arctic"

MEL_SR = 22050
HOP = 256
N_MELS = 80


def wav_path(speaker: str, utt: str) -> Path:
    return DATA_ROOT / SPEAKER_TO_ACCENT[speaker] / speaker / "wav" / f"{utt}.wav"


def transcript(speaker: str, utt: str) -> str:
    p = DATA_ROOT / SPEAKER_TO_ACCENT[speaker] / speaker / "transcript" / f"{utt}.txt"
    return p.read_text(encoding="utf-8", errors="ignore").strip()


def load_wav(path, sr: int) -> torch.Tensor:
    """Моно-волна (N,) float32 на частоте sr."""
    x, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    w = torch.from_numpy(x.mean(1))
    return AF.resample(w, file_sr, sr) if file_sr != sr else w


def wav_to_mel(wav_22k: torch.Tensor) -> torch.Tensor:
    """(N,) на 22.05 кГц → лог-мел (80, T) в параметрах HiFi-GAN."""
    from speechbrain.lobes.models.HifiGAN import mel_spectogram

    return mel_spectogram(
        sample_rate=MEL_SR, hop_length=HOP, win_length=1024, n_fft=1024, n_mels=N_MELS,
        f_min=0.0, f_max=8000.0, power=1, normalized=False, norm="slaney", mel_scale="slaney",
        compression=True, audio=wav_22k,
    )


def utt_id(feature_file: Path) -> str:
    """ABA_arctic_a0001.pt → arctic_a0001"""
    return feature_file.stem.split("_", 1)[1]
