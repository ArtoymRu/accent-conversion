"""
PyTorch Dataset for L2-ARCTIC: yields audio path (or tensor) and accent label.
Expects data/l2_arctic to contain unpacked speaker dirs (ABA, SKA, ...) with wav/ inside.
"""

import os
from pathlib import Path
from typing import Optional, Union

import torch
from torch.utils.data import Dataset

try:
    import soundfile as sf
except ImportError:
    sf = None

from .accent_map import get_accent, get_accent_id, ACCENTS


def _find_wav_files(root: Union[str, Path]) -> list:
    """Scan root for speaker dirs and collect (wav_path, accent)."""
    root = Path(root)
    out = []
    from .accent_map import SPEAKER_TO_ACCENT

    for name in os.listdir(root):
        path = root / name
        if not path.is_dir():
            continue
        speaker = name.upper()
        if speaker not in SPEAKER_TO_ACCENT:
            continue
        accent = SPEAKER_TO_ACCENT[speaker]
        wav_dir = path / "wav"
        if not wav_dir.is_dir():
            continue
        for f in wav_dir.glob("*.wav"):
            out.append((f.resolve(), accent))
    return out


class L2ArcticDataset(Dataset):
    """
    Dataset of L2-ARCTIC audio files with accent labels.
    __getitem__ returns (audio_tensor or path, accent_str, accent_id, path) or
    (audio_tensor, accent_id) depending on return_path.
    """

    def __init__(
        self,
        root: Union[str, Path],
        target_sample_rate: int = 16000,
        load_audio: bool = True,
        return_path: bool = True,
    ):
        self.root = Path(root)
        self.target_sr = target_sample_rate
        self.load_audio = load_audio
        self.return_path = return_path
        self.samples = _find_wav_files(self.root)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        wav_path, accent = self.samples[idx]
        accent_id = get_accent_id(accent)

        if self.load_audio and sf is not None:
            data, sr = sf.read(str(wav_path), dtype="float32")
            if sr != self.target_sr:
                import torchaudio
                data = torch.from_numpy(data).unsqueeze(0)
                data = torchaudio.functional.resample(data, sr, self.target_sr)
                data = data.squeeze(0).numpy()
            audio = torch.from_numpy(data)
        else:
            audio = wav_path

        if self.return_path:
            return audio, accent, accent_id, str(wav_path)
        return audio, accent_id
