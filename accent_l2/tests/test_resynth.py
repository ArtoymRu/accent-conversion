"""
Тесты ресинтез-модулей на синтетике (без загрузки моделей). Запуск:
    python tests/test_resynth.py       (без pytest)
    python -m pytest tests/test_resynth.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import huggingface_hub

from src.resynth.analyze_resynth import paired
from src.resynth.audio_metrics import wer
from src.resynth.decoder import HubertToMel, mel_len_for
from src.resynth.mel import HOP, MEL_SR, N_MELS, wav_to_mel
from src.resynth.sb_compat import hf_compat


def test_hf_compat_is_scoped():
    orig = huggingface_hub.hf_hub_download
    with hf_compat():
        assert huggingface_hub.hf_hub_download is not orig
    assert huggingface_hub.hf_hub_download is orig


def test_wav_to_mel_shape_and_range():
    t = torch.arange(MEL_SR) / MEL_SR
    wav = 0.3 * torch.sin(2 * torch.pi * 440 * t)
    mel = wav_to_mel(wav)
    assert mel.shape[-2] == N_MELS and abs(mel.shape[-1] - (MEL_SR // HOP + 1)) <= 1
    assert torch.isfinite(mel).all() and mel.min() >= -11.6   # log(1e-5)


def test_decoder_shapes():
    dec = HubertToMel(hidden=32, gru_hidden=16)
    x = torch.randn(2, 20, 768)
    n = mel_len_for(20)
    assert dec(x, n).shape == (2, n, N_MELS)
    assert dec.synthesize_mel(x[0], n).shape == (1, N_MELS, n)


def test_wer_values():
    assert wer(["the cat sat"], ["the cat sat"]) == 0.0
    assert abs(wer(["the cat"], ["the cat sat"]) - 1 / 3) < 1e-9          # одно удаление
    assert wer(["The, Cat sat!"], ["the cat sat"]) == 0.0                  # нормализация текста


def _frame(method, errs, p_tgt, n=12):
    rows = []
    for pair in ("a:b", "c:d"):
        for j in range(n):
            rows.append(dict(pair=pair, utt=f"u{j}", method=method, word_err=errs, n_words=10,
                             p_tgt=p_tgt, p_tgt_nap=p_tgt, p_src=0.5, spk_top1=1.0,
                             spk_capture=0.0, spk_cos_src=0.8))
    return pd.DataFrame(rows)


def test_paired_analysis_signs():
    df = pd.concat([_frame("identity", 1, 0.2), _frame("m", 2, 0.5)], ignore_index=True)
    r = paired(df, "m", n_boot=300)
    assert abs(r["dWER"] - 0.1) < 1e-9 and r["dWER_lo"] <= r["dWER"] <= r["dWER_hi"]
    assert abs(r["dp_tgt"] - 0.3) < 1e-9 and r["p_tgt_pairs+"] == "2/2"
    assert r["dspk_top1"] == 0.0 and r["spk_top1_pairs+"] == "0/2"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(fns)} passed")
