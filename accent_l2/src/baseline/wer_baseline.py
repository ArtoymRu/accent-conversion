"""
WER baseline: оценка разборчивости оригинальной речи с акцентом.
ASR модель: openai/whisper-small через transformers.
Для каждого спикера/акцента вычисляет WER на нескольких utterances
и сравнивает с Ground Truth транскриптами из L2-ARCTIC.
"""

import sys
from pathlib import Path
from typing import Optional, List, Dict
from collections import defaultdict

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.baseline.metrics import compute_wer, normalize_text

ACCENTS = ["arabic", "mandarin", "hindi", "korean", "spanish", "vietnamese"]


def _find_samples_with_transcripts(data_root: Path):
    """Returns list of (wav_path, accent, transcript) for files with transcripts."""
    from src.data.accent_map import SPEAKER_TO_ACCENT
    samples = []
    for accent_dir in sorted(data_root.iterdir()):
        if not accent_dir.is_dir():
            continue
        for speaker_dir in sorted(accent_dir.iterdir()):
            if not speaker_dir.is_dir():
                continue
            speaker = speaker_dir.name.upper()
            if speaker not in SPEAKER_TO_ACCENT:
                continue
            accent = SPEAKER_TO_ACCENT[speaker]
            wav_dir = speaker_dir / "wav"
            transcript_dir = speaker_dir / "transcript"
            if not wav_dir.is_dir() or not transcript_dir.is_dir():
                continue
            for wav_file in sorted(wav_dir.glob("*.wav")):
                txt = transcript_dir / f"{wav_file.stem}.txt"
                if txt.exists():
                    samples.append((wav_file, accent, txt.read_text(encoding="utf-8").strip()))
    return samples


def run_wer_baseline(
    data_root: Path,
    out_dir: Path,
    max_per_accent: int = 30,
    asr_model: str = "openai/whisper-small",
    device: str = "cpu",
) -> Dict:
    """
    Runs Whisper ASR on original L2-ARCTIC audio and computes WER against ground truth.
    This measures how intelligible accented speech is to an ASR system.
    """
    import pandas as pd
    import librosa
    from transformers import pipeline as hf_pipeline

    print(f"Loading Whisper ASR: {asr_model}")
    asr = hf_pipeline(
        "automatic-speech-recognition",
        model=asr_model,
        device=0 if device == "cuda" else -1,
        chunk_length_s=30,
    )

    samples = _find_samples_with_transcripts(data_root)
    print(f"Found {len(samples)} samples with transcripts")

    by_accent = defaultdict(list)
    for s in samples:
        by_accent[s[1]].append(s)

    selected = []
    np.random.seed(42)
    for accent in ACCENTS:
        group = by_accent[accent]
        np.random.shuffle(group)
        selected.extend(group[:max_per_accent])
    print(f"Processing {len(selected)} samples...")

    rows = []
    for i, (wav_path, accent, ref_text) in enumerate(selected):
        if i % 10 == 0:
            print(f"  [{i}/{len(selected)}] {wav_path.name} ({accent})")
        try:
            y, sr = librosa.load(str(wav_path), sr=16000, mono=True)
            result = asr({"array": y.astype(np.float32), "sampling_rate": sr})
            hyp = result.get("text", "").strip()
            wer_val = compute_wer(normalize_text(hyp), normalize_text(ref_text))
            rows.append({
                "accent": accent,
                "file": wav_path.stem,
                "reference": ref_text,
                "hypothesis": hyp,
                "wer": round(wer_val, 4),
            })
        except Exception as e:
            print(f"  Error: {e}")
            continue

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "wer_baseline_raw.csv", index=False)

    summary = df.groupby("accent")["wer"].agg(["mean", "std", "count"]).round(3)
    summary.to_csv(out_dir / "wer_baseline_summary.csv")

    print("\n--- WER по акцентам (оригинальная речь) ---")
    print(summary.to_string())
    print(f"\nСреднее WER по всему корпусу: {df['wer'].mean():.3f}")

    return {"df": df, "summary": summary}
