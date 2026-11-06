"""
Baseline на основе EnCodec (Meta, через transformers):
1. Загружает аудио → кодирует в дискретные токены → декодирует обратно
2. Вычисляет MCD между оригиналом и реконструкцией
3. Запускает ASR (Whisper small через transformers) на оригинале → WER

Использует EncodecModel из transformers (facebook/encodec-24khz).
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.baseline.metrics import compute_mcd_from_audio, compute_wer, normalize_text


def load_encodec(model_name: str = "facebook/encodec-24khz", device: str = "cpu"):
    """Load EnCodec model and processor from transformers."""
    from transformers import EncodecModel, AutoProcessor
    processor = AutoProcessor.from_pretrained(model_name)
    model = EncodecModel.from_pretrained(model_name).to(device)
    model.eval()
    return model, processor


def encode_decode(
    y: np.ndarray, sr: int,
    model, processor,
    device: str = "cpu",
    bandwidth: float = 6.0,
) -> Optional[np.ndarray]:
    """
    Encode audio to discrete tokens, then decode back to waveform.
    Returns reconstructed waveform as numpy array (same length as input).
    bandwidth: target bandwidth in kbps (1.5, 3.0, 6.0, 12.0, 24.0)
    """
    target_sr = processor.sampling_rate  # 24000 for encodec-24khz

    # Resample if needed
    if sr != target_sr:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

    # Prepare inputs
    inputs = processor(
        raw_audio=y, sampling_rate=target_sr, return_tensors="pt"
    )
    input_values = inputs["input_values"].to(device)
    padding_mask = inputs.get("padding_mask")
    if padding_mask is not None:
        padding_mask = padding_mask.to(device)

    with torch.no_grad():
        # Encode
        encoder_outputs = model.encode(
            input_values,
            padding_mask=padding_mask,
            bandwidth=bandwidth,
        )
        codes = encoder_outputs.audio_codes  # (1, n_quantizers, T)

        # Decode
        decoded = model.decode(codes, audio_scales=encoder_outputs.audio_scales)
        # decoded.audio_values: (1, 1, T)
        recon = decoded.audio_values[0, 0].cpu().numpy()

    # Resample back to original sr
    if sr != target_sr:
        import librosa
        recon = librosa.resample(recon, orig_sr=target_sr, target_sr=sr)

    # Trim/pad to original length
    orig_len = len(y) if sr == target_sr else int(len(y) * target_sr / sr)
    # After resample back, match input length
    original_len_orig_sr = len(y) if sr != target_sr else len(y)
    # Re-use original y at orig sr length
    import librosa as _lb
    y_orig_at_orig_sr = _lb.resample(y, orig_sr=target_sr, target_sr=sr) if sr != target_sr else y
    target_len = len(y_orig_at_orig_sr) if sr != target_sr else len(y)

    if len(recon) > target_len:
        recon = recon[:target_len]
    elif len(recon) < target_len:
        recon = np.pad(recon, (0, target_len - len(recon)))
    return recon


def load_asr_pipeline(model_name: str = "openai/whisper-small", device: str = "cpu"):
    """Load Whisper ASR pipeline (transformers)."""
    from transformers import pipeline as hf_pipeline
    asr = hf_pipeline(
        "automatic-speech-recognition",
        model=model_name,
        device=0 if device == "cuda" else -1,
        chunk_length_s=30,
    )
    return asr


def transcribe(asr_pipeline, y: np.ndarray, sr: int = 16000) -> str:
    """Run ASR on audio array, return lowercased text."""
    result = asr_pipeline({"array": y.astype(np.float32), "sampling_rate": sr})
    return result.get("text", "").strip().lower()


def run_encodec_baseline(
    data_root: Path,
    out_dir: Path,
    max_per_accent: int = 30,
    bandwidths: List[float] = [1.5, 3.0, 6.0],
    run_asr: bool = False,
    device: str = "cpu",
) -> Dict:
    """
    Main baseline runner. Returns dict with results DataFrame.
    For each bandwidth level computes average MCD across utterances per accent.
    If run_asr=True, also computes WER (slow — downloads Whisper model).
    """
    import pandas as pd
    import librosa

    print("Loading EnCodec model (facebook/encodec-24khz)...")
    model, processor = load_encodec(device=device)

    asr_pipeline = None
    if run_asr:
        print("Loading Whisper ASR (openai/whisper-small)...")
        asr_pipeline = load_asr_pipeline(device=device)

    # Collect samples
    from src.data.dataset import _find_wav_files
    samples = _find_wav_files(data_root)  # list of (Path, accent)

    from collections import defaultdict
    by_accent = defaultdict(list)
    for wav_path, accent in samples:
        by_accent[accent].append((wav_path, accent))

    # Balance
    ACCENTS = ["arabic", "mandarin", "hindi", "korean", "spanish", "vietnamese"]
    selected = []
    np.random.seed(42)
    for accent in ACCENTS:
        group = by_accent[accent]
        np.random.shuffle(group)
        selected.extend(group[:max_per_accent])

    # Find transcript paths
    def _get_transcript(wav_path: Path) -> Optional[str]:
        txt = wav_path.parent.parent / "transcript" / (wav_path.stem + ".txt")
        if txt.exists():
            return txt.read_text(encoding="utf-8").strip()
        return None

    rows = []
    n = len(selected)
    for i, (wav_path, accent) in enumerate(selected):
        if i % 20 == 0:
            print(f"  [{i}/{n}] Processing {wav_path.name} ({accent})...")
        try:
            y, sr = librosa.load(str(wav_path), sr=16000, mono=True)
        except Exception as e:
            print(f"  Load error: {e}")
            continue

        row = {"accent": accent, "file": wav_path.stem}
        ref_text = _get_transcript(wav_path)

        for bw in bandwidths:
            try:
                recon = encode_decode(y, sr, model, processor, device=device, bandwidth=bw)
                mcd = compute_mcd_from_audio(y, recon, sr=sr)
                row[f"mcd_bw{bw}"] = round(mcd, 4)

                if run_asr and asr_pipeline is not None and ref_text is not None:
                    hyp = transcribe(asr_pipeline, recon, sr=sr)
                    wer_val = compute_wer(normalize_text(hyp), normalize_text(ref_text))
                    row[f"wer_bw{bw}"] = round(wer_val, 4)
            except Exception as e:
                print(f"  EnCodec error (bw={bw}): {e}")
                row[f"mcd_bw{bw}"] = np.nan

        rows.append(row)

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "encodec_baseline_raw.csv", index=False)
    print(f"Raw results saved: {out_dir / 'encodec_baseline_raw.csv'}")

    # Summary table: mean MCD per accent per bandwidth
    mcd_cols = [f"mcd_bw{bw}" for bw in bandwidths]
    summary = df.groupby("accent")[mcd_cols].agg(["mean", "std"]).round(3)
    summary.to_csv(out_dir / "encodec_baseline_summary.csv")
    print("\n--- EnCodec Baseline: MCD по акцентам и ширине полосы ---")
    print(summary.to_string())

    overall = df[mcd_cols].mean().round(3)
    print("\n--- Среднее MCD по всему корпусу ---")
    print(overall.to_string())

    return {"df": df, "summary": summary}
