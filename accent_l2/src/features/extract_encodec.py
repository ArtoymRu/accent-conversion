"""
Извлечение дискретных кодов EnCodec для всех высказываний L2-ARCTIC.

Использует facebook/encodec_24khz (RVQ с 8 кодовыми книгами при 6 кбит/с).
Аудио ресемплируется в 24 кГц перед кодированием.

Для каждого файла при каждом nq сохраняет:
  'codes':   Tensor(nq, T) int64 — коды первых nq кодовых книг
  'speaker': str
  'accent':  str

Аргументы:
  --data_root    путь к data/l2_arctic
  --out_dir      outputs/features/encodec
  --nq           уровни квантования (default: 1 2 4 8)
  --device       cuda или cpu

Пример:
  python src/features/extract_encodec.py --nq 1 2 4 8
"""

import argparse
import sys
from pathlib import Path

import torch
import torchaudio

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.data.accent_map import SPEAKER_TO_ACCENT

ENCODEC_SR = 24_000
# 6 кбит/с → 8 кодовых книг при 24 кГц (покрывает nq 1..8)
ENCODEC_BANDWIDTH = 6.0


def find_wav_files(data_root: Path):
    entries = []
    for accent_dir in sorted(data_root.iterdir()):
        if not accent_dir.is_dir():
            continue
        for speaker_dir in sorted(accent_dir.iterdir()):
            if not speaker_dir.is_dir():
                continue
            speaker = speaker_dir.name.upper()
            if speaker not in SPEAKER_TO_ACCENT:
                continue
            wav_dir = speaker_dir / "wav"
            if not wav_dir.is_dir():
                continue
            for wav_file in sorted(wav_dir.glob("*.wav")):
                entries.append((speaker, wav_file))
    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--nq", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    data_root = Path(args.data_root) if args.data_root else _ROOT / "data" / "l2_arctic"
    out_dir = Path(args.out_dir) if args.out_dir else _ROOT / "outputs" / "features" / "encodec"
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    max_nq = max(args.nq)

    print("EnCodec feature extraction")
    print(f"  model:     facebook/encodec_24khz")
    print(f"  bandwidth: {ENCODEC_BANDWIDTH} kbps (8 codebooks)")
    print(f"  nq levels: {args.nq}  (saved as slices of full encoding)")
    print(f"  device:    {device}", flush=True)

    try:
        from transformers import EncodecModel, AutoProcessor
    except ImportError:
        raise ImportError("pip install transformers>=4.31.0")

    processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
    model = EncodecModel.from_pretrained("facebook/encodec_24khz").to(device)
    model.eval()

    for nq in args.nq:
        (out_dir / f"nq{nq}").mkdir(parents=True, exist_ok=True)

    entries = find_wav_files(data_root)
    print(f"WAV files found: {len(entries)}", flush=True)

    import soundfile as sf
    ok, skip = 0, 0
    for i, (speaker, wav_path) in enumerate(entries):
        accent = SPEAKER_TO_ACCENT[speaker]
        stem = f"{speaker}_{wav_path.stem}"

        all_done = all((out_dir / f"nq{nq}" / f"{stem}.pt").exists() for nq in args.nq)
        if all_done:
            ok += 1
            continue

        try:
            audio, sr = sf.read(str(wav_path), dtype="float32")
            audio_t = torch.from_numpy(audio).unsqueeze(0)  # (1, samples) для ресемплинга
            if sr != ENCODEC_SR:
                audio_t = torchaudio.functional.resample(audio_t, sr, ENCODEC_SR)

            inputs = processor(
                raw_audio=audio_t.squeeze(0).numpy(),  # (samples,) — процессор ожидает 1D
                sampling_rate=ENCODEC_SR,
                return_tensors="pt",
                padding=True,
            )
            input_values = inputs["input_values"].to(device)
            padding_mask = inputs.get("padding_mask")
            if padding_mask is not None:
                padding_mask = padding_mask.to(device)

            with torch.no_grad():
                encoded = model.encode(
                    input_values,
                    padding_mask=padding_mask,
                    bandwidth=ENCODEC_BANDWIDTH,
                )

            # audio_codes shape: (batch, codebooks, T) или (batch, 1, codebooks, T)
            audio_codes = encoded.audio_codes
            if audio_codes.dim() == 4:
                audio_codes = audio_codes.squeeze(1)
            codes = audio_codes[0].cpu()  # (max_codebooks, T)

            if codes.shape[0] < max_nq:
                skip += 1
                print(
                    f"  Warning: {wav_path.name}: got {codes.shape[0]} codebooks, "
                    f"need {max_nq}", flush=True
                )
                continue

            for nq in args.nq:
                out_path = out_dir / f"nq{nq}" / f"{stem}.pt"
                if out_path.exists():
                    continue
                torch.save({
                    "codes": codes[:nq].long(),  # (nq, T)
                    "speaker": speaker,
                    "accent": accent,
                }, out_path)

            ok += 1
        except Exception as e:
            skip += 1
            print(f"  Error {wav_path.name}: {e}", flush=True)
            continue

        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(entries)}] ok={ok}, skip={skip}", flush=True)

    print(f"\nГотово: сохранено {ok}, пропущено {skip}", flush=True)
    print(f"Директория: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
