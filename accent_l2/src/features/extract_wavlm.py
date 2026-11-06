"""
Извлечение скрытых представлений WavLM Base+ для всех высказываний L2-ARCTIC.

Для каждого WAV файла сохраняется .pt файл с ключами:
  'features': Tensor(1, T, 768) — скрытые состояния указанного слоя
  'speaker':  str
  'accent':   str

Аргументы:
  --data_root   путь к data/l2_arctic
  --out_dir     outputs/features/wavlm
  --layers      слои для извлечения (default: 6)
  --device      cuda или cpu

Пример:
  python src/features/extract_wavlm.py --layers 6
  python src/features/extract_wavlm.py --layers 1 2 3 4 5 6 7 8 9 10 11 12
"""

import argparse
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.data.accent_map import SPEAKER_TO_ACCENT


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
    parser.add_argument("--layers", type=int, nargs="+", default=[6])
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    data_root = Path(args.data_root) if args.data_root else _ROOT / "data" / "l2_arctic"
    out_dir = Path(args.out_dir) if args.out_dir else _ROOT / "outputs" / "features" / "wavlm"
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("WavLM Base+ feature extraction")
    print(f"  model:     microsoft/wavlm-base-plus")
    print(f"  layers:    {args.layers}")
    print(f"  device:    {device}")
    print(f"  data_root: {data_root}")
    print(f"  out_dir:   {out_dir}", flush=True)

    from transformers import WavLMModel, Wav2Vec2FeatureExtractor
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
    model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(device)
    model.eval()

    for layer_id in args.layers:
        (out_dir / f"layer{layer_id}").mkdir(parents=True, exist_ok=True)

    entries = find_wav_files(data_root)
    print(f"WAV files found: {len(entries)}", flush=True)

    import soundfile as sf
    ok, skip = 0, 0
    for i, (speaker, wav_path) in enumerate(entries):
        accent = SPEAKER_TO_ACCENT[speaker]
        stem = f"{speaker}_{wav_path.stem}"

        all_done = all(
            (out_dir / f"layer{l}" / f"{stem}.pt").exists()
            for l in args.layers
        )
        if all_done:
            ok += 1
            continue

        try:
            audio, sr = sf.read(str(wav_path), dtype="float32")
            if sr != 16000:
                import torchaudio
                audio_t = torch.from_numpy(audio).unsqueeze(0)
                audio_t = torchaudio.functional.resample(audio_t, sr, 16000)
                audio = audio_t.squeeze(0).numpy()

            inputs = feature_extractor(
                audio, sampling_rate=16000, return_tensors="pt", padding=True
            )
            input_values = inputs.input_values.to(device)

            with torch.no_grad():
                out = model(input_values, output_hidden_states=True)

            # hidden_states[0] = embedding layer, [1..12] = transformer layers
            hidden_states = out.hidden_states

            for layer_id in args.layers:
                out_path = out_dir / f"layer{layer_id}" / f"{stem}.pt"
                if out_path.exists():
                    continue
                h = hidden_states[layer_id].cpu()  # (1, T, 768)
                torch.save({"features": h, "speaker": speaker, "accent": accent}, out_path)

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
