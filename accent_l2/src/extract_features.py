"""
Extract hidden representations from HuBERT (and optionally EnCodec/WavLM) on L2-ARCTIC.
Saves .pt files with features, accent label, and source path per sample.
"""

import argparse
import os
from pathlib import Path
import sys

import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import L2ArcticDataset
from src.data.accent_map import ACCENTS


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_hubert(
    dataset: L2ArcticDataset,
    model_name: str,
    layers: list[int],
    out_dir: Path,
    device: str,
    batch_size: int = 1,
):
    from transformers import HubertModel, Wav2Vec2FeatureExtractor

    # HuBERT base has no tokenizer (vocab); use feature extractor only (Processor would require vocab_file).
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = HubertModel.from_pretrained(model_name).to(device)
    model.eval()

    out_dir.mkdir(parents=True, exist_ok=True)
    # One subdir per layer
    for layer_id in layers:
        (out_dir / f"layer{layer_id}").mkdir(parents=True, exist_ok=True)

    for idx in tqdm(range(len(dataset)), desc="extract"):
        item = dataset[idx]
        audio = item[0]
        accent = item[1]
        accent_id = item[2]
        wav_path = item[3]

        if isinstance(audio, Path):
            import soundfile as sf
            audio, sr = sf.read(str(audio), dtype="float32")
            if sr != 16000:
                import torchaudio
                audio = torch.from_numpy(audio).unsqueeze(0)
                audio = torchaudio.functional.resample(audio, sr, 16000)
                audio = audio.squeeze(0)
            else:
                audio = torch.from_numpy(audio)
        audio = audio.unsqueeze(0).to(device)

        inputs = feature_extractor(audio.squeeze(0).cpu().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(device)

        with torch.no_grad():
            out = model(input_values, output_hidden_states=True)

        hidden_states = out.hidden_states  # tuple of (1, T, D)
        wav_path_obj = Path(wav_path)
        base_name = wav_path_obj.stem
        # Speaker dir is parent of "wav/" (e.g. .../ABA/wav/arctic_a0001.wav -> ABA)
        speaker = wav_path_obj.parent.parent.name
        unique_name = f"{speaker}_{base_name}"
        for layer_id in layers:
            h = hidden_states[layer_id].cpu()  # (1, T, D)
            save_path = out_dir / f"layer{layer_id}" / f"{unique_name}.pt"
            torch.save(
                {
                    "features": h,
                    "accent": accent,
                    "accent_id": accent_id,
                    "wav_path": wav_path,
                },
                save_path,
            )


def main():
    parser = argparse.ArgumentParser(description="Extract features from HuBERT/EnCodec/WavLM")
    parser.add_argument("--config", type=str, default="configs/extract.yaml", help="Path to extract config")
    parser.add_argument("--data_config", type=str, default="configs/data.yaml", help="Path to data config")
    args = parser.parse_args()

    extract_cfg = load_config(args.config)
    data_cfg = load_config(args.data_config)

    root = Path(data_cfg["data_root"])
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[1] / root
    if not root.exists():
        raise FileNotFoundError(f"Data root not found: {root}. Unpack L2-ARCTIC first (scripts/unpack_l2_arctic.py).")

    dataset = L2ArcticDataset(
        root,
        target_sample_rate=data_cfg.get("target_sample_rate", 16000),
        load_audio=True,
        return_path=True,
    )
    if len(dataset) == 0:
        raise RuntimeError("No wav files found. Ensure archives are unpacked and speaker dirs have wav/.")

    out_dir = Path(extract_cfg["features_dir"])
    if not out_dir.is_absolute():
        out_dir = Path(__file__).resolve().parents[1] / out_dir
    model_name = extract_cfg.get("hubert_model", "facebook/hubert-base-ls960")
    layers = extract_cfg.get("layers", [6, 9, 12])
    device = extract_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    batch_size = extract_cfg.get("batch_size", 8)

    model_type = extract_cfg.get("model", "hubert")
    if model_type == "hubert":
        extract_hubert(dataset, model_name, layers, out_dir / "hubert", device, batch_size)
    else:
        raise NotImplementedError(f"Model {model_type} not implemented yet. Use hubert.")

    print(f"Features saved to {out_dir}")


if __name__ == "__main__":
    main()
