"""Train AccentFlow mel decoder with L1 reconstruction loss.

This trains the model to reconstruct the ground-truth mel spectrogram
from (unit tokens, d-vector, AIV features).  Once this works you know
the full pipeline is connected; plug in real accent pairs to enable
accent transfer.

Usage
-----
python scripts/train.py --data_dir /path/to/wavs --kmeans checkpoints/kmeans.pkl
python scripts/train.py --data_dir data/l2arctic --epochs 30 --max_files 200
"""
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import librosa
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.content import ContentEncoder
from src.speaker import SpeakerEncoder
from src.accent  import AccentFeatureExtractor
from src.model   import AccentFlowModel
from config      import cfg


def audio_to_log_mel(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Log-mel spectrogram, shape (T_mel, n_mels)."""
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=16000,
        n_mels=cfg.n_mels,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
    )
    return librosa.power_to_db(mel).T.astype(np.float32)   # (T_mel, 80)


def load_dataset(data_dir: Path, content_enc, speaker_enc, accent_ext, max_files=None):
    wav_files = sorted(data_dir.rglob("*.wav"))
    if max_files:
        wav_files = wav_files[:max_files]
    print(f"Loading {len(wav_files)} utterances…")

    dataset = []
    for f in tqdm(wav_files, desc="Encoding"):
        try:
            audio, sr = librosa.load(f, sr=16000, mono=True)

            tokens  = content_enc.encode(audio, sr=16000)           # (T_tok,)
            d_vec   = speaker_enc.encode(audio, sr=16000)            # (256,)
            aiv_raw = accent_ext.extract(audio, sr=16000)            # (24,)
            mel_gt  = audio_to_log_mel(audio)                        # (T_mel, 80)

            if aiv_raw is None:
                continue

            # Align token length to mel length (HuBERT stride ≈ 20 ms, mel hop = 16 ms)
            T = min(len(tokens), mel_gt.shape[0])
            tokens = tokens[:T]
            mel_gt = mel_gt[:T]

            dataset.append({
                "tokens": torch.tensor(tokens, dtype=torch.long),
                "d_vec":  torch.tensor(d_vec,   dtype=torch.float32),
                "aiv":    torch.tensor(aiv_raw,  dtype=torch.float32),
                "mel_gt": torch.tensor(mel_gt,   dtype=torch.float32),
            })
        except Exception as e:
            print(f"  Skip {f.name}: {e}")

    print(f"Dataset: {len(dataset)} utterances")
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Train AccentFlow mel decoder")
    parser.add_argument("--data_dir",  required=True)
    parser.add_argument("--kmeans",    default=cfg.kmeans_path)
    parser.add_argument("--epochs",    type=int,   default=cfg.num_epochs)
    parser.add_argument("--lr",        type=float, default=cfg.learning_rate)
    parser.add_argument("--out",       default="checkpoints/model.pt")
    parser.add_argument("--max_files", type=int,   default=None)
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Encoders
    content_enc = ContentEncoder(kmeans_path=args.kmeans)
    speaker_enc = SpeakerEncoder()
    accent_ext  = AccentFeatureExtractor()

    dataset = load_dataset(
        Path(args.data_dir), content_enc, speaker_enc, accent_ext, args.max_files
    )
    if not dataset:
        raise RuntimeError("Empty dataset — check --data_dir")

    model     = AccentFlowModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, verbose=True
    )
    criterion = nn.L1Loss()

    best_loss = float("inf")
    out_path  = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(dataset)
        total_loss = 0.0

        for item in dataset:
            tokens = item["tokens"].unsqueeze(0)    # (1, T)
            d_vec  = item["d_vec"].unsqueeze(0)     # (1, 256)
            aiv    = item["aiv"].unsqueeze(0)        # (1, 24)
            mel_gt = item["mel_gt"].unsqueeze(0)    # (1, T, 80)

            mel_pred = model(tokens, d_vec, aiv)    # (1, T, 80)

            T    = min(mel_pred.size(1), mel_gt.size(1))
            loss = criterion(mel_pred[:, :T], mel_gt[:, :T])

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataset)
        scheduler.step(avg_loss)
        print(f"Epoch {epoch:3d}/{args.epochs}  loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), out_path)
            print(f"  ✓ Saved best model → {out_path}  (loss={best_loss:.4f})")

    print(f"\nTraining done. Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
