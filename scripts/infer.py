"""Accent transfer inference: source audio → accent-converted audio.

Usage
-----
# Reconstruct (no transfer):
python scripts/infer.py --input audio.wav --output out.wav

# Transfer to a known accent centroid:
python scripts/infer.py --input indian.wav --output british.wav \\
    --target_accent british_rp --lam 1.0

# Transfer using a reference audio clip:
python scripts/infer.py --input indian.wav --output british.wav \\
    --target_ref ref_british.wav --lam 0.75

# λ sweep to generate multiple outputs:
python scripts/infer.py --input audio.wav --output out.wav --lam 0.0 0.25 0.5 0.75 1.0
"""
import sys
import argparse
import numpy as np
import torch
import librosa
import soundfile as sf
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.content  import ContentEncoder
from src.speaker  import SpeakerEncoder
from src.accent   import AccentFeatureExtractor
from src.model    import AccentFlowModel
from config       import cfg


# ── Vocoder ──────────────────────────────────────────────────────────────────

def load_hifigan(savedir: str = "checkpoints/hifigan"):
    """Load SpeechBrain HiFi-GAN (LJSpeech checkpoint)."""
    from speechbrain.inference.vocoders import HIFIGAN
    print("[infer] Loading HiFi-GAN vocoder…")
    return HIFIGAN.from_hparams(
        source="speechbrain/tts-hifigan-ljspeech",
        savedir=savedir,
    )


def mel_to_wav(mel_np: np.ndarray, vocoder) -> np.ndarray:
    """Convert (T, 80) log-mel → waveform via HiFi-GAN.

    HiFi-GAN expects (batch, n_mels, T) float tensor.
    """
    mel_t = torch.tensor(mel_np.T, dtype=torch.float32).unsqueeze(0)   # (1, 80, T)
    with torch.no_grad():
        waveform = vocoder.decode_batch(mel_t)
    return waveform.squeeze().cpu().numpy()


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_centroids(path: str) -> dict:
    centroids = np.load(path, allow_pickle=True).item()
    return centroids   # {accent_name: (24,) ndarray}


def run_inference(
    audio: np.ndarray,
    sr: int,
    content_enc: ContentEncoder,
    speaker_enc: SpeakerEncoder,
    accent_ext:  AccentFeatureExtractor,
    model:       AccentFlowModel,
    vocoder,
    src_aiv_raw: np.ndarray,
    tgt_aiv_raw: np.ndarray,
    lam: float,
) -> np.ndarray:
    """Run one accent transfer pass. Returns waveform (float32)."""
    tokens  = content_enc.encode(audio, sr)                       # (T,)
    d_vec   = speaker_enc.encode(audio, sr)                       # (256,)

    # λ-interpolated accent embedding
    aiv_interp = ((1 - lam) * src_aiv_raw + lam * tgt_aiv_raw).astype(np.float32)

    tokens_t = torch.tensor(tokens,     dtype=torch.long).unsqueeze(0)   # (1, T)
    dvec_t   = torch.tensor(d_vec,      dtype=torch.float32).unsqueeze(0)
    aiv_t    = torch.tensor(aiv_interp, dtype=torch.float32).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        mel_pred = model(tokens_t, dvec_t, aiv_t)                 # (1, T, 80)

    mel_np = mel_pred.squeeze(0).numpy()                          # (T, 80)
    return mel_to_wav(mel_np, vocoder)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AccentFlow inference")
    parser.add_argument("--input",          required=True,            help="Source .wav")
    parser.add_argument("--output",         required=True,            help="Output .wav (or stem if multiple λ)")
    parser.add_argument("--target_accent",  default=None,             help="Key in aiv_centroids.npy")
    parser.add_argument("--target_ref",     default=None,             help="Reference .wav with target accent")
    parser.add_argument("--lam",            type=float, nargs="+",    default=[cfg.lambda_default],
                        help="Accent strength λ ∈ [0,1]; supply multiple for a sweep")
    parser.add_argument("--model",          default="checkpoints/model.pt")
    parser.add_argument("--kmeans",         default=cfg.kmeans_path)
    parser.add_argument("--centroids",      default=cfg.centroids_path)
    parser.add_argument("--hifigan_dir",    default="checkpoints/hifigan")
    args = parser.parse_args()

    # ── Load encoders ──
    print("[infer] Loading encoders…")
    content_enc = ContentEncoder(kmeans_path=args.kmeans)
    speaker_enc = SpeakerEncoder()
    accent_ext  = AccentFeatureExtractor()

    # ── Load model ──
    print("[infer] Loading AccentFlow model…")
    model = AccentFlowModel()
    ckpt  = Path(args.model)
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        print(f"  Loaded weights from {ckpt}")
    else:
        print(f"  Warning: no checkpoint at {ckpt} — using random weights")

    # ── Load vocoder ──
    vocoder = load_hifigan(args.hifigan_dir)

    # ── Encode source ──
    print(f"[infer] Encoding source: {args.input}")
    audio, sr   = librosa.load(args.input, sr=16000, mono=True)
    src_aiv_raw = accent_ext.extract(audio, sr=16000)             # (24,)

    # ── Determine target AIV ──
    if args.target_accent is not None:
        centroids = load_centroids(args.centroids)
        if args.target_accent not in centroids:
            raise ValueError(
                f"Unknown accent '{args.target_accent}'. "
                f"Available: {sorted(centroids)}"
            )
        tgt_aiv_raw = centroids[args.target_accent]
        print(f"  Using centroid for accent '{args.target_accent}'")
    elif args.target_ref is not None:
        ref_audio, ref_sr = librosa.load(args.target_ref, sr=16000, mono=True)
        tgt_aiv_raw = accent_ext.extract(ref_audio, sr=16000)
        print(f"  Using AIV from reference: {args.target_ref}")
    else:
        tgt_aiv_raw = src_aiv_raw
        print("  No target specified — reconstructing source accent (λ has no effect)")

    # ── Inference (one or multiple λ) ──
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for lam in args.lam:
        print(f"[infer] λ={lam:.2f}")
        wav = run_inference(
            audio, sr,
            content_enc, speaker_enc, accent_ext,
            model, vocoder,
            src_aiv_raw, tgt_aiv_raw,
            lam,
        )
        if len(args.lam) > 1:
            stem    = out_path.stem
            outfile = out_path.with_name(f"{stem}_lam{lam:.2f}{out_path.suffix}")
        else:
            outfile = out_path

        sf.write(outfile, wav, 16000)
        print(f"  Saved → {outfile}")


if __name__ == "__main__":
    main()
