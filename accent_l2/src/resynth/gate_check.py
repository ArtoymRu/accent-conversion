"""
Gate перед OT-ресинтезом: достаточно ли хорош канал «кадры HuBERT → мел → HiFi-GAN»,
чтобы WER/speaker/accent-метрики на выходе OT были осмысленными.

На held-out дикторах фолда сравнивает:
  original      — исходная запись (16 кГц)
  copy_synth    — wav → мел → HiFi-GAN (верхняя граница вокодера)
  decoder       — кадры HuBERT → декодер → HiFi-GAN (identity-ресинтез; baseline для OT)
по WER (Whisper-small) и сходству диктора с исходной записью (ECAPA cos), сохраняет
по несколько wav для прослушивания.

Пример:
  python src/resynth/gate_check.py --fold 3 --n_per_speaker 8
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.ot.data import fold_heldout
from src.resynth.audio_metrics import Asr, SpeakerEcapa, Vocoder, to_16k, wer
from src.resynth.decoder import HubertToMel
from src.resynth.mel import MEL_SR, load_wav, transcript, utt_id, wav_path, wav_to_mel


def load_decoder(fold: int, layer: int = 6, device: str = "cuda"):
    ck = _ROOT / "outputs" / "checkpoints" / "resynth" / f"hubert_L{layer}_fold{fold}_decoder.pt"
    if not ck.exists():
        return None
    d = torch.load(ck, map_location=device, weights_only=False)
    m = HubertToMel().to(device)
    m.load_state_dict(d["state"])
    return m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=3)
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--n_per_speaker", type=int, default=8)
    ap.add_argument("--save", type=int, default=2, help="сколько wav сохранить на диктора")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda"
    rng = np.random.default_rng(args.seed)
    feat_dir = _ROOT / "outputs" / "features" / "hubert" / f"layer{args.layer}"
    out_dir = _ROOT / "outputs" / "resynth" / f"gate_fold{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    voc, asr, spk = Vocoder(dev), Asr(device=dev), SpeakerEcapa(dev)
    dec = load_decoder(args.fold, args.layer, dev)
    print("декодер:", "найден" if dec is not None else "НЕТ чекпойнта (только original/copy_synth)", flush=True)

    systems = {"original": [], "copy_synth": []}
    if dec is not None:
        systems["decoder"] = []
    refs, speakers = [], []
    for s in sorted(fold_heldout(args.fold)):
        files = sorted(feat_dir.glob(f"{s}_*.pt"))
        for k, i in enumerate(rng.choice(len(files), args.n_per_speaker, replace=False)):
            uid = utt_id(files[i])
            w22 = load_wav(wav_path(s, uid), MEL_SR)
            mel = wav_to_mel(w22.to(dev))
            wav16 = to_16k(w22, MEL_SR)
            systems["original"].append(wav16)
            systems["copy_synth"].append(voc(mel.unsqueeze(0)))
            if dec is not None:
                f = torch.load(files[i], map_location="cpu", weights_only=True)["features"].squeeze(0).to(dev)
                systems["decoder"].append(voc(dec.synthesize_mel(f, mel.shape[-1])))
            refs.append(transcript(s, uid)); speakers.append(s)
            if k < args.save:
                for name, ws in systems.items():
                    sf.write(out_dir / f"{s}_{uid}_{name}.wav", ws[-1], 16000)

    emb = {n: spk.embed(ws) for n, ws in systems.items()}
    print(f"\n{'система':12s} {'WER':>7s}  {'cos к оригиналу':>16s}  (n={len(refs)})")
    for n, ws in systems.items():
        w = wer(asr.transcribe(ws), refs)
        cos = float((emb[n] * emb["original"]).sum(1).mean())
        print(f"{n:12s} {w:7.3f}  {cos:16.3f}", flush=True)
    print(f"\nwav для прослушивания: {out_dir}")


if __name__ == "__main__":
    main()
