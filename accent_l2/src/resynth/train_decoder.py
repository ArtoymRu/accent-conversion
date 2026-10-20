"""
Обучение декодера «кадры HuBERT L6 → лог-мел» для ресинтеза после OT.

Speaker-independent: учится только на train-дикторах фолда (held-out дикторы
фолда в обучении декодера не участвуют, как и в k-means / OT / классификаторах).

Кэш выборки (кадры fp16 + мел fp16) пишется в outputs/resynth/cache_fold<f>.pt.
Чекпойнт: outputs/checkpoints/resynth/hubert_L<layer>_fold<f>_decoder.pt

Пример:
  python src/resynth/train_decoder.py --fold 3 --n_utt 300 --epochs 30
"""

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.data.accent_map import SPEAKER_TO_ACCENT
from src.ot.data import fold_heldout
from src.resynth.decoder import HubertToMel
from src.resynth.mel import MEL_SR, load_wav, utt_id, wav_path, wav_to_mel


def build_cache(feat_dir: Path, speakers, n_utt: int, seed: int, device: str):
    rng = random.Random(seed)
    items = []
    for spk in speakers:
        files = sorted(feat_dir.glob(f"{spk}_*.pt"))
        rng.shuffle(files)
        items += [(spk, f) for f in files[:n_utt]]
    out = []
    t0 = time.time()
    for i, (spk, f) in enumerate(items):
        feats = torch.load(f, map_location="cpu", weights_only=True)["features"].squeeze(0)
        wav = load_wav(wav_path(spk, utt_id(f)), MEL_SR).to(device)
        mel = wav_to_mel(wav).squeeze(0).cpu()            # (80, T_m)
        out.append((spk, utt_id(f), feats.half(), mel.half()))
        if (i + 1) % 500 == 0:
            print(f"  кэш: {i + 1}/{len(items)}  {time.time() - t0:.0f}s", flush=True)
    return out


def collate(batch, device):
    T_h = max(b[2].shape[0] for b in batch)
    T_m = max(b[3].shape[1] for b in batch)
    B = len(batch)
    feats = torch.zeros(B, T_h, batch[0][2].shape[1])
    mel = torch.zeros(B, T_m, batch[0][3].shape[0])
    mask = torch.zeros(B, T_m)
    for i, (_, _, f, m) in enumerate(batch):
        feats[i, : f.shape[0]] = f.float()
        mel[i, : m.shape[1]] = m.float().T
        mask[i, : m.shape[1]] = 1
    return feats.to(device), mel.to(device), mask.to(device), T_m


def run_epoch(model, data, bs, device, opt=None, sched=None, shuffle=True):
    model.train(opt is not None)
    idx = list(range(len(data)))
    if shuffle:
        random.shuffle(idx)
    # сортировка внутри «корзин» по длине для эффективного паддинга
    chunks = [idx[i:i + bs * 20] for i in range(0, len(idx), bs * 20)]
    batches = []
    for ch in chunks:
        ch.sort(key=lambda j: data[j][3].shape[1])
        batches += [ch[i:i + bs] for i in range(0, len(ch), bs)]
    if shuffle:
        random.shuffle(batches)
    tot, n = 0.0, 0.0
    for b in batches:
        feats, mel, mask, T_m = collate([data[j] for j in b], device)
        with torch.set_grad_enabled(opt is not None):
            pred = model(feats, T_m)
            target = (mel - model.mel_mean) / model.mel_std
            l1 = ((pred - target).abs().mean(-1) * mask).sum() / mask.sum()
        if opt is not None:
            opt.zero_grad()
            l1.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if sched is not None:
                sched.step()
        tot += l1.item() * mask.sum().item()
        n += mask.sum().item()
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hubert")
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--fold", type=int, default=3)
    ap.add_argument("--n_utt", type=int, default=300, help="высказываний на train-диктора")
    ap.add_argument("--n_val", type=int, default=20, help="высказываний на held-out диктора для контроля")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feat_dir = _ROOT / "outputs" / "features" / args.model / f"layer{args.layer}"
    held = fold_heldout(args.fold)
    train_spk = sorted(s for s in SPEAKER_TO_ACCENT if s not in held)
    cache_path = _ROOT / "outputs" / "resynth" / f"cache_L{args.layer}_fold{args.fold}_n{args.n_utt}.pt"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        train, val = torch.load(cache_path, weights_only=False)
        print(f"Кэш: {cache_path.name}: train {len(train)}, val {len(val)}", flush=True)
    else:
        print("Строю кэш train…", flush=True)
        train = build_cache(feat_dir, train_spk, args.n_utt, args.seed, device)
        print("Строю кэш val (held-out дикторы)…", flush=True)
        val = build_cache(feat_dir, sorted(held), args.n_val, args.seed + 1, device)
        torch.save((train, val), cache_path)

    model = HubertToMel().to(device)
    allmel = torch.cat([m.float() for *_, m in train], dim=1)            # (80, sum T)
    model.mel_mean.copy_(allmel.mean(1)); model.mel_std.copy_(allmel.std(1).clamp_min(1e-3))
    del allmel

    steps = args.epochs * (len(train) // args.bs + 20)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    print(f"Параметры: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M; устройство {device}", flush=True)

    ckpt_dir = _ROOT / "outputs" / "checkpoints" / "resynth"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / f"{args.model}_L{args.layer}_fold{args.fold}_decoder.pt"
    best = float("inf")
    for ep in range(1, args.epochs + 1):
        t = time.time()
        tr = run_epoch(model, train, args.bs, device, opt, sched)
        va = run_epoch(model, val, args.bs, device, shuffle=False)
        flag = ""
        if va < best:
            best = va
            torch.save({"state": model.state_dict(), "args": vars(args), "val_l1": va, "epoch": ep}, ckpt)
            flag = " *"
        print(f"epoch {ep:2d}  train L1 {tr:.4f}  val L1 {va:.4f}  {time.time() - t:.0f}s{flag}", flush=True)
    print(f"Лучший val L1 (нормированный мел) = {best:.4f}; чекпойнт {ckpt}")


if __name__ == "__main__":
    main()
