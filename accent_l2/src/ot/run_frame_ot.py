"""
Эксперимент: frame-level OT (Bures–Wasserstein и энтропийная барицентрическая
проекция) между акцентами L2-ARCTIC в пространстве HuBERT.

Protocol (speaker-independent):
  fit   — кадры train-дикторов (по n_fit_utt высказываний на диктора);
  eval  — полные высказывания held-out дикторов src-акцента, отображённые
          в целевой акцент; сравнение с кадрами held-out диктора tgt-акцента.

--speaker_norm: из кадров вычитается среднее диктора перед fit/применением,
после применения добавляется обратно среднее исходного диктора
(перенос акцента без переноса голоса). Метрики расстояний считаются в том же
пространстве, в котором обучалась карта.

Метрики на пару (src→tgt) и метод:
  bw_to_tgt, sw_to_tgt  — W_2^2 (гауссианы) / sliced W до кадров held-out tgt
  p_tgt, p_src, top1_tgt — LogReg-классификатор акцента на средних по высказыванию
                           (обучен на train-дикторах, сырые признаки)
  unit_change           — доля кадров, у которых изменилась k-means единица
  rel_disp              — средняя ||T(x)-x|| / ||x||

Результаты: outputs/ot/frame/<model>_L<layer>_<metric>_eps<eps>[_spknorm]/

Примеры:
  python src/ot/run_frame_ot.py --speaker_norm
  python src/ot/run_frame_ot.py --pairs arabic:mandarin hindi:spanish --eps 0.1
"""

import argparse
import itertools
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.data.accent_map import SPEAKER_TO_ACCENT
from src.ot import (
    EntropicOTMap,
    GaussianOTMap,
    bures_wasserstein_sq,
    fit_gaussian,
    quantize,
    sliced_wasserstein,
)
from src.utils.splits import ACCENTS, HELD_OUT_SPEAKERS

HELD_OUT = set(HELD_OUT_SPEAKERS.values())


def load_utts(feat_dir: Path, speaker: str, n: int, rng: np.random.Generator):
    files = sorted(feat_dir.glob(f"{speaker}_*.pt"))
    if len(files) > n:
        files = [files[i] for i in rng.choice(len(files), n, replace=False)]
    out = []
    for f in files:
        x = torch.load(f, map_location="cpu", weights_only=True)["features"]
        out.append(x.squeeze(0).numpy().astype(np.float32))
    return out


def sample_frames(utts, per_utt: int, rng, center=None):
    chunks = []
    for u in utts:
        idx = rng.choice(len(u), min(per_utt, len(u)), replace=False)
        chunks.append(u[idx] - (center if center is not None else 0.0))
    return np.concatenate(chunks)


def subsample(X: np.ndarray, n: int, rng):
    return X if len(X) <= n else X[rng.choice(len(X), n, replace=False)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["hubert", "wavlm"], default="hubert")
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--k", type=int, default=200, help="k-means для подсчёта unit_change")
    ap.add_argument("--metric", default="sqeuclidean", choices=["sqeuclidean", "cosine"])
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--shrink", type=float, default=1e-2)
    ap.add_argument("--n_max", type=int, default=1000, help="кадров для entropic-карты")
    ap.add_argument("--n_fit_utt", type=int, default=100)
    ap.add_argument("--frames_per_utt", type=int, default=60)
    ap.add_argument("--n_eval_utt", type=int, default=40)
    ap.add_argument("--lams", type=float, nargs="+", default=[1.0, 0.5])
    ap.add_argument("--speaker_norm", action="store_true")
    ap.add_argument("--pairs", nargs="*", help="src:tgt ... (по умолчанию все)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    feat_dir = _ROOT / "outputs" / "features" / args.model / f"layer{args.layer}"
    km_path = _ROOT / "outputs" / "kmeans_models" / f"{args.model}_layer{args.layer}_k{args.k}.pkl"
    for p in (feat_dir, km_path):
        if not p.exists():
            sys.exit(f"Не найдено: {p}")
    centroids = pickle.load(open(km_path, "rb")).cluster_centers_
    rng = np.random.default_rng(args.seed)

    # ── загрузка ────────────────────────────────────────────────────────────
    t0 = time.time()
    train_utts, eval_utts = {}, {}
    for spk, acc in SPEAKER_TO_ACCENT.items():
        if spk in HELD_OUT:
            eval_utts[acc] = load_utts(feat_dir, spk, args.n_eval_utt, rng)
        else:
            train_utts.setdefault(acc, {})[spk] = load_utts(feat_dir, spk, args.n_fit_utt, rng)
    print(f"Загружено за {time.time() - t0:.0f}s", flush=True)

    spk_mean = {}  # среднее диктора (для speaker_norm)
    for acc, d in train_utts.items():
        for spk, utts in d.items():
            spk_mean[spk] = np.concatenate([u for u in utts]).mean(0)
    heldout_mean = {acc: np.concatenate(u).mean(0) for acc, u in eval_utts.items()}

    def cloud(acc):  # облако кадров для fit
        parts = []
        for spk, utts in train_utts[acc].items():
            parts.append(sample_frames(utts, args.frames_per_utt, rng,
                                       spk_mean[spk] if args.speaker_norm else None))
        return np.concatenate(parts)

    clouds = {a: cloud(a) for a in ACCENTS}
    ref_frames = {a: np.concatenate(eval_utts[a]) - (heldout_mean[a] if args.speaker_norm else 0.0)
                  for a in ACCENTS}

    # ── классификатор акцента (utterance mean, train-дикторы, сырые признаки) ─
    Xtr, ytr = [], []
    for ai, acc in enumerate(ACCENTS):
        for utts in train_utts[acc].values():
            Xtr += [u.mean(0) for u in utts]
            ytr += [ai] * len(utts)
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=500, C=0.5).fit(scaler.transform(Xtr), ytr)

    def clf_probs(utts):
        return clf.predict_proba(scaler.transform(np.stack([u.mean(0) for u in utts])))

    acc_id = np.mean([(clf_probs(eval_utts[a]).argmax(1) == i).mean() for i, a in enumerate(ACCENTS)])
    print(f"Классификатор: acc на held-out высказываниях (без переноса) = {acc_id:.3f}", flush=True)

    # ── пары ────────────────────────────────────────────────────────────────
    pairs = ([tuple(p.split(":")) for p in args.pairs] if args.pairs
             else list(itertools.permutations(ACCENTS, 2)))

    tag = f"{args.model}_L{args.layer}_{args.metric}_eps{args.eps}" + ("_spknorm" if args.speaker_norm else "")
    out = _ROOT / "outputs" / "ot" / "frame" / tag
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for src, tgt in pairs:
        t1 = time.time()
        maps = {"bw": GaussianOTMap(args.shrink).fit(clouds[src], clouds[tgt]),
                "ent": EntropicOTMap(args.metric, args.eps, args.n_max, seed=args.seed).fit(clouds[src], clouds[tgt])}
        mu = heldout_mean[src]
        ref = subsample(ref_frames[tgt], 3000, rng)
        g_ref = fit_gaussian(ref, args.shrink)
        ti = ACCENTS.index(tgt)
        si = ACCENTS.index(src)

        variants = [("identity", None, 0.0)] + [(f"{m}_lam{l}", m, l) for m in maps for l in args.lams]
        for name, m, lam in variants:
            mapped_utts, mapped_c, changed, disp = [], [], [], []
            for u in eval_utts[src]:
                xc = u - (mu if args.speaker_norm else 0.0)
                if m is None:
                    z = xc
                else:
                    z = maps[m](xc, lam=lam).float().numpy()
                mapped_c.append(z)
                mapped_utts.append(z + (mu if args.speaker_norm else 0.0))
                changed.append((quantize(mapped_utts[-1], centroids) != quantize(u, centroids)).mean())
                disp.append((np.linalg.norm(z - xc, axis=1) / np.linalg.norm(xc, axis=1).clip(1e-8)).mean())
            Z = subsample(np.concatenate(mapped_c), 3000, rng)
            P = clf_probs(mapped_utts)
            rows.append({
                "src": src, "tgt": tgt, "method": name,
                "bw_to_tgt": bures_wasserstein_sq(fit_gaussian(Z, args.shrink), g_ref),
                "sw_to_tgt": float(sliced_wasserstein(torch.as_tensor(Z), torch.as_tensor(ref))),
                "p_tgt": P[:, ti].mean(), "p_src": P[:, si].mean(),
                "top1_tgt": (P.argmax(1) == ti).mean(),
                "unit_change": float(np.mean(changed)), "rel_disp": float(np.mean(disp)),
            })
        print(f"  {src}→{tgt}: {time.time() - t1:.0f}s  "
              f"BW²(src,tgt)={maps['bw'].distance_sq:.1f}", flush=True)

    ev = pd.DataFrame(rows)
    ev.to_csv(out / "mapping_eval.csv", index=False)
    cols = ["bw_to_tgt", "sw_to_tgt", "p_tgt", "p_src", "top1_tgt", "unit_change", "rel_disp"]
    print("\nСреднее по парам:\n", ev.groupby("method")[cols].mean().round(4), flush=True)
    print(f"\nСохранено в {out}")


if __name__ == "__main__":
    main()
