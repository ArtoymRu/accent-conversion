"""
Эксперимент: unit-level OT между акцентами L2-ARCTIC.

1. Распределения единиц по акцентам (fit — train-дикторы, speaker-independent).
2. Матрицы расстояний между акцентами: Sinkhorn-OT и JS-baseline.
3. Для каждой пары (src → tgt): план OT по train-гистограммам; применение к
   высказываниям held-out диктора src-акцента; оценка на гистограмме held-out
   диктора tgt-акцента (не участвовал в fit):
     W(src→tgt) до/после переноса, JS до/после, доля изменённых токенов.

Результаты: outputs/ot/unit/<model>_L<layer>_k<K>_<metric>_eps<eps>/
  accent_distance_ot.csv, accent_distance_js.csv, mapping_eval.csv

Пример:
  python src/ot/run_unit_ot.py --k 200 --metric cosine --eps 0.05
  python src/ot/run_unit_ot.py --k 200 --metric sqeuclidean --eps 0.1 --dedup
"""

import argparse
import itertools
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.data.accent_map import SPEAKER_TO_ACCENT
from src.ot import (
    change_rate,
    fit_unit_transport,
    js_divergence,
    load_accent_units,
    ot_distance,
    unit_cost_matrix,
    unit_histogram,
)
from src.utils.splits import ACCENTS, HELD_OUT_SPEAKERS

HELD_OUT = set(HELD_OUT_SPEAKERS.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["hubert", "wavlm"], default="hubert")
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--metric", default="cosine",
                    choices=["sqeuclidean", "euclidean", "cosine"])
    ap.add_argument("--eps", type=float, default=0.05,
                    help="Sinkhorn eps (стоимость нормирована на среднее)")
    ap.add_argument("--dedup", action="store_true", help="схлопнуть повторы токенов")
    ap.add_argument("--lams", type=float, nargs="+", default=[0.5, 1.0])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    token_dir = _ROOT / "outputs" / "features" / f"{args.model}_discrete" / f"k{args.k}"
    km_path = _ROOT / "outputs" / "kmeans_models" / f"{args.model}_layer{args.layer}_k{args.k}.pkl"
    for p in (token_dir, km_path):
        if not p.exists():
            sys.exit(f"Не найдено: {p}\nСначала запустите src/features/kmeans_quantize.py")

    with open(km_path, "rb") as f:
        centroids = pickle.load(f).cluster_centers_
    K = args.k

    train = load_accent_units(token_dir, SPEAKER_TO_ACCENT, exclude_speakers=HELD_OUT, dedup=args.dedup)
    test = load_accent_units(token_dir, SPEAKER_TO_ACCENT, include_speakers=HELD_OUT, dedup=args.dedup)
    train_h = {a: unit_histogram(train[a], K) for a in ACCENTS}
    test_h = {a: unit_histogram(test[a], K) for a in ACCENTS}
    print(f"Акценты: {ACCENTS}; train-высказываний: {sum(len(v) for v in train.values())}, "
          f"test: {sum(len(v) for v in test.values())}", flush=True)

    C = unit_cost_matrix(centroids, args.metric)
    out = (_ROOT / "outputs" / "ot" / "unit" /
           f"{args.model}_L{args.layer}_k{K}_{args.metric}_eps{args.eps}{'_dedup' if args.dedup else ''}")
    out.mkdir(parents=True, exist_ok=True)

    # ── 1. расстояния между акцентами (train-гистограммы) ───────────────────
    d_ot = pd.DataFrame(index=ACCENTS, columns=ACCENTS, dtype=float)
    d_js = d_ot.copy()
    for a, b in itertools.product(ACCENTS, ACCENTS):
        d_ot.loc[a, b] = ot_distance(train_h[a], train_h[b], C, args.eps)
        d_js.loc[a, b] = js_divergence(train_h[a], train_h[b])
    d_ot.to_csv(out / "accent_distance_ot.csv")
    d_js.to_csv(out / "accent_distance_js.csv")
    print("\nOT-расстояния между акцентами (train):\n", d_ot.round(4), flush=True)

    # ── 2. перенос src → tgt, оценка на held-out ────────────────────────────
    rng = np.random.default_rng(args.seed)
    rows = []
    for src, tgt in itertools.permutations(ACCENTS, 2):
        tr = fit_unit_transport(train_h[src], train_h[tgt], centroids, args.metric, args.eps)
        toks = np.concatenate(test[src])
        ref = test_h[tgt]
        variants = {"identity": toks, "hard": tr.map_hard(toks), "soft": tr.map_soft(toks, rng)}
        for lam in args.lams:
            variants[f"interp{lam}"] = tr.map_interpolated(toks, lam)
        for name, mapped in variants.items():
            h = unit_histogram([mapped], K)
            rows.append({
                "src": src, "tgt": tgt, "mode": name,
                "ot_to_tgt": ot_distance(h, ref, C, args.eps),
                "js_to_tgt": js_divergence(h, ref),
                "change_rate": change_rate(toks, mapped),
            })
    ev = pd.DataFrame(rows)
    ev.to_csv(out / "mapping_eval.csv", index=False)

    summary = ev.groupby("mode")[["ot_to_tgt", "js_to_tgt", "change_rate"]].mean()
    print("\nСреднее по парам (меньше ot/js — ближе к целевому акценту; change_rate — цена):\n",
          summary.round(4), flush=True)
    print(f"\nСохранено в {out}")


if __name__ == "__main__":
    main()
