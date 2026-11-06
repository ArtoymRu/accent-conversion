"""
Эксперимент: frame-level OT (Bures–Wasserstein и энтропийная барицентрическая
проекция, глобальные и условные по k-means единице) между акцентами L2-ARCTIC
в пространстве HuBERT, с leave-one-speaker-out (LOSO) fit.

Фолд f: held-out диктор каждого акцента = sorted(speakers)[f] (src.ot.data.fold_heldout).
  fit  — кадры остальных 3 дикторов каждого акцента;
  eval — высказывания held-out диктора src-акцента, отображённые в целевой акцент;
         сравнение с кадрами held-out диктора tgt-акцента.
K-means для единиц переобучается на train-кадрах фолда (--kmeans refit), чтобы
held-out дикторы не участвовали ни в одном шаге обучения.

--speaker_norm: из кадров вычитается среднее диктора, после применения
добавляется среднее исходного диктора. Внимание: центрирование убирает
дикторо-уровневое среднее, а именно в нём сидит основная часть акцентного
сдвига для классификаторов на средних по высказыванию; такие карты по
построению не меняют среднее высказывания.

Метрики на пару (src→tgt), метод и фолд:
  bw_to_tgt, sw_to_tgt — W_2^2 (гауссианы) / sliced W до кадров held-out tgt
  cond_gap             — зазор по средним внутри единицы до held-out tgt
  p_tgt, top1_tgt      — классификатор на средних по высказыванию, сырой (r=0)
  p_tgt_nap            — то же для NAP-классификатора (SpeakerRobustAccentClassifier)
  unit_change          — доля кадров, у которых изменилась k-means единица
  rel_disp             — средняя ||T(x)-x|| / ||x||
  spk_top1/cos_src/capture/margin — прокси сохранности диктора (src.ot.speaker_probe):
                         ближайший из 24 дикторов = src; сходство с src; ближайший
                         — train-диктор цели (подмена голоса); отрыв от ближайшего чужого

Результаты: outputs/ot/frame/<model>_L<layer>_<metric>_eps<eps>[_spknorm][_<out_tag>]/
  mapping_eval.csv (все фолды), summary.csv

Примеры:
  python src/ot/run_frame_ot.py --folds 0 1 2 3 --lams 1.0
  python src/ot/run_frame_ot.py --folds 3 --pairs arabic:mandarin --speaker_norm
"""

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.data.accent_map import SPEAKER_TO_ACCENT
from src.ot import (
    EntropicOTMap,
    GaussianOTMap,
    UnitConditionalOTMap,
    bures_wasserstein_sq,
    fit_gaussian,
    quantize,
    sliced_wasserstein,
)
from src.ot.data import fold_heldout, load_all_speakers, sample_frames, subsample, utterance_means
from src.ot.robust_classifier import SpeakerRobustAccentClassifier
from src.ot.speaker_probe import SpeakerEmbedder
from src.utils.splits import ACCENTS

COND_CFG = {"ubw-mean": dict(kind="bw", cov_mode="mean"), "ubw-diag": dict(kind="bw", cov_mode="diag"),
            "ubw-full": dict(kind="bw", cov_mode="full"), "uent": dict(kind="ent")}


def run_fold(args, fold: int, utts: dict, rng: np.random.Generator) -> list:
    held = fold_heldout(fold)
    train_utts = {a: {} for a in ACCENTS}
    eval_utts = {}
    for spk, us in utts.items():
        acc = SPEAKER_TO_ACCENT[spk]
        if spk in held:
            eval_utts[acc] = us[: args.n_eval_utt]
        else:
            train_utts[acc][spk] = us
    spk_mean = {s: np.concatenate(us).mean(0) for d in train_utts.values() for s, us in d.items()}
    heldout_mean = {a: np.concatenate(u).mean(0) for a, u in eval_utts.items()}

    # ── облака кадров (сырые) и k-means на train ────────────────────────────
    raw_clouds = {a: [sample_frames(us, args.frames_per_utt, rng) for us in train_utts[a].values()]
                  for a in ACCENTS}
    if args.kmeans == "refit":
        cache = (_ROOT / "outputs" / "ot" / "cache" /
                 f"kmeans_{args.model}_L{args.layer}_k{args.k}_fold{fold}_seed{args.seed}.npy")
        if cache.exists():
            centroids = np.load(cache)
            print(f"  k-means: из кэша {cache.name}", flush=True)
        else:
            t = time.time()
            allf = np.concatenate([c for a in ACCENTS for c in raw_clouds[a]])
            km = MiniBatchKMeans(args.k, batch_size=20_000, n_init=1, max_iter=30, max_no_improvement=10,
                                 random_state=args.seed).fit(subsample(allf, 100_000, rng))
            centroids = km.cluster_centers_
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, centroids)
            print(f"  k-means refit: {time.time() - t:.0f}s", flush=True)
    else:
        import pickle
        km_path = _ROOT / "outputs" / "kmeans_models" / f"{args.model}_layer{args.layer}_k{args.k}.pkl"
        centroids = pickle.load(open(km_path, "rb")).cluster_centers_

    clouds, cloud_units = {}, {}
    for a in ACCENTS:
        fit, units = [], []
        for (spk, _), raw in zip(train_utts[a].items(), raw_clouds[a]):
            fit.append(raw - (spk_mean[spk] if args.speaker_norm else 0.0))
            units.append(quantize(raw, centroids))
        clouds[a], cloud_units[a] = np.concatenate(fit), np.concatenate(units)
    ref_frames, ref_units = {}, {}
    for a in ACCENTS:
        raw = subsample(np.concatenate(eval_utts[a]), 3000, rng)
        ref_units[a] = quantize(raw, centroids)
        ref_frames[a] = raw - (heldout_mean[a] if args.speaker_norm else 0.0)

    # ── классификаторы акцента (utterance mean, train-дикторы) ───────────────
    Xtr, ytr, str_ = [], [], []
    for ai, a in enumerate(ACCENTS):
        for spk, us in train_utts[a].items():
            m = utterance_means(us)
            Xtr.append(m); ytr += [ai] * len(m); str_ += [spk] * len(m)
    Xtr, ytr, str_ = np.concatenate(Xtr), np.array(ytr), np.array(str_)
    clf_raw = SpeakerRobustAccentClassifier(n_nuisance=0).fit(Xtr, ytr, str_)
    clf_nap = SpeakerRobustAccentClassifier(n_nuisance=args.nap_r).fit(Xtr, ytr, str_)
    Xte = np.concatenate([utterance_means(eval_utts[a]) for a in ACCENTS])
    yte = np.concatenate([[i] * len(eval_utts[a]) for i, a in enumerate(ACCENTS)])
    acc_raw = (clf_raw.predict(Xte) == yte).mean()
    acc_nap = (clf_nap.predict(Xte) == yte).mean()
    print(f"  фолд {fold}: held-out {sorted(held)}; acc raw={acc_raw:.3f} nap={acc_nap:.3f}", flush=True)

    # ── прокси сохранности диктора: LDA-эмбеддер на train-дикторах; held-out зачисляются
    #    первой половиной высказываний, вторая половина — тестовая для retention ──
    held_spk = {SPEAKER_TO_ACCENT[s]: s for s in held}
    embedder = SpeakerEmbedder().fit({s: us for a in ACCENTS for s, us in train_utts[a].items()})
    n_enr = {a: len(us) // 2 for a, us in eval_utts.items()}
    for a, us in eval_utts.items():
        embedder.enroll(held_spk[a], us[: n_enr[a]])

    def probs(clf, mapped_utts):
        return clf.predict_proba(np.stack([u.mean(0) for u in mapped_utts]))

    pairs = ([tuple(p.split(":")) for p in args.pairs] if args.pairs
             else list(itertools.permutations(ACCENTS, 2)))
    rows = []
    for src, tgt in pairs:
        t1 = time.time()
        maps = {"bw": GaussianOTMap(args.shrink).fit(clouds[src], clouds[tgt])}
        if not args.no_global_ent:
            maps["ent"] = EntropicOTMap(args.metric, args.eps, args.n_max, seed=args.seed).fit(clouds[src], clouds[tgt])
        cond_kw = dict(min_frames=args.min_frames, shrink=args.cond_shrink, metric=args.metric,
                       eps=args.eps, n_max=args.cond_n_max, seed=args.seed)
        coverage = {}
        for cm in args.cond_methods:
            maps[cm] = UnitConditionalOTMap(**COND_CFG[cm], **cond_kw).fit(
                clouds[src], clouds[tgt], cloud_units[src], cloud_units[tgt])
            coverage[cm] = maps[cm].coverage
        mu = heldout_mean[src]
        ref, ref_u = ref_frames[tgt], ref_units[tgt]
        g_ref = fit_gaussian(ref, args.shrink)
        ti, si = ACCENTS.index(tgt), ACCENTS.index(src)
        shift = mu if args.speaker_norm else 0.0

        variants = [("identity", None, 0.0)] + [(f"{m}_lam{l}", m, l) for m in maps for l in args.lams]
        for name, m, lam in variants:
            mapped_utts, mapped_c, orig_units, changed, disp = [], [], [], [], []
            for u in eval_utts[src]:
                xc = u - shift
                uu = quantize(u, centroids)
                if m is None:
                    z = xc
                elif isinstance(maps[m], UnitConditionalOTMap):
                    z = maps[m](xc, uu, lam=lam).float().numpy()
                else:
                    z = maps[m](xc, lam=lam).float().numpy()
                mapped_c.append(z)
                orig_units.append(uu)
                mapped_utts.append(z + shift)
                changed.append((quantize(mapped_utts[-1], centroids) != uu).mean())
                disp.append((np.linalg.norm(z - xc, axis=1) / np.linalg.norm(xc, axis=1).clip(1e-8)).mean())
            Zfull, Ufull = np.concatenate(mapped_c), np.concatenate(orig_units)
            Z = Zfull[rng.choice(len(Zfull), min(3000, len(Zfull)), replace=False)]
            gap = wsum = 0.0
            for un in np.unique(Ufull):
                zi, ri = Zfull[Ufull == un], ref[ref_u == un]
                if len(zi) >= 10 and len(ri) >= 10:
                    gap += len(zi) * float(((zi.mean(0) - ri.mean(0)) ** 2).sum())
                    wsum += len(zi)
            Pr, Pn = probs(clf_raw, mapped_utts), probs(clf_nap, mapped_utts)
            ret = embedder.retention(mapped_utts[n_enr[src]:], held_spk[src], train_utts[tgt])
            rows.append({
                **{f"spk_{k}": v for k, v in ret.items()},
                "fold": fold, "src": src, "tgt": tgt, "method": name,
                "bw_to_tgt": bures_wasserstein_sq(fit_gaussian(Z, args.shrink), g_ref),
                "sw_to_tgt": float(sliced_wasserstein(torch.as_tensor(Z), torch.as_tensor(ref))),
                "cond_gap": gap / max(wsum, 1),
                "p_tgt": Pr[:, ti].mean(), "top1_tgt": (Pr.argmax(1) == ti).mean(),
                "p_tgt_nap": Pn[:, ti].mean(),
                "unit_change": float(np.mean(changed)), "rel_disp": float(np.mean(disp)),
                "coverage": coverage.get(m, 1.0) if m else 1.0,
            })
        print(f"    {src}→{tgt}: {time.time() - t1:.0f}s", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["hubert", "wavlm"], default="hubert")
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--kmeans", choices=["refit", "pretrained"], default="refit",
                    help="refit — k-means на train-кадрах фолда; pretrained — outputs/kmeans_models (fold 3 split)")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--metric", default="sqeuclidean", choices=["sqeuclidean", "cosine"])
    ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--shrink", type=float, default=1e-2)
    ap.add_argument("--n_max", type=int, default=1000, help="кадров для глобальной entropic-карты")
    ap.add_argument("--n_fit_utt", type=int, default=100)
    ap.add_argument("--frames_per_utt", type=int, default=150)
    ap.add_argument("--n_eval_utt", type=int, default=40)
    ap.add_argument("--lams", type=float, nargs="+", default=[1.0])
    ap.add_argument("--cond_methods", nargs="*", default=["ubw-mean", "ubw-diag", "uent"],
                    choices=list(COND_CFG), help="условные методы (пусто — только глобальные)")
    ap.add_argument("--no_global_ent", action="store_true", help="пропустить глобальную entropic-карту")
    ap.add_argument("--min_frames", type=int, default=30, help="мин. кадров единицы в src и tgt")
    ap.add_argument("--cond_shrink", type=float, default=0.3)
    ap.add_argument("--cond_n_max", type=int, default=300, help="кадров на единицу для uent")
    ap.add_argument("--nap_r", type=int, default=12, help="размерность дикторского подпространства NAP")
    ap.add_argument("--speaker_norm", action="store_true")
    ap.add_argument("--pairs", nargs="*", help="src:tgt ... (по умолчанию все)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_tag", default="", help="суффикс папки результатов (чтобы не перезаписывать другие прогоны)")
    args = ap.parse_args()

    feat_dir = _ROOT / "outputs" / "features" / args.model / f"layer{args.layer}"
    if not feat_dir.exists():
        sys.exit(f"Не найдено: {feat_dir}")
    rng = np.random.default_rng(args.seed)

    t0 = time.time()
    utts = load_all_speakers(feat_dir, SPEAKER_TO_ACCENT, max(args.n_fit_utt, args.n_eval_utt), rng)
    print(f"Загружено за {time.time() - t0:.0f}s; фолды: {args.folds}", flush=True)

    tag = f"{args.model}_L{args.layer}_{args.metric}_eps{args.eps}" + ("_spknorm" if args.speaker_norm else "")         + (f"_{args.out_tag}" if args.out_tag else "")
    out = _ROOT / "outputs" / "ot" / "frame" / tag
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for fold in args.folds:
        rows += run_fold(args, fold, utts, rng)
        pd.DataFrame(rows).to_csv(out / "mapping_eval.csv", index=False)  # чекпоинт после каждого фолда

    ev = pd.DataFrame(rows)
    cols = ["bw_to_tgt", "sw_to_tgt", "cond_gap", "p_tgt", "p_tgt_nap", "unit_change", "rel_disp", "coverage",
            "spk_top1", "spk_cos_src", "spk_capture", "spk_margin"]
    per_fold = ev.groupby(["method", "fold"])[cols].mean()
    summary = per_fold.groupby("method").agg(["mean", "std"])
    summary.to_csv(out / "summary.csv")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
    print("\nСреднее по фолдам (std — между фолдами):\n", summary.round(4), flush=True)
    # парная разность с identity (по фолдам): знак и согласованность
    base = per_fold.xs("identity", level="method")
    print("\nΔ к identity, среднее по фолдам [число фолдов с улучшением из N] для bw_to_tgt / cond_gap:", flush=True)
    for m in ev["method"].unique():
        if m == "identity":
            continue
        d = per_fold.xs(m, level="method") - base
        print(f"  {m:18s} bw_to_tgt {d['bw_to_tgt'].mean():+.3f} [{(d['bw_to_tgt'] < 0).sum()}/{len(d)}]   "
              f"cond_gap {d['cond_gap'].mean():+.3f} [{(d['cond_gap'] < 0).sum()}/{len(d)}]   "
              f"p_tgt {d['p_tgt'].mean():+.3f}  p_tgt_nap {d['p_tgt_nap'].mean():+.3f}  "
              f"spk_top1 {d['spk_top1'].mean():+.3f}  spk_capture {d['spk_capture'].mean():+.3f}", flush=True)
    print(f"\nСохранено в {out}")


if __name__ == "__main__":
    main()
