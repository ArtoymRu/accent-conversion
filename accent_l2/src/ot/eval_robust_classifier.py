"""
Сравнение акцентных классификаторов: сырой (n_nuisance=0) vs NAP по дикторам
(SpeakerRobustAccentClassifier) на 4 LOSO-фолдах (в каждом фолде held-out —
по одному диктору на акцент).

Метрики на held-out дикторах:
  acc / f1          — точность и macro-F1 по высказываниям (speaker-independent)
  speaker_probe     — точность различения дикторов внутри акцента по проекциям
                      (шанс 0.25; меньше — меньше дикторской информации)
  acc_nuis          — точность после добавления к высказываниям акцента A
                      дикторского смещения диктора акцента B (B≠A; смещение =
                      среднее диктора минус среднее его акцента по train).
                      Чистая смена голоса без смены акцента: устойчивый
                      классификатор не должен падать.
  p_src_drop        — падение вероятности истинного акцента при этом возмущении

Декомпозиция сдвига средних (shift_decomp.csv): Δ = mean(tgt train) - mean(src train)
раскладывается в дикторскую часть Δn (проекция на NAP-подпространство, где акцент
по построению не меняется) и остаток Δa. Показываем p_tgt классификатора на
высказываниях held-out src-диктора после добавления Δ, Δa или Δn.

Результаты: outputs/ot/classifier/robust_eval.csv

Пример:
  python src/ot/eval_robust_classifier.py --r 0 4 8 12
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.data.accent_map import SPEAKER_TO_ACCENT
from src.ot.data import N_FOLDS, fold_heldout, load_all_speakers, utterance_means
from src.ot.robust_classifier import SpeakerRobustAccentClassifier, within_accent_speaker_acc
from src.utils.splits import ACCENTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["hubert", "wavlm"], default="hubert")
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--n_utt", type=int, default=150, help="высказываний на диктора")
    ap.add_argument("--r", type=int, nargs="+", default=[0, 4, 8, 12])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    feat_dir = _ROOT / "outputs" / "features" / args.model / f"layer{args.layer}"
    rng = np.random.default_rng(args.seed)
    utts = load_all_speakers(feat_dir, SPEAKER_TO_ACCENT, args.n_utt, rng)

    X, acc, spk = [], [], []
    for s, us in utts.items():
        m = utterance_means(us)
        X.append(m)
        acc += [ACCENTS.index(SPEAKER_TO_ACCENT[s])] * len(m)
        spk += [s] * len(m)
    X, acc, spk = np.concatenate(X), np.array(acc), np.array(spk)

    rows, shift_rows = [], []
    for fold in range(N_FOLDS):
        clfs = {}
        held = fold_heldout(fold)
        te = np.isin(spk, list(held))
        tr = ~te
        # дикторские смещения held-out дикторов относительно train-среднего их акцента
        offsets = {}
        for s in held:
            a = ACCENTS.index(SPEAKER_TO_ACCENT[s])
            offsets[s] = X[spk == s].mean(0) - X[tr & (acc == a)].mean(0)

        for r in args.r:
            clf = SpeakerRobustAccentClassifier(n_nuisance=r).fit(X[tr], acc[tr], spk[tr])
            clfs[r] = clf
            P = clf.predict_proba(X[te])
            pred = clf.classes_[P.argmax(1)]
            # возмущение: смещение диктора из другого акцента
            Xp = X[te].copy()
            src_speakers = spk[te]
            for i, s in enumerate(src_speakers):
                a = ACCENTS.index(SPEAKER_TO_ACCENT[s])
                others = [o for o in held if ACCENTS.index(SPEAKER_TO_ACCENT[o]) != a]
                Xp[i] += offsets[others[(i + fold) % len(others)]]
            Pp = clf.predict_proba(Xp)
            idx = np.arange(len(P))
            p_true = P[idx, np.searchsorted(clf.classes_, acc[te])]
            p_true_p = Pp[idx, np.searchsorted(clf.classes_, acc[te])]
            rows.append({
                "fold": fold, "r": r,
                "acc": float((pred == acc[te]).mean()),
                "f1": float(f1_score(acc[te], pred, average="macro")),
                "speaker_probe": within_accent_speaker_acc(clf.transform(X), acc, spk, seed=args.seed),
                "acc_nuis": float((clf.classes_[Pp.argmax(1)] == acc[te]).mean()),
                "p_src_drop": float((p_true - p_true_p).mean()),
            })
            print(f"fold {fold} r={r:2d}: " + "  ".join(f"{k}={v:.3f}" for k, v in rows[-1].items()
                                                         if k not in ('fold', 'r')), flush=True)

        # ── декомпозиция сдвига средних ─────────────────────────────────────
        rmax = max(args.r)
        ref = clfs[rmax]
        scale = ref.scaler.scale_
        Q = ref.Q
        for si, src in enumerate(ACCENTS):
            for ti, tgt in enumerate(ACCENTS):
                if si == ti:
                    continue
                d = X[tr & (acc == ti)].mean(0) - X[tr & (acc == si)].mean(0)
                dn = scale * (Q @ (Q.T @ (d / scale)))
                da = d - dn
                Xs = X[te & (acc == si)]
                for rr in sorted({0, rmax}):
                    for name, delta in (("identity", 0.0), ("full", d), ("accent_part", da), ("speaker_part", dn)):
                        P = clfs[rr].predict_proba(Xs + delta)
                        shift_rows.append({"fold": fold, "src": src, "tgt": tgt, "clf_r": rr, "shift": name,
                                           "p_tgt": float(P[:, list(clfs[rr].classes_).index(ti)].mean()),
                                           "dnorm_frac_speaker": float((dn / scale) @ (dn / scale) / max((d / scale) @ (d / scale), 1e-12))})
        print(f"fold {fold}: shift decomposition done", flush=True)

    df = pd.DataFrame(rows)
    out = _ROOT / "outputs" / "ot" / "classifier"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "robust_eval.csv", index=False)
    pd.set_option("display.width", 200)
    sh = pd.DataFrame(shift_rows)
    sh.to_csv(out / "shift_decomp.csv", index=False)
    print("\nДекомпозиция сдвига средних (p_tgt, среднее по парам и фолдам):\n",
          sh.groupby(["clf_r", "shift"])["p_tgt"].mean().round(3).unstack(), flush=True)
    print("доля нормы Δ (в станд. пространстве), приходящаяся на дикторскую часть:",
          round(sh["dnorm_frac_speaker"].mean(), 3), flush=True)
    print("\nСреднее по фолдам:\n", df.groupby("r")[["acc", "f1", "speaker_probe", "acc_nuis", "p_src_drop"]]
          .agg(["mean", "std"]).round(3), flush=True)


if __name__ == "__main__":
    main()
