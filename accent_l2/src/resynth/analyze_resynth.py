"""
Парный анализ результатов resynth_eval.py: разности «метод − identity» по одним и тем же
высказываниям, bootstrap-доверительные интервалы и согласованность знака по парам.

  ΔWER                 — (ошибки метода − ошибки identity) / слов (корпусной)
  Δp_tgt, Δspk_top1    — средние разности по высказываниям
  пары (+/n)           — в скольких парах разность положительна (знак стабилен?)

Bootstrap — по высказываниям (ограничение: 6 пар × 20 высказываний одного held-out диктора на
пару; независимость пар не гарантирует независимость дикторов, поэтому смотрим и на знак по парам).

Несколько фолдов: передайте несколько csv (колонка fold берётся из файла или из имени «ot_fold<f>»);
пара считается как (fold, пара), а знак по парам — по всем таким парам.

Пример:
  python src/resynth/analyze_resynth.py outputs/resynth/ot_fold3/utt_eval.csv
  python src/resynth/analyze_resynth.py outputs/resynth/ot_fold{0,1,2,3}/utt_eval.csv
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["p_tgt", "p_tgt_nap", "p_src", "spk_top1", "spk_capture", "spk_cos_src"]


def paired(df: pd.DataFrame, method: str, base: str = "identity", n_boot: int = 4000, seed: int = 0):
    a = df[df.method == method].set_index(["pair", "utt"]).sort_index()
    b = df[df.method == base].set_index(["pair", "utt"]).sort_index()
    assert (a.index == b.index).all(), "наборы высказываний различаются"
    rng = np.random.default_rng(seed)
    n = len(a)
    boot = rng.integers(0, n, size=(n_boot, n))

    def ci(x):
        m = x[boot].mean(1)
        return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

    row = {"method": method}
    # WER: суммы ошибок на общем числе слов
    de = (a.word_err.values - b.word_err.values).astype(float)
    nw = a.n_words.values.astype(float)
    wer_boot = (de[boot].sum(1)) / nw[boot].sum(1)
    row["dWER"] = de.sum() / nw.sum()
    row["dWER_lo"], row["dWER_hi"] = float(np.percentile(wer_boot, 2.5)), float(np.percentile(wer_boot, 97.5))
    pairs = a.index.get_level_values("pair")
    for m in METRICS:
        d = a[m].values.astype(float) - b[m].values.astype(float)
        row[f"d{m}"] = float(d.mean())
        row[f"d{m}_lo"], row[f"d{m}_hi"] = ci(d)
        per_pair = pd.Series(d).groupby(pairs).mean()
        row[f"{m}_pairs+"] = f"{int((per_pair > 0).sum())}/{len(per_pair)}"
    return row


def load(paths):
    frames = []
    for p in paths:
        d = pd.read_csv(p)
        if "fold" not in d.columns:
            m = re.search(r"ot_fold(\d+)", str(p))
            d["fold"] = int(m.group(1)) if m else 0
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["pair"] = "f" + df["fold"].astype(str) + ":" + df["pair"]
    return df


def main():
    paths = [Path(p) for p in sys.argv[1:]]
    path = paths[0]
    df = load(paths)
    methods = [m for m in df.method.unique() if m not in ("identity", "original")]
    rows = [paired(df, m) for m in methods]
    res = pd.DataFrame(rows).set_index("method")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)

    def fmt(name, lo=None, hi=None, plus=None):
        return res[name].round(3).astype(str) + " [" + res[name + "_lo"].round(3).astype(str) + ", " + \
            res[name + "_hi"].round(3).astype(str) + "]" + ("  " + res[plus] if plus else "")

    show = pd.DataFrame({
        "dWER": fmt("dWER"),
        "dp_tgt": fmt("dp_tgt", plus="p_tgt_pairs+"),
        "dspk_top1": fmt("dspk_top1", plus="spk_top1_pairs+"),
        "dspk_capture": fmt("dspk_capture", plus="spk_capture_pairs+"),
        "dspk_cos_src": fmt("dspk_cos_src"),
    })
    print("Разность метод − identity (95 % bootstrap CI; «+/n» — пар с положительной разностью):\n")
    print(show.to_string())

    # по парам: p_tgt и WER identity vs методы
    print("\nPer-pair p_tgt (raw классификатор) и p_src по системам:")
    print(df.groupby(["pair", "method"])[["p_tgt", "p_src"]].mean().round(2).unstack("method").to_string())
    print("\nПо фолдам (Δ к identity): Δp_tgt / ΔWER / Δspk_top1")
    for f, g in df.groupby("fold"):
        line = []
        for m in methods:
            r = paired(g, m, n_boot=500)
            line.append(f"{m}: {r['dp_tgt']:+.2f} / {r['dWER']:+.3f} / {r['dspk_top1']:+.2f}")
        print(f"  fold {f}: " + "   ".join(line))
    out = path.parent / ("paired_vs_identity.csv" if len(paths) == 1 else "paired_vs_identity_all_folds.csv")
    res.to_csv(out)


if __name__ == "__main__":
    main()
