"""
Анализ ответов human eval.

Вход:  JSON-файлы ответов (по одному на участника, из веб-формы) и приватный ключ
       outputs/human_eval/private/stimulus_key.csv (sid → item, система, fold, пара, …).
Выход: outputs/human_eval/analysis/*.csv + сводка в консоль.

Шаги:
  1. QC участников (правила зафиксированы заранее, см. docs/HUMAN_EVAL_PROTOCOL.md §6):
       low_check   — средний MOS на «плохих» контролях > 3.0
       retest      — среднее |Δ| между стимулом и его повтором > 2 (по MOS и accentedness)
       fast        — медианное время ответа < 6 с
       straightline— по какому-то вопросу все ответы одинаковы (≥ 10 трёх)
       not_listened— > 10 % стимулов не дослушаны
     Участник исключается, если нарушено ≥ 2 правил.
  2. Системы: среднее и 95 % CI (bootstrap по участникам) по 4 шкалам.
  3. Контрасты «система − identity» (и original − identity): по участникам и по item'ам,
     Wilcoxon, поправка Холма внутри шкалы.
  4. Согласие: Krippendorff α (interval) по ячейкам (item, система).
  5. Корреляции с автометриками (utt_eval.csv): валидация p_tgt, spk_cos_src, WER.

Пример:
  python src/humaneval/analyze_responses.py responses/*.json --auto outputs/resynth/ot_fold{0,1,2,3}/utt_eval.csv
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

SCALES = ["accent_sim", "accentedness", "mos", "voice_sim"]
BASE = "identity"


# ── загрузка ──────────────────────────────────────────────────────────────────

def load_responses(paths, include_dev: bool = False) -> pd.DataFrame:
    rows = []
    seen = set()
    for p in paths:
        d = json.load(open(p, encoding="utf-8"))
        if d.get("dev") and not include_dev:
            continue
        rid = (d["rater"], d.get("pid", ""), d["started"])
        if rid in seen:
            continue
        seen.add(rid)
        rater = f"{d['rater']}|{d.get('pid', '')}|{d['started'][:19]}"
        for r in d["responses"]:
            rows.append(dict(rater=rater, position=d["position"], heard=(d.get("setup") or {}).get("heard"),
                             trial=r["trial_index"], block=r["block_index"], kind=r["kind"], item_id=r["item_id"],
                             sid=Path(r["stimulus"]).stem, rt_ms=r["rt_ms"], plays=r["plays"],
                             listened=bool(r["listened_fully"]), **{k: r["ratings"][k] for k in SCALES}))
    return pd.DataFrame(rows)


def attach_key(df: pd.DataFrame, key: pd.DataFrame) -> pd.DataFrame:
    k = key[["sid", "kind", "item", "system", "fold", "pair", "tgt", "src", "utt"]].rename(
        columns={"kind": "key_kind", "item": "key_item"})
    out = df.merge(k, on="sid", how="left")
    if out["system"].isna().any():
        raise ValueError(f"{out['system'].isna().sum()} ответов с неизвестным стимулом (не тот ключ?)")
    return out


# ── QC ────────────────────────────────────────────────────────────────────────

def qc(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rater, g in df.groupby("rater"):
        chk = g[g.kind == "check"]
        low = bool(len(chk) and chk["mos"].mean() > 3.0)
        # test–retest: повтор против первого показа того же item
        rep = g[g.kind == "repeat"]
        diffs = []
        for _, r in rep.iterrows():
            o = g[(g.kind == "item") & (g.sid == r.sid)]
            if len(o):
                diffs.append(np.mean([abs(r[s] - o.iloc[0][s]) for s in ("mos", "accentedness")]))
        retest = bool(len(diffs) and np.mean(diffs) > 2)
        items = g[g.kind == "item"]
        fast = bool(g.rt_ms.median() < 6000)
        straight = bool(len(items) >= 10 and any(items[s].nunique() == 1 for s in SCALES))
        notl = bool((~g.listened).mean() > 0.10)
        flags = dict(low_check=low, retest=retest, fast=fast, straightline=straight, not_listened=notl)
        rows.append(dict(rater=rater, n=len(g), **flags, n_flags=sum(flags.values()),
                         keep=sum(flags.values()) < 2, retest_diff=float(np.mean(diffs)) if diffs else np.nan,
                         check_mos=float(chk["mos"].mean()) if len(chk) else np.nan, median_rt_s=g.rt_ms.median() / 1000))
    return pd.DataFrame(rows).set_index("rater")


# ── статистика ────────────────────────────────────────────────────────────────

def boot_ci(per_rater: pd.DataFrame, scale: str, n_boot=4000, seed=0):
    """per_rater: колонки rater, sum, n для (система, шкала) → среднее и CI (bootstrap по участникам)."""
    rng = np.random.default_rng(seed)
    s, n = per_rater["sum"].values, per_rater["n"].values
    idx = rng.integers(0, len(s), size=(n_boot, len(s)))
    m = s[idx].sum(1) / n[idx].sum(1)
    return s.sum() / n.sum(), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def system_table(df: pd.DataFrame) -> pd.DataFrame:
    items = df[df.kind.isin(["item", "repeat"]) & (df.key_kind == "item")]
    rows = []
    for sysname, g in items.groupby("system"):
        row = {"system": sysname, "n_ratings": len(g), "n_raters": g.rater.nunique()}
        for s in SCALES:
            pr = g.groupby("rater")[s].agg(["sum", "count"]).rename(columns={"count": "n"})
            mean, lo, hi = boot_ci(pr, s)
            row[s], row[f"{s}_lo"], row[f"{s}_hi"] = mean, lo, hi
        rows.append(row)
    return pd.DataFrame(rows).set_index("system")


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    adj = np.empty_like(p, dtype=float)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def contrasts(df: pd.DataFrame, base: str = BASE, n_boot=4000, seed=0) -> pd.DataFrame:
    from scipy.stats import wilcoxon

    items = df[(df.kind == "item") & (df.key_kind == "item")]   # без повторов
    rng = np.random.default_rng(seed)
    out = []
    for unit, label in (("rater", "по участникам"), ("item", "по item'ам")):
        key = "rater" if unit == "rater" else "key_item"
        for s in SCALES:
            tab = items.groupby([key, "system"])[s].mean().unstack("system")
            systems = [c for c in tab.columns if c != base]
            ps, rows = [], []
            for sysname in systems:
                d = (tab[sysname] - tab[base]).dropna()
                if len(d) < 3:
                    continue
                idx = rng.integers(0, len(d), size=(n_boot, len(d)))
                bm = d.values[idx].mean(1)
                p = 1.0 if np.allclose(d.values, 0) else float(wilcoxon(d.values).pvalue)
                rows.append(dict(unit=label, scale=s, system=sysname, n=len(d), delta=float(d.mean()),
                                 lo=float(np.percentile(bm, 2.5)), hi=float(np.percentile(bm, 97.5)), p=p))
                ps.append(p)
            if ps:
                adj = holm(np.array(ps))
                for r, a in zip(rows, adj):
                    r["p_holm"] = float(a)
                out += rows
    return pd.DataFrame(out)


def krippendorff_alpha_interval(values_by_unit) -> float:
    """α (interval) для списка списков оценок; единицы с < 2 оценок игнорируются."""
    units = [np.asarray(v, dtype=float) for v in values_by_unit if len(v) >= 2]
    allv = np.concatenate(units) if units else np.array([])
    n = len(allv)
    if n < 2:
        return float("nan")
    d_o = sum(2 * (len(u) * (u ** 2).sum() - u.sum() ** 2) / (len(u) - 1) for u in units) / n
    d_e = 2 * (n * (allv ** 2).sum() - allv.sum() ** 2) / (n * (n - 1))
    return float(1 - d_o / d_e) if d_e > 0 else float("nan")


def agreement(df: pd.DataFrame) -> pd.Series:
    items = df[(df.kind == "item") & (df.key_kind == "item")]
    return pd.Series({s: krippendorff_alpha_interval(
        [g[s].values for _, g in items.groupby(["key_item", "system"])]) for s in SCALES})


def auto_correlations(df: pd.DataFrame, auto: pd.DataFrame) -> pd.DataFrame:
    """Spearman между человеческими средними по ячейке (item, система) и автометриками."""
    from scipy.stats import spearmanr

    items = df[(df.kind == "item") & (df.key_kind == "item")]
    cells = items.groupby(["fold", "pair", "utt", "system"])[SCALES].mean().reset_index()
    a = auto.rename(columns={"method": "system"}).copy()
    a["wer_utt"] = a.word_err / a.n_words.clip(lower=1)
    m = cells.merge(a, on=["fold", "pair", "utt", "system"], how="inner")
    pairs = [("accent_sim", "p_tgt"), ("accent_sim", "p_tgt_nap"), ("voice_sim", "spk_cos_src"),
             ("voice_sim", "spk_top1"), ("mos", "wer_utt"), ("accentedness", "p_src")]
    rows = []
    for h, x in pairs:
        if x in m and len(m) >= 5:
            r, p = spearmanr(m[h], m[x])
            rows.append(dict(human=h, auto=x, spearman=float(r), p=float(p), n_cells=len(m)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("responses", nargs="+")
    ap.add_argument("--key", default=str(_ROOT / "outputs" / "human_eval" / "private" / "stimulus_key.csv"))
    ap.add_argument("--auto", nargs="*", default=[])
    ap.add_argument("--out", default=str(_ROOT / "outputs" / "human_eval" / "analysis"))
    ap.add_argument("--include_dev", action="store_true")
    ap.add_argument("--no_exclude", action="store_true", help="не исключать участников по QC (для диагностики)")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    key = pd.read_csv(args.key)
    df = attach_key(load_responses(args.responses, args.include_dev), key)
    q = qc(df)
    q.to_csv(out / "qc_raters.csv")
    keep = q.index if args.no_exclude else q.index[q.keep]
    print(f"Участников: {len(q)}, прошли QC: {int(q.keep.sum())}")
    print(q[["n", "n_flags", "keep", "check_mos", "retest_diff", "median_rt_s"]].round(2).to_string())
    d = df[df.rater.isin(keep)]

    st = system_table(d); st.to_csv(out / "systems.csv")
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", None)
    print("\nСистемы (среднее [95 % CI по участникам]):")
    for sysname, r in st.iterrows():
        print(f"  {sysname:24s} " + "  ".join(f"{s}={r[s]:.2f} [{r[s + '_lo']:.2f},{r[s + '_hi']:.2f}]" for s in SCALES)
              + f"  (n={int(r.n_ratings)})")
    c = contrasts(d); c.to_csv(out / "contrasts_vs_identity.csv", index=False)
    print(f"\nКонтрасты к {BASE} (Δ [95 % CI], Wilcoxon p, Holm):")
    print(c.round(3).to_string(index=False))
    al = agreement(d); al.to_csv(out / "agreement_alpha.csv")
    print("\nKrippendorff α (interval) по ячейкам (item, система):", al.round(3).to_dict())
    if args.auto:
        auto = pd.concat([pd.read_csv(p) for p in args.auto], ignore_index=True)
        ac = auto_correlations(d, auto); ac.to_csv(out / "auto_vs_human.csv", index=False)
        print("\nКорреляции человек ↔ автометрики (Spearman по ячейкам):\n", ac.round(3).to_string(index=False))
    print(f"\nСохранено в {out}")


if __name__ == "__main__":
    main()
