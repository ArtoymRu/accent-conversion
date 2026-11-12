"""E1 and E3. Where accent lives, and the placebo level of the accent terms.

For every layer: the variance components of model (1), their range when each
speaker is left out in turn, the same components with accent labels shuffled
across speakers (the placebo), and how well a
held-out speaker's accent is identified from the conditional view (profile)
and from the global view (centre).

Prediction: `accent x content` lies above its placebo band and peaks at
phonetic layers. Refuted if it sits inside the band at every layer.

    python -m experiments.e1_variance [--synthetic]
"""
import numpy as np

from experiments.common import load_table, setup, show, style, write_csv
from otaccent import metrics, variance

PARTS = ["content", "speaker", "accent (global)", "speaker x content", "accent x content", "residual"]
ACCENT = ["accent x content", "accent (global)"]      # the terms a shuffle of accent labels should kill


def main():
    cfg = setup(__doc__)
    rows, ids = [], []
    for layer in cfg.layers:
        tab = load_table(cfg, layer)
        p = tab.content()
        est = variance.components(tab, p)
        null = variance.permutation_null(tab, p, n_perm=200, seed=cfg.seed)
        loo = variance.leave_one_speaker_out(tab, p)
        for k in PARTS:
            lo, hi = np.quantile([n[k] for n in null], [0.025, 0.975]) if k in ACCENT else (np.nan, np.nan)
            shares = [r[k] / r["total"] for r in loo]
            rows.append(dict(layer=layer, component=k, variance=est[k], share=est[k] / est["total"],
                             loo_min=min(shares), loo_max=max(shares),
                             placebo_lo=lo / est["total"], placebo_hi=hi / est["total"]))
        ids.append(dict(layer=layer, **metrics.loso_accent_id(tab, p)))
    show(rows, ["layer", "component", "share", "loo_min", "loo_max", "placebo_lo", "placebo_hi"], "{:.4f}")
    print("\nleave-one-speaker-out accent identification (chance = 1 / number of accents)")
    show(ids, ["layer", "conditional", "global"])
    write_csv(cfg.results / "e1_variance.csv", rows)
    write_csv(cfg.results / "e1_accent_id.csv", ids)

    plt = style()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9, 3.6), gridspec_kw={"wspace": 0.3})
    x, bottom = np.arange(len(cfg.layers)), np.zeros(len(cfg.layers))
    for k in PARTS:
        share = np.array([r["share"] for r in rows if r["component"] == k])
        ax.bar(x, share, bottom=bottom, label=k)
        bottom += share
    ax.set(xticks=x, xticklabels=cfg.layers, xlabel="layer", ylabel="share of variance", title="Where the variance is")
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    for k, marker in zip(ACCENT, "os"):
        sel = [r for r in rows if r["component"] == k]
        line, = bx.plot(x, [r["share"] for r in sel], marker=marker, label=k)
        lo, hi = np.array([r["placebo_lo"] for r in sel]), np.array([r["placebo_hi"] for r in sel])
        bx.fill_between(x, lo, hi, color=line.get_color(), alpha=0.2, lw=0)
        bx.errorbar(x, (lo + hi) / 2, (hi - lo) / 2, fmt="none", capsize=3, color=line.get_color(), alpha=0.5)
    bx.axhline(0, color="k", lw=0.5)
    bx.set(xticks=x, xticklabels=cfg.layers, xlabel="layer", ylabel="share of variance",
           title="Accent terms against their placebo (band)")
    bx.legend(fontsize=7, frameon=False)
    fig.savefig(cfg.results / "e1_variance.pdf")
    print(f"wrote {cfg.results / 'e1_variance.pdf'}")


if __name__ == "__main__":
    main()
