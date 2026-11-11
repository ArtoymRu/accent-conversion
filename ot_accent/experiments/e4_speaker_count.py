"""E4. How many speakers does an accent direction need? (Proposition 1)

A delta is estimated twice, from disjoint sets of S speakers per accent. Half
the squared distance between the two is the error of one estimate; their inner
product is the squared norm of the true delta.

Prediction: error = (2/S) x the speaker variance for the global delta, and
(2/S) x the speaker-x-content variance for the conditional one, with both
variances taken from E1 (plus the small sampling noise of the means). A direction is usable once its error is below its
signal. With four speakers per accent only S = 1, 2 can be measured; a corpus
with more speakers per accent extends the curve.

    python -m experiments.e4_speaker_count [--synthetic]
"""
import numpy as np

from experiments.common import load_table, setup, show, style, write_csv
from otaccent import variance
from otaccent.synthetic import World

KINDS = {"global": "speaker", "conditional": "speaker x content"}


def main():
    cfg = setup(__doc__)
    if cfg.synthetic:        # more speakers than any cache, to see the whole curve
        tab = World(n_accents=7, speakers_per_accent=32, n_phones=39, dim=64, seed=cfg.seed).table(60)
    else:
        tab = load_table(cfg, cfg.edit_layer)
    p, comp, noise = tab.content(), variance.components(tab), variance.sampling_noise(tab)
    targets = [a for a in np.unique(tab.accents) if a != cfg.source_accent]
    max_S = min(len(tab.idx(a)) for a in np.unique(tab.accents)) // 2
    rows = []
    for S in [s for s in (1, 2, 4, 8, 16, 32) if s <= max_S]:
        e = [variance.delta_error(tab, cfg.source_accent, t, S, p, seed=cfg.seed) for t in targets]
        for kind, part in KINDS.items():
            rows.append(dict(S=S, delta=kind,
                             error=float(np.mean([x[f"{kind} error"] for x in e])),
                             predicted=2 / S * (comp[part] + noise[part]),
                             signal=float(np.mean([x[f"{kind} signal"] for x in e]))))
    show(rows, ["S", "delta", "error", "predicted", "signal"], "{:.4f}")
    write_csv(cfg.results / "e4_speaker_count.csv", rows)

    plt = style()
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for kind in KINDS:
        sel = [r for r in rows if r["delta"] == kind]
        S = np.array([r["S"] for r in sel])
        line, = ax.loglog(S, [r["error"] for r in sel], "o", label=f"{kind}: measured error")
        ax.loglog(S, [r["predicted"] for r in sel], "-", color=line.get_color(), label="(2/S) x variance from E1")
        ax.axhline(np.mean([r["signal"] for r in sel]), ls=":", color=line.get_color(), label="signal")
    ax.set(xlabel="reference speakers per accent, S", ylabel="squared error of the delta", xticks=S,
           xticklabels=S, title="Error of an accent direction")
    ax.legend(fontsize=6, frameon=False)
    fig.savefig(cfg.results / "e4_speaker_count.pdf")
    print(f"wrote {cfg.results / 'e4_speaker_count.pdf'}")


if __name__ == "__main__":
    main()
