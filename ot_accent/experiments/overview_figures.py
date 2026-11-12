"""Overview figures for talks and the README, drawn from the result files.

Reads the CSVs written by e1_variance, e4_speaker_count and e2_maps and draws
five figures into results/overview/: the idea (a two-dimensional toy), where
the variance is, how many speakers a direction needs, what strength does, and
a scorecard of all maps at strength 1.

    python -m experiments.overview_figures [--synthetic]
"""
import numpy as np

from experiments.common import load_table, read_csv, setup, style
from otaccent import metrics, variance

BLUE, ORANGE, AQUA, VIOLET, GREY, INK, MUTED, GRID = \
    "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#a3a299", "#262624", "#6b6a63", "#e6e5df"
SURFACE = "#fcfcfb"


def axes(plt, n, size, **kw):
    fig, ax = plt.subplots(1, n, figsize=size, facecolor=SURFACE, **kw)
    for a in np.atleast_1d(ax):
        a.set_facecolor(SURFACE)
        a.tick_params(colors=MUTED, length=0)
        a.grid(color=GRID, lw=0.8)
        a.set_axisbelow(True)
        for s in a.spines.values():
            s.set_visible(False)
    return fig, ax


def idea(plt, out):
    """Two accents, three speakers each, four phones, in two dimensions."""
    rng = np.random.default_rng(3)
    ang = np.pi / 4 + np.arange(4) * np.pi / 2
    mu = 1.6 * np.c_[np.cos(ang), np.sin(ang)]
    v = np.array([[0.5, 0.5], [0.1, -0.7], [0.6, 0.1], [-0.6, 0.3]])              # accent B bends each phone
    u = {0: np.array([[-3.9, 2.0], [-0.9, -2.3], [-4.3, -2.0]]), 1: np.array([[3.5, 2.3], [4.3, -1.7], [0.5, 2.2]])}
    w = rng.normal(0, 0.08, (2, 3, 4, 2))
    pts = {a: mu[None] + u[a][:, None] + a * v[None] + w[a] for a in (0, 1)}      # (speakers, phones, 2)
    col, names = {0: BLUE, 1: ORANGE}, {0: "accent A", 1: "accent B"}
    closed = lambda q: np.r_[q, q[:1]].T
    fig, (l, r) = axes(plt, 2, (10, 4.4), gridspec_kw={"wspace": 0.06})
    for a in (0, 1):
        for s in range(3):
            q = pts[a][s]
            l.fill(*q.T, facecolor=col[a], alpha=0.1, edgecolor="none")
            l.plot(*closed(q), color=col[a], lw=1.3)
            l.scatter(*q.T, s=30, color=col[a], edgecolor=SURFACE, lw=1, zorder=3, label=names[a] if s == 0 else None)
            l.scatter(*q.mean(0), marker="+", s=80, color=col[a], lw=1.8, zorder=4)
            r.plot(*closed(q - q.mean(0)), color=col[a], lw=0.8, alpha=0.4)
    c0, c1 = pts[0].mean((0, 1)), pts[1].mean((0, 1))
    l.annotate("", c1, c0, arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.2))
    l.text(*((c0 + c1) / 2 + [0, 0.45]), "one global delta", color=INK, ha="center", fontsize=9.5,
           bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))
    l.set_title("As recorded: one shape per speaker", color=INK, fontsize=10.5, loc="left")
    l.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK)
    mean = {a: (pts[a] - pts[a].mean(1, keepdims=True)).mean(0) for a in (0, 1)}
    for a in (0, 1):
        r.plot(*closed(mean[a]), color=col[a], lw=2.2)
        r.scatter(*mean[a].T, s=40, color=col[a], edgecolor=SURFACE, lw=1, zorder=3)
    for k in range(4):
        r.annotate("", mean[1][k], mean[0][k], arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.2, shrinkA=4, shrinkB=4))
        r.text(*(mean[0][k] + [[0.5, -0.2], [-0.42, 0.34], [-0.42, -0.34], [0.42, -0.34]][k]),
               ["/t/", "/w/", "/ae/", "/r/"][k], color=MUTED, ha="center", va="center", fontsize=11)
    r.set_title("Speaker centres removed: one delta per phone", color=INK, fontsize=10.5, loc="left")
    for ax in (l, r):
        ax.set(xticks=[], yticks=[])
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(False)
        ax.margins(0.1)
    fig.savefig(out / "fig_idea.png", dpi=170, facecolor=SURFACE, bbox_inches="tight")


def variance_figure(plt, cfg, out):
    tab = load_table(cfg, cfg.edit_layer)
    est, null = variance.components(tab), variance.permutation_null(tab, n_perm=400, seed=cfg.seed)
    parts = ["content", "residual", "speaker", "speaker x content", "accent x content", "accent (global)"]
    colour = {"accent x content": BLUE, "accent (global)": ORANGE}
    fig, (a, b, c) = axes(plt, 3, (11, 3.1), gridspec_kw={"width_ratios": [1.25, 1, 0.8], "wspace": 0.45})
    share = [est[k] / est["total"] for k in parts]
    a.barh(parts[::-1], share[::-1], color=[colour.get(k, GREY) for k in parts[::-1]], height=0.62)
    for y, s in enumerate(share[::-1]):
        a.text(s + 0.006, y, f"{100 * s:.1f}%", va="center", color=INK, fontsize=8.5)
    a.set_xlim(0, max(share) * 1.2)
    a.grid(axis="y", visible=False)
    a.tick_params(axis="y", labelcolor=INK)
    a.set_xlabel("share of the variance of frames", color=MUTED)
    a.set_title("Where the variance is", color=INK, fontsize=10, loc="left")

    obs = est["accent x content"] / est["total"]
    b.hist([n["accent x content"] / est["total"] for n in null], bins=30, color=GREY)
    b.axvline(obs, color=BLUE, lw=2)
    top = b.get_ylim()[1]
    b.text(obs, top * 0.93, "measured  ", color=INK, ha="right", fontsize=8.5)
    b.text(obs * 0.22, top * 0.6, "accent labels\nshuffled", color=MUTED, ha="left", fontsize=8.5)
    b.set_yticks([])
    b.grid(axis="y", visible=False)
    b.set_xlabel("share of accent x content", color=MUTED)
    b.set_title("Against its placebo", color=INK, fontsize=10, loc="left")

    acc = metrics.loso_accent_id(tab)
    c.bar(["speaker\ncentre", "phone\nprofile"], [acc["global"], acc["conditional"]], color=[ORANGE, BLUE], width=0.55)
    chance = 1 / len(np.unique(tab.accents))
    c.axhline(chance, color=MUTED, ls=(0, (4, 3)), lw=1)
    c.text(0.5, chance + 0.02, "chance", color=MUTED, fontsize=8.5, ha="center")
    for x, v in enumerate([acc["global"], acc["conditional"]]):
        c.text(x, v + 0.02, f"{v:.2f}", ha="center", color=INK, fontsize=8.5)
    c.set_ylim(0, 1.12)
    c.grid(axis="x", visible=False)
    c.tick_params(axis="x", labelcolor=INK)
    c.set_title("Accent of an unseen speaker,\nidentified from", color=INK, fontsize=10, loc="left")
    fig.savefig(out / "fig_variance.png", dpi=170, facecolor=SURFACE, bbox_inches="tight")


def speaker_count(plt, cfg, out):
    rows = read_csv(cfg.results / "e4_speaker_count.csv")
    fig, ax = axes(plt, 1, (5.6, 3.9))
    names = {"global": "one global delta", "conditional": "one delta per phone"}
    for kind, col in (("global", ORANGE), ("conditional", BLUE)):
        sel = [r for r in rows if r["delta"] == kind]
        S = np.array([r["S"] for r in sel])
        grid = np.geomspace(S.min(), max(S.max(), 32), 50)
        ax.loglog(grid, sel[0]["predicted"] * S[0] / grid, color=col, lw=2)
        ax.loglog(S, [r["error"] for r in sel], "o", color=col, ms=7, mec=SURFACE, mew=1.2)
        signal = np.mean([r["signal"] for r in sel])
        ax.axhline(signal, color=col, ls=(0, (2, 2)), lw=1.3)
        up = kind == "conditional"
        ax.text(grid[-1], signal * (1.12 if up else 0.89), f"true size, {names[kind][4:]}", color=INK,
                ha="right", va="bottom" if up else "top", fontsize=8)
        ax.text(grid[-1] * 1.08, sel[0]["predicted"] * S[0] / grid[-1], names[kind], color=INK, fontsize=9, va="center")
    ax.set_xticks([1, 2, 4, 8, 16, 32], [1, 2, 4, 8, 16, 32])
    ax.minorticks_off()
    ax.text(0.02, 0.03, "points: measured    lines: predicted    dotted: true size of the delta",
            transform=ax.transAxes, color=MUTED, fontsize=7.5)
    ax.set_xlabel("reference speakers per accent, S", color=MUTED)
    ax.set_ylabel("squared error of the estimated delta", color=MUTED)
    ax.set_title("Error of an accent direction falls as 1/S", color=INK, fontsize=10.5, loc="left")
    fig.savefig(out / "fig_speaker_count.png", dpi=170, facecolor=SURFACE, bbox_inches="tight")


def _e2(cfg):
    rows = read_csv(cfg.results / "e2_maps.csv")
    base = {(r["fold"], r["source"], r["target"]): r for r in rows if r["system"] == "identity"}
    for r in rows:
        r["d_accent"] = r["accent"] - base[(r["fold"], r["source"], r["target"])]["accent"]
    return rows


def strength(plt, cfg, out):
    rows = _e2(cfg)
    series = [("conditional shift", BLUE), ("global shift", ORANGE), ("conditional barycentre", AQUA), ("placebo", GREY)]
    panels = [("d_accent", "Accent moved towards the target", "change in profile cosine"),
              ("centre_shift", "Voice centre moved", "typical distances between speakers"),
              ("profile_kept", "Personal phone profile kept", "cosine to the unedited speaker")]
    fig, ax = axes(plt, 3, (11, 3.2), gridspec_kw={"wspace": 0.3})
    for a, (key, title, unit) in zip(ax, panels):
        for name, col in series:
            lam = sorted({r["strength"] for r in rows if r["system"] == name})
            y = [np.mean([r[key] for r in rows if r["system"] == name and r["strength"] == s]) for s in lam]
            zero = 1.0 if key == "profile_kept" else 0.0
            wide = name == "conditional shift"          # drawn wider so that a coinciding line stays visible
            a.plot([0] + lam, [zero] + y, "-o", color=col, lw=4.5 if wide else 2, ms=5, mec=SURFACE, mew=1,
                   label=name, zorder=2 if wide else 3)
        a.set_xlabel("strength", color=MUTED)
        a.set_ylabel(unit, color=MUTED, fontsize=8.5)
        a.set_title(title, color=INK, fontsize=10, loc="left")
    ax[0].legend(frameon=False, fontsize=8, labelcolor=INK, loc="center right")
    fig.savefig(out / "fig_strength.png", dpi=170, facecolor=SURFACE, bbox_inches="tight")


def scorecard(plt, cfg, out):
    rows = [r for r in _e2(cfg) if r["strength"] == 1.0]
    order = ["placebo", "global shift", "global affine", "global barycentre", "conditional barycentre",
             "conditional barycentre, centred", "conditional affine", "conditional shift"]
    panels = [("d_accent", "Accent moved towards the target"), ("centre_shift", "Voice centre moved"),
              ("profile_kept", "Personal phone profile kept")]
    fig, ax = axes(plt, 3, (11, 3.3), sharey=True, gridspec_kw={"wspace": 0.12})
    for a, (key, title) in zip(ax, panels):
        val = [np.mean([r[key] for r in rows if r["system"] == s]) for s in order]
        a.barh(order, val, color=[BLUE if s == "conditional shift" else GREY for s in order], height=0.62)
        for y, v in enumerate(val):
            a.text(v + 0.02 * max(val), y, f"{v:.2f}", va="center", color=INK, fontsize=8)
        a.set_xlim(0, max(val) * 1.18)
        a.grid(axis="y", visible=False)
        a.set_title(title, color=INK, fontsize=10, loc="left")
    ax[0].tick_params(axis="y", labelcolor=INK)
    fig.savefig(out / "fig_scorecard.png", dpi=170, facecolor=SURFACE, bbox_inches="tight")


def main():
    cfg = setup(__doc__)
    plt = style()
    out = cfg.results / "overview"
    out.mkdir(exist_ok=True)
    idea(plt, out)
    variance_figure(plt, cfg, out)
    speaker_count(plt, cfg, out)
    strength(plt, cfg, out)
    scorecard(plt, cfg, out)
    print(f"wrote five figures to {out}")


if __name__ == "__main__":
    main()
