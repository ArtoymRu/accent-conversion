"""Shared plumbing: configuration, the per-speaker cache, a synthetic stand-in for it."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from otaccent.stats import Table
from otaccent.synthetic import World


def setup(description) -> SimpleNamespace:
    """Parse the command line and read the config. `--synthetic` swaps the corpora for a known world."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--synthetic", action="store_true", help="run on a synthetic world, no data or GPU needed")
    args = ap.parse_args()
    cfg = SimpleNamespace(**yaml.safe_load(open(args.config)), synthetic=args.synthetic)
    if cfg.synthetic:
        cfg.cache, cfg.results = "cache_synthetic", "results_synthetic"
        cfg.layers, cfg.source_accent = [cfg.edit_layer], "acc0"
        if not list(Path(cfg.cache).glob("*.npz")):
            write_synthetic_cache(cfg)
    cfg.cache, cfg.results = Path(cfg.cache), Path(cfg.results)
    cfg.results.mkdir(parents=True, exist_ok=True)
    return cfg


def load_table(cfg, layer) -> Table:
    """Assemble the Table of one layer from the per-speaker cache files."""
    files = sorted(cfg.cache.glob("*.npz"))
    assert files, f"no cache in {cfg.cache}: run experiments/extract.py (or pass --synthetic)"
    rows = [np.load(f) for f in files]
    i = list(rows[0]["layers"]).index(layer)
    return Table(np.array([f.stem for f in files]), np.array([str(r["accent"]) for r in rows]),
                 *(np.stack([r[k][i] for r in rows]) for k in ("n", "sx", "sxx")))


def load_frames(cfg, speaker) -> SimpleNamespace:
    """Frames of the cached utterances of one speaker, at the edit layer.

    .frames (N, d), .labels (N,), .utt (N,) utterance index of each frame,
    .is_test (U,), .texts (U,), .wavs list of U waveforms (empty for pool utterances).
    """
    z = np.load(cfg.cache / f"{speaker}.npz")
    off = z["wav_offsets"]
    return SimpleNamespace(frames=z["frames"].astype(np.float32), labels=z["labels"].astype(int),
                           utt=z["utt"].astype(int), is_test=z["is_test"], texts=z["texts"],
                           wavs=[z["wav"][a:b].astype(np.float32) for a, b in zip(off[:-1], off[1:])])


def save_speaker(path, accent, layers, stats, frames, labels, is_test, texts, wavs):
    """One cache file. `stats` is {layer: (n, sx, sxx)} for this speaker."""
    np.savez_compressed(
        path, accent=accent, layers=np.array(layers),
        n=np.stack([stats[l][0] for l in layers]), sx=np.stack([stats[l][1] for l in layers]),
        sxx=np.stack([stats[l][2] for l in layers]),
        frames=np.concatenate(frames).astype(np.float16), labels=np.concatenate(labels).astype(np.int8),
        utt=np.concatenate([np.full(len(l), i) for i, l in enumerate(labels)]).astype(np.int16),
        is_test=np.array(is_test), texts=np.array(texts),
        wav=np.concatenate(wavs).astype(np.float16), wav_offsets=np.cumsum([0] + [len(w) for w in wavs]))


def write_synthetic_cache(cfg, speakers_per_accent=4, seed=0):
    """The same cache layout, filled from a World: 7 accents, the first one playing `native`."""
    world = World(n_accents=7, speakers_per_accent=speakers_per_accent, n_phones=39, dim=64, seed=seed)
    Path(cfg.cache).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    sentence = lambda: np.r_[-np.ones(5, int), np.repeat(rng.permutation(world.K)[:25], 6), -np.ones(5, int)]
    full = world.table(frames_per_cell=60)
    for s, name in enumerate(full.speakers):
        labels = [sentence() for _ in range(cfg.pool_utts + cfg.test_utts)]
        frames = [world.frames(s, l) for l in labels]
        is_test = [i >= cfg.pool_utts for i in range(len(labels))]
        stats = {cfg.edit_layer: (full.n[s], full.sx[s], full.sxx[s])}
        save_speaker(Path(cfg.cache) / f"{name}.npz", full.accents[s], [cfg.edit_layer], stats, frames, labels,
                     is_test, [""] * len(labels), [np.zeros(0)] * len(labels))


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": 9})
    return plt


def write_csv(path, rows):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


def read_csv(path) -> list[dict]:
    import csv
    def num(v):
        try:
            return float(v)
        except ValueError:
            return v
    return [{k: num(v) for k, v in r.items()} for r in csv.DictReader(open(path))] if Path(path).exists() else []


def show(rows, columns, fmt="{:.3f}"):
    """Print rows as an aligned table."""
    cell = lambda v: fmt.format(v) if isinstance(v, float) else str(v)
    table = [columns] + [[cell(r[c]) for c in columns] for r in rows]
    width = [max(len(row[i]) for row in table) for i in range(len(columns))]
    for row in table:
        print("  ".join(v.ljust(w) for v, w in zip(row, width)))
