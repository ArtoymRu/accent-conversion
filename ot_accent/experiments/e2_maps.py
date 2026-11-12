"""E2, with the placebo of E3 and the strength sweep of E6. Every map, on held-out speakers.

Fold f holds out the f-th speaker of every accent. For each (source, target)
accent pair, the test utterances of the held-out source speaker are edited
towards the target accent by maps fitted on the remaining speakers, vocoded,
encoded again, and scored:

  accent        cosine of the speaker-centred phone profile to the target accent's
  accent_hit    1 if the target is the nearest accent
  centre_shift  how far the voice centre moved, in units of the typical distance between speakers
  profile_kept  cosine between the speaker's personal phone profile after and before the edit
  secs, ecapa   speaker cosine to the original recording (Resemblyzer, ECAPA)   (audio_metrics)
  wer           Whisper word error rate                                          (audio_metrics)
  effort        mean squared displacement of the edited frames

Hypotheses: conditional maps beat global ones on accent for held-out
speakers; translations beat barycentres on identity at equal accent; the
placebo moves accent by nothing.

    python -m experiments.e2_maps [--synthetic]

With --synthetic there is no audio: edited frames are scored directly.
Finished rows are appended to results/e2_maps.csv, so a run can be resumed.
"""
import csv

import numpy as np

from experiments.common import accent_pairs, load_frames, load_table, read_csv, setup, show, style, write_csv
from otaccent import maps, metrics
from otaccent.stats import centres, conditional_delta, global_delta, moments, profiles

COLUMNS = ["fold", "source", "target", "system", "strength", "accent", "accent_hit", "centre_shift",
           "profile_kept", "secs", "ecapa", "wer", "effort"]


def build_systems(tab, p, s, src, tgt, pool, k):
    """name -> editor(X, c, strength). All maps are fitted on speakers `src` and `tgt` only."""
    assert len(src) > 1, "the placebo needs at least two reference speakers of the source accent"
    m = tab.means()
    cen = centres(m, p)
    Delta, delta = conditional_delta(m, p, src, tgt), global_delta(m, p, src, tgt)
    placebo = conditional_delta(m, p, src[1:], src[:1])        # same accent on both sides: speaker leak only
    whole = dict(centred=False, conditional=False)
    g_src, g_tgt = moments(tab, src, p, **whole), moments(tab, tgt, p, **whole)
    c_src, c_tgt = moments(tab, src, p), moments(tab, tgt, p)
    pool_X, pool_c, pool_centred = pool
    return {
        "placebo": lambda X, c, a: maps.shift(X, c, placebo, lam=a),
        "global shift": lambda X, c, a: maps.shift(X, c, delta=delta, kappa=a),
        "conditional shift": lambda X, c, a: maps.shift(X, c, Delta, lam=a),
        "conditional + global shift": lambda X, c, a: maps.shift(X, c, Delta, delta, lam=a, kappa=a),
        "global affine": lambda X, c, a: maps.affine(X, c, g_src, g_tgt, lam=a),
        "conditional affine": lambda X, c, a: maps.affine(X, c, c_src, c_tgt, centre=cen[s], lam=a),
        "global barycentre": lambda X, c, a: maps.barycentre(X, c, pool_X, pool_c, k, a, conditional=False),
        "conditional barycentre": lambda X, c, a: maps.barycentre(X, c, pool_X, pool_c, k, a),
        "conditional barycentre, centred":
            lambda X, c, a: maps.barycentre(X - cen[s], c, pool_centred, pool_c, k, a) + cen[s],
    }


def main():
    cfg = setup(__doc__)
    tab = load_table(cfg, cfg.edit_layer)
    p, m = tab.content(), tab.means()
    cen, names = centres(m, p), np.unique(tab.accents)
    pairs = accent_pairs(cfg, names)
    speaker_scale = np.sqrt(np.mean([np.sum((cen[i] - cen[j]) ** 2)
                                     for i in range(len(cen)) for j in range(i)]))
    frames = {s: load_frames(cfg, s) for s in tab.speakers}
    real = not cfg.synthetic
    audio = real and cfg.audio_metrics
    if real:
        import soundfile as sf
        from otaccent import features
        vocoder = features.Vocoder(cfg.device, cfg.prematched_vocoder)
        encoder = features.Encoder(cfg.device)
    if audio:
        asr = metrics.Asr(cfg.device, cfg.whisper)
        voices = {"secs": metrics.Resemblyzer(cfg.device), "ecapa": metrics.Ecapa(cfg.device)}

    def channel(Y):
        """Edited frames -> waveform -> frames again. Without audio the channel is the identity."""
        if not real:
            return None, Y
        wav = vocoder(Y)
        return wav, encoder(wav, (cfg.edit_layer,))[cfg.edit_layer]

    out_path = cfg.results / "e2_maps.csv"
    done = {(r["fold"], r["source"], r["target"], r["system"], r["strength"]) for r in read_csv(out_path)}
    if not done:
        csv.writer(open(out_path, "w", newline="")).writerow(COLUMNS)
    n_folds = min(len(tab.idx(a)) for a in names)
    for fold in range(n_folds):
        held = [tab.speakers[tab.idx(a)[fold]] for a in names]             # one speaker per accent
        refs = np.stack([profiles(m, p)[tab.idx(a, exclude=held)].mean(0) for a in names])
        for source in sorted({a for a, _ in pairs}):
            s = tab.idx(source)[fold]
            f = frames[tab.speakers[s]]
            utts = [(f.frames[f.utt == u], f.labels[f.utt == u], u) for u in np.flatnonzero(f.is_test)]
            identity = [channel(X) for X, _, _ in utts]
            original = {k: [enc(f.wavs[u]) for _, _, u in utts] for k, enc in voices.items()} if audio else {}

            def gather(outputs):
                Z = np.concatenate([z[:len(c)] for (_, z), (_, c, _) in zip(outputs, utts)])
                lab = np.concatenate([c[:len(z)] for (_, z), (_, c, _) in zip(outputs, utts)])
                return Z, lab

            def score(outputs, target):
                Z, lab = gather(outputs)
                Z0, lab0 = gather(identity)
                prof, present = metrics.profile(Z, lab, p)
                scores = metrics.accent_scores(prof, present, refs, p)
                t = list(names).index(target)
                row = dict(accent=scores[t], accent_hit=float(scores.argmax() == t), secs=np.nan, ecapa=np.nan, wer=np.nan)
                row["centre_shift"] = np.linalg.norm(metrics.centre(Z, lab, p) - metrics.centre(Z0, lab0, p)) / speaker_scale
                row["profile_kept"] = metrics.profile_kept(prof, metrics.profile(Z0, lab0, p)[0], present, refs, p)
                if audio:
                    for k, enc in voices.items():
                        row[k] = float(np.mean([enc(w) @ e for (w, _), e in zip(outputs, original[k])]))
                    row["wer"] = asr.wer([str(f.texts[u]) for _, _, u in utts], [asr(w) for w, _ in outputs])
                return row

            for target in [b for a, b in pairs if a == source]:
                src, tgt = tab.idx(source, exclude=held), tab.idx(target, exclude=held)
                pf = [frames[tab.speakers[t]] for t in tgt]
                pool = (np.concatenate([q.frames[~q.is_test[q.utt]] for q in pf]),
                        np.concatenate([q.labels[~q.is_test[q.utt]] for q in pf]),
                        np.concatenate([q.frames[~q.is_test[q.utt]] - cen[t] for q, t in zip(pf, tgt)]))
                systems = build_systems(tab, p, s, src, tgt, pool, cfg.knn)
                chosen = [n for n in systems if cfg.systems is None or n in cfg.systems]
                runs = [("identity", 0.0, lambda X, c, a: X)] + \
                       [(name, a, systems[name]) for name in chosen for a in cfg.strengths]
                for name, a, fn in runs:
                    if (float(fold), source, target, name, a) in done:
                        continue
                    edited = [fn(X, c, a) for X, c, _ in utts]
                    outputs = identity if name == "identity" else [channel(Y) for Y in edited]
                    effort = np.mean(np.concatenate([((Y - X) ** 2).sum(1)[c >= 0] for Y, (X, c, _) in zip(edited, utts)]))
                    row = dict(fold=fold, source=source, target=target, system=name, strength=a,
                               effort=effort, **score(outputs, target))
                    csv.writer(open(out_path, "a", newline="")).writerow([row[k] for k in COLUMNS])
                    if real and fold == 0 and a in (0.0, 1.0):
                        d = cfg.results / "e2_audio" / f"{source}_to_{target}"
                        d.mkdir(parents=True, exist_ok=True)
                        sf.write(d / f"{name.replace(' ', '_').replace(',', '')}.wav", outputs[0][0], features.SR)
                print(f"fold {fold}: {source} -> {target} done")
    report(cfg, read_csv(out_path))


def report(cfg, rows):
    """Summary at strength 1 (differences to identity, mean and standard error over items) and the frontier."""
    base = {(r["fold"], r["source"], r["target"]): r for r in rows if r["system"] == "identity"}
    for r in rows:
        b = base[(r["fold"], r["source"], r["target"])]
        for k in ("accent", "secs", "ecapa", "wer"):
            r["d_" + k] = r[k] - b[k]
    systems = list(dict.fromkeys(r["system"] for r in rows if r["system"] != "identity"))
    keys = ["d_accent", "accent_hit", "centre_shift", "profile_kept", "d_secs", "d_ecapa", "d_wer", "effort"]
    summary = []
    for name in systems:
        sel = [r for r in rows if r["system"] == name and r["strength"] == 1.0]
        if sel:
            row = dict(system=name, items=len(sel))
            for k in keys:
                v = np.array([r[k] for r in sel])
                row[k], row[k + "_sem"] = float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan
            summary.append(row)
    print("\nstrength 1, differences to identity resynthesis, mean over folds and pairs (standard errors in the csv)")
    show(summary, ["system", "items"] + keys)
    write_csv(cfg.results / "e2_summary.csv", summary)

    audio = not np.isnan(rows[0]["secs"])
    ykey, ylabel = ("d_secs", "change in speaker cosine (Resemblyzer)") if audio else \
                   ("centre_shift", "shift of the voice centre (speaker distances)")
    mean = lambda sel, k: float(np.mean([r[k] for r in sel]))
    plt = style()
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for name in systems:
        strengths = sorted({r["strength"] for r in rows if r["system"] == name})
        pts = [[r for r in rows if r["system"] == name and r["strength"] == a] for a in strengths]
        ax.plot([mean(q, "d_accent") for q in pts], [mean(q, ykey) for q in pts], "o-", ms=3, lw=1, label=name)
    if not audio:
        ax.invert_yaxis()
    ax.set(xlabel="accent moved towards target (change in profile cosine)", ylabel=ylabel,
           title="Accent against identity; strength grows along each line")
    ax.legend(fontsize=6, frameon=False)
    fig.savefig(cfg.results / "e2_frontier.pdf")
    print(f"wrote {cfg.results / 'e2_frontier.pdf'}")


if __name__ == "__main__":
    main()
