"""E2, with the placebo of E3 and the strength sweep of E6. Every map, on held-out speakers.

Fold f holds out the f-th speaker of every accent. The test utterances of the
held-out source speaker are edited towards each target accent by maps fitted
on the remaining speakers, vocoded, encoded again, and scored:

  accent        cosine of the speaker-centred phone profile to the target accent's
  accent_hit    1 if the target is the nearest accent
  centre_shift  how far the voice centre moved, in units of the typical distance between speakers
  profile_kept  cosine between the speaker's personal phone profile after and before the edit
  speaker_cos   ECAPA cosine to the original recording        (audio_metrics)
  wer           Whisper word error rate                        (audio_metrics)
  effort        mean squared displacement of the edited frames

Predictions: conditional maps beat global ones on accent for held-out
speakers; translations beat barycentres on identity at equal accent; the
placebo moves accent by nothing.

    python -m experiments.e2_maps [--synthetic]

With --synthetic there is no audio: edited frames are scored directly.
Finished rows are appended to results/e2_maps.csv, so a run can be resumed.
"""
import csv

import numpy as np

from experiments.common import load_frames, load_table, read_csv, setup, show, style
from otaccent import maps, metrics
from otaccent.stats import centres, conditional_delta, global_delta, moments, profiles

COLUMNS = ["fold", "source", "target", "system", "strength", "accent", "accent_hit", "centre_shift",
           "profile_kept", "speaker_cos", "wer", "effort"]


def build_systems(tab, p, s, src, tgt, pool, k):
    """name -> editor(X, c, strength). All maps are fitted on speakers `src` and `tgt` only."""
    m = tab.means()
    cen = centres(m, p)
    Delta, delta = conditional_delta(m, p, src, tgt), global_delta(m, p, src, tgt)
    placebo = conditional_delta(m, p, src[1:], src[:1])        # same accent on both sides: speaker leak only
    assert len(src) > 1, "the placebo needs at least two reference speakers of the source accent"
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
    speaker_scale = np.sqrt(np.mean([np.sum((cen[i] - cen[j]) ** 2)
                                     for i in range(len(cen)) for j in range(i)]))
    frames = {s: load_frames(cfg, s) for s in tab.speakers}
    real = not cfg.synthetic
    if real:
        import soundfile as sf
        from otaccent import features
        vocoder = features.Vocoder(cfg.device, cfg.prematched_vocoder)
        encoder = features.Encoder(cfg.device)
    if real and cfg.audio_metrics:
        asr, voice = metrics.Asr(cfg.device), metrics.SpeakerEncoder(cfg.device)

    def channel(Y):
        """Edited frames -> waveform -> frames again. Without audio the channel is the identity."""
        if not real:
            return None, Y
        wav = vocoder(Y)
        return wav, encoder(wav, (cfg.edit_layer,))[cfg.edit_layer]

    out_path = cfg.results / "e2_maps.csv"
    done = {(r["fold"], r["target"], r["system"], r["strength"]) for r in read_csv(out_path)}
    if not done:
        csv.writer(open(out_path, "w", newline="")).writerow(COLUMNS)
    n_folds = min(len(tab.idx(a)) for a in names)
    for fold in range(n_folds):
        held = [tab.speakers[tab.idx(a)[fold]] for a in names]             # one speaker per accent
        s = tab.idx(cfg.source_accent)[fold]
        f = frames[tab.speakers[s]]
        utts = [(f.frames[f.utt == u], f.labels[f.utt == u], u) for u in np.flatnonzero(f.is_test)]
        labels = np.concatenate([c for _, c, _ in utts])
        refs = np.stack([profiles(m, p)[tab.idx(a, exclude=held)].mean(0) for a in names])
        identity = [channel(X) for X, _, _ in utts]

        def score(outputs, target):
            Z = np.concatenate([z[:len(c)] for (_, z), (_, c, _) in zip(outputs, utts)])
            lab = np.concatenate([c[:len(z)] for (_, z), (_, c, _) in zip(outputs, utts)])
            prof, present = metrics.profile(Z, lab, p)
            scores = metrics.accent_scores(prof, present, refs, p)
            t = list(names).index(target)
            row = dict(accent=scores[t], accent_hit=float(scores.argmax() == t), speaker_cos=np.nan, wer=np.nan)
            Z0 = np.concatenate([z[:len(c)] for (_, z), (_, c, _) in zip(identity, utts)])
            row["centre_shift"] = np.linalg.norm(metrics.centre(Z, lab, p) - metrics.centre(Z0, lab, p)) / speaker_scale
            row["profile_kept"] = metrics.profile_kept(prof, metrics.profile(Z0, lab, p)[0], present, refs, p)
            if real and cfg.audio_metrics:
                row["speaker_cos"] = float(np.mean([voice(w) @ voice(f.wavs[u]) for (w, _), (_, _, u) in zip(outputs, utts)]))
                row["wer"] = asr.wer([str(f.texts[u]) for _, _, u in utts], [asr(w) for w, _ in outputs])
            return row

        for target in [a for a in names if a != cfg.source_accent]:
            src, tgt = tab.idx(cfg.source_accent, exclude=held), tab.idx(target, exclude=held)
            pf = [frames[tab.speakers[t]] for t in tgt]
            pool = (np.concatenate([q.frames[~q.is_test[q.utt]] for q in pf]),
                    np.concatenate([q.labels[~q.is_test[q.utt]] for q in pf]),
                    np.concatenate([q.frames[~q.is_test[q.utt]] - cen[t] for q, t in zip(pf, tgt)]))
            systems = build_systems(tab, p, s, src, tgt, pool, cfg.knn)
            runs = [("identity", 0.0, lambda X, c, a: X)] + \
                   [(name, a, fn) for name, fn in systems.items() for a in cfg.strengths]
            for name, a, fn in runs:
                if (float(fold), target, name, a) in done:
                    continue
                edited = [fn(X, c, a) for X, c, _ in utts]
                outputs = identity if name == "identity" else [channel(Y) for Y in edited]
                effort = np.mean(np.concatenate([((Y - X) ** 2).sum(1)[c >= 0] for Y, (X, c, _) in zip(edited, utts)]))
                row = dict(fold=fold, source=cfg.source_accent, target=target, system=name, strength=a,
                           effort=effort, **score(outputs, target))
                csv.writer(open(out_path, "a", newline="")).writerow([row[k] for k in COLUMNS])
                if real and fold == 0 and a in (0.0, 1.0):
                    d = cfg.results / "e2_audio" / target
                    d.mkdir(parents=True, exist_ok=True)
                    sf.write(d / f"{name.replace(' ', '_').replace(',', '')}.wav", outputs[0][0], features.SR)
            print(f"fold {fold}: {cfg.source_accent} -> {target} done")
    report(cfg, read_csv(out_path))


def report(cfg, rows):
    """Table at strength 1 (differences to identity) and the accent-identity frontier."""
    base = {(r["fold"], r["target"]): r for r in rows if r["system"] == "identity"}
    for r in rows:
        b = base[(r["fold"], r["target"])]
        r["d_accent"], r["d_wer"] = r["accent"] - b["accent"], r["wer"] - b["wer"]
        r["d_speaker_cos"] = r["speaker_cos"] - b["speaker_cos"]
    systems = list(dict.fromkeys(r["system"] for r in rows if r["system"] != "identity"))
    mean = lambda sel, k: float(np.mean([r[k] for r in sel]))
    keys = ["d_accent", "accent_hit", "centre_shift", "profile_kept", "d_speaker_cos", "d_wer", "effort"]
    table = []
    for name in systems:
        sel = [r for r in rows if r["system"] == name and r["strength"] == 1.0]
        if sel:
            table.append(dict(system=name, **{k: mean(sel, k) for k in keys}))
    print("\nstrength 1, mean over folds and targets, differences to identity resynthesis")
    show(table, ["system"] + keys)

    audio = not np.isnan(rows[0]["speaker_cos"])
    ykey, ylabel = ("d_speaker_cos", "change in speaker cosine") if audio else \
                   ("centre_shift", "shift of the voice centre (speaker distances)")
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
