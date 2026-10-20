"""
Тесты анализа human eval на СИМУЛИРОВАННЫХ ответах (проверка конвейера, не результаты исследования).
Запуск:
    python tests/test_humaneval.py     (без pytest)
    python -m pytest tests/test_humaneval.py
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.humaneval.analyze_responses import (
    SCALES, agreement, attach_key, contrasts, holm, krippendorff_alpha_interval, load_responses, qc, system_table)

SYSTEMS = ["original", "identity", "ubw-mean_l1.0_spknorm", "uent_l0.5", "uent_l1.0"]
# «истинные» средние по системам: (accent_sim, accentedness, mos, voice_sim)
TRUE = {"original": (2.0, 4.0, 4.6, 4.7), "identity": (2.5, 4.5, 3.9, 3.8),
        "ubw-mean_l1.0_spknorm": (3.4, 4.3, 3.7, 3.2), "uent_l0.5": (3.6, 5.0, 3.0, 2.2), "uent_l1.0": (4.8, 5.8, 1.9, 1.3)}
N_ITEMS = 10


def make_key():
    rows = []
    for i in range(N_ITEMS):
        for s in SYSTEMS:
            rows.append(dict(sid=f"s_{i}_{s}", kind="item", item=f"i{i:02d}", system=s, fold=i % 4,
                             pair=f"p{i % 6}", tgt="korean", src="hindi", utt=f"u{i}"))
    for c in range(2):
        rows.append(dict(sid=f"s_chk{c}", kind="check_low", item=f"i{c:02d}", system="degraded_original",
                         fold=0, pair="p0", tgt="korean", src="hindi", utt="u0"))
    return pd.DataFrame(rows)


def simulate(rater_idx, rng, bad=False):
    pos = rater_idx % 5
    resp = []

    def rating(mean, noise):
        return int(np.clip(round(mean + rng.normal(0, noise)), 1, 9))

    def trial(sid, kind, item, k, truth):
        if bad:
            r = {"accent_sim": 4, "accentedness": 5, "mos": 5, "voice_sim": 5}
        else:
            r = {"accent_sim": min(7, rating(truth[0], 0.8)), "accentedness": rating(truth[1], 1.0),
                 "mos": min(5, rating(truth[2], 0.6)), "voice_sim": min(5, rating(truth[3], 0.6))}
        resp.append(dict(trial_index=k, block_index=0, tgt="korean", kind=kind, item_id=item, stimulus=f"audio/{sid}.wav",
                         voice_ref="audio/v.wav", ratings=r, plays=1, voice_ref_plays=1, listened_fully=not bad,
                         rt_ms=3000 if bad else int(rng.integers(9000, 20000)), t_end="2026-01-01T00:00:00Z"))

    k = 0
    for i in range(N_ITEMS):
        s = SYSTEMS[(i + pos) % 5]
        trial(f"s_{i}_{s}", "item", f"i{i:02d}", k, TRUE[s]); k += 1
    for c in range(2):                                   # «плохие» контроли
        trial(f"s_chk{c}", "check", f"c{c}", k, (6.0, 8.0, 1.3, 3.0)); k += 1
    s0 = SYSTEMS[pos % 5]                                # повтор первого item
    trial(f"s_0_{s0}", "repeat", "i00", k, TRUE[s0])
    return dict(study="t", manifest_version=1, manifest_seed=0, rater=rater_idx, pid=f"P{rater_idx}", dev=False,
                position=pos, started=f"2026-01-01T00:{rater_idx % 60:02d}:00Z", finished="x", user_agent="t",
                setup={"heard": "yes"}, demographics={}, responses=resp)


def build_df(n_good=30, with_bad=True):
    rng = np.random.default_rng(0)
    tmp = Path(tempfile.mkdtemp())
    for r in range(n_good):
        json.dump(simulate(r, rng), open(tmp / f"r{r}.json", "w"))
    if with_bad:
        json.dump(simulate(n_good, rng, bad=True), open(tmp / "bad.json", "w"))
    return attach_key(load_responses(sorted(tmp.glob("*.json"))), make_key())


def test_qc_flags_bad_rater_only():
    df = build_df()
    q = qc(df)
    bad = [r for r in q.index if r.startswith("30|")]
    assert len(bad) == 1 and not q.loc[bad[0], "keep"] and q.loc[bad[0], "n_flags"] >= 2
    assert q.drop(bad).keep.all()


def test_system_means_recover_truth():
    df = build_df()
    d = df[df.rater.isin(qc(df).query("keep").index)]
    st = system_table(d)
    for s in SYSTEMS:
        for j, sc in enumerate(SCALES):
            assert abs(st.loc[s, sc] - TRUE[s][j]) < 0.6, (s, sc, st.loc[s, sc])
        assert st.loc[s, "mos_lo"] <= st.loc[s, "mos"] <= st.loc[s, "mos_hi"]


def test_contrasts_direction_and_significance():
    df = build_df()
    d = df[df.rater.isin(qc(df).query("keep").index)]
    c = contrasts(d)
    r = c[(c.system == "uent_l1.0") & (c.scale == "mos") & (c.unit == "по участникам")].iloc[0]
    assert r.delta < -1.0 and r.hi < 0 and r.p_holm < 0.01
    r2 = c[(c.system == "original") & (c.scale == "mos") & (c.unit == "по участникам")].iloc[0]
    assert r2.delta > 0.4 and r2.p_holm < 0.05
    # уровень item'ов тоже даёт направление (мощность ниже: 10 item'ов)
    r3 = c[(c.system == "uent_l1.0") & (c.scale == "voice_sim") & (c.unit == "по item'ам")].iloc[0]
    assert r3.delta < -1.5


def test_agreement_positive_for_simulated_effects():
    df = build_df()
    d = df[df.rater.isin(qc(df).query("keep").index)]
    a = agreement(d)
    assert a["mos"] > 0.5 and a["voice_sim"] > 0.5


def test_krippendorff_extremes():
    assert abs(krippendorff_alpha_interval([[3, 3], [5, 5], [1, 1]]) - 1.0) < 1e-9
    rng = np.random.default_rng(1)
    units = [list(rng.integers(1, 6, size=4)) for _ in range(400)]
    assert abs(krippendorff_alpha_interval(units)) < 0.1


def test_holm_monotone():
    p = np.array([0.01, 0.04, 0.03])
    adj = holm(p)
    assert np.allclose(adj, [0.03, 0.06, 0.06]) and (adj >= p).all()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(fns)} passed")
