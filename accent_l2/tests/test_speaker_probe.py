"""
Тесты прокси сохранности диктора на синтетике. Запуск:
    python tests/test_speaker_probe.py     (без pytest)
    python -m pytest tests/test_speaker_probe.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ot.speaker_probe import SpeakerEmbedder, pool

D = 24


def _speaker_vecs(n, seed=0):
    rng = np.random.default_rng(seed)
    return {f"s{i}": rng.normal(size=D) * 2.0 for i in range(n)}


def _utts(vec, n=30, T=60, seed=0):
    rng = np.random.default_rng(seed)
    return [vec + rng.normal(size=(T, D)) for _ in range(n)]


def _setup():
    vecs = _speaker_vecs(10)
    train = {f"s{i}": _utts(vecs[f"s{i}"], seed=i) for i in range(6)}
    emb = SpeakerEmbedder().fit(train)
    for i in range(6, 10):                      # неувиденные диктора — зачисление отдельными высказываниями
        emb.enroll(f"s{i}", _utts(vecs[f"s{i}"], n=15, seed=100 + i))
    return vecs, emb


def test_pool_shape():
    assert pool(_utts(np.zeros(D), n=3)).shape == (3, 2 * D)


def test_identity_retention_unseen_speaker():
    vecs, emb = _setup()
    r = emb.retention(_utts(vecs["s7"], n=20, seed=200), "s7", ["s0", "s1"])
    assert r["top1"] > 0.9 and r["margin"] > 0 and r["capture"] < 0.1


def test_voice_swap_is_detected_as_capture():
    vecs, emb = _setup()
    swapped = [u - vecs["s7"] + vecs["s1"] for u in _utts(vecs["s7"], n=20, seed=201)]
    r = emb.retention(swapped, "s7", ["s0", "s1"])
    assert r["top1"] < 0.2 and r["capture"] > 0.8 and r["margin"] < 0


def test_small_shift_keeps_speaker():
    vecs, emb = _setup()
    shifted = [u + 0.3 * np.ones(D) for u in _utts(vecs["s7"], n=20, seed=202)]
    r = emb.retention(shifted, "s7", ["s0"])
    assert r["top1"] > 0.8


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(fns)} passed")
