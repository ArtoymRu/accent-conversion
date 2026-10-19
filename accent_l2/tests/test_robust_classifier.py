"""
Тесты LOSO-фолдов и диктор-устойчивого классификатора на синтетике. Запуск:
    python tests/test_robust_classifier.py     (без pytest)
    python -m pytest tests/test_robust_classifier.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.accent_map import SPEAKER_TO_ACCENT
from src.ot.data import N_FOLDS, fold_heldout
from src.ot.robust_classifier import SpeakerRobustAccentClassifier, within_accent_speaker_acc
from src.utils.splits import ACCENT_TO_SPEAKERS, HELD_OUT_SPEAKERS


# ── фолды ─────────────────────────────────────────────────────────────────────

def test_folds_partition_speakers():
    all_held = [s for f in range(N_FOLDS) for s in fold_heldout(f)]
    assert sorted(all_held) == sorted(SPEAKER_TO_ACCENT)          # каждый диктор ровно в одном фолде
    for f in range(N_FOLDS):
        held = fold_heldout(f)
        assert len(held) == len(ACCENT_TO_SPEAKERS)               # по одному на акцент
        assert {SPEAKER_TO_ACCENT[s] for s in held} == set(ACCENT_TO_SPEAKERS)


def test_last_fold_matches_default_split():
    assert fold_heldout(3) == set(HELD_OUT_SPEAKERS.values())


def test_fold_out_of_range():
    for bad in (-1, N_FOLDS):
        try:
            fold_heldout(bad)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


# ── классификатор ─────────────────────────────────────────────────────────────

def _synthetic(n_acc=3, n_spk=4, n_utt=60, d=40, seed=0, spk_scale=4.0, acc_scale=1.5):
    """Акцент — сдвиг по осям 0..1; диктор — большой случайный сдвиг по осям 2..9."""
    rng = np.random.default_rng(seed)
    X, acc, spk = [], [], []
    for a in range(n_acc):
        acc_vec = np.zeros(d)
        acc_vec[:2] = acc_scale * rng.normal(size=2)
        for s in range(n_spk):
            spk_vec = np.zeros(d)
            spk_vec[2:10] = spk_scale * rng.normal(size=8)
            X.append(acc_vec + spk_vec + rng.normal(size=(n_utt, d)))
            acc += [a] * n_utt
            spk += [f"a{a}s{s}"] * n_utt
    return np.concatenate(X), np.array(acc), np.array(spk)


def _split_loso(X, acc, spk, held_idx=3):
    held = {f"a{a}s{held_idx}" for a in np.unique(acc)}
    te = np.isin(spk, list(held))
    return ~te, te


def test_nap_basis_orthonormal_and_probe_drops():
    X, acc, spk = _synthetic()
    tr, te = _split_loso(X, acc, spk)
    raw = SpeakerRobustAccentClassifier(n_nuisance=0).fit(X[tr], acc[tr], spk[tr])
    nap = SpeakerRobustAccentClassifier(n_nuisance=6).fit(X[tr], acc[tr], spk[tr])
    assert raw.Q is None and nap.Q.shape == (X.shape[1], 6)
    assert np.allclose(nap.Q.T @ nap.Q, np.eye(6), atol=1e-8)
    p_raw = within_accent_speaker_acc(raw.transform(X), acc, spk)
    p_nap = within_accent_speaker_acc(nap.transform(X), acc, spk)
    assert p_raw > 0.95            # дикторы легко различимы
    assert p_nap < p_raw - 0.1     # NAP убирает заметную часть дикторской информации


def test_nap_not_worse_than_raw_on_heldout_speakers():
    accs = {0: [], 8: []}
    for seed in range(5):
        X, acc, spk = _synthetic(seed=seed)
        tr, te = _split_loso(X, acc, spk)
        for r in accs:
            clf = SpeakerRobustAccentClassifier(n_nuisance=r).fit(X[tr], acc[tr], spk[tr])
            accs[r].append((clf.predict(X[te]) == acc[te]).mean())
    assert np.mean(accs[8]) >= np.mean(accs[0]) - 0.01
    assert np.mean(accs[8]) > 0.5   # заметно выше шанса 1/3


def test_predict_proba_shape_and_classes():
    X, acc, spk = _synthetic()
    tr, te = _split_loso(X, acc, spk)
    clf = SpeakerRobustAccentClassifier(n_nuisance=4).fit(X[tr], acc[tr], spk[tr])
    P = clf.predict_proba(X[te])
    assert P.shape == (te.sum(), 3) and np.allclose(P.sum(1), 1.0)
    assert clf.classes_.tolist() == [0, 1, 2]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(fns)} passed")
