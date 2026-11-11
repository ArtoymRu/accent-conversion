"""The propositions of the paper, checked on a world where the truth is known."""
import numpy as np
import pytest

from otaccent import maps, metrics, variance
from otaccent.stats import centres, conditional_delta, global_delta, moments, profiles
from otaccent.synthetic import World


@pytest.fixture(scope="module")
def world():
    return World(n_accents=12, speakers_per_accent=8, seed=1)


@pytest.fixture(scope="module")
def table(world):
    return world.table(frames_per_cell=40)


def test_variance_components_are_recovered(world, table):
    est, var, shrink = variance.components(table), world.var, (world.K - 1) / world.K
    assert est["content"] == pytest.approx(var["content"] * shrink, rel=0.25)
    assert est["speaker"] == pytest.approx(var["speaker"], rel=0.15)
    assert est["accent (global)"] == pytest.approx(var["accent"], rel=0.35)
    assert est["accent x content"] == pytest.approx(var["accent_x_content"] * shrink, rel=0.15)
    assert est["speaker x content"] == pytest.approx(var["speaker_x_content"] * shrink, rel=0.15)
    assert est["residual"] == pytest.approx(var["residual"], rel=0.05)
    parts = sum(v for k, v in est.items() if k != "total")
    assert parts == pytest.approx(est["total"], rel=1e-9)


def test_no_accent_means_no_accent_component():
    tab = World(n_accents=12, speakers_per_accent=8, seed=2,
                var={"accent": 0.0, "accent_x_content": 0.0}).table()
    est = variance.components(tab)
    null = variance.permutation_null(tab, n_perm=100)
    for key in ("accent (global)", "accent x content"):
        lo, hi = np.quantile([n[key] for n in null], [0.005, 0.995])
        assert lo <= est[key] <= hi


def test_accent_component_exceeds_its_placebo(table):
    est, null = variance.components(table), variance.permutation_null(table, n_perm=100)
    assert est["accent x content"] > max(n["accent x content"] for n in null)


def test_conditional_delta_ignores_speaker_translations(table):
    """Proposition 1, last claim: x -> x + t_s changes the global delta, not the conditional one."""
    p, m = table.content(), table.means()
    t = np.random.default_rng(0).normal(size=(len(m), 1, m.shape[2]))
    src, tgt = table.idx("acc0"), table.idx("acc1")
    assert np.allclose(conditional_delta(m, p, src, tgt), conditional_delta(m + t, p, src, tgt))
    assert not np.allclose(global_delta(m, p, src, tgt), global_delta(m + t, p, src, tgt))


def test_delta_error_decays_as_one_over_S():
    """Proposition 1: error = (2/S) tr Sigma, with Sigma_u for global and Sigma_w for conditional."""
    world = World(n_accents=2, speakers_per_accent=64, seed=3)
    tab, shrink = world.table(frames_per_cell=200), (world.K - 1) / world.K
    noise = variance.sampling_noise(tab)
    for S in (1, 2, 4, 8, 16):
        e = variance.delta_error(tab, "acc0", "acc1", S, n_draws=300)
        assert e["global error"] == pytest.approx(2 / S * (world.var["speaker"] + noise["speaker"]), rel=0.2)
        assert e["conditional error"] == pytest.approx(
            2 / S * (world.var["speaker_x_content"] * shrink + noise["speaker x content"]), rel=0.2)
        assert e["conditional signal"] == pytest.approx(((world.v[1] - world.v[0]) ** 2).sum() / world.K, rel=0.2)


def test_translation_carries_the_voice_barycentre_replaces_it(world, table):
    """Proposition 3 on the frames of one source speaker moved from accent 0 to accent 1."""
    p, m = table.content(), table.means()
    src, tgt = table.idx("acc0"), table.idx("acc1")
    s = src[0]
    labels = np.tile(np.arange(world.K), 200)
    X = world.frames(s, labels)
    voice = lambda Y: metrics.profile(Y, labels, p)[0]     # personal profile, noise averaged out
    centre = lambda Y: np.array([Y[labels == k].mean(0) for k in range(world.K)]).mean(0)

    true_Delta = world.v[1] - world.v[0]
    Y = maps.shift(X, labels, Delta=true_Delta)
    assert np.allclose(centre(Y), centre(X), atol=1e-5)                    # speaker centre untouched
    assert np.allclose(voice(Y) - voice(X), true_Delta, atol=1e-4)         # only the accent term moved

    pool_labels = np.tile(np.arange(world.K), 100 * len(tgt))
    pool = np.concatenate([world.frames(t, np.tile(np.arange(world.K), 100)) for t in tgt])
    Z = maps.barycentre(X, labels, pool, pool_labels, k=8)
    # the source centre is replaced by a convex combination of target speakers' centres:
    # it travels most of the way to their mean (nearest neighbours favour the closest of them)
    target_centre = (world.u[tgt] + world.m[1]).mean(0) + world.mu.mean(0)
    assert np.linalg.norm(centre(Z) - target_centre) < 0.5 * np.linalg.norm(centre(X) - target_centre)
    assert Z.var(0).sum() < pool.var(0).sum()                              # Cov T(X) below Cov Y


def test_shifted_speaker_is_identified_as_target_accent(world, table):
    p, m = table.content(), table.means()
    names = np.unique(table.accents)
    src, tgt = table.idx("acc0", exclude=["spk000"]), table.idx("acc1")
    Delta = conditional_delta(m, p, src, tgt)                              # fitted without speaker 0
    labels = np.tile(np.arange(world.K), 100)
    X = world.frames(0, labels)
    refs = np.stack([profiles(m, p)[table.idx(a, exclude=["spk000"])].mean(0) for a in names])
    score = lambda Y: accent_scores_of(Y, labels, refs, p)
    assert names[score(X).argmax()] == "acc0"
    assert names[score(maps.shift(X, labels, Delta=Delta)).argmax()] == "acc1"
    assert score(maps.shift(X, labels, Delta=Delta, lam=0.5))[1] < score(maps.shift(X, labels, Delta=Delta))[1]


def accent_scores_of(Y, labels, refs, p):
    prof, present = metrics.profile(Y, labels, p)
    return metrics.accent_scores(prof, present, refs, p)


def test_editors_are_identity_at_zero_strength_and_spare_silence(world, table):
    p = table.content()
    labels = np.r_[np.full(10, -1), np.tile(np.arange(world.K), 5)]
    X = world.frames(0, labels)
    src, tgt = table.idx("acc0"), table.idx("acc1")
    Delta, delta = conditional_delta(table.means(), p, src, tgt), global_delta(table.means(), p, src, tgt)
    mom = lambda i: moments(table, i, p)
    edits = [maps.shift(X, labels, Delta, delta, lam=1, kappa=1),
             maps.affine(X, labels, mom(src), mom(tgt), centre=centres(table.means(), p)[0]),
             maps.barycentre(X, labels, X[::-1].copy(), labels[::-1].copy())]
    for Y in edits:
        assert np.array_equal(Y[:10], X[:10])
        assert not np.allclose(Y[10:], X[10:])
    assert np.allclose(maps.shift(X, labels, Delta, delta, lam=0, kappa=0), X)
    assert np.allclose(maps.affine(X, labels, mom(src), mom(tgt), lam=0), X)


def test_loso_accent_identification_prefers_the_conditional_view():
    """With a large speaker variance and few speakers, the centre is a poor accent cue; the profile is not."""
    tab = World(n_accents=6, speakers_per_accent=4, seed=4, var={"speaker": 6.0}).table()
    acc = metrics.loso_accent_id(tab)
    assert acc["conditional"] > 0.9 > acc["global"]
