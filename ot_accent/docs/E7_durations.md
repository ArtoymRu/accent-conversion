# E7: rhythm. Durations by one-dimensional transport

Out of scope until E0–E2 have run on real data and the main systems are frozen.
Symbols and experiment names are those of the [README](../README.md).
Estimate: 2–3 days. Part A needs no GPU and decides whether part B is worth
doing.

### Why

The frame edit of the README (section 3.7) cannot change timing: a frame stays where it was.
But accent is also rhythm: which phones are stretched, which are clipped.
Phone durations are scalars, and in one dimension optimal transport is exact
and non-trivial: it is the quantile map. This is the one place in the paper
where transport does more than shift a mean.

### The idea

Treat the duration of a phone exactly as the README treats a frame.

```math
z \;=\; \log d \;-\; r_s ,
\qquad
r_s \;=\; \sum_c p_c \,\overline{\log d}_{s,c}
```

$d$ is the duration in seconds of one phone token, $r_s$ the speaker's own
rate (the content-balanced mean, the analogue of the speaker centre). Tempo
is part of the voice and must not move. Two maps, per phone $c$:

```math
\text{shift:}\quad z' = z + \lambda\,(m_{b,c} - m_{a,c})
\qquad\qquad
\text{quantile:}\quad z' = z + \lambda\,\bigl(F_{b,c}^{-1}(F_{a,c}(z)) - z\bigr)
```

with $m$ the mean and $F$ the distribution function of $z$ over the reference
speakers of an accent. The new duration is $d' = \exp(r_s + z')$. The shift
is the analogue of `conditional shift`; the quantile map is the full
one-dimensional transport.

### Part A: is there accent in durations at all? (no audio)

The existing code works on durations unchanged: a duration table is a `Table`
with `d = 1`, where a "frame" is one phone token and its value is
$\log d$. (Checked as code on the synthetic world: the estimators run unchanged with `dim=1`.)

1. `experiments/extract_durations.py`: read the alignments only (no audio, no
   GPU), skip the shared test utterances exactly as `extract.py` does, and
   save one array of tokens: speaker, phone, half, $\log d$. Durations come
   from the alignment in seconds, not from frame counts.
2. Build the `Table` with `TableBuilder(39, 1)` and run, as in E1 and E4:
   `variance.components`, `variance.permutation_null`,
   `metrics.loso_accent_id`, `variance.delta_error`.

| Prediction | Refuted if |
|---|---|
| accent × content share of duration variance is above its placebo band | it is inside the band: **stop here**, report it in one sentence |
| a held-out speaker's accent is identified from the duration profile above chance (1/7) | it is at chance |
| the error of the duration delta falls as 2/S × the speaker × content variance | — |

**Output:** `results/e7_duration_variance.csv`, one figure in the style of E1.

### Part B: move the durations, in audio

New file `otaccent/durations.py`, about 80 lines:

```python
def runs(labels):
    """Phone segments of a frame-label sequence: [(phone, first_frame, n_frames)]."""

def fit_quantile(z, phone, speakers_a, speakers_b, n_quantiles=51, min_tokens=50):
    """(K, n_quantiles) quantile grids of the two accents; a phone with too few tokens maps to itself."""

def new_durations(z, phone, grids=None, shift=None, lam=1.0):
    """z' for every token: quantile map via np.interp on the grids, or the shift."""

def warp(frames, labels, ratio):
    """Time-warp an utterance. `ratio[i]` is new/old duration of segment i.
    One monotone piecewise-linear time map over the whole utterance; frames are
    linearly interpolated along it; labels are carried by nearest neighbour.
    Returns (frames', labels')."""
```

Then `experiments/e7_durations.py`, on the protocol of E2 (same folds, same
test utterances, differences to identity):

| System | Frames | Durations |
|---|---|---|
| `identity` | — | — |
| `conditional shift` | shift | — |
| `durations (shift)` | — | shift |
| `durations (quantile)` | — | quantile |
| `conditional shift + durations (quantile)` | shift | quantile |

Order of operations: edit the frames, then warp, then vocode. The two commute
because the frame edit depends only on the label.

**Measure:** WER, speaker cosine, `profile_kept` (with the warped labels), and
the duration-profile cosine to the target accent, computed from the durations
we imposed. **Listen** to fold 0 before trusting any number.

| Prediction | Refuted if |
|---|---|
| on held-out speakers the transported durations are closer to the target accent's profile; quantile is at least as close as shift | no gain over identity |
| warping costs little: WER and speaker cosine stay near those of `conditional shift` alone | WER rises sharply even with the clamp below |
| in the listening test, accent similarity is higher with durations than without | no difference: rhythm does not matter at this scale, say so |

### Rules that are easy to get wrong

- **Silence is not touched.** Pauses carry accent too, but not in version 1.
- **Clamp** every ratio to [0.5, 2] and keep at least one frame per segment.
- **Rounding:** let the continuous time map decide the output length; do not
  round segment by segment, the errors add up.
- **Rate is content-balanced.** $r_s$ is the $p$-weighted mean over phones of
  the per-phone means, not the raw mean over tokens. Otherwise a speaker who
  happens to say more long vowels looks slow.
- **Same aligner on both sides.** Boundaries from two different aligners are
  not comparable. Run E7 on accent → accent pairs inside L2-ARCTIC only.
- **Interpolated frames are a guess.** kNN-VC rescales duration by linear
  interpolation of frames too, but it then matches them to real frames; here
  they go straight to the vocoder. If stretched phones sound smeared, repeat
  frames instead of interpolating and compare.

### Tests to write first

On the synthetic world with `dim=1`:

1. the quantile map sends the source sample onto the target sample (sorted values agree);
2. `lam = 0` is the identity for both maps;
3. `warp` with all ratios 1 returns the input; with ratio 2 everywhere the length doubles; the time map is monotone;
4. the shift map leaves the speaker's rate $r_s$ unchanged.

### Done when

- [ ] the tests pass
- [ ] part A figure on real data, and a decision: go or stop
- [ ] `results/e7_durations.csv` with the five systems
- [ ] audio of fold 0 in `results/e7_audio/`, listened to by both of us
- [ ] three sentences for the paper: what was predicted, what was found, what it costs
