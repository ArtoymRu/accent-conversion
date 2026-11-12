# Accent Is a Conditional

Experiments for the paper *"Accent Is a Conditional: Content-Conditional Optimal
Transport for Accent Transfer in Frozen Speech Representations"*
(target: Interspeech 2027).

> [!IMPORTANT]
> **Status, 11 Nov 2026.** The code runs and its mathematics is tested on a
> synthetic world. **No real accented speech has been through it yet.** Every
> figure on this page is synthetic and shows the *shape* of a result, not a
> result. The first real run is step 1 of [Next steps](#7-next-steps).

**Contents:** [0 In plain words](#0-in-plain-words) ·
[1 The idea](#1-the-idea) ·
[2 Three claims](#2-three-claims-three-experiments) ·
[3 What changes from `accent_l2`](#3-what-changes-from-accent_l2) ·
[4 The family of maps](#4-the-family-of-maps) ·
[5 Code](#5-code) ·
[6 Experiments](#6-experiments) ·
[7 Next steps](#7-next-steps) ·
[8 Decisions for the call](#8-decisions-for-the-call) ·
[9 What is checked](#9-what-is-checked-and-what-is-not) ·
[10 Spec for E7](#10-spec-for-e7-rhythm)

---

## 0. In plain words

Read this first. The rest of the page is the same thing with formulas.

**Where it started.** AccentFlow moved accent with one vector: the difference
between the centroids of two groups of speakers. The accent moved, and the
voice moved with it. This paper asks why, and how to avoid it.

**The main thought.** An accent is not "everything sounds a bit different".
It is "particular sounds are pronounced differently". A Hindi speaker has a
different /t/ and a different /w/, while many vowels are almost like anyone
else's. Add the same vector to every frame and you shift the speaker's
average, which is their voice, not the individual sounds.

**The model in words.** Every frame (a WavLM vector) is a sum of four things:

1. which sound it is (the phone);
2. whose voice it is (a shift common to all sounds of that speaker);
3. how this accent bends this particular sound;
4. how this person bends this particular sound.

The accent we want to move is the third.

**Why one vector is bad.** We have 4 speakers per accent. The difference of
the averages of two groups is "the difference between accents" plus "which
people we happened to record". With four people the second part is large and
the two cannot be told apart.

**What we do instead.**

1. Subtract from every speaker their own average. The voice is gone.
2. For each of the 39 phones, see where it sits relative to the speaker's centre.
3. Average over the speakers of an accent and take the difference between two
   accents. The result is a table of 39 vectors, one per phone.
4. To edit, add to every frame the vector of its phone.

The speaker's average does not change, so the voice stays where it was.

**Where optimal transport comes in.** All methods are versions of one task,
"move the frames of accent A onto the frames of accent B":

- a shift (move the means only): our method;
- a shift and a scale (the variances too);
- replacing a frame by the nearest frames of target speakers (kNN, Sinkhorn).

The third takes frames from other people and brings their voice along. That
is what the pilot saw: the entropic map gave the strongest accent shift and
replaced the voice.

**The experiments are four questions.**

| | Question |
|---|---|
| **E0** | Does speech still sound normal after WavLM → vocoder when nothing is changed? |
| **E1** | What share of the spread of frames belongs to each of the four terms? Control: shuffle the accent labels between speakers; the accent shares must drop to zero. |
| **E4** | How many speakers does an accent direction need? Prediction: the error falls as 1/S, and it is much smaller for the per-phone table than for the single vector. |
| **E2** | Every map, on speakers that were not used to fit it: how much accent was added, how much voice was lost. |

**What is needed now.** Only the first step: run the feature extraction and
E0 on Colab and listen. Everything else depends on it.

---

## 1. The idea

A Hindi accent is not "more of something" in every frame. It is a different
/t/, a different /w/, a different vowel in *cat*. So accent is a property of
how **each sound** is realised, and moving it means moving each sound
separately. Anything that moves all frames together mostly moves the speaker.

We write one frame of speaker $s$ saying phone $c$ as

```math
x \;=\; \underbrace{\mu_c}_{\text{the sound}}
 \;+\; \underbrace{u_s}_{\text{the voice}}
 \;+\; \underbrace{v_{a(s),c}}_{\text{accent bends the sound}}
 \;+\; \underbrace{w_{s,c}}_{\text{this speaker bends the sound}}
 \;+\; \varepsilon
```

and the editor is a translation with two knobs:

```math
T_{\lambda,\kappa}(x, c) \;=\; x \;+\; \lambda\,\Delta_c \;+\; \kappa\,\delta
```

| Symbol | What it is | How it is estimated |
|---|---|---|
| $`\Delta_c`$ | conditional delta: how the target accent bends phone $c$ relative to the source accent | difference of **speaker-centred** phone means, averaged over reference speakers |
| $\delta$ | global delta: difference of the mean voices of the two groups | difference of speaker centres (this is AccentFlow at frame level) |
| $\lambda$ | segmental strength | leaves the speaker's centre exactly where it was |
| $\kappa$ | global strength | moves the voice centre; cannot be told apart from "the average voice of the people we recorded" |

<p align="center"><img src="docs/fig_idea.png" width="760"><br>
<em>A toy in two dimensions. Left: the two groups differ mostly by who is in them. Right: with each speaker's centre removed, the accent is a different displacement for each phone.</em></p>

The whole pipeline, with nothing trained:

```mermaid
flowchart LR
    A["waveform"] --> B["WavLM-Large, layer 6<br/>frozen"]
    T["transcript"] --> F["forced alignment<br/>canonical phone per frame"]
    B --> E["x + λ·Δ_c + κ·δ"]
    F --> E
    S["reference speakers<br/>per-phone means"] --> E
    E --> V["HiFi-GAN of kNN-VC<br/>frozen"]
    V --> W["waveform, new accent"]
```

---

## 2. Three claims, three experiments

| | Claim | In the paper | Experiment | Refuted if |
|---|---|---|---|---|
| **1** | Accent lives in the **accent × content** term. The global accent term cannot be separated from the speaker with 4 speakers per accent. | Proposition 1 | [E1](#e1--where-accent-lives-with-the-placebo-of-e3), [E4](#e4--how-many-speakers-does-a-direction-need) | accent × content is inside its placebo band at every layer |
| **2** | Delta vectors, per-unit steering, Gaussian maps and retrieval are **one conditional transport problem** with different restrictions. | Proposition 2 | [E2](#e2--every-map-on-held-out-speakers) compares them on one protocol | conditional maps are no better than global ones on held-out speakers |
| **3** | **Translations carry the voice; barycentres replace it.** That is why retrieval editors lose identity as accent grows. | Proposition 3 | [E2](#e2--every-map-on-held-out-speakers), identity columns | barycentres keep identity as well as translations at equal accent |

---

## 3. What changes from `accent_l2`

The pilot in `accent_l2/` already found the pattern the paper explains:
unit-conditional mean shift moved accent-ID by +0.19 at ΔWER +0.013 and
speaker top-1 −0.12; the entropic map moved it by +0.58 and replaced the
voice; global maps did nothing on held-out speakers. This folder keeps the
protocol and changes four things.

| | `accent_l2` (pilot) | `ot_accent` (paper) | Why |
|---|---|---|---|
| **Channel** | HuBERT-base L6 → own mel decoder → HiFi-GAN (LJSpeech) | WavLM-Large L6 → HiFi-GAN of kNN-VC, both frozen | The decoder was the bottleneck ("robovoice", speaker cosine 0.60–0.68). The kNN-VC vocoder inverts these frames directly; nothing to train. |
| **Content** | k-means units, refit per fold | canonical phones from the transcript (forced alignment shipped with L2-ARCTIC) | A cluster index is already accented: a retroflex /t/ can fall in another cluster and hide the difference we want to move. In a translation cascade the transcript is free. |
| **Direction** | accent → accent (6 pairs) | native → accent (CMU ARCTIC speakers → 6 accents) | It is what the cascade needs, the least covered in the literature, and the prompts are the same sentences, so content is parallel. |
| **Accent metric** | accent-ID on utterance means, raw and NAP | cosine of the speaker-centred **phone profile** to the target accent | A centred map does not change the utterance mean by construction (`run_frame_ot.py` says so in its docstring). An utterance-mean classifier cannot see it. |

Kept as they are, to be reused:

| From `accent_l2` | Use here |
|---|---|
| LOSO folds, `fold_heldout(f) = sorted(speakers)[f]` | identical rule in `e2_maps.py` |
| `identity` as the baseline, not `original` | same: every number is a difference to identity resynthesis |
| `gate_check.py` | became `e0_channel.py` |
| `resynth/audio_metrics.py` (Whisper, ECAPA; tested) | fallback if `otaccent/metrics.py` misbehaves; it has not been run |
| `EntropicOTMap`, full Bures–Wasserstein | to port as two more systems (step 4) |
| NAP classifier, `speaker_probe.py` | second opinion on accent and identity (step 4) |
| `docs/HUMAN_EVAL_PROTOCOL.md` and the web form | the listening test E8, with new systems |

New here: the variance decomposition with a placebo (E1), the 1/S law (E4),
the placebo map, and `profile_kept`, which measures whether the speaker's
personal way of saying each phone survived.

---

## 4. The family of maps

All of these solve the same problem, *move the frames of accent a onto those
of accent b without moving content*, under different restrictions. Parameters
grow downwards, and so does the number of speakers needed to estimate them.

| Content cells | Map | Name in `e2_maps.py` | In the literature | Parameters |
|---|---|---|---|---|
| 1 | translation | `global shift` | delta vector (AccentFlow), activation steering | $d$ |
| 1 | diagonal affine | `global affine` | — | $2d$ |
| 1 | barycentric | `global barycentre` | kNN-VC | none |
| 39 phones | translation | **`conditional shift`** | `ubw-mean` + `speaker_norm` in the pilot | $39\,d$ |
| 39 phones | diagonal affine | `conditional affine` | `ubw-diag` | $78\,d$ |
| 39 phones | barycentric | `conditional barycentre` | `uent`; retrieval editing (PACE, to confirm) | none |
| same accent on both sides | translation | `placebo` | — | measures pure speaker leak |

---

## 5. Code

About 1,250 lines. The library is numpy; torch is needed only for the encoder,
the vocoder and the audio metrics.

```
otaccent/
  corpus.py      L2-ARCTIC and CMU ARCTIC: speakers, accents, phone alignments
  features.py    the channel: WavLM-Large layer 6 <-> HiFi-GAN (kNN-VC checkpoints)
  stats.py       Table of sufficient statistics; the two deltas
  maps.py        editors: shift, diagonal Gaussian map, barycentre
  variance.py    variance components, placebo, the 1/S law
  metrics.py     accent from the phone profile; WER; speaker cosine
  synthetic.py   a world that obeys the model, with known components
experiments/     extract.py, e0_channel.py, e1_variance.py, e2_maps.py, e4_speaker_count.py,
                 overview_figures.py (the figures on this page, from the result files)
tests/           the propositions, checked on the synthetic world
config.yaml      paths and the few numbers that matter
```

One idea carries most of it: **everything is a function of one table**, the
sum and sum of squares of frames per (speaker, half of the utterances, phone).
`extract.py` fills it once on a GPU; E1, E4 and every fitted map then run in
seconds on a laptop.

Fitting and applying the main map is four lines:

```python
from otaccent import maps
from otaccent.stats import conditional_delta, global_delta

p, m = tab.content(), tab.means()                  # content distribution, (speakers, 39, 1024) phone means
src, tgt = tab.idx("native", exclude=held), tab.idx("hindi", exclude=held)
Delta = conditional_delta(m, p, src, tgt)          # (39, 1024)
delta = global_delta(m, p, src, tgt)               # (1024,)
Y = maps.shift(X, c, Delta, delta, lam=1.0, kappa=0.0)    # X: (T, 1024) frames, c: (T,) phone ids
wav = vocoder(Y)
```

---

## 6. Experiments

Each script states its prediction in its docstring, writes a CSV and a figure
to `results/`, and runs on the synthetic world with `--synthetic`.

### E0 — the channel gate

**Question.** Is WavLM → HiFi-GAN resynthesis clean before anything is edited?
**Pass if** WER and speaker cosine stay close to the original and
`results/e0_audio/` does not sound robotic. If it fails, no audio claim
downstream means anything.

```bash
python -m experiments.e0_channel
```

### E1 — where accent lives, with the placebo of E3

**Question.** How much variance does each term of the model carry, per layer?
**Method.** Each component is a difference of three averaged inner products:
the same speaker across two halves of their utterances, two speakers of the
same accent, two speakers of different accents. Noise cancels, so the
estimates are unbiased. The placebo shuffles accent labels across speakers.
**Prediction.** accent × content is above its placebo band and peaks at
phonetic layers.
Also reports leave-one-speaker-out accent identification from the phone
profile (conditional view) and from the speaker centre (global view).

```bash
python -m experiments.e1_variance
```

<p align="center"><img src="docs/fig_variance.png" width="860"><br>
<em>Left: share of variance per term. Middle: accent × content against the same quantity with accent labels shuffled. Right: accent of an unseen speaker identified from the global and from the conditional view.</em></p>

### E4 — how many speakers does a direction need?

**Question.** How wrong is a delta estimated from $S$ speakers per accent?
**Method.** Estimate it twice from disjoint speaker sets. Half their squared
distance is the error; their inner product is the squared norm of the true
delta.
**Prediction.**

```math
\text{error}(S) \;=\; \frac{2}{S}\,\mathrm{tr}\,\Sigma
\qquad
\Sigma = \Sigma_u \text{ (speaker) for } \delta,
\quad
\Sigma = \Sigma_w \text{ (speaker} \times \text{content) for } \Delta_c
```

with both variances read from E1. A direction is usable once its error falls
below its signal (dotted lines).

```bash
python -m experiments.e4_speaker_count
```

<p align="center"><img src="docs/fig_speaker_count.png" width="460"><br>
<em>Points: measured. Lines: predicted from E1. Dotted: size of the true delta. A direction is usable to the right of where its line crosses its dotted level.</em></p>

> [!NOTE]
> L2-ARCTIC has 4 speakers per accent, so only $S = 1, 2$ can be measured. The
> rest of the curve is the prediction until a corpus with more speakers per
> accent is added (VCTK regional accents, Speech Accent Archive).

### E2 — every map, on held-out speakers

**Protocol.** Fold $f$ holds out the $f$-th speaker of every accent. The 20
test utterances of the held-out native speaker are edited towards each of the
6 accents with maps fitted on the other speakers, vocoded, encoded again and
scored. Strengths 0.5, 1, 1.5, 2 give the sweep of E6; the `placebo` system is
E3 in audio.

| Column | Meaning |
|---|---|
| `accent`, `accent_hit` | cosine of the phone profile to the target accent; 1 if the target is the nearest accent |
| `centre_shift` | how far the voice centre moved, in typical distances between speakers |
| `profile_kept` | cosine between the speaker's personal phone profile after and before |
| `speaker_cos` | ECAPA cosine to the original recording |
| `wer` | Whisper word error rate |
| `effort` | mean squared displacement of the edited frames |

**Predictions.** Conditional beats global on accent. Translations beat
barycentres on identity at equal accent. The placebo moves accent by nothing.

```bash
python -m experiments.e2_maps        # resumable; rows are appended to results/e2_maps.csv
```

<p align="center"><img src="docs/fig_strength.png" width="860"><br>
<em>What strength does, for four maps (frames only, no vocoder).</em></p>

<p align="center"><img src="docs/fig_scorecard.png" width="860"><br>
<em>All maps at strength 1 (frames only, no vocoder).</em></p>

### Running

```bash
# no data, laptop
pip install numpy scipy pyyaml matplotlib pytest
python -m pytest -q
python -m experiments.e1_variance --synthetic

# Colab, T4
pip install -r requirements.txt
# config.yaml: set l2_arctic, cmu_arctic; point cache/ at Google Drive
python -m experiments.extract        # once, resumable, about 30 min
python -m experiments.e0_channel
python -m experiments.e1_variance
python -m experiments.e4_speaker_count
python -m experiments.e2_maps        # first look: strengths: [1.0], test_utts: 10
```

Data layout expected:
`<l2_arctic>/<SPEAKER>/{wav,transcript,textgrid}/` (or
`<l2_arctic>/<accent>/<SPEAKER>/...`) and
`<cmu_arctic>/cmu_us_<spk>_arctic/{wav,lab,etc}/` for bdl, slt, clb, rms.

---

## 7. Next steps

```mermaid
flowchart TD
    S1["1. extract + E0<br/>does resynthesis sound clean?"] -->|yes| S2["2. E1 + E4<br/>is accent × content above placebo?"]
    S1 -->|no| X1["paper becomes analysis only<br/>no audio claims"]
    S2 -->|yes| S3["3. E2, quick then full<br/>listen to e2_audio"]
    S2 -->|no| X2["claim is false<br/>write up the negative result"]
    S3 --> S4["4. second opinions<br/>independent encoder, Sinkhorn, NAP"]
    S4 --> S5["5. listening test"]
    S5 --> S6["6. write"]
```

| # | Task | Who (proposed) | Done when |
|---|---|---|---|
| 1 | Download CMU ARCTIC (bdl, slt, clb, rms). Check that our L2-ARCTIC copy has `textgrid/`. Run `extract.py` and `e0_channel.py` on Colab. Fix the loaders if they miss. | rendei | `results/e0_channel.csv` exists and we have both listened to `e0_audio/` |
| 2 | Run E1 and E4. | rendei | the two figures, on real data |
| 3 | Run E2 with `strengths: [1.0]`, `test_utts: 10`; listen; then the full grid. | rendei | `e2_maps.csv`, `e2_frontier.pdf`, `e2_audio/` |
| 4 | Port `EntropicOTMap` and full Bures–Wasserstein from `accent_l2/src/ot/frame_ot.py` as two more systems. Add HuBERT as an independent encoder for the accent score. Add the NAP classifier as a second accent metric. | rendei | three more rows in the E2 table |
| 5 | Read PACE (arXiv:2609.22031) in full. Decide whether to reproduce it or place it in the table of section 4 with evidence. | Artem | a paragraph in Related Work we both agree with |
| 6 | Listening test: reuse `HUMAN_EVAL_PROTOCOL.md` and the web form with systems `original`, `identity`, `global shift`, `conditional shift`, `conditional barycentre`. | both | pilot with 5–10 listeners |
| 7 | Write. Figures come straight from `results/`. | Artem | draft in the Interspeech 2027 template |

Not written yet, and not blocking: E5 (k-means units against canonical
phones), E7 (durations by one-dimensional transport; the spec is in
[section 10](#10-spec-for-e7-rhythm)), E9 (cascade demo), the speaker-count
curve on a corpus with more speakers.

---

## 8. Decisions for the call

Proposed answers to the open questions of the note of 3 Nov. To confirm or
change together.

| Question in the note | Proposal |
|---|---|
| 1. Which pipeline: units + XTTS, or our own resynthesis? | Neither. The frozen kNN-VC channel. Our decoder stays in `accent_l2` as a control. |
| 2. Venue and main contribution | Interspeech 2027. The contribution is the account (why global fails, why retrieval loses the voice, the 1/S law, the placebo), then the frontier, then a small listening test. |
| 3. Human-eval protocol | Keep the design. Change the systems (step 6). |
| 7. Improve the decoder before human eval? | No. Drop it. Pilot only after E0 passes. |
| 11. Position against PACE | Read it in full first (step 5). Working hypothesis: it is a barycentric editor, and Proposition 3 explains its trade-off. |
| 12. Tempo and intonation | Durations as optional E7 in this paper if time allows. Pitch needs a vocoder with explicit F0: a separate work. |

Open, to discuss:

- Is native → accent the right main direction, or do we keep accent → accent
  as a second table for comparison with the pilot and with PACE?
- The accent score lives in the space we edit. Which independent encoder do we
  trust: HuBERT-large, Whisper encoder, or the NAP classifier?
- Four speakers per accent. Which corpus gives us the 1/S curve beyond S = 2?
- **`native` is also a different corpus.** CMU ARCTIC was recorded elsewhere
  and its phone labels come from a different aligner than L2-ARCTIC's. A
  systematic difference in where boundaries are placed would look like an
  accent × content effect, and the placebo cannot catch it. Two controls:
  report E1 on the six L2 accents alone, and re-align CMU ARCTIC with the
  aligner used for L2-ARCTIC (Montreal Forced Aligner).

---

## 9. What is checked, and what is not

| Checked | How |
|---|---|
| The estimators and the propositions | `python -m pytest`: 9 tests on a synthetic world with known components |
| The channel | on real speech, on CPU: `Encoder` returns exactly the frames of the kNN-VC fast path; both vocoders produce audio |
| The pipeline | `extract → e1 → e4 → e2` end to end with the real encoder and vocoder, on a fabricated mini-corpus in the two layouts |

| Not checked | Consequence |
|---|---|
| Loaders on the real L2-ARCTIC and CMU ARCTIC | If `extract.py` finds no utterances, look at `corpus.py` first. The `textgrid/` alignments and the CMU `.lab` format are written from memory. |
| `metrics.Asr`, `metrics.SpeakerEncoder` | E0 and the `wer`, `speaker_cos` columns of E2 have never run. `accent_l2/src/resynth/audio_metrics.py` is a tested alternative. |
| Anything about accents | No real accented speech has been through this code. |

Known limits of the measurements: the accent score is taken on WavLM layer 6
after the vocoder round trip, so it shows that an edit survives resynthesis in
the space where it was made; Whisper and ECAPA hear what a listener would, and
neither replaces the listening test.

---

## 10. Spec for E7: rhythm

For rendei. Optional for the paper; start only after E0–E2 run on real data.
Estimate: 2–3 days. Part A needs no GPU and decides whether part B is worth
doing.

### Why

The frame edit of section 1 cannot change timing: a frame stays where it was.
But accent is also rhythm: which phones are stretched, which are clipped.
Phone durations are scalars, and in one dimension optimal transport is exact
and non-trivial: it is the quantile map. This is the one place in the paper
where transport does more than shift a mean.

### The idea

Treat the duration of a phone exactly as section 1 treats a frame.

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
$\log d$. This was checked on the synthetic world.

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
- **Same aligner on both sides.** For durations the corpus confound of
  section 8 is fatal: boundaries from two different aligners are not
  comparable. Run E7 on accent → accent pairs inside L2-ARCTIC, or re-align
  CMU ARCTIC first.
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

