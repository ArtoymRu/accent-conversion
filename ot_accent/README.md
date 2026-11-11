# ot_accent — experiments for "Accent Is a Conditional"

Code for paper A (`paperA_conditional_ot_accent.tex`): accent transfer as
content-conditional optimal transport on frozen WavLM frames. Nothing is
trained. The parameters of every map are per-phone means and variances.

Section and equation numbers in the docstrings refer to the paper.

## Layout

```
otaccent/            the library (about 600 lines)
  corpus.py          L2-ARCTIC and CMU ARCTIC: speakers, accents, phone alignments
  features.py        the channel: WavLM-Large layer 6  <->  HiFi-GAN (kNN-VC checkpoints)
  stats.py           Table of sufficient statistics; the deltas of eqs. (2) and (3)
  maps.py            editors: shift (eq. 6), diagonal Gaussian map (eq. 5), barycentre (kNN)
  variance.py        variance components of model (1), placebo, the 1/S law
  metrics.py         accent from the phone profile; WER; speaker cosine
  synthetic.py       a world that obeys model (1) with known components
experiments/         one script per experiment; each writes a csv and a figure
tests/               the propositions, checked on the synthetic world
config.yaml          paths and the few numbers that matter
```

## Experiments

| Paper | Script | Needs | Output |
|---|---|---|---|
| — | `extract.py` | corpora, GPU | `cache/<speaker>.npz` |
| E0 channel gate | `e0_channel.py` | cache, GPU | `e0_channel.csv`, `e0_audio/` |
| E1 where accent lives, E3 placebo | `e1_variance.py` | cache | `e1_variance.csv`, `e1_accent_id.csv`, figure |
| E4 speaker count | `e4_speaker_count.py` | cache | `e4_speaker_count.csv`, figure |
| E2 maps, E3 placebo, E6 strength | `e2_maps.py` | cache, GPU | `e2_maps.csv`, `e2_frontier.pdf`, `e2_audio/` |

Not written yet: E5 (k-means units instead of canonical phones), E7 (durations),
E8 (listening test), E9 (cascade demo).

## Run

Without any data, on a laptop:

```bash
pip install numpy scipy pyyaml matplotlib pytest
python -m pytest -q                                # the propositions on a synthetic world
python -m experiments.e1_variance --synthetic
python -m experiments.e4_speaker_count --synthetic
python -m experiments.e2_maps --synthetic          # frames only, no audio
```

`--synthetic` replaces the corpora by `otaccent.synthetic.World`. It checks
the code and shows what each figure looks like. It is not evidence: that
world obeys the model by construction.

With the corpora, on Colab (T4 is enough):

```bash
pip install -r requirements.txt
# edit config.yaml: l2_arctic, cmu_arctic, and point cache/ at Google Drive
python -m experiments.extract            # once; resumable; about 30 min for everything
python -m experiments.e0_channel         # listen to results/e0_audio before going on
python -m experiments.e1_variance
python -m experiments.e4_speaker_count
python -m experiments.e2_maps            # resumable; hours with audio metrics, see below
```

E2 runs 4 folds x 6 targets x (9 systems x 4 strengths + identity) x 20
utterances through the vocoder, the encoder, Whisper and ECAPA. For a first
look set `strengths: [1.0]` and `test_utts: 10`.

## Data

- **L2-ARCTIC** (request form at psi.engr.tamu.edu/l2-arctic-corpus). Expected
  layout: `<l2_arctic>/<SPEAKER>/{wav,transcript,textgrid}/arctic_a0001.*`
  (or `<l2_arctic>/<accent>/<SPEAKER>/...`, as `accent_l2` arranges it).
  The forced alignments in `textgrid/` give the canonical phones.
- **CMU ARCTIC** (festvox.org/cmu_arctic), speakers bdl, slt, clb, rms as the
  `native` accent. Expected layout:
  `<cmu_arctic>/cmu_us_<spk>_arctic/{wav,lab,etc}`. The same prompts as
  L2-ARCTIC, so content is parallel.

The same sentence ids are held out as test utterances for every speaker.

## What was checked, and what was not

Checked here:
- `python -m pytest`: 9 tests. Variance components recovered; accent terms vanish under
  the placebo; conditional delta invariant to speaker translations; error of
  a delta equals (2/S) x variance; a translation keeps the speaker centre and
  personal profile, a barycentre replaces them; editors are the identity at
  zero strength and never touch silence.
- The channel on real speech, on CPU: `Encoder` returns exactly the frames of
  the kNN-VC fast path; both vocoders produce audio.
- `extract -> e1 -> e4 -> e2` end to end with the real encoder and vocoder on
  a fabricated mini-corpus written in the two layouts above.

Not checked:
- The loaders against the real L2-ARCTIC and CMU ARCTIC files. The layouts
  and label formats are written from memory of those corpora. If
  `extract.py` finds no utterances or `content()` is all zeros, look at
  `corpus.py` first; it is short.
- `metrics.Asr` and `metrics.SpeakerEncoder` (Whisper, ECAPA), hence E0 and
  the audio columns of E2. Their weights could not be downloaded where this
  was written.
- Anything about accents. No real accented speech has been through this code.

## Known limits of the measurements

- The accent score is computed on WavLM layer 6 after the vocoder round trip:
  it shows that an edit survives resynthesis, in the space where it was made.
  An independent encoder for scoring is the next thing to add.
- Whisper and ECAPA see the same audio a listener would; neither replaces E8.
- Four speakers per accent: E4 measures S = 1, 2 only. The curve beyond that
  is the prediction from E1 until a corpus with more speakers is added.
