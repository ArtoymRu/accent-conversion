# AccentFlow

**Disentangled Accent Transfer in End-to-End Speech-to-Speech Translation via Discrete Acoustic Representations and Controllable Prosodic Embeddings**

---

## Overview

AccentFlow is a three-stream speech synthesis model that transfers accent independently of speaker identity and linguistic content. Given a source utterance, it re-synthesises the speech with a target accent while preserving the original speaker's voice and the spoken words.

```
Source audio
    ├── Content stream  →  HuBERT-base (layer 6) → k-means (K=200) → unit tokens
    ├── Speaker stream  →  GE2E d-vector encoder                    → 256-dim embedding
    └── Accent stream   →  24-dim AIV features   → GRU encoder      → 64-dim embedding
                                                            ↓ fuse + concat
                                                    GRU mel decoder  → 80-band mel
                                                            ↓
                                                        HiFi-GAN     → waveform
```

**Accent control** is a single scalar λ ∈ [0, 1]:

```
a(λ) = (1 − λ) · a_source  +  λ · a_target
```

λ = 0 → source accent unchanged · λ = 1 → full target accent transfer.

---

## Project Structure

```
accent-conversion/
├── requirements.txt
├── config.py                   # all hyperparameters in one place
├── src/
│   ├── content.py              # HuBERT feature extraction + k-means quantization
│   ├── speaker.py              # GE2E d-vector extraction (resemblyzer)
│   ├── accent.py               # 24-feature AIV extraction + accent centroid util
│   ├── model.py                # AIVEncoder (GRU) + MelDecoder (GRU) + AccentFlowModel
│   └── evaluate.py             # WER, WhisperScore, SpeakerSim metrics
├── scripts/
│   ├── extract_kmeans.py       # fit k-means on HuBERT features from L2-ARCTIC
│   ├── build_aiv_centroids.py  # compute per-accent AIV centroids, save to disk
│   ├── train.py                # train mel decoder (L1 reconstruction loss)
│   └── infer.py                # accent transfer: audio in → audio out
└── checkpoints/                # saved weights (git-ignored)
```

---

## Accent Identity Vector (AIV)

Each utterance is described by a 24-dimensional feature vector:

| Feature group | Dim | Description |
|---|---|---|
| F0 trajectory | 6 | mean, std, p10, p50, p90, linear slope |
| Formant transitions | 16 | ΔF1, ΔF2, mean-F1, mean-F2 per 4 temporal segments |
| Rhythm (nPVI) | 2 | vowel-interval nPVI, consonant-interval nPVI |

Per-accent centroids are the robust mean over all training utterances for that accent group. A 2-layer GRU encoder maps the 24-dim vector to a 64-dim accent embedding.

---

## Installation

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `transformers`, `resemblyzer`, `librosa`, `praat-parselmouth`, `speechbrain`, `openai-whisper`.

---

## Quickstart

### 1. Train k-means on your data

```bash
python scripts/extract_kmeans.py --data_dir /path/to/l2arctic --max_files 500
# → checkpoints/kmeans.pkl
```

### 2. Build accent centroids

Organise audio into subdirectories named by accent:
```
data/
  hindi/    utt001.wav  utt002.wav ...
  mandarin/ utt001.wav  ...
```

```bash
python scripts/build_aiv_centroids.py --data_dir data/
# → checkpoints/aiv_centroids.npy
```

### 3. Train the mel decoder

```bash
python scripts/train.py --data_dir /path/to/wavs --kmeans checkpoints/kmeans.pkl
# → checkpoints/model.pt
```

### 4. Run accent transfer

```bash
# Transfer to a known accent centroid
python scripts/infer.py \
    --input  source.wav \
    --output output.wav \
    --target_accent hindi \
    --lam 1.0

# Use a reference audio clip as the target
python scripts/infer.py \
    --input      source.wav \
    --output     output.wav \
    --target_ref reference.wav \
    --lam 0.75

# λ sweep — produces output_lam0.00.wav … output_lam1.00.wav
python scripts/infer.py \
    --input source.wav --output out.wav \
    --lam 0.0 0.25 0.5 0.75 1.0
```

---

## Evaluation Metrics

| Metric | What it measures |
|---|---|
| WER | Intelligibility (Whisper-large-v3 transcription) |
| WhisperScore | Accent intensity — P(target lang \| audio) via Whisper LID |
| SpeakerSim | Voice preservation — d-vector cosine similarity |
| UTMOS | Naturalness (neural MOS estimator) |
| ACA | Accent classifier accuracy (ResNet-34) |

---

## Audio Examples

The repository includes accent conversion examples recorded as part of earlier work:

| File | Description |
|---|---|
| `indian_en_en .wav` | Indian-accented English |
| `nigerian_en_en .wav` | Nigerian-accented English |
| `canada_fr .wav` | Québécois French |
| `canada_fr_fr.wav` | Québécois → Parisian French conversion |
| `german_man_sp .wav` | German-accented Spanish |
| `german_man_sp_fr.wav` | German-accented Spanish → French conversion |

---

## Citation

> Anonymous (2025). *Disentangled Accent Transfer in End-to-End Speech-to-Speech Translation via Discrete Acoustic Representations and Controllable Prosodic Embeddings*.
