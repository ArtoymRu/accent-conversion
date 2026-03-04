"""Accent stream: 24-feature AIV extraction and accent centroid computation.

Feature layout (24 total)
--------------------------
 [0:6]   F0 trajectory stats       — mean, std, p10, p50, p90, slope
 [6:22]  Formant transitions       — ΔF1, ΔF2, mean-F1, mean-F2 per 4 temporal
                                     segments (4 segs × 4 values = 16)
 [22:24] Rhythm (nPVI)             — vowel-interval nPVI, consonant-interval nPVI
"""
import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call
from scipy.stats import linregress
from pathlib import Path
from typing import Optional


class AccentFeatureExtractor:
    """Extract a 24-dim per-utterance Accent Identity Vector (AIV)."""

    def __init__(self, sr: int = 16000, f0_floor: float = 70.0, f0_ceiling: float = 400.0):
        self.sr = sr
        self.f0_floor = f0_floor
        self.f0_ceiling = f0_ceiling

    # ── F0 features (6) ────────────────────────────────────────────────────

    def _extract_f0(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """pyin F0 estimation. Returns (T,) array with NaN for unvoiced frames."""
        f0, _voiced_flag, _voiced_probs = librosa.pyin(
            audio,
            fmin=self.f0_floor,
            fmax=self.f0_ceiling,
            sr=sr,
            hop_length=256,
        )
        return f0  # NaN where unvoiced

    def _f0_features(self, f0: np.ndarray) -> np.ndarray:
        """6-dim: mean, std, p10, p50, p90, linear slope (over voiced frames)."""
        voiced = f0[~np.isnan(f0)]
        if len(voiced) < 4:
            return np.zeros(6, dtype=np.float32)
        mean_f0 = float(np.mean(voiced))
        std_f0  = float(np.std(voiced))
        p10, p50, p90 = np.percentile(voiced, [10, 50, 90]).tolist()
        voiced_idx = np.where(~np.isnan(f0))[0].astype(float)
        if len(voiced_idx) > 1:
            slope, *_ = linregress(voiced_idx, voiced)
        else:
            slope = 0.0
        return np.array([mean_f0, std_f0, p10, p50, p90, slope], dtype=np.float32)

    # ── Formant features (16) ───────────────────────────────────────────────

    def _extract_formants(self, audio: np.ndarray, sr: int):
        """F1 and F2 trajectories at 10 ms intervals via Praat Burg method."""
        snd = parselmouth.Sound(audio, sampling_frequency=float(sr))
        formants = snd.to_formant_burg(
            time_step=0.01,
            max_number_of_formants=5,
            maximum_formant=5500.0,
            window_length=0.025,
        )
        duration = snd.get_total_duration()
        times = np.arange(0.01, duration, 0.01)

        f1 = np.array(
            [call(formants, "Get value at time", 1, t, "hertz", "linear") for t in times],
            dtype=np.float32,
        )
        f2 = np.array(
            [call(formants, "Get value at time", 2, t, "hertz", "linear") for t in times],
            dtype=np.float32,
        )
        # Replace undefined (0 or very small) with NaN
        f1 = np.where((f1 == 0) | np.isnan(f1), np.nan, f1)
        f2 = np.where((f2 == 0) | np.isnan(f2), np.nan, f2)
        return f1, f2

    def _formant_features(self, f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
        """16-dim: [ΔF1, ΔF2, mean-F1, mean-F2] × 4 temporal quartile-segments."""
        n = len(f1)
        if n < 8:
            return np.zeros(16, dtype=np.float32)

        q_size = n // 4
        features = []
        prev_m1 = prev_m2 = 0.0

        for q in range(4):
            seg1 = f1[q * q_size:(q + 1) * q_size]
            seg2 = f2[q * q_size:(q + 1) * q_size]
            m1 = float(np.nanmean(seg1)) if not np.all(np.isnan(seg1)) else 0.0
            m2 = float(np.nanmean(seg2)) if not np.all(np.isnan(seg2)) else 0.0
            delta1 = m1 - prev_m1 if q > 0 else 0.0
            delta2 = m2 - prev_m2 if q > 0 else 0.0
            features.extend([delta1, delta2, m1, m2])
            prev_m1, prev_m2 = m1, m2

        return np.array(features, dtype=np.float32)  # (16,)

    # ── Rhythm features (2) ─────────────────────────────────────────────────

    @staticmethod
    def _npvi(intervals: list) -> float:
        """Normalised Pairwise Variability Index over a list of durations."""
        if len(intervals) < 2:
            return 0.0
        arr = np.asarray(intervals, dtype=float)
        denom = (arr[:-1] + arr[1:]) / 2.0
        valid = denom > 0
        if not np.any(valid):
            return 0.0
        return float(100.0 * np.mean(np.abs(arr[:-1][valid] - arr[1:][valid]) / denom[valid]))

    def _rhythm_features(self, audio: np.ndarray, sr: int, f0: np.ndarray) -> np.ndarray:
        """2-dim: [vowel nPVI, consonant nPVI]."""
        hop = 256
        energy = librosa.feature.rms(y=audio, hop_length=hop)[0]
        voiced = ~np.isnan(f0)

        # Align to shorter length
        min_len = min(len(energy), len(voiced))
        energy = energy[:min_len]
        voiced = voiced[:min_len]

        energy_threshold = np.percentile(energy, 40)
        vowel_frames = voiced & (energy > energy_threshold)
        consonant_frames = ~vowel_frames

        def segment_durations(mask: np.ndarray) -> list:
            durs, count, in_seg = [], 0, False
            for v in mask:
                if v:
                    in_seg, count = True, count + 1
                elif in_seg:
                    durs.append(count * hop / sr)
                    in_seg, count = False, 0
            if in_seg and count > 0:
                durs.append(count * hop / sr)
            return durs

        return np.array(
            [self._npvi(segment_durations(vowel_frames)),
             self._npvi(segment_durations(consonant_frames))],
            dtype=np.float32,
        )

    # ── Public API ──────────────────────────────────────────────────────────

    def extract(self, audio: np.ndarray, sr: int = 16000) -> Optional[np.ndarray]:
        """Return 24-dim AIV feature vector, or None on failure."""
        if sr != self.sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sr)
            sr = self.sr

        # F0 (6)
        f0 = self._extract_f0(audio, sr)
        f0_feat = self._f0_features(f0)

        # Formants (16)
        try:
            f1, f2 = self._extract_formants(audio, sr)
            formant_feat = self._formant_features(f1, f2)
        except Exception:
            formant_feat = np.zeros(16, dtype=np.float32)

        # Rhythm (2)
        rhythm_feat = self._rhythm_features(audio, sr, f0)

        return np.concatenate([f0_feat, formant_feat, rhythm_feat])  # (24,)


def compute_accent_centroid(audio_paths: list, extractor: AccentFeatureExtractor) -> np.ndarray:
    """Robust mean of AIV vectors across a list of audio files.

    Args:
        audio_paths: list of str/Path to .wav files
        extractor:   AccentFeatureExtractor instance

    Returns:
        centroid: (24,) float32 array
    """
    import librosa as _librosa

    vecs = []
    for p in audio_paths:
        try:
            audio, sr = _librosa.load(str(p), sr=16000, mono=True)
            feat = extractor.extract(audio, sr=16000)
            if feat is not None:
                vecs.append(feat)
        except Exception as e:
            print(f"  [centroid] Skip {p}: {e}")
    if not vecs:
        raise ValueError("No valid utterances to compute centroid.")
    return np.mean(vecs, axis=0).astype(np.float32)
