"""Evaluation metrics: WER, WhisperScore, SpeakerSim, (UTMOS placeholder).

Lazy-loads heavy models on first use to keep import time low.
"""
from __future__ import annotations

import numpy as np
import torch
import librosa
from typing import Optional
from pathlib import Path


class Evaluator:
    """Unified interface for all AccentFlow evaluation metrics."""

    def __init__(
        self,
        whisper_eval_model:  str = "base",
        whisper_score_model: str = "large-v3",
    ):
        self._wm_name  = whisper_eval_model
        self._ws_name  = whisper_score_model
        self._wm       = None   # Whisper base  (WER)
        self._ws       = None   # Whisper large (WhisperScore)
        self._spk_enc  = None   # resemblyzer VoiceEncoder

    # ── Lazy loaders ────────────────────────────────────────────────────────

    @property
    def _whisper_base(self):
        if self._wm is None:
            import whisper
            print(f"[Evaluator] Loading Whisper {self._wm_name}…")
            self._wm = whisper.load_model(self._wm_name)
        return self._wm

    @property
    def _whisper_large(self):
        if self._ws is None:
            import whisper
            print(f"[Evaluator] Loading Whisper {self._ws_name}…")
            self._ws = whisper.load_model(self._ws_name)
        return self._ws

    @property
    def _speaker_enc(self):
        if self._spk_enc is None:
            from resemblyzer import VoiceEncoder
            print("[Evaluator] Loading GE2E speaker encoder…")
            self._spk_enc = VoiceEncoder(device="cpu")
        return self._spk_enc

    # ── WER ─────────────────────────────────────────────────────────────────

    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> str:
        """Transcribe audio with Whisper base."""
        import whisper
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        result = self._whisper_base.transcribe(audio.astype(np.float32))
        return result["text"].strip().lower()

    @staticmethod
    def wer(hyp: str, ref: str) -> float:
        """Token-level Word Error Rate (edit distance / ref length)."""
        h = hyp.split()
        r = ref.split()
        if not r:
            return 0.0
        d = np.zeros((len(r) + 1, len(h) + 1), dtype=int)
        d[:, 0] = np.arange(len(r) + 1)
        d[0, :] = np.arange(len(h) + 1)
        for i in range(1, len(r) + 1):
            for j in range(1, len(h) + 1):
                cost = 0 if r[i - 1] == h[j - 1] else 1
                d[i, j] = min(d[i-1, j] + 1, d[i, j-1] + 1, d[i-1, j-1] + cost)
        return d[len(r), len(h)] / len(r)

    # ── WhisperScore ────────────────────────────────────────────────────────

    def whisper_score(
        self, audio: np.ndarray, target_lang: str, sr: int = 16000
    ) -> float:
        """P(target_lang | audio) from Whisper language-id probabilities.

        WhisperScore = P_Whisper(l_t | x) as defined in the paper.
        """
        import whisper
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        audio_t = torch.from_numpy(audio.astype(np.float32))
        # Pad/trim to 30 s as Whisper expects
        audio_t = whisper.pad_or_trim(audio_t)
        mel = whisper.log_mel_spectrogram(audio_t).unsqueeze(0)   # (1, 80, T)
        _, probs = self._whisper_large.detect_language(mel)
        return float(probs.get(target_lang, 0.0))

    def whisper_score_reg(
        self, audio: np.ndarray, target_lang: str, sr: int = 16000
    ) -> float:
        """WhisperScore-Reg: same metric but using base model for regional accents."""
        import whisper
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        audio_t = whisper.pad_or_trim(torch.from_numpy(audio.astype(np.float32)))
        mel = whisper.log_mel_spectrogram(audio_t).unsqueeze(0)
        _, probs = self._whisper_base.detect_language(mel)
        return float(probs.get(target_lang, 0.0))

    # ── SpeakerSim ──────────────────────────────────────────────────────────

    def speaker_sim(
        self,
        audio1: np.ndarray,
        audio2: np.ndarray,
        sr: int = 16000,
    ) -> float:
        """Cosine similarity between d-vectors of two utterances ∈ [-1, 1]."""
        from resemblyzer import preprocess_wav

        def _embed(a, s):
            if s != 16000:
                a = librosa.resample(a, orig_sr=s, target_sr=16000)
            wav = preprocess_wav(a.astype(np.float32), source_sr=16000)
            return self._speaker_enc.embed_utterance(wav)

        e1 = _embed(audio1, sr)
        e2 = _embed(audio2, sr)
        return float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-9))

    # ── Convenience batch evaluator ─────────────────────────────────────────

    def evaluate(
        self,
        synth_audio: np.ndarray,
        ref_audio:   Optional[np.ndarray],
        ref_text:    Optional[str],
        target_lang: str,
        sr: int = 16000,
    ) -> dict:
        """Run all available metrics and return a results dict."""
        results: dict = {}

        # WER
        if ref_text is not None:
            hyp = self.transcribe(synth_audio, sr)
            results["wer"] = self.wer(hyp, ref_text)
            results["transcription"] = hyp

        # WhisperScore
        results["whisper_score"] = self.whisper_score(synth_audio, target_lang, sr)

        # SpeakerSim
        if ref_audio is not None:
            results["speaker_sim"] = self.speaker_sim(synth_audio, ref_audio, sr)

        return results
