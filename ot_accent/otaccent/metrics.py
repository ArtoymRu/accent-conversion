"""What we measure.

Accent is measured where the paper says it lives: in the speaker-centred phone
profile. An utterance-level mean cannot see a conditional edit at all, because
sum_c p_c Delta_c = 0 by construction.
"""
import numpy as np

from .stats import centres, profiles


# ---- accent, from frames ------------------------------------------------------

def profile(X, c, p):
    """Speaker-centred phone means of a set of frames. Returns (K, d) and the phones present."""
    K = len(p)
    n = np.bincount(c[c >= 0], minlength=K)
    present = (n > 0) & (p > 0)
    means = np.zeros((K, X.shape[1]))
    np.add.at(means, c[c >= 0], X[c >= 0])
    means /= np.maximum(n, 1)[:, None]
    q = np.where(present, p, 0) / p[present].sum()
    return (means - q @ means) * present[:, None], present


def centre(X, c, p) -> np.ndarray:
    """Content-balanced centre of a set of frames: sum_c p_c mean_c, over the phones present."""
    n = np.bincount(c[c >= 0], minlength=len(p))
    present = (n > 0) & (p > 0)
    means = np.zeros((len(p), X.shape[1]))
    np.add.at(means, c[c >= 0], X[c >= 0])
    return (np.where(present, p, 0) / p[present].sum()) @ (means / np.maximum(n, 1)[:, None])


def accent_scores(prof, present, refs, p) -> np.ndarray:
    """Cosine between a profile and each reference accent profile, both taken
    relative to the mean reference (the part every accent shares is content)."""
    w = (np.sqrt(p) * present)[:, None]
    common = refs.mean(0)
    a = ((prof - common) * w).ravel()
    B = ((refs - common) * w).reshape(len(refs), -1)
    return B @ a / (np.linalg.norm(B, axis=1) * np.linalg.norm(a) + 1e-12)


def profile_kept(prof, prof0, present, refs, p) -> float:
    """How much of a speaker's personal way of saying each phone survived an edit.

    Both profiles are stripped of the accent subspace (the span of the reference
    accent profiles); what is left is the speaker's own term w_{s,c}. Returns
    the cosine between what is left after and before the edit: 1 for a
    translation along accent directions, low when the voice was replaced.
    """
    w = (np.sqrt(p) * present)[:, None]
    common = refs.mean(0)
    B = ((refs - common) * w).reshape(len(refs), -1)
    Q = np.linalg.qr(B.T)[0][:, :len(refs) - 1]                # refs - common has rank A - 1
    rest = lambda r: (lambda a: a - Q @ (Q.T @ a))(((r - common) * w).ravel())
    a, b = rest(prof), rest(prof0)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def loso_accent_id(tab, p=None) -> dict:
    """Leave-one-speaker-out accent identification, from the profile and from the centre.

    The profile is the conditional view of a speaker, the centre the global one.
    """
    p = tab.content() if p is None else p
    m, names = tab.means(), np.unique(tab.accents)
    cen, pro = centres(m, p), profiles(m, p)
    hit = {"conditional": [], "global": []}
    for s in range(len(tab.speakers)):
        others = [np.setdiff1d(tab.idx(a), [s]) for a in names]
        refs = np.stack([pro[i].mean(0) for i in others])
        hit["conditional"].append(names[accent_scores(pro[s], p > 0, refs, p).argmax()] == tab.accents[s])
        cref = np.stack([cen[i].mean(0) for i in others])
        a, B = cen[s] - cref.mean(0), cref - cref.mean(0)
        hit["global"].append(names[(B @ a / np.linalg.norm(B, axis=1)).argmax()] == tab.accents[s])
    return {k: float(np.mean(v)) for k, v in hit.items()}


# ---- intelligibility and identity, from audio ---------------------------------

class Asr:
    """Whisper. asr(wav) -> text; Asr.wer(refs, hyps) with Whisper's English normaliser."""

    def __init__(self, device="cuda", name="base"):
        import whisper
        from whisper.normalizers import EnglishTextNormalizer
        self.model, self.norm = whisper.load_model(name, device=device), EnglishTextNormalizer()
        self.fp16 = device != "cpu"

    def __call__(self, wav) -> str:
        return self.model.transcribe(np.asarray(wav, dtype=np.float32), language="en", fp16=self.fp16)["text"]

    def wer(self, refs, hyps) -> float:
        import jiwer
        return jiwer.wer([self.norm(r) for r in refs], [self.norm(h) for h in hyps])


class Resemblyzer:
    """GE2E d-vectors (the "SECS" of PACE). enc(wav at 16 kHz) -> unit vector."""

    def __init__(self, device="cuda"):
        from resemblyzer import VoiceEncoder
        self.model = VoiceEncoder(device=device, verbose=False)

    def __call__(self, wav) -> np.ndarray:
        from resemblyzer import preprocess_wav
        e = self.model.embed_utterance(preprocess_wav(np.asarray(wav, dtype=np.float32), source_sr=16000))
        return e / np.linalg.norm(e)


class Ecapa:
    """ECAPA-TDNN trained on VoxCeleb. enc(wav at 16 kHz) -> unit vector."""

    def __init__(self, device="cuda"):
        import torch
        from speechbrain.inference.speaker import EncoderClassifier
        self.torch = torch
        self.model = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb",
                                                    run_opts={"device": device})

    def __call__(self, wav) -> np.ndarray:
        e = self.model.encode_batch(self.torch.from_numpy(np.asarray(wav, dtype=np.float32))[None])
        e = e.squeeze().cpu().numpy()
        return e / np.linalg.norm(e)
