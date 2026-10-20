"""
OT → ресинтез → метрики на аудио.

Для held-out диктора src-акцента: кадры HuBERT L6 отображаются OT-картой в акцент tgt,
озвучиваются декодером (src/resynth/train_decoder.py) + HiFi-GAN, затем на аудио 16 кГц:
  WER                 — Whisper-small против транскрипта (разборчивость),
  speaker (ECAPA)     — top1 / capture / cos_src / margin по 24 дикторам с зачислением
                        по ОРИГИНАЛЬНОМУ аудио (как speaker_probe, но на настоящих embeddings),
  accent-ID           — HuBERT L6 от ресинтезированного аудио → классификаторы акцента
                        (сырой и NAP), обученные на оригинальных признаках train-дикторов.

Системы: original (исходное аудио), identity (декодер без OT — baseline канала),
OT-варианты. Выводы делаются относительно identity, а не original: канал декодер+вокодер
сам меняет WER, голос и признаки HuBERT.

Результаты: outputs/resynth/ot_fold<f>[_tag]/utt_eval.csv, summary.csv, wav/

Пример:
  python src/resynth/resynth_eval.py --fold 3 --n_utt 20
"""

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.data.accent_map import SPEAKER_TO_ACCENT
from src.ot import GaussianOTMap, UnitConditionalOTMap, quantize
from src.ot.data import fold_heldout, load_utts_ids, sample_frames, utterance_means
from src.ot.robust_classifier import SpeakerRobustAccentClassifier
from src.resynth.audio_metrics import Asr, HubertFeatures, SpeakerEcapa, Vocoder, to_16k
from src.resynth.decoder import HubertToMel
from src.resynth.mel import MEL_SR, HOP, load_wav, transcript, wav_path
from src.utils.splits import ACCENTS

# имя метода → (тип карты, speaker_norm, lam); None — без OT (identity)
METHODS = {
    "identity": None,
    "uent_l0.5": ("uent", False, 0.5),
    "uent_l1.0": ("uent", False, 1.0),
    "uent_l1.0_spknorm": ("uent", True, 1.0),
    "ubw-mean_l1.0_spknorm": ("ubw-mean", True, 1.0),
}
DEFAULT_PAIRS = ["arabic:mandarin", "hindi:spanish", "korean:arabic",
                 "vietnamese:hindi", "mandarin:korean", "spanish:vietnamese"]


def word_errors(ref: str, hyp: str):
    from jiwer import process_words

    from src.baseline.metrics import normalize_text
    r, h = normalize_text(ref), normalize_text(hyp)
    if not r:
        return 0, 0
    o = process_words(r, h if h else " ")
    return o.substitutions + o.deletions + o.insertions, len(r.split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=3)
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--pairs", nargs="*", default=DEFAULT_PAIRS)
    ap.add_argument("--methods", nargs="*", default=list(METHODS))
    ap.add_argument("--n_utt", type=int, default=20, help="тестовых высказываний на пару")
    ap.add_argument("--n_enr_audio", type=int, default=10, help="высказываний на диктора для зачисления ECAPA")
    ap.add_argument("--n_fit_utt", type=int, default=100)
    ap.add_argument("--n_eval", type=int, default=40, help="загружаемых высказываний held-out диктора")
    ap.add_argument("--frames_per_utt", type=int, default=150)
    ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--nap_r", type=int, default=12)
    ap.add_argument("--min_frames", type=int, default=30)
    ap.add_argument("--cond_shrink", type=float, default=0.3)
    ap.add_argument("--cond_n_max", type=int, default=300)
    ap.add_argument("--n_wav", type=int, default=1, help="сколько wav на (пара, метод) сохранить")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_tag", default="")
    args = ap.parse_args()

    dev = "cuda"
    rng = np.random.default_rng(args.seed)
    feat_dir = _ROOT / "outputs" / "features" / "hubert" / f"layer{args.layer}"
    out = _ROOT / "outputs" / "resynth" / f"ot_fold{args.fold}" f"{'_' + args.out_tag if args.out_tag else ''}"
    (out / "wav").mkdir(parents=True, exist_ok=True)

    # ── данные фолда ────────────────────────────────────────────────────────
    held = fold_heldout(args.fold)
    held_spk = {SPEAKER_TO_ACCENT[s]: s for s in held}
    ids, utts = {}, {}
    for spk in SPEAKER_TO_ACCENT:
        ids[spk], utts[spk] = load_utts_ids(feat_dir, spk, max(args.n_fit_utt, args.n_eval), rng)
    train_utts = {a: {s: utts[s][: args.n_fit_utt] for s in utts if SPEAKER_TO_ACCENT[s] == a and s not in held}
                  for a in ACCENTS}
    train_ids = {s: ids[s][: args.n_fit_utt] for s in utts if s not in held}
    half = args.n_eval // 2
    mu_held = {a: np.concatenate(utts[held_spk[a]][:half]).mean(0) for a in ACCENTS}   # среднее по зачисляемой половине

    cache = _ROOT / "outputs" / "ot" / "cache" / f"kmeans_hubert_L{args.layer}_k{args.k}_fold{args.fold}_seed{args.seed}.npy"
    if not cache.exists():
        sys.exit(f"Нет k-means фолда: {cache}\nСначала запустите src/ot/run_frame_ot.py --folds {args.fold}")
    centroids = np.load(cache)

    spk_mean = {s: np.concatenate(us).mean(0) for a in ACCENTS for s, us in train_utts[a].items()}
    raw_clouds = {a: [(s, sample_frames(us, args.frames_per_utt, rng)) for s, us in train_utts[a].items()] for a in ACCENTS}
    clouds_raw, clouds_c, cloud_units = {}, {}, {}
    for a in ACCENTS:
        clouds_raw[a] = np.concatenate([f for _, f in raw_clouds[a]])
        clouds_c[a] = np.concatenate([f - spk_mean[s] for s, f in raw_clouds[a]])
        cloud_units[a] = np.concatenate([quantize(f, centroids) for _, f in raw_clouds[a]])

    # ── классификаторы акцента (оригинальные признаки train-дикторов) ───────
    Xtr, ytr, str_ = [], [], []
    for ai, a in enumerate(ACCENTS):
        for s, us in train_utts[a].items():
            m = utterance_means(us)
            Xtr.append(m); ytr += [ai] * len(m); str_ += [s] * len(m)
    Xtr, ytr, str_ = np.concatenate(Xtr), np.array(ytr), np.array(str_)
    clf_raw = SpeakerRobustAccentClassifier(n_nuisance=0).fit(Xtr, ytr, str_)
    clf_nap = SpeakerRobustAccentClassifier(n_nuisance=args.nap_r).fit(Xtr, ytr, str_)

    # ── модели аудио-стороны ────────────────────────────────────────────────
    t0 = time.time()
    voc, asr, ecapa, hub = Vocoder(dev), Asr(device=dev), SpeakerEcapa(dev), HubertFeatures(args.layer, device=dev)
    ck = _ROOT / "outputs" / "checkpoints" / "resynth" / f"hubert_L{args.layer}_fold{args.fold}_decoder.pt"
    dec = HubertToMel().to(dev)
    dec.load_state_dict(torch.load(ck, map_location=dev, weights_only=False)["state"])
    dec.eval()
    print(f"Модели загружены за {time.time() - t0:.0f}s; декодер: {ck.name}", flush=True)

    def wav16_of(spk, uid):
        return load_wav(wav_path(spk, uid), 16000).numpy()

    # ── зачисление дикторов по ОРИГИНАЛЬНОМУ аудио (ECAPA) ──────────────────
    centroid_names, centroid_vecs = [], []
    for s in SPEAKER_TO_ACCENT:
        enr_ids = (train_ids[s] if s not in held else ids[s][:half])[: args.n_enr_audio]
        e = ecapa.embed([wav16_of(s, u) for u in enr_ids]).mean(0)
        centroid_names.append(s); centroid_vecs.append(e / np.linalg.norm(e))
    C = np.stack(centroid_vecs)
    print("ECAPA: зачислено 24 диктора", flush=True)

    # ── пары ────────────────────────────────────────────────────────────────
    rows = []
    for pair in args.pairs:
        src, tgt = pair.split(":")
        t1 = time.time()
        si, ti = ACCENTS.index(src), ACCENTS.index(tgt)
        sspk = held_spk[src]
        tgt_train = list(train_utts[tgt])
        tgt_idx = np.array([centroid_names.index(s) for s in tgt_train])
        s_idx = centroid_names.index(sspk)

        # карты OT (нужные методам)
        need = {(m[0], m[1]) for n, m in METHODS.items() if n in args.methods and m}
        maps = {}
        for kind, norm in need:
            cl = clouds_c if norm else clouds_raw
            if kind == "uent":
                mp = UnitConditionalOTMap("ent", min_frames=args.min_frames, eps=args.eps, n_max=args.cond_n_max, seed=args.seed)
            else:
                mp = UnitConditionalOTMap("bw", "mean", shrink=args.cond_shrink, min_frames=args.min_frames)
            maps[(kind, norm)] = mp.fit(cl[src], cl[tgt], cloud_units[src], cloud_units[tgt])

        test_ids = ids[sspk][half: half + args.n_utt]
        test_utts = utts[sspk][half: half + args.n_utt]
        wavs = {m: [] for m in args.methods}
        for uid, u in zip(test_ids, test_utts):
            mel_len = load_wav(wav_path(sspk, uid), MEL_SR).shape[0] // HOP + 1
            units = quantize(u, centroids)
            for name in args.methods:
                spec = METHODS[name]
                if spec is None:
                    z = u
                else:
                    kind, norm, lam = spec
                    mp = maps[(kind, norm)]
                    mu = mu_held[src] if norm else 0.0
                    z = mp(u - mu, units, lam=lam).float().numpy() + mu
                mel = dec.synthesize_mel(torch.as_tensor(z, dtype=torch.float32, device=dev), mel_len)
                wavs[name].append(voc(mel))
        wavs["original"] = [wav16_of(sspk, uid) for uid in test_ids]
        order = list(wavs)

        refs = [transcript(sspk, uid) for uid in test_ids]
        for name in order:
            ws = wavs[name]
            hyps = asr.transcribe(ws)
            E = ecapa.embed(ws)
            sim = E @ C.T
            nearest = sim.argmax(1)
            other = np.delete(sim, s_idx, axis=1).max(1)
            Hm = np.stack([hub(w).mean(0) for w in ws])
            Pr, Pn = clf_raw.predict_proba(Hm), clf_nap.predict_proba(Hm)
            for j, uid in enumerate(test_ids):
                err, nw = word_errors(refs[j], hyps[j])
                rows.append({
                    "fold": args.fold, "pair": pair, "src": src, "tgt": tgt, "method": name, "utt": uid, "ref": refs[j], "hyp": hyps[j],
                    "word_err": err, "n_words": nw,
                    "spk_top1": float(nearest[j] == s_idx), "spk_capture": float(np.isin(nearest[j], tgt_idx)),
                    "spk_cos_src": float(sim[j, s_idx]), "spk_margin": float(sim[j, s_idx] - other[j]),
                    "p_tgt": Pr[j, ti], "p_tgt_nap": Pn[j, ti], "p_src": Pr[j, si], "p_src_nap": Pn[j, si],
                })
            for j in range(min(args.n_wav, len(ws))):
                sf.write(out / "wav" / f"{src}-{tgt}_{test_ids[j]}_{name}.wav", ws[j], 16000)
        pd.DataFrame(rows).to_csv(out / "utt_eval.csv", index=False)
        print(f"  {pair}: {time.time() - t1:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    agg = df.groupby("method").apply(lambda g: pd.Series({
        "WER": g.word_err.sum() / max(g.n_words.sum(), 1),
        "spk_top1": g.spk_top1.mean(), "spk_capture": g.spk_capture.mean(),
        "spk_cos_src": g.spk_cos_src.mean(), "spk_margin": g.spk_margin.mean(),
        "p_tgt": g.p_tgt.mean(), "p_tgt_nap": g.p_tgt_nap.mean(),
        "p_src": g.p_src.mean(), "n": len(g),
    }), include_groups=False)
    agg = agg.loc[[m for m in ["original"] + args.methods if m in agg.index]]
    agg.to_csv(out / "summary.csv")
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", None)
    print("\nСводка по системам (среднее по парам):\n", agg.round(3), flush=True)
    print(f"\nСохранено в {out}")


if __name__ == "__main__":
    main()
