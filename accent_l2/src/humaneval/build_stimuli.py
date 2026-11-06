"""
Сборка набора стимулов и сайта для human eval (оценка акцента, MOS, сходство диктора).

Берёт уже ресинтезированные wav из outputs/resynth/ot_fold<f>/wav/ (по одному высказыванию на
(фолд, пару), без отбора — первое тестовое высказывание пары, как записано в resynth_eval) и собирает:

  outputs/human_eval/site/            — статический сайт (index.html, app.js, style.css, manifest.json, audio/)
  outputs/human_eval/private/stimulus_key.csv — ключ sid → (item, система …); НЕ выкладывать на сервер

Дизайн (подробно — docs/HUMAN_EVAL_PROTOCOL.md):
  5 систем: original (скрытый верхний референс), identity (канал без OT), ubw-mean (speaker_norm),
  uent λ=0.5, uent λ=1.0. 24 item'а = 4 фолда × 6 пар. Каждый рater видит каждый item один раз,
  система определяется латинским квадратом: система(item i, rater r) = SYSTEMS[(i + r) % 5].
  Имена файлов — хэши; порядок sid в manifest — циклический сдвиг, не раскрывающий систему без ключа.
  Контроли: 6 «плохих» стимулов (original + шум 0 дБ SNR + полоса 3.4 кГц), 2 на рater;
  повторы 2 стимулов для test–retest (в app.js).
  Громкость всех клипов приведена к одному уровню (RMS по речевым кадрам, −23 dBFS, пик ≤ −1 dBFS).

Пример:
  python src/humaneval/build_stimuli.py
"""

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.data.accent_map import SPEAKER_TO_ACCENT
from src.ot.data import N_FOLDS, fold_heldout
from src.resynth.mel import wav_path
from src.utils.splits import ACCENT_TO_SPEAKERS, ACCENTS

SYSTEMS = ["original", "identity", "ubw-mean_l1.0_spknorm", "uent_l0.5", "uent_l1.0"]
PAIRS = ["arabic:mandarin", "hindi:spanish", "korean:arabic",
         "vietnamese:hindi", "mandarin:korean", "spanish:vietnamese"]
SR = 16000
TARGET_DBFS = -23.0
PEAK_LIMIT = 10 ** (-1.0 / 20)


def speech_rms(x: np.ndarray, sr: int = SR) -> float:
    """RMS по речевым кадрам (кадры > −35 дБ от самого громкого), чтобы паузы не занижали уровень."""
    n = int(0.025 * sr)
    frames = x[: len(x) // n * n].reshape(-1, n)
    r = np.sqrt((frames ** 2).mean(1) + 1e-12)
    keep = r > r.max() * 10 ** (-35 / 20)
    return float(np.sqrt((frames[keep] ** 2).mean()))


def normalize(x: np.ndarray) -> np.ndarray:
    gain = 10 ** (TARGET_DBFS / 20) / max(speech_rms(x), 1e-6)
    y = x * gain
    peak = np.abs(y).max()
    if peak > PEAK_LIMIT:
        y = y * (PEAK_LIMIT / peak)
    fade = int(0.01 * SR)
    y[:fade] *= np.linspace(0, 1, fade)
    y[-fade:] *= np.linspace(1, 0, fade)
    return np.concatenate([np.zeros(int(0.1 * SR), dtype=y.dtype), y, np.zeros(int(0.1 * SR), dtype=y.dtype)])


def degrade(x: np.ndarray, rng: np.random.Generator, snr_db: float = 0.0) -> np.ndarray:
    """«Плохой» контроль: полоса 3.4 кГц + белый шум при заданном SNR."""
    from scipy.signal import butter, sosfilt

    y = sosfilt(butter(6, 3400, btype="low", fs=SR, output="sos"), x)
    noise = rng.normal(size=len(y))
    noise *= speech_rms(y) / 10 ** (snr_db / 20) / max(noise.std(), 1e-9)
    return (y + noise).astype(np.float32)


def read16k(path: Path) -> np.ndarray:
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != SR:
        import torch
        import torchaudio.functional as AF

        x = AF.resample(torch.from_numpy(x), sr, SR).numpy()
    return x


def pick_clip(speaker: str, rng: np.random.Generator, exclude=(), min_s=2.5, max_s=6.0):
    d = wav_path(speaker, "x").parent
    files = sorted(d.glob("*.wav"))
    for i in rng.permutation(len(files)):
        f = files[i]
        if f.stem in exclude:
            continue
        info = sf.info(str(f))
        if min_s <= info.duration <= max_s:
            return f.stem
    raise RuntimeError(f"нет подходящего клипа у {speaker}")


def sid_for(seed: int, key: str) -> str:
    return "s_" + hashlib.sha1(f"{seed}:{key}".encode()).hexdigest()[:10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(_ROOT / "outputs" / "human_eval"))
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    site, private = out / "site", out / "private"
    if site.exists():
        shutil.rmtree(site)
    (site / "audio").mkdir(parents=True)
    private.mkdir(parents=True, exist_ok=True)

    # ── item'ы: порядок «целевой акцент → фолд» (4 item'а подряд на целевой акцент) ──
    raw_items = []
    for pair in PAIRS:
        src, tgt = pair.split(":")
        for f in range(N_FOLDS):
            wd = _ROOT / "outputs" / "resynth" / f"ot_fold{f}" / "wav"
            cands = sorted(wd.glob(f"{src}-{tgt}_*_original.wav"))
            if len(cands) != 1:
                sys.exit(f"ожидался ровно один original-клип для {pair} в фолде {f}: {cands}")
            uid = cands[0].stem[len(f"{src}-{tgt}_"): -len("_original")]
            spk = next(s for s in sorted(fold_heldout(f)) if SPEAKER_TO_ACCENT[s] == src)
            raw_items.append(dict(fold=f, pair=pair, src=src, tgt=tgt, utt=uid, speaker=spk, wav_dir=wd))
    raw_items.sort(key=lambda d: (ACCENTS.index(d["tgt"]), d["fold"], ACCENTS.index(d["src"])))

    key_rows, items, checks = [], [], []
    for i, it in enumerate(raw_items):
        item_id = f"i{i:02d}"
        sids = {}
        for sysname in SYSTEMS:
            w = it["wav_dir"] / f"{it['src']}-{it['tgt']}_{it['utt']}_{sysname}.wav"
            x = read16k(w)
            sid = sid_for(args.seed, f"{item_id}:{sysname}")
            sf.write(site / "audio" / f"{sid}.wav", normalize(x), SR, subtype="PCM_16")
            sids[sysname] = sid
            key_rows.append(dict(sid=sid, kind="item", item=item_id, system=sysname, fold=it["fold"],
                                 pair=it["pair"], src=it["src"], tgt=it["tgt"], speaker=it["speaker"], utt=it["utt"]))
        # референс голоса: другое высказывание того же held-out диктора (оригинал)
        ref_utt = pick_clip(it["speaker"], rng, exclude={it["utt"]})
        ref_sid = sid_for(args.seed, f"{item_id}:voice")
        sf.write(site / "audio" / f"{ref_sid}.wav", normalize(read16k(wav_path(it["speaker"], ref_utt))), SR, subtype="PCM_16")
        key_rows.append(dict(sid=ref_sid, kind="voice_ref", item=item_id, system="original", fold=it["fold"],
                             pair=it["pair"], src=it["src"], tgt=it["tgt"], speaker=it["speaker"], utt=ref_utt))
        # циклический сдвиг: stimuli[p] — система SYSTEMS[(i + p) % 5]; рater r берёт позицию r % 5
        items.append(dict(id=item_id, tgt=it["tgt"], voice_ref=f"audio/{ref_sid}.wav",
                          stimuli=[f"audio/{sids[SYSTEMS[(i + p) % len(SYSTEMS)]]}.wav" for p in range(len(SYSTEMS))]))
        it["id"] = item_id
        it["orig_sid"] = sids["original"]

    # ── контроли: «плохие» клипы, по одному на целевой акцент (из item'ов этого акцента) ──
    for ti, tgt in enumerate(ACCENTS):
        it = next(d for d in raw_items if d["tgt"] == tgt)
        w = it["wav_dir"] / f"{it['src']}-{it['tgt']}_{it['utt']}_original.wav"
        sid = sid_for(args.seed, f"check:{tgt}")
        sf.write(site / "audio" / f"{sid}.wav", normalize(degrade(read16k(w), rng)), SR, subtype="PCM_16")
        checks.append(dict(id=f"c{ti}", tgt=tgt, stimulus=f"audio/{sid}.wav", voice_ref=next(x for x in items if x["id"] == it["id"])["voice_ref"]))
        key_rows.append(dict(sid=sid, kind="check_low", item=it["id"], system="degraded_original", fold=it["fold"],
                             pair=it["pair"], src=it["src"], tgt=tgt, speaker=it["speaker"], utt=it["utt"]))

    # ── якоря целевого акцента: 3 клипа от 3 разных дикторов акцента ──
    anchors = {}
    for tgt in ACCENTS:
        spks = sorted(ACCENT_TO_SPEAKERS[tgt])[:3]
        anchors[tgt] = []
        for spk in spks:
            utt = pick_clip(spk, rng)
            sid = sid_for(args.seed, f"anchor:{tgt}:{spk}")
            sf.write(site / "audio" / f"{sid}.wav", normalize(read16k(wav_path(spk, utt))), SR, subtype="PCM_16")
            anchors[tgt].append(f"audio/{sid}.wav")
            key_rows.append(dict(sid=sid, kind="anchor", item="", system="original", fold="", pair="",
                                 src="", tgt=tgt, speaker=spk, utt=utt))

    labels = {a: f"{a.capitalize()}-accented English" for a in ACCENTS}
    manifest = dict(version=1, seed=args.seed, n_systems=len(SYSTEMS), items=items, checks=checks,
                    anchors=anchors, accent_labels=labels, accent_order=ACCENTS)
    (site / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(private / "stimulus_key.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(key_rows[0]))
        w.writeheader(); w.writerows(key_rows)
    (private / "systems_order.json").write_text(json.dumps(SYSTEMS), encoding="utf-8")

    web = Path(__file__).resolve().parent / "web"
    for f in web.iterdir():
        shutil.copy(f, site / f.name)
    total = sum(p.stat().st_size for p in (site / "audio").glob("*.wav")) / 1e6
    print(f"item'ов: {len(items)}, контролей: {len(checks)}, якорей: {sum(map(len, anchors.values()))}; "
          f"аудио {len(list((site / 'audio').glob('*.wav')))} файлов, {total:.1f} МБ")
    print(f"сайт: {site}\nключ (приватный): {private / 'stimulus_key.csv'}")


if __name__ == "__main__":
    main()
