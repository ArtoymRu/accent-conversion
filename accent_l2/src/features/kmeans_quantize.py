"""
K-means квантизация HuBERT/WavLM признаков в дискретные токены.

Для каждого размера словаря k (число кластеров) обучает MiniBatchKMeans
на признаках обучающих дикторов (speaker-independent: held-out спикеры исключаются)
и применяет ко всем высказываниям.

Для каждого файла сохраняет:
  'tokens':  Tensor(T,) int64 — индексы кластеров
  'speaker': str
  'accent':  str

Аргументы:
  --model   hubert или wavlm
  --layer   слой модели (default: 6)
  --k       размеры словарей (default: 50 100 200 500)
  --max_frames  максимум фреймов для обучения k-means (default: 200000)

Пример:
  python src/features/kmeans_quantize.py --model hubert --layer 6
  python src/features/kmeans_quantize.py --model wavlm  --layer 6
"""

import argparse
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.data.accent_map import SPEAKER_TO_ACCENT

# Held-out test speakers (не участвуют в обучении k-means)
HELD_OUT_SPEAKERS = {"ZHAA", "TXHC", "TNI", "YKWK", "NJS", "TLV"}


def load_feature_dir(feat_dir: Path):
    """Возвращает список (pt_path, speaker) для всех .pt файлов."""
    entries = []
    for pt_file in sorted(feat_dir.glob("*.pt")):
        parts = pt_file.stem.split("_")
        speaker = parts[0].upper()
        if speaker in SPEAKER_TO_ACCENT:
            entries.append((pt_file, speaker))
    return entries


def collect_train_features(feat_dir: Path, held_out: set, max_frames: int) -> np.ndarray:
    """Собирает признаки обучающих дикторов (до max_frames фреймов)."""
    entries = [(p, s) for p, s in load_feature_dir(feat_dir) if s not in held_out]
    random.seed(42)
    random.shuffle(entries)

    all_feats = []
    total = 0
    for pt_path, _ in entries:
        if total >= max_frames:
            break
        try:
            data = torch.load(pt_path, map_location="cpu", weights_only=True)
            feats = data["features"]
            if feats.dim() == 3:
                feats = feats.squeeze(0)  # (T, D)
            arr = feats.numpy()
            all_feats.append(arr)
            total += len(arr)
        except Exception:
            continue

    if not all_feats:
        raise RuntimeError(f"Не удалось загрузить признаки из {feat_dir}")

    combined = np.concatenate(all_feats, axis=0)[:max_frames]
    print(f"    Собрано {len(combined)} фреймов из {len(all_feats)} файлов", flush=True)
    return combined


def apply_kmeans(feat_dir: Path, km: MiniBatchKMeans, out_dir: Path) -> int:
    """Применяет k-means ко всем файлам, сохраняет дискретные токены."""
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = load_feature_dir(feat_dir)
    ok = 0
    for pt_path, speaker in entries:
        out_path = out_dir / pt_path.name
        if out_path.exists():
            ok += 1
            continue
        try:
            data = torch.load(pt_path, map_location="cpu", weights_only=True)
            feats = data["features"]
            if feats.dim() == 3:
                feats = feats.squeeze(0)
            tokens = km.predict(feats.numpy()).astype(np.int64)
            # speaker берём из имени файла — HuBERT-файлы не содержат ключ 'speaker'
            accent = data.get("accent", SPEAKER_TO_ACCENT.get(speaker, ""))
            torch.save({
                "tokens": torch.from_numpy(tokens),
                "speaker": speaker,
                "accent": accent,
            }, out_path)
            ok += 1
        except Exception as e:
            print(f"  Error {pt_path.name}: {e}", flush=True)
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["hubert", "wavlm"], default="hubert")
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--k", type=int, nargs="+", default=[50, 100, 200, 500])
    parser.add_argument("--max_frames", type=int, default=200_000)
    args = parser.parse_args()

    feat_dir = _ROOT / "outputs" / "features" / args.model / f"layer{args.layer}"
    km_dir = _ROOT / "outputs" / "kmeans_models"
    km_dir.mkdir(parents=True, exist_ok=True)

    if not feat_dir.exists():
        print(f"Директория признаков не найдена: {feat_dir}")
        print("Сначала запустите extract_wavlm.py или убедитесь, что HuBERT-признаки извлечены.")
        return

    n_files = len(list(feat_dir.glob("*.pt")))
    print(f"K-means квантизация: {args.model} layer{args.layer}")
    print(f"  feat_dir:   {feat_dir}  ({n_files} файлов)")
    print(f"  k values:   {args.k}")
    print(f"  max_frames: {args.max_frames}", flush=True)

    for k in args.k:
        print(f"\n── k={k} {'─'*40}", flush=True)

        km_path = km_dir / f"{args.model}_layer{args.layer}_k{k}.pkl"
        if km_path.exists():
            print(f"  Загружаем сохранённую модель: {km_path}", flush=True)
            with open(km_path, "rb") as f:
                km = pickle.load(f)
        else:
            print(f"  Сбор обучающих признаков (max {args.max_frames} фреймов)...", flush=True)
            train_feats = collect_train_features(feat_dir, HELD_OUT_SPEAKERS, args.max_frames)

            print(f"  Обучение MiniBatchKMeans(k={k})...", flush=True)
            km = MiniBatchKMeans(
                n_clusters=k,
                batch_size=min(50_000, max(k * 50, 10_000)),
                n_init=5,
                random_state=42,
                verbose=0,
            )
            km.fit(train_feats)
            inertia = km.inertia_
            print(f"  Inertia: {inertia:.2f}", flush=True)

            with open(km_path, "wb") as f:
                pickle.dump(km, f)
            print(f"  Модель сохранена: {km_path}", flush=True)

        out_dir = _ROOT / "outputs" / "features" / f"{args.model}_discrete" / f"k{k}"
        n = apply_kmeans(feat_dir, km, out_dir)
        print(f"  Токенизировано: {n} файлов → {out_dir}", flush=True)

    print("\nГотово.", flush=True)


if __name__ == "__main__":
    main()
