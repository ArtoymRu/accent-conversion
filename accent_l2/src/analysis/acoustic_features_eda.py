"""
Акустический EDA корпуса L2-ARCTIC.
Анализирует паралингвистические признаки речи по акцентам:
- Основной тон (F0/pitch): средний, диапазон, вариабельность
- Энергия (RMS): динамика интенсивности
- Темп речи (speaking rate): слов/сек из транскриптов
- Спектральные признаки: MFCC, spectral centroid
Запускать: venv/Scripts/python src/analysis/acoustic_features_eda.py
"""

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

try:
    import librosa
    import librosa.display
except ImportError:
    raise ImportError("librosa not installed. Run: pip install librosa")

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    plt = None; sns = None

from scipy import stats as scipy_stats

SPEAKER_TO_ACCENT = {
    "ABA": "arabic", "SKA": "arabic", "YBAA": "arabic", "ZHAA": "arabic",
    "BWC": "mandarin", "LXC": "mandarin", "NCC": "mandarin", "TXHC": "mandarin",
    "ASI": "hindi", "RRBI": "hindi", "SVBI": "hindi", "TNI": "hindi",
    "HJK": "korean", "HKK": "korean", "YDCK": "korean", "YKWK": "korean",
    "EBVS": "spanish", "ERMS": "spanish", "MBMPS": "spanish", "NJS": "spanish",
    "HQTV": "vietnamese", "PNV": "vietnamese", "THV": "vietnamese", "TLV": "vietnamese",
}
ACCENTS = ["arabic", "mandarin", "hindi", "korean", "spanish", "vietnamese"]
ACCENT_COLORS = {
    "arabic": "#e41a1c", "mandarin": "#ff7f00", "hindi": "#4daf4a",
    "korean": "#377eb8", "spanish": "#984ea3", "vietnamese": "#a65628",
}


def _find_all_samples(data_root: Path) -> list:
    """Returns list of (wav_path, speaker, accent, transcript_path_or_None)."""
    samples = []
    for accent_dir in sorted(data_root.iterdir()):
        if not accent_dir.is_dir():
            continue
        for speaker_dir in sorted(accent_dir.iterdir()):
            if not speaker_dir.is_dir():
                continue
            speaker = speaker_dir.name.upper()
            if speaker not in SPEAKER_TO_ACCENT:
                continue
            accent = SPEAKER_TO_ACCENT[speaker]
            wav_dir = speaker_dir / "wav"
            transcript_dir = speaker_dir / "transcript"
            if not wav_dir.is_dir():
                continue
            for wav_file in sorted(wav_dir.glob("*.wav")):
                stem = wav_file.stem
                txt_path = transcript_dir / f"{stem}.txt" if transcript_dir.is_dir() else None
                if txt_path and not txt_path.exists():
                    txt_path = None
                samples.append((wav_file, speaker, accent, txt_path))
    return samples


def _load_audio(path: Path, sr: int = 16000):
    """Load audio file with librosa, return (y, sr)."""
    y, orig_sr = librosa.load(str(path), sr=sr, mono=True)
    return y, sr


def _extract_f0(y: np.ndarray, sr: int, fmin: float = 75.0, fmax: float = 600.0):
    """Extract fundamental frequency using PYIN. Returns array of voiced F0 values."""
    try:
        f0, voiced_flag, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr, frame_length=2048)
        voiced_f0 = f0[voiced_flag == 1] if voiced_flag is not None else f0[~np.isnan(f0)]
        return voiced_f0[~np.isnan(voiced_f0)]
    except Exception:
        return np.array([])


def _extract_rms(y: np.ndarray, frame_length: int = 2048, hop_length: int = 512):
    """Return per-frame RMS energy."""
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    return rms


def _extract_mfcc(y: np.ndarray, sr: int, n_mfcc: int = 13):
    """Return mean MFCCs (shape: n_mfcc)."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc.mean(axis=1)


def _extract_spectral_centroid(y: np.ndarray, sr: int):
    """Return mean spectral centroid (Hz)."""
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    return float(np.mean(sc))


def _count_words(txt_path: Optional[Path]) -> Optional[int]:
    """Count words in transcript file."""
    if txt_path is None:
        return None
    try:
        text = txt_path.read_text(encoding="utf-8").strip()
        return len(text.split())
    except Exception:
        return None


def collect_acoustic_features(data_root: Path, max_per_accent: int = 100, sr: int = 16000):
    """
    Collects acoustic features for up to max_per_accent utterances per accent.
    Returns list of dicts with fields: accent, speaker, f0_mean, f0_std, f0_range,
    rms_mean, rms_std, speaking_rate, spectral_centroid, mfcc_1..13, duration.
    """
    samples = _find_all_samples(data_root)
    print(f"Total samples found: {len(samples)}")

    # Balance: take max_per_accent per accent
    from collections import defaultdict
    by_accent = defaultdict(list)
    for s in samples:
        by_accent[s[2]].append(s)

    selected = []
    for accent in ACCENTS:
        group = by_accent[accent]
        np.random.seed(42)
        np.random.shuffle(group)
        selected.extend(group[:max_per_accent])

    print(f"Selected {len(selected)} samples ({max_per_accent} per accent)")

    rows = []
    for i, (wav_path, speaker, accent, txt_path) in enumerate(selected):
        if i % 50 == 0:
            print(f"  Processing {i}/{len(selected)}...")
        try:
            y, _ = _load_audio(wav_path, sr=sr)
            duration = len(y) / sr

            f0_vals = _extract_f0(y, sr)
            f0_mean = float(np.mean(f0_vals)) if len(f0_vals) > 0 else np.nan
            f0_std = float(np.std(f0_vals)) if len(f0_vals) > 0 else np.nan
            f0_range = float(np.ptp(f0_vals)) if len(f0_vals) > 0 else np.nan

            rms = _extract_rms(y)
            rms_mean = float(np.mean(rms))
            rms_std = float(np.std(rms))

            mfcc = _extract_mfcc(y, sr)
            sc = _extract_spectral_centroid(y, sr)

            n_words = _count_words(txt_path)
            speaking_rate = n_words / duration if (n_words and duration > 0) else np.nan

            row = {
                "accent": accent,
                "speaker": speaker,
                "duration": duration,
                "f0_mean": f0_mean,
                "f0_std": f0_std,
                "f0_range": f0_range,
                "rms_mean": rms_mean,
                "rms_std": rms_std,
                "spectral_centroid": sc,
                "speaking_rate": speaking_rate,
            }
            for j, v in enumerate(mfcc, start=1):
                row[f"mfcc_{j}"] = float(v)
            rows.append(row)
        except Exception as e:
            print(f"  Error on {wav_path.name}: {e}")
            continue

    return rows


def plot_f0_analysis(df, out_dir: Path):
    """F0 (pitch) distribution per accent."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Violin plot: F0 mean
    order = [a for a in ACCENTS if a in df["accent"].values]
    palette = {a: ACCENT_COLORS[a] for a in order}

    sns.violinplot(data=df, x="accent", y="f0_mean", order=order,
                   hue="accent", palette=palette, ax=axes[0], inner="box", legend=False)
    axes[0].set_title("Средний F0 (Гц) по акцентам", fontsize=12)
    axes[0].set_xlabel("Акцент")
    axes[0].set_ylabel("F0 mean (Гц)")
    axes[0].tick_params(axis="x", rotation=30)

    sns.violinplot(data=df, x="accent", y="f0_std", order=order,
                   hue="accent", palette=palette, ax=axes[1], inner="box", legend=False)
    axes[1].set_title("Вариабельность F0 (интонация) по акцентам", fontsize=12)
    axes[1].set_xlabel("Акцент")
    axes[1].set_ylabel("F0 std (Гц)")
    axes[1].tick_params(axis="x", rotation=30)

    sns.violinplot(data=df, x="accent", y="f0_range", order=order,
                   hue="accent", palette=palette, ax=axes[2], inner="box", legend=False)
    axes[2].set_title("Диапазон F0 (интонационный контур)", fontsize=12)
    axes[2].set_xlabel("Акцент")
    axes[2].set_ylabel("F0 range (Гц)")
    axes[2].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(out_dir / "acoustic_f0_analysis.png", dpi=150)
    plt.close()
    print("Saved: acoustic_f0_analysis.png")


def plot_tempo_energy(df, out_dir: Path):
    """Speaking rate and RMS energy per accent."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    order = [a for a in ACCENTS if a in df["accent"].values]
    palette = {a: ACCENT_COLORS[a] for a in order}

    # Speaking rate
    df_sr = df.dropna(subset=["speaking_rate"])
    if len(df_sr) > 0:
        sns.boxplot(data=df_sr, x="accent", y="speaking_rate", order=order,
                    hue="accent", palette=palette, ax=axes[0], legend=False)
        axes[0].set_title("Темп речи (слов/сек) по акцентам", fontsize=12)
        axes[0].set_xlabel("Акцент")
        axes[0].set_ylabel("Слов в секунду")
        axes[0].tick_params(axis="x", rotation=30)

    sns.boxplot(data=df, x="accent", y="rms_mean", order=order,
                hue="accent", palette=palette, ax=axes[1], legend=False)
    axes[1].set_title("Средняя энергия (RMS) по акцентам", fontsize=12)
    axes[1].set_xlabel("Акцент")
    axes[1].set_ylabel("RMS energy")
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(out_dir / "acoustic_tempo_energy.png", dpi=150)
    plt.close()
    print("Saved: acoustic_tempo_energy.png")


def plot_mfcc_heatmap(df, out_dir: Path):
    """Heatmap of mean MFCC coefficients per accent."""
    mfcc_cols = [f"mfcc_{i}" for i in range(1, 14)]
    accent_mfcc = df.groupby("accent")[mfcc_cols].mean().reindex(ACCENTS).dropna()

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(accent_mfcc, annot=True, fmt=".1f", cmap="RdBu_r",
                center=0, ax=ax, linewidths=0.5)
    ax.set_title("Средние значения MFCC коэффициентов по акцентам", fontsize=13)
    ax.set_xlabel("MFCC коэффициент")
    ax.set_ylabel("Акцент")
    plt.tight_layout()
    plt.savefig(out_dir / "acoustic_mfcc_heatmap.png", dpi=150)
    plt.close()
    print("Saved: acoustic_mfcc_heatmap.png")


def plot_spectral_centroid(df, out_dir: Path):
    """Spectral centroid distribution per accent."""
    fig, ax = plt.subplots(figsize=(9, 5))
    order = [a for a in ACCENTS if a in df["accent"].values]
    palette = {a: ACCENT_COLORS[a] for a in order}
    sns.violinplot(data=df, x="accent", y="spectral_centroid", order=order,
                   hue="accent", palette=palette, ax=ax, inner="box", legend=False)
    ax.set_title("Спектральный центроид по акцентам (яркость звука)", fontsize=12)
    ax.set_xlabel("Акцент")
    ax.set_ylabel("Спектральный центроид (Гц)")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(out_dir / "acoustic_spectral_centroid.png", dpi=150)
    plt.close()
    print("Saved: acoustic_spectral_centroid.png")


def plot_acoustic_summary(df, out_dir: Path):
    """Summary table plot: mean ± std for key metrics per accent."""
    metrics = {
        "f0_mean": "F0 среднее (Гц)",
        "f0_std": "F0 std (Гц)",
        "speaking_rate": "Темп (сл/с)",
        "rms_mean": "Энергия (RMS)",
        "spectral_centroid": "Спектр. центроид (Гц)",
    }
    order = [a for a in ACCENTS if a in df["accent"].values]

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5))
    for ax, (col, label) in zip(axes, metrics.items()):
        data_col = df.dropna(subset=[col])
        means = data_col.groupby("accent")[col].mean().reindex(order)
        stds = data_col.groupby("accent")[col].std().reindex(order)
        colors = [ACCENT_COLORS[a] for a in order]
        ax.bar(order, means, yerr=stds, color=colors, edgecolor="black", capsize=4)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)
    plt.suptitle("Акустические признаки: среднее ± ст.откл. по акцентам", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir / "acoustic_summary_bars.png", dpi=150)
    plt.close()
    print("Saved: acoustic_summary_bars.png")


def run_statistical_tests(df, out_dir: Path):
    """Kruskal-Wallis test for each acoustic feature across accents."""
    print("\n" + "=" * 60)
    print("СТАТИСТИЧЕСКИЕ ТЕСТЫ (Kruskal-Wallis H)")
    print("=" * 60)
    metrics = ["f0_mean", "f0_std", "f0_range", "rms_mean", "speaking_rate", "spectral_centroid"]
    results = []
    order = [a for a in ACCENTS if a in df["accent"].values]
    for metric in metrics:
        groups = [df[df["accent"] == a][metric].dropna().values for a in order]
        groups = [g for g in groups if len(g) > 2]
        if len(groups) < 2:
            continue
        stat, p = scipy_stats.kruskal(*groups)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"  {metric:<22}: H={stat:.2f}, p={p:.4f} {sig}")
        results.append({"metric": metric, "H_stat": round(stat, 3), "p_value": round(p, 5), "sig": sig})
    if pd is not None:
        pd.DataFrame(results).to_csv(out_dir / "acoustic_kruskal_tests.csv", index=False)
    print()


def print_accent_stats(df, out_dir: Path):
    """Print and save mean acoustic stats per accent."""
    print("\n" + "=" * 60)
    print("СРЕДНИЕ АКУСТИЧЕСКИЕ ПРИЗНАКИ ПО АКЦЕНТАМ")
    print("=" * 60)
    cols = ["f0_mean", "f0_std", "f0_range", "rms_mean", "speaking_rate", "spectral_centroid"]
    stats = df.groupby("accent")[cols].agg(["mean", "std"]).round(2)
    print(stats.to_string())
    if pd is not None:
        df.groupby("accent")[cols].agg(["mean", "std"]).round(3).to_csv(
            out_dir / "acoustic_stats_by_accent.csv"
        )
        print(f"\nСохранено: acoustic_stats_by_accent.csv")


def main():
    data_root = _ROOT / "data" / "l2_arctic"
    out_dir = _ROOT / "outputs" / "exploratory_analysis" / "acoustic"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_root.exists():
        print(f"Данные не найдены: {data_root}")
        return

    print("Извлечение акустических признаков (librosa)...")
    rows = collect_acoustic_features(data_root, max_per_accent=150)
    if not rows:
        print("Нет данных.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "acoustic_features_raw.csv", index=False)
    print(f"Сохранено: acoustic_features_raw.csv ({len(df)} записей)")

    print("\nГенерация графиков...")
    plot_f0_analysis(df, out_dir)
    plot_tempo_energy(df, out_dir)
    plot_mfcc_heatmap(df, out_dir)
    plot_spectral_centroid(df, out_dir)
    plot_acoustic_summary(df, out_dir)

    print_accent_stats(df, out_dir)
    run_statistical_tests(df, out_dir)

    print(f"\nВсе результаты в: {out_dir}")


if __name__ == "__main__":
    main()
