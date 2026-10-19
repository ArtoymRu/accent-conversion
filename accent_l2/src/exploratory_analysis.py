"""
Эксплораторный анализ корпуса L2-ARCTIC.
1) Статистический анализ: объёмы данных, распределение, средняя, медиана, мода и др.
2) Визуальный анализ: графики распределений и сравнений.
3) Корреляционный анализ: связь меток и признаков (длительность, акцент, спикер).
4) Выводы.
Запуск без torch: сканирует data/l2_arctic по правилам L2-ARCTIC (спикер -> акцент).
"""

import os
import sys
from pathlib import Path

import numpy as np

try:
    import yaml
except ImportError:
    yaml = None

# Add project root for optional dataset import
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
sys.path.insert(0, str(_project_root))

# L2-ARCTIC speaker -> accent (дублируем из accent_map, чтобы скрипт работал без torch)
SPEAKER_TO_ACCENT = {
    "ABA": "arabic", "SKA": "arabic", "YBAA": "arabic", "ZHAA": "arabic",
    "BWC": "mandarin", "LXC": "mandarin", "NCC": "mandarin", "TXHC": "mandarin",
    "ASI": "hindi", "RRBI": "hindi", "SVBI": "hindi", "TNI": "hindi",
    "HJK": "korean", "HKK": "korean", "YDCK": "korean", "YKWK": "korean",
    "EBVS": "spanish", "ERMS": "spanish", "MBMPS": "spanish", "NJS": "spanish",
    "HQTV": "vietnamese", "PNV": "vietnamese", "THV": "vietnamese", "TLV": "vietnamese",
}
ACCENTS = ["arabic", "mandarin", "hindi", "korean", "spanish", "vietnamese"]


def get_accent_id(accent: str) -> int:
    try:
        return ACCENTS.index(accent.lower())
    except ValueError:
        return -1

try:
    import soundfile as sf
except ImportError:
    sf = None

import wave as _wave

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    plt = None
    sns = None

try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None


def load_config(path: str = "configs/data.yaml") -> dict:
    if yaml is None:
        return {"data_root": "data/l2_arctic"}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_speaker_from_path(wav_path: str) -> str:
    """Из пути .../ABA/wav/file.wav извлекает код спикера ABA."""
    return Path(wav_path).parent.parent.name.upper()


def _find_wav_samples(root: Path) -> list:
    """Сканирует root на папки спикеров. Поддерживает плоскую и вложенную структуры:
    - плоская: root/SPEAKER/wav/*.wav
    - вложенная: root/accent/SPEAKER/wav/*.wav
    """
    out = []

    def _try_speaker_dir(path: Path):
        speaker = path.name.upper()
        if speaker not in SPEAKER_TO_ACCENT:
            return False
        wav_dir = path / "wav"
        if not wav_dir.is_dir():
            return False
        accent = SPEAKER_TO_ACCENT[speaker]
        for f in wav_dir.glob("*.wav"):
            out.append((str(f.resolve()), accent))
        return True

    for name in os.listdir(root):
        path = root / name
        if not path.is_dir():
            continue
        if not _try_speaker_dir(path):
            # Try one level deeper (accent subdirectory)
            for sub_name in os.listdir(path):
                sub_path = path / sub_name
                if sub_path.is_dir():
                    _try_speaker_dir(sub_path)
    return out


def collect_corpus_stats(data_root: str, use_soundfile: bool = True):
    """
    Собирает по каждому файлу: путь, акцент, accent_id, спикер, длительность (сек).
    Если use_soundfile=False или soundfile недоступен — длительности не считаем.
    """
    root = Path(data_root)
    if not root.is_absolute():
        root = _project_root / data_root
    samples = _find_wav_samples(root)
    rows = []
    for wav_path, accent in samples:
        speaker = get_speaker_from_path(wav_path)
        accent_id = get_accent_id(accent)
        duration_sec = np.nan
        if use_soundfile:
            if sf is not None:
                try:
                    info = sf.info(wav_path)
                    duration_sec = info.duration
                except Exception:
                    pass
            # Fallback: wave (когда soundfile недоступен или вернул 0)
            if np.isnan(duration_sec) or (isinstance(duration_sec, (int, float)) and duration_sec <= 0):
                try:
                    with _wave.open(str(Path(wav_path).resolve()), "rb") as w:
                        n = w.getnframes()
                        r = w.getframerate()
                        if r > 0:
                            duration_sec = n / float(r)
                except Exception:
                    pass
        rows.append({
            "path": wav_path,
            "accent": accent,
            "accent_id": accent_id,
            "speaker": speaker,
            "duration_sec": duration_sec,
        })
    return rows


def run_statistical_analysis(rows: list, out_dir: Path):
    """Статистический анализ: объёмы, распределение, средняя, медиана, мода и др."""
    if pd is None:
        print("pandas не установлен — статистика будет упрощённой.")
        _stats_fallback(rows)
        return None

    df = pd.DataFrame(rows)
    has_duration = df["duration_sec"].notna().any()

    # --- Общие объёмы ---
    n_total = len(df)
    n_accents = df["accent"].nunique()
    n_speakers = df["speaker"].nunique()

    print("\n" + "=" * 60)
    print("1. СТАТИСТИЧЕСКИЙ АНАЛИЗ КОРПУСА")
    print("=" * 60)
    print(f"Всего записей (файлов):     {n_total}")
    print(f"Число акцентов (классов):   {n_accents}")
    print(f"Число спикеров:             {n_speakers}")

    # --- По акцентам ---
    by_accent = df.groupby("accent").agg(
        count=("accent", "count"),
        duration_sec_sum=("duration_sec", "sum"),
        duration_sec_mean=("duration_sec", "mean"),
        duration_sec_median=("duration_sec", "median"),
        duration_sec_std=("duration_sec", "std"),
        duration_sec_min=("duration_sec", "min"),
        duration_sec_max=("duration_sec", "max"),
    ).reset_index()
    by_accent["n_speakers"] = df.groupby("accent")["speaker"].nunique().values

    print("\n--- Распределение по акцентам ---")
    print(by_accent.to_string(index=False))

    # --- По спикерам ---
    by_speaker = df.groupby("speaker").agg(
        accent=("accent", "first"),
        count=("accent", "count"),
        duration_sec_sum=("duration_sec", "sum"),
        duration_sec_mean=("duration_sec", "mean"),
    ).reset_index()

    print("\n--- Распределение по спикерам (первые 10) ---")
    print(by_speaker.head(10).to_string(index=False))

    # --- Общие статистические показатели по длительности ---
    if has_duration:
        d = df["duration_sec"].dropna()
        print("\n--- Показатели длительности записей (сек) по всему корпусу ---")
        print(f"  Среднее (mean):    {d.mean():.2f}")
        print(f"  Медиана (median):  {d.median():.2f}")
        print(f"  Мода (mode):       {d.mode().values[0]:.2f}" if len(d.mode()) else "  Мода: N/A")
        print(f"  Ст. откл. (std):   {d.std():.2f}")
        print(f"  Мин:               {d.min():.2f}")
        print(f"  Макс:              {d.max():.2f}")
        print(f"  Квартили: Q1={d.quantile(0.25):.2f}, Q2={d.quantile(0.5):.2f}, Q3={d.quantile(0.75):.2f}")

    # Сохраняем таблицы в CSV
    out_dir.mkdir(parents=True, exist_ok=True)
    by_accent.to_csv(out_dir / "stats_by_accent.csv", index=False)
    by_speaker.to_csv(out_dir / "stats_by_speaker.csv", index=False)

    return df


def _stats_fallback(rows: list):
    """Минимальная статистика без pandas."""
    from collections import Counter
    accents = [r["accent"] for r in rows]
    speakers = [r["speaker"] for r in rows]
    durations = [r["duration_sec"] for r in rows if r["duration_sec"] is not None and not np.isnan(r["duration_sec"])]
    print("Всего записей:", len(rows))
    print("По акцентам:", Counter(accents))
    print("По спикерам (число уникальных):", len(set(speakers)))
    if durations:
        a = np.array(durations)
        print("Длительность: mean=%.2f, median=%.2f, std=%.2f" % (np.mean(a), np.median(a), np.std(a)))


def run_visual_analysis(df, out_dir: Path):
    """Визуальный анализ: минимум 2 графика (распределение по акцентам, длительности)."""
    if plt is None or sns is None or df is None:
        print("matplotlib/seaborn или данные недоступны — визуализация пропущена.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    has_duration = df["duration_sec"].notna().any()

    # График 1: количество записей по акцентам (столбчатая диаграмма)
    fig, ax = plt.subplots(figsize=(10, 5))
    counts = df["accent"].value_counts().reindex(ACCENTS)
    counts.plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
    ax.set_title("Распределение количества записей по акцентам (L2-ARCTIC)")
    ax.set_xlabel("Акцент")
    ax.set_ylabel("Количество записей")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "plot1_counts_by_accent.png", dpi=150)
    plt.close()
    print("Сохранён график: plot1_counts_by_accent.png")

    # График 2: распределение длительности по акцентам (boxplot)
    if has_duration:
        fig, ax = plt.subplots(figsize=(10, 5))
        order = [a for a in ACCENTS if a in df["accent"].values]
        sns.boxplot(data=df, x="accent", y="duration_sec", hue="accent", order=order, ax=ax, palette="Set2", legend=False)
        ax.set_title("Распределение длительности записей по акцентам")
        ax.set_xlabel("Акцент")
        ax.set_ylabel("Длительность, с")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(out_dir / "plot2_duration_by_accent.png", dpi=150)
        plt.close()
        print("Сохранён график: plot2_duration_by_accent.png")

    # График 3: количество записей по спикерам (горизонтальная столбчатая)
    fig, ax = plt.subplots(figsize=(10, 8))
    speaker_counts = df["speaker"].value_counts()
    speaker_counts.plot(kind="barh", ax=ax, color="coral", edgecolor="black")
    ax.set_title("Количество записей по дикторам")
    ax.set_xlabel("Количество записей")
    ax.set_ylabel("Диктор")
    plt.tight_layout()
    plt.savefig(out_dir / "plot3_counts_by_speaker.png", dpi=150)
    plt.close()
    print("Сохранён график: plot3_counts_by_speaker.png")

    # График 4: гистограмма длительности по всему корпусу
    if has_duration:
        fig, ax = plt.subplots(figsize=(8, 5))
        df["duration_sec"].dropna().hist(bins=50, ax=ax, color="seagreen", edgecolor="black")
        ax.set_title("Гистограмма длительности записей (весь корпус)")
        ax.set_xlabel("Длительность (сек)")
        ax.set_ylabel("Частота")
        plt.tight_layout()
        plt.savefig(out_dir / "plot4_histogram_duration.png", dpi=150)
        plt.close()
        print("Сохранён график: plot4_histogram_duration.png")


def run_correlational_analysis(df, out_dir: Path):
    """Корреляционный анализ: связь меток (акцент) и признаков (длительность, спикер)."""
    if df is None or not df["duration_sec"].notna().any():
        print("Нет данных или длительностей для корреляционного анализа.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Числовые признаки для корреляции (на уровне записей)
    # accent_id — метка; duration_sec — признак
    X = df[["accent_id", "duration_sec"]].dropna()
    if len(X) < 2:
        return

    # Матрица корреляций Пирсона (accent_id и duration_sec)
    corr = X.corr()
    print("\n" + "=" * 60)
    print("3. КОРРЕЛЯЦИОННЫЙ АНАЛИЗ")
    print("=" * 60)
    print("Корреляция Пирсона (accent_id, duration_sec):")
    print(corr.to_string())

    if sns is not None and plt is not None:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(corr, annot=True, fmt=".3f", cmap="RdBu_r", center=0, ax=ax, square=True)
        ax.set_title("Корреляция: accent_id и duration_sec")
        plt.tight_layout()
        plt.savefig(out_dir / "correlation_accent_duration.png", dpi=150)
        plt.close()
        print("Сохранён график: correlation_accent_duration.png")

    # Агрегация по спикерам: корреляция между признаками спикеров
    by_speaker = df.groupby("speaker").agg(
        accent_id=("accent_id", "first"),
        n_files=("accent", "count"),
        total_duration=("duration_sec", "sum"),
        mean_duration=("duration_sec", "mean"),
    ).reset_index()
    by_speaker = by_speaker.dropna(subset=["total_duration", "mean_duration"])
    if len(by_speaker) >= 2:
        corr_speaker = by_speaker[["accent_id", "n_files", "total_duration", "mean_duration"]].corr()
        print("\nКорреляция на уровне спикеров (accent_id, n_files, total_duration, mean_duration):")
        print(corr_speaker.to_string())
        if sns is not None and plt is not None:
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(corr_speaker, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, square=True)
            ax.set_title("Корреляция признаков по спикерам")
            plt.tight_layout()
            plt.savefig(out_dir / "correlation_by_speaker.png", dpi=150)
            plt.close()
            print("Сохранён график: correlation_by_speaker.png")


def print_conclusions():
    """Краткие выводы по эксплораторному анализу."""
    print("\n" + "=" * 60)
    print("4. ВЫВОДЫ")
    print("=" * 60)
    print("""
• Объём корпуса: L2-ARCTIC содержит тысячи записей по 6 акцентам и 24 спикерам,
  что достаточно для обучения и оценки моделей классификации акцента.

• Распределение по акцентам и спикерам может быть неравномерным (разное число
  записей на акцент/спикера); при обучении классификаторов целесообразно учитывать
  дисбаланс (взвешивание классов или стратификацию).

• Длительность записей — полезный признак для анализа; её связь с меткой акцента
  (accent_id) обычно слабая, так как длительность в большей степени зависит от
  фразы, а не от языка L1.

• Корреляция признаков на уровне спикеров (число файлов, суммарная/средняя
  длительность) помогает проверить однородность корпуса по спикерам и выявить
  возможные выбросы.

• Визуализации (столбчатые диаграммы, boxplot, гистограмма) наглядно показывают
  распределение данных и поддерживают решение о предобработке и стратегии обучения.
""")


def main():
    config = load_config()
    data_root = config.get("data_root", "data/l2_arctic")
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / data_root
    out_dir = project_root / "outputs" / "exploratory_analysis"

    if not data_path.is_dir():
        print(f"Директория данных не найдена: {data_path}")
        return

    print("Сбор метаданных корпуса (длительности через soundfile)...")
    # Сбор длительностей: soundfile при наличии, иначе wave (стандартная библиотека)
    rows = collect_corpus_stats(str(data_path), use_soundfile=True)
    if not rows:
        print("Нет записей в корпусе.")
        return

    df = run_statistical_analysis(rows, out_dir)
    run_visual_analysis(df, out_dir)
    run_correlational_analysis(df, out_dir)
    print_conclusions()

    print(f"\nРезультаты сохранены в: {out_dir}")


if __name__ == "__main__":
    main()
