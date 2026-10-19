"""
Сравнительный анализ Speaker Clustering Purity для трёх дискретных представлений речи.

Вычисляет метрику для всех доступных условий:
  HuBERT Base+ k-means:  k = 50, 100, 200, 500
  WavLM  Base+ k-means:  k = 50, 100, 200, 500
  EnCodec 24kHz RVQ:     nq = 1, 2, 4, 8

Сохраняет:
  outputs/discrete_analysis/speaker_purity.csv
  outputs/discrete_analysis/speaker_purity.png

Пример:
  python src/analysis/compare_representations.py
"""

import sys
from pathlib import Path

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.analysis.speaker_purity import compute_purity_from_dir

N_SPEAKERS = 24


def collect_results() -> pd.DataFrame:
    feats_root = _ROOT / "outputs" / "features"
    rows = []

    # HuBERT k-means
    for k in [50, 100, 200, 500]:
        d = feats_root / "hubert_discrete" / f"k{k}"
        if not d.exists():
            print(f"  [skip] HuBERT k={k}: директория не найдена ({d})", flush=True)
            continue
        print(f"  HuBERT k={k} ...", flush=True)
        purity, n_frames, vocab = compute_purity_from_dir(d, mode="kmeans")
        rows.append({
            "model": "HuBERT",
            "condition": f"k={k}",
            "vocab_size": k,
            "log2_vocab": float(np.log2(k)),
            "purity": purity,
            "n_frames": n_frames,
            "active_vocab": vocab,
        })
        print(f"    purity={purity:.4f}  frames={n_frames:,}  active_vocab={vocab}", flush=True)

    # WavLM k-means
    for k in [50, 100, 200, 500]:
        d = feats_root / "wavlm_discrete" / f"k{k}"
        if not d.exists():
            print(f"  [skip] WavLM k={k}: директория не найдена ({d})", flush=True)
            continue
        print(f"  WavLM k={k} ...", flush=True)
        purity, n_frames, vocab = compute_purity_from_dir(d, mode="kmeans")
        rows.append({
            "model": "WavLM",
            "condition": f"k={k}",
            "vocab_size": k,
            "log2_vocab": float(np.log2(k)),
            "purity": purity,
            "n_frames": n_frames,
            "active_vocab": vocab,
        })
        print(f"    purity={purity:.4f}  frames={n_frames:,}  active_vocab={vocab}", flush=True)

    # EnCodec RVQ
    for nq in [1, 2, 4, 8]:
        d = feats_root / "encodec" / f"nq{nq}"
        if not d.exists():
            print(f"  [skip] EnCodec nq={nq}: директория не найдена ({d})", flush=True)
            continue
        print(f"  EnCodec nq={nq} ...", flush=True)
        # Теоретический размер словаря: 1024^nq
        vocab_theoretical = 1024 ** nq
        purity, n_frames, vocab = compute_purity_from_dir(d, mode="encodec", encodec_nq=nq)
        rows.append({
            "model": "EnCodec",
            "condition": f"nq={nq}",
            "vocab_size": vocab_theoretical,
            "log2_vocab": math.log2(vocab_theoretical),
            "purity": purity,
            "n_frames": n_frames,
            "active_vocab": vocab,
        })
        print(f"    purity={purity:.4f}  frames={n_frames:,}  active_vocab={vocab}", flush=True)

    return pd.DataFrame(rows)


def plot_results(df: pd.DataFrame, out_dir: Path):
    colors = {"HuBERT": "#4472C4", "WavLM": "#ED7D31", "EnCodec": "#70AD47"}
    markers = {"HuBERT": "o", "WavLM": "s", "EnCodec": "^"}

    fig, ax = plt.subplots(figsize=(11, 6))

    for model, grp in df.groupby("model"):
        grp_s = grp.sort_values("log2_vocab")
        color = colors.get(model, "gray")
        ax.plot(
            grp_s["log2_vocab"],
            grp_s["purity"],
            color=color,
            marker=markers.get(model, "o"),
            linewidth=2.2,
            markersize=9,
            label=model,
            zorder=3,
        )
        for _, row in grp_s.iterrows():
            ax.annotate(
                row["condition"],
                (row["log2_vocab"], row["purity"]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8.5,
                color=color,
            )

    ax.axhline(
        1.0 / N_SPEAKERS,
        color="red", linestyle="--", linewidth=1.5, alpha=0.7,
        label=f"Случайный выбор (1/{N_SPEAKERS} ≈ {1/N_SPEAKERS:.3f})",
        zorder=2,
    )

    ax.set_xlabel("log₂(Размер словаря)", fontsize=12)
    ax.set_ylabel("Speaker Clustering Purity", fontsize=12)
    ax.set_title(
        "Сохранение информации о дикторе в дискретных представлениях\n"
        "(L2-ARCTIC, 24 диктора)",
        fontsize=13,
    )
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.35)
    ax.set_ylim(0.0, 1.05)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    plt.tight_layout()
    fig_path = out_dir / "speaker_purity.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig_path


def main():
    out_dir = _ROOT / "outputs" / "discrete_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Сравнительный анализ Speaker Clustering Purity")
    print("=" * 60, flush=True)

    df = collect_results()

    if df.empty:
        print(
            "\nНет данных. Сначала запустите скрипты извлечения:\n"
            "  python src/features/extract_wavlm.py --layers 6\n"
            "  python src/features/kmeans_quantize.py --model hubert --layer 6\n"
            "  python src/features/kmeans_quantize.py --model wavlm  --layer 6\n"
            "  python src/features/extract_encodec.py --nq 1 2 4 8"
        )
        return

    csv_path = out_dir / "speaker_purity.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nCSV сохранён: {csv_path}")

    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 60)
    display_cols = ["model", "condition", "purity", "active_vocab", "n_frames"]
    print(df.sort_values(["model", "log2_vocab"])[display_cols].to_string(index=False))

    best = df.loc[df["purity"].idxmax()]
    print(f"\nЛучший результат: {best['model']} {best['condition']} — purity={best['purity']:.4f}")

    fig_path = plot_results(df, out_dir)
    print(f"График сохранён: {fig_path}", flush=True)


if __name__ == "__main__":
    main()
