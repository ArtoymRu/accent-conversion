"""
Компиляция всех результатов экспериментов в единую таблицу (пункт 6 курсовой).

Читает CSV из:
  outputs/models/experiment_table.csv       — Audio + HuBERT (HuBERT-модели)
  outputs/models/mfcc/experiment_table.csv  — Audio + MFCC
  outputs/models/text/experiment_table.csv  — Text (char/word TF-IDF)
  outputs/multimodal/multimodal_experiment_table.csv — Fusion-модели

Создаёт:
  outputs/final_experiment_table.csv   — полная таблица
  outputs/report_figures/experiment_comparison.png — визуализация

Пример:
  python src/models/compile_results.py
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))


def load_hubert_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        model_type = "baseline" if r["model"] == "mlp" else "enhanced"
        rows.append({
            "Modality": "Audio",
            "Feature": "HuBERT (layer 6)",
            "Model": r["model"].upper(),
            "Type": model_type,
            "Test Accuracy": r["test_accuracy"],
            "Test F1": r["test_f1"],
        })
    return pd.DataFrame(rows)


def load_mfcc_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        model_type = "baseline" if "mlp" in r["model"] and "deep" not in r["model"] else "enhanced"
        rows.append({
            "Modality": "Audio",
            "Feature": "MFCC+Δ+ΔΔ (39-dim)",
            "Model": r["model"].upper(),
            "Type": model_type,
            "Test Accuracy": r["test_accuracy"],
            "Test F1": r["test_f1"],
        })
    return pd.DataFrame(rows)


def load_text_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    rows = []
    feature_map = {
        "char_tfidf": "Text: char TF-IDF (2,4)",
        "word_tfidf": "Text: word TF-IDF (1,2)",
    }
    for _, r in df.iterrows():
        feature = feature_map.get(r.get("feature", ""), r.get("feature", "unknown"))
        model_type = "baseline" if "baseline" in r["model"] else "enhanced"
        rows.append({
            "Modality": "Text",
            "Feature": feature,
            "Model": r["model"].upper(),
            "Type": model_type,
            "Test Accuracy": r["test_accuracy"],
            "Test F1": r["test_f1"],
        })
    return pd.DataFrame(rows)


def load_fusion_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "Modality": "Multimodal",
            "Feature": "Audio+Text Fusion",
            "Model": r["model"],
            "Type": "fusion",
            "Test Accuracy": r["test_accuracy"],
            "Test F1": float(r["test_f1"]),
        })
    return pd.DataFrame(rows)


def main():
    out_root = _ROOT / "outputs"
    fig_dir = out_root / "report_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    dfs = []

    # Load available results
    hubert_path = out_root / "models" / "experiment_table.csv"
    if hubert_path.exists():
        dfs.append(load_hubert_results(hubert_path))
        print(f"✓ HuBERT results: {hubert_path}")
    else:
        print(f"✗ Missing: {hubert_path}")

    mfcc_path = out_root / "models" / "mfcc" / "experiment_table.csv"
    if mfcc_path.exists():
        dfs.append(load_mfcc_results(mfcc_path))
        print(f"✓ MFCC results: {mfcc_path}")
    else:
        print(f"✗ Missing: {mfcc_path}")

    text_path = out_root / "models" / "text" / "experiment_table.csv"
    if text_path.exists():
        dfs.append(load_text_results(text_path))
        print(f"✓ Text results: {text_path}")
    else:
        print(f"✗ Missing: {text_path}")

    fusion_path = out_root / "multimodal" / "multimodal_experiment_table.csv"
    if fusion_path.exists():
        dfs.append(load_fusion_results(fusion_path))
        print(f"✓ Fusion results: {fusion_path}")
    else:
        print(f"✗ Missing: {fusion_path}")

    if not dfs:
        print("No results found."); return

    df = pd.concat(dfs, ignore_index=True)
    df["Test F1"] = df["Test F1"].astype(float).round(4)
    df["Test Accuracy"] = df["Test Accuracy"].astype(float).round(4)

    # Save unified table
    out_path = out_root / "final_experiment_table.csv"
    df.to_csv(out_path, index=False)
    print(f"\nСохранена итоговая таблица: {out_path}")
    print("\n" + "=" * 80)
    print("ИТОГОВАЯ ТАБЛИЦА ЭКСПЕРИМЕНТОВ")
    print("=" * 80)
    print(df.to_string(index=False))

    # ── Visualization ─────────────────────────────────────────────────────── #
    modality_colors = {
        "Audio":      "#4472C4",
        "Text":       "#ED7D31",
        "Multimodal": "#70AD47",
    }
    type_alpha = {
        "baseline": 0.5,
        "enhanced": 0.9,
        "fusion":   0.95,
    }

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(df))
    bars = []
    for i, row in df.iterrows():
        color = modality_colors.get(row["Modality"], "gray")
        alpha = type_alpha.get(row["Type"], 0.8)
        bar = ax.bar(i, row["Test F1"], color=color, alpha=alpha, width=0.7)
        bars.append(bar)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{r['Model']}\n({r['Feature'].split(':')[-1].strip()[:18]})"
         for _, r in df.iterrows()],
        rotation=45, ha="right", fontsize=8
    )
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Test F1-macro (speaker-independent)", fontsize=11)
    ax.set_title("Таблица экспериментов: сравнение модальностей, признаков и моделей",
                 fontsize=12)
    ax.axhline(1/6, color="red", linestyle="--", alpha=0.5, label="Случайный выбор (1/6)")
    ax.grid(axis="y", alpha=0.3)

    # Legend
    legend_handles = [
        mpatches.Patch(color=c, label=m, alpha=0.8)
        for m, c in modality_colors.items()
    ] + [plt.Line2D([0], [0], color="red", linestyle="--", label="Random baseline")]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9)

    plt.tight_layout()
    fig_path = fig_dir / "experiment_comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nСохранён график: {fig_path}")


if __name__ == "__main__":
    main()
