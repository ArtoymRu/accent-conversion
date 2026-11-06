"""
Рисунок архитектуры мультимодальной модели для курсовой (пункт 6).

Схема показывает:
  - Две модальности: Audio и Text
  - По два признака на модальность:
      Audio: HuBERT (768-dim), MFCC+Δ+ΔΔ (39-dim)
      Text:  char TF-IDF (512-dim), word TF-IDF (512-dim)
  - Классификаторы для каждой ветки (MLP baseline и enhanced)
  - Три стратегии fusion: Early, Late, CrossAttention

Вывод: outputs/report_figures/architecture_overview.png

Пример:
  python src/analysis/draw_architecture.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))


# ──────────────── Colour palette ─────────────────────────────────────────── #
C_AUDIO   = "#4472C4"   # deep blue — audio modality
C_TEXT    = "#ED7D31"   # orange — text modality
C_FUSION  = "#70AD47"   # green — fusion
C_OUTPUT  = "#7030A0"   # purple — classification output
C_BG      = "#F2F2F2"   # light grey background for boxes
C_WHITE   = "white"
C_DARK    = "#2E2E2E"


def rounded_box(ax, x, y, w, h, color, alpha=0.9, lw=1.5,
                ec="white", zorder=3):
    box = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                         boxstyle="round,pad=0.03",
                         facecolor=color, edgecolor=ec,
                         linewidth=lw, alpha=alpha, zorder=zorder)
    ax.add_patch(box)
    return box


def arrow(ax, x0, y0, x1, y1, color="#555555", lw=1.5, zorder=2,
          style="-|>"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle="arc3,rad=0.0"),
                zorder=zorder)


def label(ax, x, y, text, fontsize=9, color=C_DARK, ha="center",
          va="center", bold=False, zorder=5):
    weight = "bold" if bold else "normal"
    ax.text(x, y, text, fontsize=fontsize, color=color,
            ha=ha, va=va, fontweight=weight, zorder=zorder)


def draw_full_architecture():
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    ax.set_title(
        "Мультимодальная архитектура классификации акцентов\n"
        "(2 модальности × 2 признака + 3 стратегии объединения)",
        fontsize=13, fontweight="bold", pad=12
    )

    # ── Column x-positions ─────────────────────────────────────────────── #
    # Input → Feature Extractor → Encoder → Fusion → Output
    X_INPUT   = 1.3
    X_FEAT    = 3.5
    X_ENC     = 5.9
    X_PROJ    = 8.3
    X_FUSION  = 11.5
    X_CLF     = 14.5
    X_OUT     = 16.8

    # ── Row y-positions ─────────────────────────────────────────────────── #
    # 4 branches stacked vertically
    Y_HUBERT = 7.8
    Y_MFCC   = 5.6
    Y_CHAR   = 3.5
    Y_WORD   = 1.4

    BW, BH = 2.0, 0.65   # box width, height
    BH2     = 0.55

    # ── Section background bands ─────────────────────────────────────────── #
    audio_bg = FancyBboxPatch((0.1, 4.65), 9.8, 4.0,
                              boxstyle="round,pad=0.1",
                              facecolor="#EBF3FB", edgecolor=C_AUDIO,
                              linewidth=1.5, alpha=0.6, zorder=0)
    text_bg = FancyBboxPatch((0.1, 0.3), 9.8, 3.9,
                             boxstyle="round,pad=0.1",
                             facecolor="#FFF3EB", edgecolor=C_TEXT,
                             linewidth=1.5, alpha=0.6, zorder=0)
    ax.add_patch(audio_bg)
    ax.add_patch(text_bg)
    label(ax, 0.55, 8.45, "Аудио\nмодальность", 8, C_AUDIO, bold=True, ha="center")
    label(ax, 0.55, 2.4,  "Текстовая\nмодальность", 8, C_TEXT, bold=True, ha="center")

    # ─────────────────────────────────────────────────────────────────────── #
    #                        BRANCH 1: HuBERT                                 #
    # ─────────────────────────────────────────────────────────────────────── #
    # Input
    rounded_box(ax, X_INPUT, Y_HUBERT, 1.6, BH, C_AUDIO, 0.7)
    label(ax, X_INPUT, Y_HUBERT, "WAV файл", 8.5, C_WHITE, bold=True)

    # Feature extractor
    rounded_box(ax, X_FEAT, Y_HUBERT, BW, BH, C_AUDIO, 0.85)
    label(ax, X_FEAT, Y_HUBERT + 0.15, "HuBERT Base", 8.5, C_WHITE, bold=True)
    label(ax, X_FEAT, Y_HUBERT - 0.18, "(12 слоёв, 768-dim)", 7, C_WHITE)

    # Encoder output
    rounded_box(ax, X_ENC, Y_HUBERT, BW, BH, C_AUDIO, 0.7)
    label(ax, X_ENC, Y_HUBERT + 0.15, "Layer 6 hidden", 8.5, C_WHITE, bold=True)
    label(ax, X_ENC, Y_HUBERT - 0.18, "(T × 768)", 7, C_WHITE)

    # Projection
    rounded_box(ax, X_PROJ, Y_HUBERT, BW, BH2, C_AUDIO, 0.6)
    label(ax, X_PROJ, Y_HUBERT, "Mean-pool → 768", 8, C_WHITE)

    # Arrows in branch 1
    arrow(ax, X_INPUT + 0.8, Y_HUBERT, X_FEAT - 1.0, Y_HUBERT, C_AUDIO)
    arrow(ax, X_FEAT + 1.0, Y_HUBERT, X_ENC - 1.0, Y_HUBERT, C_AUDIO)
    arrow(ax, X_ENC + 1.0, Y_HUBERT, X_PROJ - 1.0, Y_HUBERT, C_AUDIO)

    # ─────────────────────────────────────────────────────────────────────── #
    #                        BRANCH 2: MFCC                                   #
    # ─────────────────────────────────────────────────────────────────────── #
    rounded_box(ax, X_INPUT, Y_MFCC, 1.6, BH, C_AUDIO, 0.7)
    label(ax, X_INPUT, Y_MFCC, "WAV файл", 8.5, C_WHITE, bold=True)

    rounded_box(ax, X_FEAT, Y_MFCC, BW, BH, C_AUDIO, 0.85)
    label(ax, X_FEAT, Y_MFCC + 0.15, "librosa MFCC", 8.5, C_WHITE, bold=True)
    label(ax, X_FEAT, Y_MFCC - 0.18, "(+delta +delta2)", 7, C_WHITE)

    rounded_box(ax, X_ENC, Y_MFCC, BW, BH, C_AUDIO, 0.7)
    label(ax, X_ENC, Y_MFCC + 0.15, "MFCC фреймы", 8.5, C_WHITE, bold=True)
    label(ax, X_ENC, Y_MFCC - 0.18, "(T × 39)", 7, C_WHITE)

    rounded_box(ax, X_PROJ, Y_MFCC, BW, BH2, C_AUDIO, 0.6)
    label(ax, X_PROJ, Y_MFCC, "Mean-pool → 39", 8, C_WHITE)

    arrow(ax, X_INPUT + 0.8, Y_MFCC, X_FEAT - 1.0, Y_MFCC, C_AUDIO)
    arrow(ax, X_FEAT + 1.0, Y_MFCC, X_ENC - 1.0, Y_MFCC, C_AUDIO)
    arrow(ax, X_ENC + 1.0, Y_MFCC, X_PROJ - 1.0, Y_MFCC, C_AUDIO)

    # ─────────────────────────────────────────────────────────────────────── #
    #                        BRANCH 3: char TF-IDF                            #
    # ─────────────────────────────────────────────────────────────────────── #
    rounded_box(ax, X_INPUT, Y_CHAR, 1.6, BH, C_TEXT, 0.7)
    label(ax, X_INPUT, Y_CHAR, "Транскрипт", 8.5, C_WHITE, bold=True)

    rounded_box(ax, X_FEAT, Y_CHAR, BW, BH, C_TEXT, 0.85)
    label(ax, X_FEAT, Y_CHAR + 0.15, "char TF-IDF", 8.5, C_WHITE, bold=True)
    label(ax, X_FEAT, Y_CHAR - 0.18, "(ngram 2-4)", 7, C_WHITE)

    rounded_box(ax, X_ENC, Y_CHAR, BW, BH, C_TEXT, 0.7)
    label(ax, X_ENC, Y_CHAR + 0.15, "Разреженный", 8.5, C_WHITE, bold=True)
    label(ax, X_ENC, Y_CHAR - 0.18, "вектор (512-dim)", 7, C_WHITE)

    rounded_box(ax, X_PROJ, Y_CHAR, BW, BH2, C_TEXT, 0.6)
    label(ax, X_PROJ, Y_CHAR, "Dense 512-dim", 8, C_WHITE)

    arrow(ax, X_INPUT + 0.8, Y_CHAR, X_FEAT - 1.0, Y_CHAR, C_TEXT)
    arrow(ax, X_FEAT + 1.0, Y_CHAR, X_ENC - 1.0, Y_CHAR, C_TEXT)
    arrow(ax, X_ENC + 1.0, Y_CHAR, X_PROJ - 1.0, Y_CHAR, C_TEXT)

    # ─────────────────────────────────────────────────────────────────────── #
    #                        BRANCH 4: word TF-IDF                            #
    # ─────────────────────────────────────────────────────────────────────── #
    rounded_box(ax, X_INPUT, Y_WORD, 1.6, BH, C_TEXT, 0.7)
    label(ax, X_INPUT, Y_WORD, "Транскрипт", 8.5, C_WHITE, bold=True)

    rounded_box(ax, X_FEAT, Y_WORD, BW, BH, C_TEXT, 0.85)
    label(ax, X_FEAT, Y_WORD + 0.15, "word TF-IDF", 8.5, C_WHITE, bold=True)
    label(ax, X_FEAT, Y_WORD - 0.18, "(unigram+bigram)", 7, C_WHITE)

    rounded_box(ax, X_ENC, Y_WORD, BW, BH, C_TEXT, 0.7)
    label(ax, X_ENC, Y_WORD + 0.15, "Разреженный", 8.5, C_WHITE, bold=True)
    label(ax, X_ENC, Y_WORD - 0.18, "вектор (512-dim)", 7, C_WHITE)

    rounded_box(ax, X_PROJ, Y_WORD, BW, BH2, C_TEXT, 0.6)
    label(ax, X_PROJ, Y_WORD, "Dense 512-dim", 8, C_WHITE)

    arrow(ax, X_INPUT + 0.8, Y_WORD, X_FEAT - 1.0, Y_WORD, C_TEXT)
    arrow(ax, X_FEAT + 1.0, Y_WORD, X_ENC - 1.0, Y_WORD, C_TEXT)
    arrow(ax, X_ENC + 1.0, Y_WORD, X_PROJ - 1.0, Y_WORD, C_TEXT)

    # ─────────────────────────────────────────────────────────────────────── #
    #                Individual classifiers (one per branch)                   #
    # ─────────────────────────────────────────────────────────────────────── #
    X_IND_CLF = 10.3
    BH_CLF = 0.52

    ind_configs = [
        (Y_HUBERT, C_AUDIO, "MLP / LSTM"),
        (Y_MFCC,   C_AUDIO, "MLP / LSTM"),
        (Y_CHAR,   C_TEXT,  "MLP"),
        (Y_WORD,   C_TEXT,  "MLP"),
    ]
    for (y, c, txt) in ind_configs:
        rounded_box(ax, X_IND_CLF, y, 1.7, BH_CLF, c, 0.55)
        label(ax, X_IND_CLF, y + 0.1, "Classifier", 7.5, C_WHITE, bold=True)
        label(ax, X_IND_CLF, y - 0.15, txt, 7, C_WHITE)
        arrow(ax, X_PROJ + 1.0, y, X_IND_CLF - 0.85, y, "#888888")

    # ─────────────────────────────────────────────────────────────────────── #
    #                        Fusion box                                        #
    # ─────────────────────────────────────────────────────────────────────── #
    Y_FUSE_TOP = 7.0
    Y_FUSE_BOT = 1.8
    Y_FUSE_MID = (Y_FUSE_TOP + Y_FUSE_BOT) / 2

    fusion_bg = FancyBboxPatch((X_FUSION - 1.3, Y_FUSE_BOT - 0.5), 2.6, 6.0,
                               boxstyle="round,pad=0.1",
                               facecolor="#F0FBF0", edgecolor=C_FUSION,
                               linewidth=2, alpha=0.85, zorder=1)
    ax.add_patch(fusion_bg)
    label(ax, X_FUSION, 9.0, "Fusion\n(мультимодальный\nклассификатор)", 9,
          C_FUSION, bold=True)

    # Three fusion strategy boxes
    fuse_strategies = [
        (8.0, "Early Fusion",        "concat → MLP"),
        (6.5, "Late Fusion",         "weighted logit sum"),
        (5.0, "Cross-Attention",     "Q=text, K/V=audio"),
    ]
    for (y_f, title, sub) in fuse_strategies:
        rounded_box(ax, X_FUSION, y_f, 2.4, 0.8, C_FUSION, 0.85)
        label(ax, X_FUSION, y_f + 0.17, title, 8.5, C_WHITE, bold=True)
        label(ax, X_FUSION, y_f - 0.18, sub, 7.5, C_WHITE)

    # Arrows from HuBERT+MFCC projections to each fusion strategy
    for (y_f, _, __) in fuse_strategies:
        # Audio (HuBERT)
        ax.annotate("", xy=(X_FUSION - 1.2, y_f + 0.05),
                    xytext=(X_PROJ + 1.0, Y_HUBERT),
                    arrowprops=dict(arrowstyle="-|>", color=C_AUDIO, lw=1.2,
                                   connectionstyle="arc3,rad=0.1"),
                    zorder=2)
        # Audio (MFCC)
        ax.annotate("", xy=(X_FUSION - 1.2, y_f),
                    xytext=(X_PROJ + 1.0, Y_MFCC),
                    arrowprops=dict(arrowstyle="-|>", color=C_AUDIO, lw=1.0,
                                   connectionstyle="arc3,rad=-0.05",
                                   linestyle="dashed"),
                    zorder=2)
        # Text (char)
        ax.annotate("", xy=(X_FUSION - 1.2, y_f - 0.05),
                    xytext=(X_PROJ + 1.0, Y_CHAR),
                    arrowprops=dict(arrowstyle="-|>", color=C_TEXT, lw=1.2,
                                   connectionstyle="arc3,rad=-0.1"),
                    zorder=2)

    # ─────────────────────────────────────────────────────────────────────── #
    #                  Output: softmax over 6 accents                          #
    # ─────────────────────────────────────────────────────────────────────── #
    # Individual classifiers → individual outputs
    for y in [Y_HUBERT, Y_MFCC, Y_CHAR, Y_WORD]:
        rounded_box(ax, X_CLF, y, 1.5, 0.48, C_OUTPUT, 0.6)
        label(ax, X_CLF, y, "6 accents\n(softmax)", 7, C_WHITE)
        arrow(ax, X_IND_CLF + 0.85, y, X_CLF - 0.75, y, "#999999")

    # Fusion → output
    for (y_f, _, __) in fuse_strategies:
        rounded_box(ax, X_CLF, y_f, 1.5, 0.48, C_OUTPUT, 0.75)
        label(ax, X_CLF, y_f, "6 accents\n(softmax)", 7, C_WHITE)
        arrow(ax, X_FUSION + 1.2, y_f, X_CLF - 0.75, y_f, C_FUSION)

    # ─────────────────────────────────────────────────────────────────────── #
    #                        Column labels                                     #
    # ─────────────────────────────────────────────────────────────────────── #
    col_labels = [
        (X_INPUT,   "Вход"),
        (X_FEAT,    "Извлечение\nпризнаков"),
        (X_ENC,     "Кодирование"),
        (X_PROJ,    "Проекция"),
        (X_IND_CLF, "Classifier\n(per-feature)"),
        (X_FUSION,  "Fusion"),
        (X_CLF,     "Выход"),
    ]
    for (x, lbl) in col_labels:
        ax.text(x, 9.55, lbl, ha="center", va="bottom", fontsize=8.5,
                color="#555555", fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="#CCCCCC",
                          boxstyle="round,pad=0.2", alpha=0.9))

    # ─────────────────────────────────────────────────────────────────────── #
    #                           Legend                                         #
    # ─────────────────────────────────────────────────────────────────────── #
    legend_items = [
        mpatches.Patch(color=C_AUDIO,  label="Audio модальность (HuBERT / MFCC)"),
        mpatches.Patch(color=C_TEXT,   label="Текстовая модальность (TF-IDF)"),
        mpatches.Patch(color=C_FUSION, label="Мультимодальный Fusion"),
        mpatches.Patch(color=C_OUTPUT, label="Классификация (6 акцентов)"),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=8,
              framealpha=0.9, edgecolor="#CCCCCC")

    plt.tight_layout()
    out_dir = _ROOT / "outputs" / "report_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "architecture_overview.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def draw_fusion_detail():
    """Детальная схема CrossAttention Fusion модели."""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.set_title("Cross-Attention Fusion: детальная архитектура",
                 fontsize=13, fontweight="bold", pad=10)

    def box(x, y, w, h, color, text, subtext=None, alpha=0.85):
        p = FancyBboxPatch((x - w/2, y - h/2), w, h,
                           boxstyle="round,pad=0.05",
                           facecolor=color, edgecolor="white",
                           linewidth=1.5, alpha=alpha, zorder=3)
        ax.add_patch(p)
        if subtext:
            ax.text(x, y + 0.12, text, ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold", zorder=5)
            ax.text(x, y - 0.18, subtext, ha="center", va="center",
                    fontsize=7.5, color="white", zorder=5)
        else:
            ax.text(x, y, text, ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold", zorder=5)

    def arr(x0, y0, x1, y1, c="#555", lw=1.5):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=lw,
                                   connectionstyle="arc3,rad=0.0"), zorder=2)

    # Audio branch (left)
    box(1.8, 5.5, 2.2, 0.7, C_AUDIO, "HuBERT Layer 6", "(T × 768)")
    box(1.8, 4.2, 2.2, 0.65, C_AUDIO, "Mean-pool", "→ 768-dim")
    box(1.8, 3.0, 2.0, 0.65, "#2E75B6", "Linear(768→256)", "Audio Projection")
    arr(1.8, 5.15, 1.8, 4.53)
    arr(1.8, 3.88, 1.8, 3.33)

    # Text branch (right)
    box(10.2, 5.5, 2.2, 0.7, C_TEXT, "Транскрипт", "char TF-IDF (512)")
    box(10.2, 4.2, 2.2, 0.65, C_TEXT, "TF-IDF вектор", "→ 512-dim")
    box(10.2, 3.0, 2.0, 0.65, "#C55A11", "Linear(512→256)", "Text Projection")
    arr(10.2, 5.15, 10.2, 4.53)
    arr(10.2, 3.88, 10.2, 3.33)

    # Cross-attention (centre)
    box(6.0, 3.0, 3.2, 0.8, C_FUSION, "MultiheadAttention", "Q=text, K=V=audio (4 heads)")
    # Arrows from projections to attention
    arr(2.8, 3.0, 4.4, 3.0, C_AUDIO, 2.0)   # audio → attn (K/V)
    arr(9.2, 3.0, 7.6, 3.0, C_TEXT, 2.0)    # text → attn (Q)
    ax.text(3.6, 3.2, "K, V", fontsize=8.5, color=C_AUDIO, ha="center")
    ax.text(8.4, 3.2, "Q",    fontsize=8.5, color=C_TEXT,  ha="center")

    # LayerNorm + residual
    box(6.0, 1.9, 2.8, 0.6, "#4472C4", "LayerNorm + Residual")
    arr(6.0, 2.6, 6.0, 2.2)

    # Concat + MLP classifier
    box(6.0, 1.0, 3.0, 0.65, C_OUTPUT, "Concat(attn_out, audio_proj)", "(→ 512-dim)")
    arr(6.0, 1.6, 6.0, 1.33)
    # Direct audio projection also feeds concat
    ax.annotate("", xy=(5.0, 1.0), xytext=(1.8, 2.68),
                arrowprops=dict(arrowstyle="-|>", color=C_AUDIO, lw=1.2,
                                connectionstyle="arc3,rad=0.3"), zorder=2)

    box(6.0, 0.2, 2.4, 0.55, C_OUTPUT, "MLP → Softmax (6 классов)")
    arr(6.0, 0.68, 6.0, 0.48)

    # Labels
    ax.text(1.8, 6.3,  "AUDIO\nмодальность", ha="center", va="center",
            fontsize=9, color=C_AUDIO, fontweight="bold")
    ax.text(10.2, 6.3, "TEXT\nмодальность", ha="center", va="center",
            fontsize=9, color=C_TEXT, fontweight="bold")
    ax.text(6.0, 6.3,  "Cross-Attention\nFusion", ha="center", va="center",
            fontsize=9, color=C_FUSION, fontweight="bold")

    plt.tight_layout()
    out_dir = _ROOT / "outputs" / "report_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fusion_detail.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    draw_full_architecture()
    draw_fusion_detail()
    print("Architecture diagrams saved to outputs/report_figures/")
