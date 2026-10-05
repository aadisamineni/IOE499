"""Visualize the held-out logistic regression results.

Run from the repository root:
    .venv/bin/python News_word_frequency/plot_logistic_results.py

Reads logistic_results/session_predictions.csv and results.json, then writes a
PNG and SVG to the same directory. Only the held-out test sessions are plotted.
"""

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve


HERE = Path(__file__).resolve().parent / "logistic_results"
PREDICTIONS = HERE / "session_predictions.csv"
RESULTS = HERE / "results.json"


def main():
    df = pd.read_csv(PREDICTIONS, parse_dates=["trading_date"])
    report = json.loads(RESULTS.read_text(encoding="utf-8"))
    test = df.loc[df.split.eq("test")].sort_values("trading_date").copy()
    if test.empty or test[["large_move", "baseline_probability", "news_model_probability"]].isna().any().any():
        raise ValueError("Held-out predictions are missing")
    if test.large_move.nunique() != 2:
        raise ValueError("ROC curves require both outcome classes")

    base = report["held_out_metrics"]["baseline"]
    news = report["held_out_metrics"]["with_news_words"]
    if base["n"] != len(test) or news["n"] != len(test):
        raise ValueError("The results report does not match the predictions")

    blue = "#52677d"
    green = "#3c8d45"
    ink = "#253443"
    muted = "#637487"
    fig = plt.figure(figsize=(12.4, 8.2), facecolor="white")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], width_ratios=[1, 1.05],
                           hspace=0.76, wspace=0.3, left=0.09, right=0.95,
                           top=0.85, bottom=0.17)
    ax_time = fig.add_subplot(grid[0, :])
    ax_roc = fig.add_subplot(grid[1, 0])
    ax_scores = fig.add_subplot(grid[1, 1])

    fig.suptitle("Earlier news and large NVDA price moves", x=0.09, y=0.97,
                 ha="left", fontsize=18, fontweight="semibold", color=ink)
    date_range = (test.trading_date.min().strftime("%b %d") + "–" +
                  test.trading_date.max().strftime("%b %d, %Y"))
    fig.text(0.09, 0.925,
             f"{len(test)} held-out trading sessions · {date_range} · "
             f"{int(test.large_move.sum())} moves of at least 1%",
             fontsize=10, color=muted)

    ax_time.plot(test.trading_date, test.baseline_probability, color=blue,
                 linewidth=1.8, marker="o", markersize=3.5, label="Market + article volume")
    ax_time.plot(test.trading_date, test.news_model_probability, color=green,
                 linewidth=1.8, marker="o", markersize=3.5, label="Plus news words")
    large = test.large_move.eq(1)
    ax_time.scatter(test.loc[large, "trading_date"], [1.04] * large.sum(),
                    s=32, color=ink, marker="|", linewidths=2.2,
                    label="Observed ≥1% move", zorder=4)
    ax_time.scatter(test.loc[~large, "trading_date"], [-0.04] * (~large).sum(),
                    s=35, color=ink, marker="o", facecolors="none", linewidths=1.3,
                    label="Observed <1% move", zorder=4)
    ax_time.axhline(0.5, color="#a9b3bc", linewidth=1, linestyle="--")
    ax_time.set_ylim(-0.1, 1.1)
    ax_time.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax_time.set_ylabel("Predicted probability")
    ax_time.set_title("Predictions by trading day", loc="left", fontsize=12, color=ink)
    ax_time.xaxis.set_major_locator(mdates.MonthLocator())
    ax_time.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax_time.grid(axis="y", color="#e5e9ed", linewidth=0.8)
    ax_time.legend(loc="upper center", frameon=False, ncol=4, fontsize=9,
                   bbox_to_anchor=(0.5, -0.19))

    for column, label, color, value in (
        ("baseline_probability", "Market + article volume", blue, base["auc"]),
        ("news_model_probability", "Plus news words", green, news["auc"]),
    ):
        false_positive, true_positive, _ = roc_curve(test.large_move, test[column])
        ax_roc.plot(false_positive, true_positive, color=color, linewidth=2.4,
                    label=f"{label}  ·  AUC {value:.3f}")
    ax_roc.plot([0, 1], [0, 1], color="#a9b3bc", linewidth=1, linestyle="--",
                label="Chance  ·  AUC 0.500")
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1.02)
    ax_roc.set_xlabel("False-positive rate")
    ax_roc.set_ylabel("True-positive rate")
    ax_roc.set_title("ROC curve", loc="left", fontsize=12, color=ink)
    ax_roc.grid(color="#e5e9ed", linewidth=0.8)
    ax_roc.text(0.7, 0.67, "Chance", fontsize=9, color=muted)

    ax_scores.axis("off")
    ax_scores.set_title("Held-out scores", loc="left", fontsize=12, color=ink)
    ax_scores.text(0.02, 0.86, "Metric", color=muted, transform=ax_scores.transAxes)
    ax_scores.text(0.61, 0.86, "Baseline", color=blue, ha="right",
                   transform=ax_scores.transAxes)
    ax_scores.text(0.98, 0.86, "+ news words", color=green, ha="right",
                   transform=ax_scores.transAxes)
    for y, label, key in ((0.66, "AUC  ·  higher is better", "auc"),
                          (0.48, "Brier  ·  lower is better", "brier"),
                          (0.30, "Accuracy at 0.5", "accuracy")):
        ax_scores.text(0.02, y, label, color=ink, transform=ax_scores.transAxes)
        ax_scores.text(0.61, y, f"{base[key]:.3f}", color=ink, ha="right",
                       transform=ax_scores.transAxes)
        ax_scores.text(0.98, y, f"{news[key]:.3f}", color=ink, ha="right",
                       transform=ax_scores.transAxes)
    ax_scores.axhline(0.78, color="#e5e9ed", linewidth=1)

    fig.text(0.09, 0.065,
             "A large move means an absolute adjusted close-to-close return ≥1%. "
             "This small test period is exploratory.",
             fontsize=9, color=muted)

    for extension in ("png", "svg"):
        path = HERE / f"held_out_model_comparison.{extension}"
        fig.savefig(path, dpi=200, facecolor="white")
        print(f"Wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
