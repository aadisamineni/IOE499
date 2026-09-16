"""Local Matplotlib plots for news word frequency and next-session NVDA returns.

Run from the project root after installing:
    python3 -m pip install -r News_word_frequency/requirements.txt
    python3 News_word_frequency/plot_word_frequency_matplotlib.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
INPUT = HERE / "lagged_word_stock_dataset.csv"
OUTPUT = HERE / "matplotlib_plots"
WORDS = ["ai", "market", "nvda", "chip", "nasdaq", "data"]


def main():
    OUTPUT.mkdir(exist_ok=True)
    df = pd.read_csv(INPUT)
    df["news_date_new_york"] = pd.to_datetime(df["news_date_new_york"])
    df["next_day_return_pct"] = 100 * pd.to_numeric(df["next_day_close_to_close_return"])
    available = [word for word in WORDS if word in set(df["word"])]

    # One layered chart: frequency and future return share the same date axis.
    fig, ax_word = plt.subplots(figsize=(16, 8))
    ax_return = ax_word.twinx()
    for word in available:
        part = df[df["word"] == word].sort_values("news_date_new_york")
        ax_word.plot(part["news_date_new_york"], part["word_occurrences"], linewidth=1.5, label=f"{word} frequency")
    daily_return = (df[["news_date_new_york", "next_day_return_pct"]]
                    .drop_duplicates("news_date_new_york")
                    .sort_values("news_date_new_york"))
    ax_return.plot(daily_return["news_date_new_york"], daily_return["next_day_return_pct"], color="black", linewidth=1.2, alpha=.8, label="Next-session NVDA return (%)")
    ax_return.axhline(0, color="black", linewidth=.7, alpha=.4)
    ax_word.set_xlabel("News publication date (New York)")
    ax_word.set_ylabel("Word occurrences on news date")
    ax_return.set_ylabel("Next-session NVDA close-to-close return (%)")
    ax_word.set_title("News word frequency and next-session NVDA return")
    handles1, labels1 = ax_word.get_legend_handles_labels()
    handles2, labels2 = ax_return.get_legend_handles_labels()
    ax_word.legend(handles1 + handles2, labels1 + labels2, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(OUTPUT / "layered_frequency_and_return.png", dpi=180)
    plt.close(fig)

    # Correlation view: one scatter panel for each word.
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=True)
    for ax, word in zip(axes.flat, available):
        part = df[df["word"] == word]
        x = part["word_occurrences"]
        y = part["next_day_return_pct"]
        ax.scatter(x, y, s=14, alpha=.55, color="#76b900")
        if len(part) > 1 and x.nunique() > 1:
            slope, intercept = __import__("numpy").polyfit(x, y, 1)
            ax.plot(x, slope * x + intercept, color="#c0392b", linewidth=1.2)
            corr = x.corr(y)
            ax.text(.04, .94, f"r = {corr:.2f}", transform=ax.transAxes, va="top")
        ax.set_title(word)
        ax.set_xlabel("Word occurrences")
        ax.grid(alpha=.2)
    axes[0, 0].set_ylabel("Next-session return (%)")
    axes[1, 0].set_ylabel("Next-session return (%)")
    fig.suptitle("Word frequency versus next-session NVDA return", fontsize=16)
    fig.tight_layout()
    fig.savefig(OUTPUT / "word_frequency_return_scatter.png", dpi=180)
    plt.close(fig)
    print(f"Saved plots to {OUTPUT}")


if __name__ == "__main__":
    main()
