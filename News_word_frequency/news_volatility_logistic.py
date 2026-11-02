"""Test whether earlier news words improve prediction of large NVDA moves.

Run from the repository root:
    .venv/bin/python News_word_frequency/news_volatility_logistic.py

One observation is one trading session. The outcome is an adjusted close-to-close
absolute return of at least 1% (up or down). Article counts and word occurrences
come from the 72 elapsed hours ending *before* the previous NASDAQ close. Thus
all predictors precede the return interval. Only sessions with verified news
collection throughout that window are analyzed.

Two regularized logistic regressions are fitted on the earliest 70% of sessions:
market controls plus article volume, and those features plus six prespecified
news-word frequencies. The remaining 30% are held out in chronological order.
Results are exploratory associations, not causal estimates or trading signals.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, brier_score_loss,
    log_loss, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
STOCK = ROOT / "data/processed/NVDA_daily_features_2025-10-01_2026-05-31.csv"
NEWS = ROOT / "news data pull/processed/all_classified_articles.csv"
COVERAGE = ROOT / "data/matched/2025-09-15_2026-05-15/collection_coverage_by_day.csv"
OUTPUT = Path(__file__).resolve().parent / "logistic_results"
WORDS = ("ai", "chip", "earnings", "infrastructure", "demand", "china")
TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
EXPLICIT_ZONE = re.compile(r"(?:Z|[+-]\d{2}:?\d{2})$")
REVISION_FIELDS = ("updated_at_utc", "modified_at_utc", "revised_at_utc", "first_available_at_utc")
BASE_FEATURES = ("prior_abs_return", "prior_volatility_20d", "article_count")
WORD_FEATURES = tuple(f"word_{word}" for word in WORDS)


def timestamp(value):
    """Accept only an explicit date, time, and offset, as in the matcher."""
    value = str(value).strip()
    if not re.search(r"[T ]\d{2}:\d{2}", value) or not EXPLICIT_ZONE.search(value):
        return pd.NaT
    try:
        return pd.Timestamp(value).tz_convert("UTC")
    except (TypeError, ValueError, OverflowError):
        return pd.NaT


def read_inputs():
    for path in (STOCK, NEWS, COVERAGE):
        if not path.is_file():
            raise FileNotFoundError(f"Required input is missing: {path}")
    stock = pd.read_csv(STOCK, dtype={"trading_date": str}).sort_values("trading_date")
    if stock.trading_date.duplicated().any() or not stock.ticker.eq("NVDA").all():
        raise ValueError("Stock dates must be unique and all rows must be NVDA")
    news = pd.read_csv(NEWS, dtype=str, keep_default_na=False)
    news = news.loc[news.relevance_type.isin(("direct", "indirect"))].drop_duplicates().copy()
    news["published"] = news.published_at_utc.map(timestamp)
    rejected_timestamps = int(news.published.isna().sum())
    news = news.loc[news.published.notna()].copy()
    for field in REVISION_FIELDS:
        if field in news:
            news[field + "_parsed"] = news[field].map(lambda v: timestamp(v) if v else pd.NaT)
            # An ambiguous known revision or availability time cannot safely be
            # placed before the session cutoff.
            news = news.loc[news[field].eq("") | news[field + "_parsed"].notna()].copy()
    news = news.sort_values("published")
    coverage = pd.read_csv(COVERAGE, dtype=str, keep_default_na=False)
    if coverage.duplicated(["coverage_date_utc", "ticker"]).any():
        raise ValueError("Coverage audit has duplicate date/ticker entries")
    tickers = set(coverage.ticker.unique())
    good_days = set(coverage.loc[coverage.coverage_state.eq("success")]
                    .groupby("coverage_date_utc").filter(lambda day: set(day.ticker) == tickers)
                    .coverage_date_utc)
    return stock, news, good_days, rejected_timestamps


def build_dataset(stock, news, good_days, threshold, lookback_hours):
    first, last = stock.trading_date.iloc[0], stock.trading_date.iloc[-1]
    schedule = mcal.get_calendar("NASDAQ").schedule(
        start_date=pd.Timestamp(first) - pd.Timedelta(days=10), end_date=last)
    stock_days = pd.DatetimeIndex(pd.to_datetime(stock.trading_date))
    if not stock_days.isin(schedule.index).all():
        raise ValueError("Stock input contains a date outside the NASDAQ schedule")
    if not schedule.index[(schedule.index >= stock_days[0]) &
                          (schedule.index <= stock_days[-1])].equals(stock_days):
        raise ValueError("Stock input is missing a NASDAQ session")

    records = []
    excluded_coverage = 0
    excluded_controls = 0
    for row in stock.itertuples(index=False):
        position = schedule.index.get_loc(pd.Timestamp(row.trading_date))
        cutoff = schedule.iloc[position - 1].market_close
        start = cutoff - pd.Timedelta(hours=lookback_hours)
        days = pd.date_range(start.floor("D"), (cutoff - pd.Timedelta(nanoseconds=1)).floor("D"))
        if not all(str(day.date()) in good_days for day in days):
            excluded_coverage += 1
            continue
        values = (row.adjusted_close_return_1d,
                  row.adjusted_close_return_1d_lag1_session,
                  row.daily_return_volatility_20d_lag1_session)
        if not all(np.isfinite(float(value)) for value in values):
            excluded_controls += 1
            continue
        outcome_return, prior_return, prior_volatility = map(float, values)
        label = int(abs(outcome_return) >= threshold)
        if threshold == 0.01 and label != int(row.price_move_1pct != 0):
            raise ValueError(f"1% label disagrees with stock flag on {row.trading_date}")
        window = news.loc[(news.published >= start) & (news.published < cutoff)]
        for field in REVISION_FIELDS:
            parsed = field + "_parsed"
            if parsed in window:
                window = window.loc[window[field].eq("") | (window[parsed] < cutoff)]
        counts = Counter()
        for article in window.itertuples(index=False):
            counts.update(TOKEN.findall(f"{article.headline} {article.summary}".lower()))
        records.append({
            "trading_date": row.trading_date,
            "previous_close_utc": cutoff.isoformat(),
            "news_window_start_utc": start.isoformat(),
            "adjusted_close_return_1d": outcome_return,
            "large_move": label,
            "prior_abs_return": abs(prior_return),
            "prior_volatility_20d": prior_volatility,
            "article_count": len(window),
            **{f"word_{word}": counts[word] for word in WORDS},
        })
    frame = pd.DataFrame(records).sort_values("trading_date").reset_index(drop=True)
    if len(frame) < 30 or frame.large_move.nunique() < 2:
        raise ValueError("Too few usable sessions or only one outcome class")
    return frame, excluded_coverage, excluded_controls


def metrics(y, probabilities):
    predicted = probabilities >= 0.5
    return {
        "n": len(y),
        "large_moves": int(y.sum()),
        "auc": float(roc_auc_score(y, probabilities)) if len(np.unique(y)) == 2 else None,
        "brier": float(brier_score_loss(y, probabilities)),
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1])),
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
    }


def fit_and_score(frame, features, split):
    x = frame.loc[:, features].copy()
    for field in ("article_count", *WORD_FEATURES):
        if field in x:
            x[field] = np.log1p(x[field])
    y_train = frame.large_move.iloc[:split].to_numpy()
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    model.fit(x.iloc[:split], y_train)
    probabilities = model.predict_proba(x.iloc[split:])[:, 1]
    coefficients = model.named_steps["logisticregression"].coef_[0]
    odds_ratios = {name: float(np.exp(coef)) for name, coef in zip(features, coefficients)}
    return probabilities, odds_ratios


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=0.01,
                        help="Absolute return threshold as a fraction (default: 0.01)")
    parser.add_argument("--lookback-hours", type=int, default=72,
                        help="Hours of news before the prior close (default: 72)")
    args = parser.parse_args()
    if not 0 < args.threshold < 1 or args.lookback_hours < 1:
        parser.error("threshold must be between 0 and 1 and lookback-hours must be positive")

    stock, news, good_days, bad_timestamps = read_inputs()
    frame, no_coverage, no_controls = build_dataset(
        stock, news, good_days, args.threshold, args.lookback_hours)
    split = int(len(frame) * 0.7)
    if frame.large_move.iloc[:split].nunique() < 2 or frame.large_move.iloc[split:].nunique() < 2:
        raise ValueError("Both chronological partitions need both outcome classes")
    baseline, base_odds = fit_and_score(frame, BASE_FEATURES, split)
    full, full_odds = fit_and_score(frame, BASE_FEATURES + WORD_FEATURES, split)
    y_test = frame.large_move.iloc[split:].to_numpy()
    prior = np.full(len(y_test), frame.large_move.iloc[:split].mean())
    prior_metrics = metrics(y_test, prior)
    base_metrics, full_metrics = metrics(y_test, baseline), metrics(y_test, full)

    frame["split"] = np.where(frame.index < split, "train", "test")
    frame["training_rate_probability"] = np.nan
    frame["baseline_probability"] = np.nan
    frame["news_model_probability"] = np.nan
    frame.loc[split:, "training_rate_probability"] = prior
    frame.loc[split:, "baseline_probability"] = baseline
    frame.loc[split:, "news_model_probability"] = full
    report = {
        "outcome": f"abs(adjusted_close_return_1d) >= {args.threshold}",
        "news_window": f"{args.lookback_hours} hours before previous NASDAQ close",
        "words_prespecified": list(WORDS),
        "features": {"baseline": list(BASE_FEATURES), "with_news_words": list(BASE_FEATURES + WORD_FEATURES)},
        "model": "L2 logistic regression; log1p counts; predictors scaled using training data only; C=1",
        "sample": {
            "usable_sessions": len(frame), "train_sessions": split, "test_sessions": len(frame) - split,
            "train_large_moves": int(frame.large_move.iloc[:split].sum()),
            "test_large_moves": int(y_test.sum()),
            "train_date_range": [frame.trading_date.iloc[0], frame.trading_date.iloc[split - 1]],
            "test_date_range": [frame.trading_date.iloc[split], frame.trading_date.iloc[-1]],
            "excluded_for_incomplete_news_collection": no_coverage,
            "excluded_for_missing_market_controls": no_controls,
            "rejected_news_publication_timestamps": bad_timestamps,
        },
        "held_out_metrics": {
            "training_rate_only": prior_metrics,
            "baseline": base_metrics,
            "with_news_words": full_metrics,
        },
        "difference_news_minus_baseline": {
            "auc": None if base_metrics["auc"] is None else full_metrics["auc"] - base_metrics["auc"],
            "brier": full_metrics["brier"] - base_metrics["brier"],
            "log_loss": full_metrics["log_loss"] - base_metrics["log_loss"],
        },
        "training_odds_ratio_per_one_standard_deviation": {
            "baseline": base_odds, "with_news_words": full_odds,
        },
        "limitations": [
            "Small, single chronological test period; results do not establish statistical significance.",
            "Odds ratios are from regularized models and do not have p-values.",
            "News was collected retrospectively; historical article availability and text revisions are not fully verifiable.",
            "Verified collection windows do not guarantee complete publisher coverage.",
        ],
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT / "session_predictions.csv", index=False)
    (OUTPUT / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Usable sessions: {len(frame)} (train {split}, test {len(y_test)})")
    print(f"Test large-move rate: {y_test.mean():.1%}")
    print(f"Held-out Brier, training rate only: {prior_metrics['brier']:.3f}")
    print(f"Held-out AUC: baseline {base_metrics['auc']:.3f}; with words {full_metrics['auc']:.3f}")
    print(f"Held-out Brier: baseline {base_metrics['brier']:.3f}; with words {full_metrics['brier']:.3f}")
    print(f"Held-out log loss: baseline {base_metrics['log_loss']:.3f}; with words {full_metrics['log_loss']:.3f}")
    print(f"Held-out accuracy: baseline {base_metrics['accuracy']:.3f}; with words {full_metrics['accuracy']:.3f}")
    print(f"Wrote {OUTPUT / 'results.json'} and {OUTPUT / 'session_predictions.csv'}")


if __name__ == "__main__":
    main()
