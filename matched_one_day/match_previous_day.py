"""Match each NVDA trading session to news from the prior New York calendar day.

Run from the repository root:
    .venv/bin/python matched_one_day/match_previous_day.py

The output keeps every stock session, including sessions with no matching news.
Only articles already classified as direct or indirect are eligible. Publication
dates are derived from explicit timestamps in America/New_York; timestamps are
never assigned an assumed timezone.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STOCK = ROOT / "data/processed/NVDA_daily_features_2025-10-01_2026-05-31.csv"
NEWS = ROOT / "news data pull/processed/all_classified_articles.csv"
CONFIG = ROOT / "news data pull/config.json"
OUTPUT = Path(__file__).resolve().parent / "output"
TZ = "America/New_York"
DEFAULT_THRESHOLD = 0.02
ELIGIBLE_RELEVANCE = ("direct", "indirect")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_timestamp(value: str) -> tuple[pd.Timestamp | None, str]:
    """Parse a timestamp only when it contains a time and explicit UTC offset."""
    text = str(value).strip()
    if not text:
        return None, "missing_timestamp"
    if not re.search(r"[T ]\d{2}:\d{2}", text):
        return None, "date_only_or_unsupported_timestamp"
    if not re.search(r"(?:Z|[+-]\d{2}:?\d{2})$", text):
        return None, "missing_or_ambiguous_timezone"
    try:
        parsed = pd.Timestamp(text)
        if pd.isna(parsed) or parsed.tzinfo is None:
            return None, "invalid_timestamp"
        return parsed.tz_convert("UTC"), ""
    except (TypeError, ValueError, OverflowError):
        return None, "invalid_timestamp"


def classify_move(value: float, threshold: float = DEFAULT_THRESHOLD) -> int:
    """Return -1/0/+1 for inclusive adjusted-close return thresholds."""
    if not np.isfinite(value):
        raise ValueError("Cannot classify a missing or non-finite stock return")
    if value <= -threshold:
        return -1
    if value >= threshold:
        return 1
    return 0


def prepare_stock(stock: pd.DataFrame, start: str | None, end: str | None,
                  threshold: float) -> pd.DataFrame:
    required = {"trading_date", "ticker", "adjusted_close_return_1d"}
    missing = required.difference(stock.columns)
    if missing:
        raise ValueError(f"Stock input is missing columns: {sorted(missing)}")
    result = stock.copy()
    parsed_dates = pd.to_datetime(result.trading_date, format="%Y-%m-%d", errors="coerce")
    if parsed_dates.isna().any():
        raise ValueError("Stock input contains a missing or invalid trading date")
    if result.trading_date.duplicated().any():
        raise ValueError("Stock input contains duplicate trading dates")
    if not result.ticker.eq("NVDA").all():
        raise ValueError("Stock input contains a non-NVDA ticker")
    if not result.trading_date.is_monotonic_increasing:
        raise ValueError("Stock input must be sorted by trading date")
    if start is not None:
        result = result.loc[result.trading_date.ge(start)].copy()
    if end is not None:
        result = result.loc[result.trading_date.le(end)].copy()
    if result.empty:
        raise ValueError("No stock sessions fall in the requested date range")
    returns = pd.to_numeric(result.adjusted_close_return_1d, errors="coerce")
    if not np.isfinite(returns).all():
        raise ValueError("Stock input contains a missing or non-finite daily return")
    result["adjusted_close_return_1d_pct"] = returns * 100
    result["price_move_2pct"] = returns.map(lambda value: classify_move(value, threshold))
    result["matched_news_date"] = (
        pd.to_datetime(result.trading_date, format="%Y-%m-%d") - pd.Timedelta(days=1)
    ).dt.strftime("%Y-%m-%d")
    return result


def prepare_articles(news: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    required = {"published_at_utc", "relevance_type", "article_id", "headline"}
    missing = required.difference(news.columns)
    if missing:
        raise ValueError(f"News input is missing columns: {sorted(missing)}")
    selected = news.loc[news.relevance_type.isin(ELIGIBLE_RELEVANCE)].copy()
    before_deduplication = len(selected)
    selected = selected.drop_duplicates().copy()
    exact_duplicates_removed = before_deduplication - len(selected)
    articles = []
    review = []
    for row in selected.to_dict("records"):
        published, problem = parse_timestamp(row.get("published_at_utc", ""))
        if problem:
            review.append({**row, "timestamp_review_reason": problem})
            continue
        stable = json.dumps(row, sort_keys=True, separators=(",", ":"))
        row["article_match_key"] = hashlib.sha256(stable.encode()).hexdigest()[:24]
        row["publication_timestamp_utc"] = published.isoformat()
        local = published.tz_convert(TZ)
        row["publication_timestamp_new_york"] = local.isoformat()
        row["news_date_new_york"] = str(local.date())
        articles.append(row)
    article_columns = list(selected.columns) + [
        "article_match_key", "publication_timestamp_utc",
        "publication_timestamp_new_york", "news_date_new_york",
    ]
    review_columns = list(selected.columns) + ["timestamp_review_reason"]
    return (pd.DataFrame(articles, columns=article_columns),
            pd.DataFrame(review, columns=review_columns), exact_duplicates_removed)


def configured_day_coverage(news_date: str, collection: dict) -> tuple[str, str, int]:
    """Check whether the complete New York day fits in the configured UTC range."""
    local_start = pd.Timestamp(news_date, tz=TZ)
    local_end = local_start + pd.DateOffset(days=1)
    start_utc = local_start.tz_convert("UTC")
    end_utc = local_end.tz_convert("UTC")
    collection_start = pd.Timestamp(collection["start_date"], tz="UTC")
    collection_end = pd.Timestamp(collection["end_date"], tz="UTC") + pd.Timedelta(days=1)
    complete = int(start_utc >= collection_start and end_utc <= collection_end)
    return start_utc.isoformat(), end_utc.isoformat(), complete


def build_matches(stock: pd.DataFrame, articles: pd.DataFrame,
                  collection: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    stock = stock.copy()
    bounds = stock.matched_news_date.map(lambda day: configured_day_coverage(day, collection))
    stock["matched_news_day_start_utc"] = bounds.map(lambda values: values[0])
    stock["matched_news_day_end_utc"] = bounds.map(lambda values: values[1])
    stock["within_configured_utc_collection_range"] = bounds.map(lambda values: values[2])

    matches = stock.merge(
        articles, left_on="matched_news_date", right_on="news_date_new_york",
        how="inner", validate="one_to_many", suffixes=("", "_article"),
    )
    if not matches.empty:
        matches = matches.sort_values(
            ["trading_date", "publication_timestamp_utc", "article_match_key"]
        ).reset_index(drop=True)

    counts = matches.groupby("trading_date").size() if not matches.empty else pd.Series(dtype=int)
    direct = (matches.loc[matches.relevance_type.eq("direct")].groupby("trading_date").size()
              if not matches.empty else pd.Series(dtype=int))
    indirect = (matches.loc[matches.relevance_type.eq("indirect")].groupby("trading_date").size()
                if not matches.empty else pd.Series(dtype=int))
    summary = stock.copy()
    summary["matched_article_count"] = summary.trading_date.map(counts).fillna(0).astype(int)
    summary["matched_direct_article_count"] = summary.trading_date.map(direct).fillna(0).astype(int)
    summary["matched_indirect_article_count"] = summary.trading_date.map(indirect).fillna(0).astype(int)
    return summary, matches


def validate_results(stock: pd.DataFrame, summary: pd.DataFrame,
                     matches: pd.DataFrame, threshold: float) -> dict:
    assert summary.trading_date.is_unique
    assert summary.trading_date.tolist() == stock.trading_date.tolist()
    expected_labels = pd.to_numeric(summary.adjusted_close_return_1d).map(
        lambda value: classify_move(value, threshold))
    assert summary.price_move_2pct.astype(int).equals(expected_labels.astype(int))
    if not matches.empty:
        expected_dates = (
            pd.to_datetime(matches.trading_date, format="%Y-%m-%d") - pd.Timedelta(days=1)
        ).dt.strftime("%Y-%m-%d")
        assert matches.matched_news_date.equals(expected_dates)
        assert matches.news_date_new_york.equals(expected_dates)
        assert not matches.duplicated(["trading_date", "article_match_key"]).any()
        assert matches.price_move_2pct.astype(int).equals(
            pd.to_numeric(matches.adjusted_close_return_1d).map(
                lambda value: classify_move(value, threshold)).astype(int))
    counts = matches.groupby("trading_date").size() if not matches.empty else pd.Series(dtype=int)
    observed = summary.trading_date.map(counts).fillna(0).astype(int)
    assert summary.matched_article_count.equals(observed)
    assert (summary.matched_article_count ==
            summary.matched_direct_article_count + summary.matched_indirect_article_count).all()
    return {
        "all_stock_sessions_present": True,
        "one_summary_row_per_stock_session": True,
        "matches_use_exact_previous_new_york_calendar_date": True,
        "price_move_2pct_recalculated_and_verified": True,
        "unique_stock_session_article_pairs": True,
        "summary_article_counts_consistent": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", help="First stock trading date, inclusive")
    parser.add_argument("--end", help="Last stock trading date, inclusive")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="Absolute daily return threshold as a fraction (default: 0.02)")
    parser.add_argument("--output", type=Path, default=OUTPUT,
                        help=f"Output directory (default: {OUTPUT.relative_to(ROOT)})")
    args = parser.parse_args()
    if args.start and args.end and pd.Timestamp(args.start) > pd.Timestamp(args.end):
        parser.error("--start must not be after --end")
    if not 0 < args.threshold < 1:
        parser.error("--threshold must be between 0 and 1")

    source_paths = [STOCK, NEWS, CONFIG]
    hashes_before = {str(path.relative_to(ROOT)): digest(path) for path in source_paths}
    raw_stock, raw_news = read_csv(STOCK), read_csv(NEWS)
    stock = prepare_stock(raw_stock, args.start, args.end, args.threshold)
    articles, timestamp_review, duplicates_removed = prepare_articles(raw_news)
    collection = json.loads(CONFIG.read_text(encoding="utf-8"))["collection"]
    summary, matches = build_matches(stock, articles, collection)
    checks = validate_results(stock, summary, matches, args.threshold)
    hashes_after = {str(path.relative_to(ROOT)): digest(path) for path in source_paths}
    assert hashes_before == hashes_after, "Source files changed during matching"

    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "daily_stock_news_summary.csv", index=False)
    matches.to_csv(output / "stock_news_matches.csv", index=False)
    timestamp_review.to_csv(output / "timestamp_review.csv", index=False)

    label_counts = {
        str(label): int(count)
        for label, count in summary.price_move_2pct.value_counts().sort_index().items()
    }
    example = summary.loc[summary.trading_date.eq("2025-12-02")]
    december_example = example[["trading_date", "matched_news_date", "price_move_2pct",
                                "matched_article_count"]].to_dict("records")
    metadata = {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "stock_path": str(STOCK.relative_to(ROOT)),
        "news_path": str(NEWS.relative_to(ROOT)),
        "source_sha256": hashes_before,
        "source_files_unchanged": True,
        "package_versions": {"pandas": version("pandas"), "numpy": version("numpy")},
        "matching_rule": (
            "For stock trading date t, include direct/indirect articles whose explicit "
            "publication timestamp has New York calendar date t minus one calendar day."
        ),
        "weekend_rule": "A Monday stock session matches Sunday only, not the entire weekend.",
        "feature": {
            "name": "price_move_2pct",
            "source": "adjusted_close_return_1d",
            "encoding": {
                "-1": f"return <= -{args.threshold:.2%}",
                "0": f"-{args.threshold:.2%} < return < {args.threshold:.2%}",
                "1": f"return >= {args.threshold:.2%}",
            },
        },
        "stock_session_count": len(summary),
        "stock_date_range": [summary.trading_date.min(), summary.trading_date.max()],
        "eligible_unique_article_count": len(articles),
        "event_article_pair_count": len(matches),
        "stock_sessions_with_matches": int(summary.matched_article_count.gt(0).sum()),
        "stock_sessions_without_matches": int(summary.matched_article_count.eq(0).sum()),
        "sessions_outside_complete_configured_utc_collection_range": int(
            summary.within_configured_utc_collection_range.eq(0).sum()),
        "price_move_2pct_counts": label_counts,
        "exact_duplicate_news_rows_removed": duplicates_removed,
        "timestamp_review_rows": len(timestamp_review),
        "december_2_example": december_example,
        "validation": checks,
        "limitations": [
            "A previous calendar day is not a rolling 24-hour window and is not the previous trading session.",
            "Only existing direct/indirect relevance classifications are included.",
            "Configured collection range coverage does not prove complete publisher coverage.",
            "News was retrieved retrospectively; publication timestamps do not guarantee historical article versions.",
        ],
    }
    (output / "validation_report.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_directory": str(output.relative_to(ROOT)),
        "stock_session_count": len(summary),
        "event_article_pair_count": len(matches),
        "stock_sessions_with_matches": metadata["stock_sessions_with_matches"],
        "price_move_2pct_counts": label_counts,
        "december_2_example": december_example,
        "validation": checks,
    }, indent=2))


if __name__ == "__main__":
    main()
