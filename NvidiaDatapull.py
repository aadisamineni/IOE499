"""Download and validate NVDA research data.

Run the existing daily feature pipeline:
    .venv/bin/python NvidiaDatapull.py

Download regular-session hourly bars without replacing daily files:
    .venv/bin/python NvidiaDatapull.py --hourly
"""

import argparse
from datetime import date, datetime, timedelta, timezone
from importlib.metadata import version
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf

ROOT = Path(__file__).resolve().parent
START, END = date(2025, 10, 1), date(2026, 5, 31)
WARM_START = START - timedelta(days=90)
TICKER = "NVDA"
STEM = f"{TICKER}_daily_{START}_{END}"
RENAME = {
    "Open": "open", "High": "high", "Low": "low", "Close": "close",
    "Adj Close": "adjusted_close", "Volume": "volume",
    "Dividends": "dividends", "Stock Splits": "stock_splits",
}
SETTINGS = dict(interval="1d", auto_adjust=False, back_adjust=False,
                actions=True, repair=False, keepna=True, rounding=False,
                prepost=False, timeout=30)
HOURLY_SETTINGS = dict(SETTINGS, interval="1h")
FEATURES = ["adjusted_close_return_1d", "adjusted_open_to_close_return",
            "adjusted_close_return_5d", "daily_return_volatility_20d",
            "average_volume_20d", "price_move_5pct", "price_move_1pct"]
MOVE_DEFINITION = (
    "Adjusted close versus previous exchange session: +1 if at least 5% higher, "
    "-1 if at least 5% lower, otherwise 0; missing previous close remains missing. "
    "Same-day classification, available after close; lag1_session uses prior session."
)
MOVE_1PCT_DEFINITION = MOVE_DEFINITION.replace("5%", "1%")
CONVENTIONS = {
    "ohlc": "Yahoo provider OHLC, generally split-adjusted, not dividend-adjusted; "
            "auto_adjust=False does not restore original pre-split traded prices.",
    "adjusted_close": "Yahoo adjusted close reflects splits and dividend distributions; "
                      "historical values may be revised after retrieval.",
    "adjusted_open": "Computed internally as open * adjusted_close / close; "
                     "intraday return = adjusted_close / adjusted_open - 1.",
    "actions": "Provider dividends (USD/share, ex-date) and stock split ratios "
               "(new shares / old shares); provider zero denotes no action.",
    "volume": "Provider-reported daily share volume, preserved without transformation.",
}
TIMING = (
    "Session dates use America/New_York. Same-day close, high, low, volume and all "
    "unlagged derived features are available only after session end. Each _lag1_session "
    "feature on date t uses the preceding exchange session's feature, including warm-up. "
    "No future targets or news joins. This is a retrospectively adjusted snapshot, "
    "not a point-in-time archive of historical provider revisions."
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def sessions(start, end):
    return mcal.get_calendar("NASDAQ").schedule(
        start_date=start, end_date=end).index


def download(start, end_exclusive, settings=SETTINGS):
    frame = yf.Ticker(TICKER).history(
        start=str(start), end=str(end_exclusive), **settings)
    if frame.empty:
        raise RuntimeError(f"Yahoo returned no rows for {start} to {end_exclusive} exclusive")
    missing = set(RENAME) - set(frame.columns)
    if missing:
        raise RuntimeError(f"Provider omitted required columns: {sorted(missing)}")
    if frame.index.tz is None:
        raise RuntimeError("Provider returned dates without an exchange timezone")
    frame = frame[list(RENAME)].rename(columns=RENAME).copy()
    frame.index = frame.index.tz_convert("America/New_York")
    if settings["interval"] == "1d":
        frame.index = frame.index.tz_localize(None).normalize()
        frame.index.name = "trading_date"
    else:
        frame.index.name = "trading_timestamp"
    frame.insert(0, "ticker", TICKER)
    return frame


def date_list(index):
    return index.strftime("%Y-%m-%d").tolist()


def validate(frame, start, end):
    expected = sessions(start, end)
    numeric = frame[list(RENAME.values())]
    prices = frame[["open", "high", "low", "close", "adjusted_close"]]
    checks = {
        "sorted_dates": bool(frame.index.is_monotonic_increasing),
        "unique_dates": bool(frame.index.is_unique),
        "nonempty": bool(len(frame)),
        "no_missing_source_values": bool(frame.notna().all().all()),
        "finite_numeric_values": bool(np.isfinite(numeric.to_numpy()).all()),
        "positive_prices": bool((prices > 0).all().all()),
        "nonnegative_volume": bool((frame.volume >= 0).all()),
        "valid_ohlc_bounds": bool(((frame.high >= frame.open) &
            (frame.high >= frame.close) & (frame.low <= frame.open) &
            (frame.low <= frame.close) & (frame.high >= frame.low)).all()),
    }
    missing = expected.difference(frame.index)
    unexpected = frame.index.difference(expected)
    checks["all_expected_sessions_present"] = len(missing) == 0
    checks["no_unexpected_sessions"] = len(unexpected) == 0
    actions = frame.loc[(frame.dividends.ne(0) | frame.stock_splits.ne(0)),
                        ["dividends", "stock_splits"]].reset_index()
    # JSON conversion represents any missing values as null rather than inventing values.
    actions["trading_date"] = actions.trading_date.dt.strftime("%Y-%m-%d")
    return {
        "passed": all(checks.values()), "checks": checks,
        "expected_session_count": len(expected), "row_count": len(frame),
        "first_trading_date": str(frame.index.min().date()) if len(frame) else None,
        "last_trading_date": str(frame.index.max().date()) if len(frame) else None,
        "missing_sessions": date_list(missing),
        "unexpected_sessions": date_list(unexpected),
        "missing_values": {k: int(v) for k, v in frame.isna().sum().items()},
        "corporate_actions_for_review": json.loads(actions.to_json(orient="records")),
    }


def validate_hourly(frame, start, end):
    """Validate regular-session 1-hour bars, including early-close counts."""
    schedule = mcal.get_calendar("NASDAQ").schedule(start_date=start, end_date=end)
    numeric = frame[list(RENAME.values())]
    prices = frame[["open", "high", "low", "close", "adjusted_close"]]
    local_index = frame.index.tz_convert("America/New_York")
    utc_index = local_index.tz_convert("UTC")
    local_dates = local_index.tz_localize(None).normalize()
    observed_sessions = pd.DatetimeIndex(local_dates.unique()).sort_values()
    missing_sessions = schedule.index.difference(observed_sessions)
    unexpected_sessions = observed_sessions.difference(schedule.index)

    within_regular_session = []
    aligned_to_hourly_grid = []
    for stamp_utc, session_date in zip(utc_index, local_dates):
        if session_date not in schedule.index:
            within_regular_session.append(False)
            aligned_to_hourly_grid.append(False)
            continue
        market_open = schedule.loc[session_date, "market_open"]
        market_close = schedule.loc[session_date, "market_close"]
        within_regular_session.append(bool(market_open <= stamp_utc < market_close))
        offset_seconds = (stamp_utc - market_open).total_seconds()
        aligned_to_hourly_grid.append(bool(offset_seconds >= 0 and offset_seconds % 3600 == 0))

    actual_counts = pd.Series(local_dates).value_counts().sort_index()
    expected_counts = {
        session_date: int(np.ceil(
            (row.market_close - row.market_open).total_seconds() / 3600))
        for session_date, row in schedule.iterrows()
    }
    bar_count_review = []
    for session_date, expected_count in expected_counts.items():
        actual_count = int(actual_counts.get(session_date, 0))
        if actual_count != expected_count:
            bar_count_review.append({
                "trading_date": str(session_date.date()),
                "expected_bar_count": expected_count,
                "actual_bar_count": actual_count,
            })

    missing_price_bars = []
    for timestamp, row in frame.loc[prices.isna().any(axis=1)].iterrows():
        missing_price_bars.append({
            "trading_timestamp": timestamp.isoformat(),
            "missing_fields": [column for column in prices.columns if pd.isna(row[column])],
            "volume": int(row.volume),
        })

    checks = {
        "sorted_timestamps": bool(frame.index.is_monotonic_increasing),
        "unique_timestamps": bool(frame.index.is_unique),
        "timezone_aware_timestamps": bool(frame.index.tz is not None),
        "nonempty": bool(len(frame)),
        "no_missing_source_values": bool(frame.notna().all().all()),
        "finite_numeric_values": bool(np.isfinite(numeric.to_numpy()).all()),
        "positive_prices": bool((prices > 0).all().all()),
        "nonnegative_volume": bool((frame.volume >= 0).all()),
        "valid_ohlc_bounds": bool(((frame.high >= frame.open) &
            (frame.high >= frame.close) & (frame.low <= frame.open) &
            (frame.low <= frame.close) & (frame.high >= frame.low)).all()),
        "all_expected_sessions_present": len(missing_sessions) == 0,
        "no_unexpected_sessions": len(unexpected_sessions) == 0,
        "all_bars_within_regular_session": bool(all(within_regular_session)),
        "all_bars_aligned_to_market_open_hour_grid": bool(all(aligned_to_hourly_grid)),
        "expected_bar_count_each_session": len(bar_count_review) == 0,
    }
    structural_check_names = [
        "sorted_timestamps", "unique_timestamps", "timezone_aware_timestamps",
        "nonempty", "nonnegative_volume", "all_expected_sessions_present",
        "no_unexpected_sessions", "all_bars_within_regular_session",
        "all_bars_aligned_to_market_open_hour_grid", "expected_bar_count_each_session",
    ]
    actions = frame.loc[(frame.dividends.ne(0) | frame.stock_splits.ne(0)),
                        ["dividends", "stock_splits"]].reset_index()
    actions["trading_timestamp"] = actions.trading_timestamp.map(
        lambda value: value.isoformat())
    return {
        "passed": all(checks.values()),
        "structural_validation_passed": all(checks[name] for name in structural_check_names),
        "complete_price_data": len(missing_price_bars) == 0,
        "checks": checks,
        "expected_session_count": len(schedule),
        "observed_session_count": len(observed_sessions),
        "row_count": len(frame),
        "first_timestamp": frame.index.min().isoformat() if len(frame) else None,
        "last_timestamp": frame.index.max().isoformat() if len(frame) else None,
        "missing_sessions": date_list(missing_sessions),
        "unexpected_sessions": date_list(unexpected_sessions),
        "bar_count_review": bar_count_review,
        "missing_price_bar_count": len(missing_price_bars),
        "usable_price_bar_count": len(frame) - len(missing_price_bars),
        "missing_price_bar_review": missing_price_bars,
        "missing_values": {key: int(value) for key, value in frame.isna().sum().items()},
        "corporate_actions_for_review": json.loads(
            actions.to_json(orient="records", date_format="iso")),
    }


def derive_features(frame):
    result = frame.copy()
    adj = frame.adjusted_close
    adjusted_open = frame.open * adj / frame.close
    result[FEATURES[0]] = adj.pct_change(fill_method=None)
    result[FEATURES[1]] = adj / adjusted_open - 1
    result[FEATURES[2]] = adj.pct_change(periods=5, fill_method=None)
    result[FEATURES[3]] = result[FEATURES[0]].rolling(20, min_periods=20).std(ddof=1)
    result[FEATURES[4]] = frame.volume.rolling(20, min_periods=20).mean()
    previous = adj.shift(1)
    for percent in (5, 1):
        threshold = percent / 100
        move = pd.Series(0.0, index=frame.index)
        move.loc[adj >= previous * (1 + threshold)] = 1
        move.loc[adj <= previous * (1 - threshold)] = -1
        result[f"price_move_{percent}pct"] = move.where(adj.notna() & previous.notna())
    for feature in FEATURES:
        result[f"{feature}_lag1_session"] = result[feature].shift(1)
    return result


def hourly_main():
    """Download a separate raw hourly dataset and validation metadata."""
    raw_dir = ROOT / "data/raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{TICKER}_hourly_{START}_{END}"
    paths = {
        "raw": raw_dir / f"{stem}.csv",
        "metadata": raw_dir / f"{stem}_metadata.json",
    }
    failure_path = raw_dir / f"{stem}_failure.json"
    metadata = {
        "status": "started",
        "source": "Yahoo Finance via yfinance.Ticker.history",
        "retrieval_started_utc": utc_now(),
        "ticker": TICKER,
        "currency": "USD",
        "interval": "1h",
        "timestamp_column": "trading_timestamp",
        "timestamp_timezone": "America/New_York with explicit UTC offset",
        "session_scope": "regular NASDAQ session only; prepost=False",
        "bar_timestamp_convention": "Provider timestamp marks the beginning of each hourly bar.",
        "package_versions": {package: version(package) for package in
            ["yfinance", "pandas", "numpy", "pandas_market_calendars"]},
        "python_version": sys.version,
        "requested_dates": {
            "start_inclusive": str(START),
            "end_inclusive": str(END),
            "end_exclusive_sent_to_provider": str(END + timedelta(days=1)),
        },
        "download_settings": HOURLY_SETTINGS,
        "exception_settings": {"yf.config.debug.hide_exceptions": False},
        "adjustment_conventions": {
            **CONVENTIONS,
            "volume": "Provider-reported share volume for each hourly bar.",
        },
        "calendar": "pandas_market_calendars NASDAQ; includes early-close sessions",
        "paths": {key: str(value.relative_to(ROOT)) for key, value in paths.items()},
        "documentation": [
            "https://ranaroussi.github.io/yfinance/reference/yfinance.price_history.html",
            "https://help.yahoo.com/kb/SLN28256.html",
            "https://pandas-market-calendars.readthedocs.io/en/latest/usage.html",
        ],
    }
    try:
        yf.set_tz_cache_location(str(ROOT / ".cache/yfinance"))
        yf.config.debug.hide_exceptions = False
        hourly = download(START, END + timedelta(days=1), settings=HOURLY_SETTINGS)
        metadata["retrieval_time_utc"] = utc_now()
        # The ISO-like format keeps the New York UTC offset on every row. This
        # avoids ambiguous local times across daylight-saving transitions.
        hourly.to_csv(paths["raw"], date_format="%Y-%m-%dT%H:%M:%S%z")
        report = validate_hourly(hourly, START, END)
        metadata["validation"] = report
        metadata["actual_timestamps"] = {
            "first": report["first_timestamp"],
            "last": report["last_timestamp"],
        }
        metadata["row_count"] = len(hourly)
        if not report["structural_validation_passed"]:
            raise RuntimeError(
                "Hourly source structure validation failed; raw download was retained for review")
        metadata["status"] = (
            "validated" if report["complete_price_data"] else
            "validated_with_provider_gaps")
        metadata["provider_gap_policy"] = (
            "Missing provider bars are retained with null prices and zero reported volume; "
            "no price is imputed and no placeholder row is silently removed.")
        write_json(paths["metadata"], metadata)
        failure_path.unlink(missing_ok=True)
        print(json.dumps({
            "paths": metadata["paths"],
            "actual_timestamps": metadata["actual_timestamps"],
            "row_count": len(hourly),
            "usable_price_bar_count": report["usable_price_bar_count"],
            "missing_price_bar_count": report["missing_price_bar_count"],
            "observed_session_count": report["observed_session_count"],
            "status": metadata["status"],
            "validation": report["checks"],
        }, indent=2))
        print("\nFirst five hourly rows:\n" + hourly.head().to_string())
        return 0
    except Exception as exc:
        metadata.update(
            status="failed", failure_time_utc=utc_now(),
            error=f"{type(exc).__name__}: {exc}")
        write_json(failure_path, metadata)
        print(f"FAILED: {metadata['error']}\nFailure details: {failure_path}", file=sys.stderr)
        return 1


def main():
    raw_dir, processed_dir = ROOT / "data/raw", ROOT / "data/processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "raw": raw_dir / f"{STEM}.csv",
        "warmup": raw_dir / f"NVDA_daily_warmup_{WARM_START}_{START - timedelta(days=1)}.csv",
        "processed": processed_dir / f"NVDA_daily_features_{START}_{END}.csv",
        "metadata": raw_dir / f"{STEM}_metadata.json",
    }
    metadata = {
        "status": "started", "source": "Yahoo Finance via yfinance.Ticker.history",
        "retrieval_started_utc": utc_now(), "ticker": TICKER, "currency": "USD",
        "package_versions": {p: version(p) for p in
            ["yfinance", "pandas", "numpy", "pandas_market_calendars"]},
        "python_version": sys.version,
        "requested_dates": {"start_inclusive": str(START), "end_inclusive": str(END)},
        "download_settings": SETTINGS,
        "exception_settings": {"yf.config.debug.hide_exceptions": False},
        "requests": {
            "warmup": {"start_inclusive": str(WARM_START), "end_exclusive": str(START)},
            "main": {"start_inclusive": str(START), "end_exclusive": str(END + timedelta(days=1))}},
        "warmup_calendar_days": (START - WARM_START).days,
        "adjustment_conventions": CONVENTIONS, "feature_timing": TIMING,
        "price_move_5pct_definition": MOVE_DEFINITION,
        "price_move_1pct_definition": MOVE_1PCT_DEFINITION,
        "calendar": "pandas_market_calendars NASDAQ; includes early-close sessions",
        "unusual_return_review_threshold": "abs(adjusted close daily return) >= 0.10",
        "documentation": [
            "https://ranaroussi.github.io/yfinance/reference/yfinance.price_history.html",
            "https://help.yahoo.com/kb/SLN28256.html",
            "https://github.com/ranaroussi/yfinance/discussions/1682",
            "https://pandas-market-calendars.readthedocs.io/en/latest/usage.html"],
        "paths": {k: str(v.relative_to(ROOT)) for k, v in paths.items()},
    }
    try:
        # Keep yfinance cookies/timezone caches local to this project too.
        yf.set_tz_cache_location(str(ROOT / ".cache/yfinance"))
        yf.config.debug.hide_exceptions = False
        warm = download(WARM_START, START)
        metadata["warmup_retrieved_utc"] = utc_now()
        warm.to_csv(paths["warmup"], date_format="%Y-%m-%d")
        raw = download(START, END + timedelta(days=1))
        metadata["retrieval_time_utc"] = utc_now()
        raw.to_csv(paths["raw"], date_format="%Y-%m-%d")
        reports = {
            "warmup": validate(warm, WARM_START, START - timedelta(days=1)),
            "requested": validate(raw, START, END),
        }
        metadata["validation"] = reports
        metadata["actual_dates"] = {
            "first": reports["requested"]["first_trading_date"],
            "last": reports["requested"]["last_trading_date"]}
        metadata["row_count"] = len(raw)
        if not all(r["passed"] for r in reports.values()):
            raise RuntimeError("Source validation failed; inspect validation in failure JSON. "
                               "Raw downloads retained; processed output not refreshed.")
        combined = pd.concat([warm, raw])
        featured = derive_features(combined)
        processed = featured.loc[str(START):str(END)].copy()
        derived = processed.drop(columns=raw.columns)
        metadata["processed_missing_values"] = {
            k: int(v) for k, v in processed.isna().sum().items()}
        if not np.isfinite(derived.to_numpy()).all():
            raise RuntimeError("Derived features contain missing/nonfinite values despite warm-up")
        for column in ("price_move_5pct", "price_move_5pct_lag1_session",
                       "price_move_1pct", "price_move_1pct_lag1_session"):
            processed[column] = processed[column].astype("int64")
        unusual = featured.loc[featured[FEATURES[0]].abs() >= 0.10, [FEATURES[0]]].reset_index()
        unusual["trading_date"] = unusual.trading_date.dt.strftime("%Y-%m-%d")
        metadata["unusual_returns_for_review_including_warmup"] = json.loads(
            unusual.to_json(orient="records"))
        processed.to_csv(paths["processed"], date_format="%Y-%m-%d")
        metadata["processed_generated_utc"] = utc_now()
        metadata["status"] = "validated"
        write_json(paths["metadata"], metadata)
        print(json.dumps({"paths": metadata["paths"], "actual_dates": metadata["actual_dates"],
                          "row_count": len(raw)}, indent=2))
        print("\nFirst five raw rows:\n" + raw.head().to_string())
        print("\nMissing-value summary (processed includes all source columns):\n" +
              processed.isna().sum().to_string())
        print("\nValidation findings:\n" + json.dumps(reports, indent=2))
        print("\nUnusual returns for review:\n" + unusual.to_string(index=False))
        return 0
    except Exception as exc:
        metadata.update(status="failed", failure_time_utc=utc_now(),
                        error=f"{type(exc).__name__}: {exc}")
        failure_path = raw_dir / f"{STEM}_failure.json"
        write_json(failure_path, metadata)
        print(f"FAILED: {metadata['error']}\nFailure details: {failure_path}", file=sys.stderr)
        return 1


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hourly", action="store_true",
        help="download a separate regular-session 1-hour raw dataset")
    args = parser.parse_args()
    return hourly_main() if args.hourly else main()


if __name__ == "__main__":
    sys.exit(cli())
