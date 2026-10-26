#!/usr/bin/env python3
"""Build compact NVDA minute-market caches from Alpaca SIP partitions.

Minute bars are inexpensive and are always materialized. Quote enrichment is
optional because the source contains hundreds of millions of records; each UTC
day is streamed independently and can be processed concurrently on a cluster.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    import orjson
except ImportError:  # The standard library fallback is slower but fully supported.
    orjson = None

from common import (
    DEFAULT_BARS,
    DEFAULT_OUTPUT,
    DEFAULT_QUOTES,
    atomic_json,
    atomic_parquet,
    file_sha256,
)


PARTITION_RE = re.compile(
    r"^(?P<symbol>[A-Za-z0-9._-]+)_(?P<kind>bars|quotes)_"
    r"(?P<date>\d{4}-\d{2}-\d{2})\.jsonl\.gz$"
)


@dataclass
class PartitionResult:
    utc_date: str
    source: str
    output: str
    source_rows: int
    retained_rows: int
    rejected_rows: int
    reused: bool = False


def discover_partitions(directory: Path, symbol: str, kind: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in directory.glob(f"{symbol.upper()}_{kind}_*.jsonl.gz"):
        match = PARTITION_RE.match(path.name)
        if (
            match
            and match.group("symbol").upper() == symbol.upper()
            and match.group("kind") == kind
        ):
            result[match.group("date")] = path
    return result


def _read_records(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = orjson.loads(line) if orjson is not None else json.loads(line)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object in {path}:{line_number}")
            yield value


def _partition_is_current(source: Path, output: Path) -> bool:
    metadata = output.with_suffix(".summary.json")
    if not output.is_file() or not metadata.is_file():
        return False
    try:
        saved = json.loads(metadata.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    stat = source.stat()
    return (
        saved.get("status") == "complete"
        and saved.get("source_bytes") == stat.st_size
        and saved.get("source_mtime_ns") == stat.st_mtime_ns
    )


def _write_partition(
    frame: pd.DataFrame,
    source: Path,
    output: Path,
    result: PartitionResult,
) -> None:
    atomic_parquet(frame, output)
    stat = source.stat()
    atomic_json(
        {
            "status": "complete",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_bytes": stat.st_size,
            "source_mtime_ns": stat.st_mtime_ns,
            **asdict(result),
        },
        output.with_suffix(".summary.json"),
    )


def build_bar_partition(task: tuple[str, str, str, bool]) -> PartitionResult:
    source_text, output_text, symbol, force = task
    source = Path(source_text)
    output = Path(output_text)
    match = PARTITION_RE.match(source.name)
    if not match:
        raise ValueError(f"Unexpected bar filename: {source}")
    utc_date = match.group("date")
    if not force and _partition_is_current(source, output):
        metadata = json.loads(output.with_suffix(".summary.json").read_text())
        return PartitionResult(
            utc_date=utc_date,
            source=str(source),
            output=str(output),
            source_rows=int(metadata["source_rows"]),
            retained_rows=int(metadata["retained_rows"]),
            rejected_rows=int(metadata["rejected_rows"]),
            reused=True,
        )

    rows: list[dict[str, object]] = []
    source_rows = rejected_rows = 0
    for record in _read_records(source):
        source_rows += 1
        if str(record.get("_kind", "bars")) != "bars":
            continue
        if str(record.get("_symbol", symbol)).upper() != symbol.upper():
            continue
        try:
            timestamp = pd.Timestamp(str(record["t"]))
            values = {
                "open": float(record["o"]),
                "high": float(record["h"]),
                "low": float(record["l"]),
                "close": float(record["c"]),
                "volume": float(record.get("v", 0.0)),
                "trade_count": float(record.get("n", 0.0)),
                "vwap": float(record.get("vw", np.nan)),
            }
        except (KeyError, TypeError, ValueError):
            rejected_rows += 1
            continue
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        prices = [values[name] for name in ("open", "high", "low", "close")]
        if not all(np.isfinite(value) and value > 0 for value in prices):
            rejected_rows += 1
            continue
        rows.append(
            {
                "bar_start_utc": timestamp,
                "minute_end_utc": timestamp + pd.Timedelta(minutes=1),
                **values,
            }
        )

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values("minute_end_utc").drop_duplicates(
            "minute_end_utc", keep="last"
        )
    result = PartitionResult(
        utc_date=utc_date,
        source=str(source),
        output=str(output),
        source_rows=source_rows,
        retained_rows=len(frame),
        rejected_rows=rejected_rows,
    )
    _write_partition(frame, source, output, result)
    return result


def build_quote_partition(task: tuple[str, str, str, bool]) -> PartitionResult:
    """Stream a tick quote file into final-valid-quote-per-minute rows."""
    source_text, output_text, symbol, force = task
    source = Path(source_text)
    output = Path(output_text)
    match = PARTITION_RE.match(source.name)
    if not match:
        raise ValueError(f"Unexpected quote filename: {source}")
    utc_date = match.group("date")
    if not force and _partition_is_current(source, output):
        metadata = json.loads(output.with_suffix(".summary.json").read_text())
        return PartitionResult(
            utc_date=utc_date,
            source=str(source),
            output=str(output),
            source_rows=int(metadata["source_rows"]),
            retained_rows=int(metadata["retained_rows"]),
            rejected_rows=int(metadata["rejected_rows"]),
            reused=True,
        )

    rows: list[dict[str, object]] = []
    source_rows = rejected_rows = 0
    current_minute: str | None = None
    last_valid: dict[str, object] | None = None
    valid_updates = 0

    def flush() -> None:
        if current_minute is None or last_valid is None:
            return
        bid = float(last_valid["bid_price"])
        ask = float(last_valid["ask_price"])
        midpoint = (bid + ask) / 2.0
        rows.append(
            {
                "minute_end_utc": pd.Timestamp(f"{current_minute}:00Z")
                + pd.Timedelta(minutes=1),
                "midpoint": midpoint,
                "bid_price": bid,
                "ask_price": ask,
                "bid_size": float(last_valid["bid_size"]),
                "ask_size": float(last_valid["ask_size"]),
                "spread_bps": (ask - bid) / midpoint * 10_000.0,
                "last_quote_utc": pd.Timestamp(str(last_valid["timestamp"])),
                "valid_quote_updates": valid_updates,
            }
        )

    for record in _read_records(source):
        source_rows += 1
        if str(record.get("_kind", "quotes")) != "quotes":
            continue
        if str(record.get("_symbol", symbol)).upper() != symbol.upper():
            continue
        timestamp = str(record.get("t", ""))
        if len(timestamp) < 17 or timestamp[10] != "T":
            rejected_rows += 1
            continue
        minute = timestamp[:16]
        if current_minute is not None and minute < current_minute:
            raise ValueError(f"Out-of-order quote in {source}: {minute} < {current_minute}")
        if minute != current_minute:
            flush()
            current_minute = minute
            last_valid = None
            valid_updates = 0
        try:
            bid = float(record.get("bp", 0.0))
            ask = float(record.get("ap", 0.0))
            bid_size = float(record.get("bs", 0.0))
            ask_size = float(record.get("as", 0.0))
        except (TypeError, ValueError):
            rejected_rows += 1
            continue
        if not (
            np.isfinite(bid)
            and np.isfinite(ask)
            and bid > 0
            and ask >= bid
            and np.isfinite(bid_size)
            and np.isfinite(ask_size)
        ):
            rejected_rows += 1
            continue
        last_valid = {
            "bid_price": bid,
            "ask_price": ask,
            "bid_size": bid_size,
            "ask_size": ask_size,
            "timestamp": timestamp,
        }
        valid_updates += 1
    flush()

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values("minute_end_utc").drop_duplicates(
            "minute_end_utc", keep="last"
        )
    result = PartitionResult(
        utc_date=utc_date,
        source=str(source),
        output=str(output),
        source_rows=source_rows,
        retained_rows=len(frame),
        rejected_rows=rejected_rows,
    )
    _write_partition(frame, source, output, result)
    return result


def _run_tasks(
    function: Any,
    tasks: list[tuple[str, str, str, bool]],
    workers: int,
    label: str,
) -> list[PartitionResult]:
    if workers <= 1:
        results = []
        for index, task in enumerate(tasks, start=1):
            result = function(task)
            print(f"{label} {index}/{len(tasks)} {result.utc_date}", flush=True)
            results.append(result)
        return results

    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_task = {executor.submit(function, task): task for task in tasks}
        completed = 0
        for future in as_completed(future_to_task):
            result = future.result()
            results.append(result)
            completed += 1
            print(f"{label} {completed}/{len(tasks)} {result.utc_date}", flush=True)
    return sorted(results, key=lambda result: result.utc_date)


def build_market_cache(
    bars_dir: Path,
    quotes_dir: Path,
    output_dir: Path,
    symbol: str = "NVDA",
    include_quotes: bool = False,
    workers: int = 1,
    force: bool = False,
) -> tuple[pd.DataFrame, dict[str, object]]:
    cache_root = output_dir / "market_cache"
    bar_partitions = discover_partitions(bars_dir, symbol, "bars")
    if not bar_partitions:
        raise FileNotFoundError(f"No {symbol} bar partitions found in {bars_dir}")
    bar_tasks = [
        (
            str(source),
            str(cache_root / "bars" / f"{symbol}_bars_{utc_date}.parquet"),
            symbol,
            force,
        )
        for utc_date, source in sorted(bar_partitions.items())
    ]
    bar_results = _run_tasks(build_bar_partition, bar_tasks, workers, "bars")
    bar_frames = [pd.read_parquet(result.output) for result in bar_results]
    market = pd.concat(bar_frames, ignore_index=True).sort_values("minute_end_utc")
    market["minute_end_utc"] = pd.to_datetime(market.minute_end_utc, utc=True)
    market["bar_start_utc"] = pd.to_datetime(market.bar_start_utc, utc=True)

    quote_results: list[PartitionResult] = []
    if include_quotes:
        quote_partitions = discover_partitions(quotes_dir, symbol, "quotes")
        missing = sorted(set(bar_partitions).difference(quote_partitions))
        if missing:
            raise FileNotFoundError(
                "Missing quote partitions corresponding to bar dates: " + ", ".join(missing)
            )
        quote_tasks = [
            (
                str(quote_partitions[utc_date]),
                str(cache_root / "quotes" / f"{symbol}_quotes_{utc_date}.parquet"),
                symbol,
                force,
            )
            for utc_date in sorted(bar_partitions)
        ]
        quote_results = _run_tasks(
            build_quote_partition, quote_tasks, workers, "quotes"
        )
        quote_frames = [pd.read_parquet(result.output) for result in quote_results]
        quotes = pd.concat(quote_frames, ignore_index=True)
        quotes["minute_end_utc"] = pd.to_datetime(quotes.minute_end_utc, utc=True)
        quotes["last_quote_utc"] = pd.to_datetime(quotes.last_quote_utc, utc=True)
        market = market.merge(
            quotes,
            on="minute_end_utc",
            how="left",
            validate="one_to_one",
        )

    market = market.drop_duplicates("minute_end_utc", keep="last").reset_index(drop=True)
    if "midpoint" in market.columns:
        quote_staleness = (
            market.minute_end_utc - pd.to_datetime(market.last_quote_utc, utc=True)
        ).dt.total_seconds()
        valid_midpoint = (
            market.midpoint.notna()
            & market.midpoint.gt(0)
            & quote_staleness.between(0, 120)
        )
        market["quote_staleness_seconds"] = quote_staleness
        market["price"] = market.midpoint.where(valid_midpoint, market.close)
        market["price_source"] = np.where(valid_midpoint, "quote_midpoint", "bar_close")
    else:
        market["price"] = market.close
        market["price_source"] = "bar_close"
    market_path = output_dir / "market_minutes.parquet"
    atomic_parquet(market, market_path)
    audit: dict[str, object] = {
        "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol.upper(),
        "include_quotes": include_quotes,
        "source": {
            "bars_dir": str(bars_dir.resolve()),
            "quotes_dir": str(quotes_dir.resolve()) if include_quotes else None,
        },
        "bar_partitions": [asdict(result) for result in bar_results],
        "quote_partitions": [asdict(result) for result in quote_results],
        "output": {
            "path": str(market_path.resolve()),
            "rows": len(market),
            "first_minute_end_utc": str(market.minute_end_utc.iloc[0]),
            "last_minute_end_utc": str(market.minute_end_utc.iloc[-1]),
            "sha256": file_sha256(market_path),
        },
    }
    atomic_json(audit, output_dir / "market_cache_audit.json")
    return market, audit


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars-dir", type=Path, default=DEFAULT_BARS)
    parser.add_argument("--quotes-dir", type=Path, default=DEFAULT_QUOTES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--symbol", default="NVDA")
    parser.add_argument("--include-quotes", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    build_market_cache(
        bars_dir=args.bars_dir,
        quotes_dir=args.quotes_dir,
        output_dir=args.output_dir,
        symbol=args.symbol,
        include_quotes=args.include_quotes,
        workers=args.workers,
        force=args.force,
    )


if __name__ == "__main__":
    main()
