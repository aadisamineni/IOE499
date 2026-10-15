"""Stream Alpaca quote partitions into compact one-minute midpoint files."""

from __future__ import annotations

import gzip
import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import orjson
import pandas as pd


QUOTE_FILE_RE = re.compile(
    r"^(?P<symbol>[A-Za-z0-9._-]+)_quotes_(?P<date>\d{4}-\d{2}-\d{2})\.jsonl\.gz$"
)

MINUTE_COLUMNS = [
    "minute_end_utc",
    "midpoint",
    "bid_price",
    "ask_price",
    "bid_size",
    "ask_size",
    "spread_bps",
    "last_quote_utc",
    "valid_quote_updates",
]


@dataclass
class CacheBuildResult:
    symbol: str
    utc_date: str
    source: str
    cache: str
    source_rows: int
    valid_quote_rows: int
    invalid_quote_rows: int
    minute_rows: int
    first_quote_utc: str | None
    last_quote_utc: str | None
    reused: bool = False


def discover_quote_partitions(quotes_dir: Path, symbol: str) -> dict[str, Path]:
    """Map UTC dates to daily gzip quote partitions for one symbol."""
    symbol = symbol.upper()
    result: dict[str, Path] = {}
    if not quotes_dir.exists():
        return result
    for path in quotes_dir.glob(f"{symbol}_quotes_*.jsonl.gz"):
        match = QUOTE_FILE_RE.match(path.name)
        if match and match.group("symbol").upper() == symbol:
            result[match.group("date")] = path
    return result


def _minute_row(
    minute_key: str,
    bid_price: float,
    ask_price: float,
    bid_size: float,
    ask_size: float,
    last_quote_utc: str,
    valid_quote_updates: int,
) -> dict[str, object]:
    midpoint = (bid_price + ask_price) / 2.0
    minute_start = pd.Timestamp(f"{minute_key}:00Z")
    return {
        "minute_end_utc": minute_start + pd.Timedelta(minutes=1),
        "midpoint": midpoint,
        "bid_price": bid_price,
        "ask_price": ask_price,
        "bid_size": bid_size,
        "ask_size": ask_size,
        "spread_bps": (ask_price - bid_price) / midpoint * 10_000.0,
        "last_quote_utc": pd.Timestamp(last_quote_utc),
        "valid_quote_updates": valid_quote_updates,
    }


def _cache_is_current(source: Path, cache: Path, metadata: Path) -> bool:
    if not cache.exists() or not metadata.exists():
        return False
    try:
        saved = json.loads(metadata.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    stat = source.stat()
    return (
        saved.get("source_size") == stat.st_size
        and saved.get("source_mtime_ns") == stat.st_mtime_ns
        and saved.get("status") == "complete"
    )


def build_daily_minute_cache(
    source: Path,
    cache: Path,
    symbol: str,
    force: bool = False,
) -> CacheBuildResult:
    """Build one daily minute file without loading a tick partition into memory."""
    match = QUOTE_FILE_RE.match(source.name)
    if not match:
        raise ValueError(f"Unexpected quote partition name: {source}")
    utc_date = match.group("date")
    metadata = cache.with_suffix(".summary.json")

    if not force and _cache_is_current(source, cache, metadata):
        saved = json.loads(metadata.read_text(encoding="utf-8"))
        return CacheBuildResult(
            symbol=symbol.upper(),
            utc_date=utc_date,
            source=str(source),
            cache=str(cache),
            source_rows=int(saved["source_rows"]),
            valid_quote_rows=int(saved["valid_quote_rows"]),
            invalid_quote_rows=int(saved["invalid_quote_rows"]),
            minute_rows=int(saved["minute_rows"]),
            first_quote_utc=saved.get("first_quote_utc"),
            last_quote_utc=saved.get("last_quote_utc"),
            reused=True,
        )

    cache.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    source_rows = 0
    valid_rows = 0
    invalid_rows = 0
    first_quote: str | None = None
    final_quote: str | None = None

    current_minute: str | None = None
    last_bid = last_ask = last_bid_size = last_ask_size = np.nan
    last_timestamp: str | None = None
    updates_in_minute = 0

    def flush_current() -> None:
        if current_minute is None or last_timestamp is None:
            return
        rows.append(
            _minute_row(
                current_minute,
                float(last_bid),
                float(last_ask),
                float(last_bid_size),
                float(last_ask_size),
                last_timestamp,
                updates_in_minute,
            )
        )

    with gzip.open(source, "rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            source_rows += 1
            try:
                record = orjson.loads(line)
            except orjson.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {source}:{line_number}") from exc

            if str(record.get("_kind", "quotes")) != "quotes":
                continue
            record_feed = str(record.get("_feed", "sip")).lower()
            if record_feed != "sip":
                raise ValueError(
                    f"Expected SIP quotes but found feed={record_feed!r} in {source}"
                )
            record_symbol = str(record.get("_symbol", symbol)).upper()
            if record_symbol != symbol.upper():
                continue

            timestamp = str(record.get("t", ""))
            if len(timestamp) < 17 or timestamp[10] != "T":
                invalid_rows += 1
                continue
            minute_key = timestamp[:16]
            if current_minute is not None and minute_key < current_minute:
                raise ValueError(
                    f"Quote timestamps are not sorted in {source}: "
                    f"{minute_key} follows {current_minute}"
                )
            if minute_key != current_minute:
                flush_current()
                current_minute = minute_key
                last_timestamp = None
                updates_in_minute = 0

            try:
                bid = float(record.get("bp", 0.0))
                ask = float(record.get("ap", 0.0))
                bid_size = float(record.get("bs", 0.0))
                ask_size = float(record.get("as", 0.0))
            except (TypeError, ValueError):
                invalid_rows += 1
                continue
            if not (
                np.isfinite(bid)
                and np.isfinite(ask)
                and bid > 0
                and ask > 0
                and ask >= bid
            ):
                invalid_rows += 1
                continue

            last_bid = bid
            last_ask = ask
            last_bid_size = bid_size
            last_ask_size = ask_size
            last_timestamp = timestamp
            updates_in_minute += 1
            valid_rows += 1
            if first_quote is None:
                first_quote = timestamp
            final_quote = timestamp

    flush_current()
    frame = pd.DataFrame(rows, columns=MINUTE_COLUMNS)
    if not frame.empty:
        frame = frame.sort_values("minute_end_utc").reset_index(drop=True)
        if frame.minute_end_utc.duplicated().any():
            raise ValueError(f"Duplicate minute endpoints while processing {source}")
    frame.to_parquet(cache, index=False)

    result = CacheBuildResult(
        symbol=symbol.upper(),
        utc_date=utc_date,
        source=str(source),
        cache=str(cache),
        source_rows=source_rows,
        valid_quote_rows=valid_rows,
        invalid_quote_rows=invalid_rows,
        minute_rows=len(frame),
        first_quote_utc=first_quote,
        last_quote_utc=final_quote,
    )
    stat = source.stat()
    summary = {
        "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        **asdict(result),
    }
    metadata.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return result


def materialize_minute_cache(
    quotes_dir: Path,
    cache_root: Path,
    symbol: str,
    utc_dates: Iterable[str],
    force: bool = False,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[Path], list[CacheBuildResult], list[str]]:
    """Ensure requested daily caches exist and report unavailable partitions."""
    partitions = discover_quote_partitions(quotes_dir, symbol)
    cache_paths: list[Path] = []
    results: list[CacheBuildResult] = []
    missing: list[str] = []
    for utc_date in sorted(set(utc_dates)):
        source = partitions.get(utc_date)
        if source is None:
            missing.append(utc_date)
            if progress is not None:
                progress(f"missing {symbol.upper()} quote partition for {utc_date}")
            continue
        cache = cache_root / symbol.upper() / f"{symbol.upper()}_minute_quotes_{utc_date}.parquet"
        if progress is not None:
            action = "rebuild" if force or not _cache_is_current(
                source, cache, cache.with_suffix(".summary.json")
            ) else "reuse"
            progress(f"{action} {symbol.upper()} minute cache for {utc_date}")
        result = build_daily_minute_cache(source, cache, symbol, force=force)
        cache_paths.append(cache)
        results.append(result)
    return cache_paths, results, missing


def load_minute_cache(paths: Iterable[Path], symbol: str) -> pd.DataFrame:
    """Load small daily minute files and prefix non-key fields by symbol."""
    frames = [pd.read_parquet(path) for path in paths]
    if not frames:
        columns = ["minute_end_utc"] + [
            f"{symbol.lower()}_{column}" for column in MINUTE_COLUMNS if column != "minute_end_utc"
        ]
        return pd.DataFrame(columns=columns)
    frame = pd.concat(frames, ignore_index=True)
    frame["minute_end_utc"] = pd.to_datetime(frame.minute_end_utc, utc=True)
    frame["last_quote_utc"] = pd.to_datetime(frame.last_quote_utc, utc=True)
    frame = frame.sort_values("minute_end_utc").drop_duplicates(
        "minute_end_utc", keep="last"
    )
    prefix = symbol.lower()
    return frame.rename(
        columns={
            column: f"{prefix}_{column}"
            for column in frame.columns
            if column != "minute_end_utc"
        }
    ).reset_index(drop=True)
