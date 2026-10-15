#!/usr/bin/env python3
"""Build clustered news/price labels from historical news and Alpaca SIP quotes.

This command performs preprocessing only. It contains no model training code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from label_events import assign_chronological_split, label_event_clusters
from news_clusters import build_fixed_clusters, prepare_news
from quote_minutes import load_minute_cache, materialize_minute_cache


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_NEWS = ROOT / "news data pull/processed/all_classified_articles.csv"
DEFAULT_QUOTES = ROOT / "alpaca data pull/raw/sip/quotes"
DEFAULT_OUTPUT = HERE / "output"


def parse_horizons(value: str) -> list[int]:
    try:
        horizons = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("horizons must be comma-separated integers") from exc
    if not horizons or min(horizons) < 1:
        raise argparse.ArgumentTypeError("horizons must contain positive integers")
    return horizons


def file_identity(path: Path, include_sha256: bool = False) -> dict[str, object]:
    stat = path.stat()
    result: dict[str, object] = {
        "path": str(path.resolve()),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if include_sha256:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        result["sha256"] = digest.hexdigest()
    return result


def atomic_parquet(frame: pd.DataFrame, destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".partial")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, destination)


def atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".partial")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, destination)


def json_counts(series: pd.Series) -> dict[str, int]:
    counts = series.astype("string").fillna("missing").value_counts(dropna=False)
    return {str(key): int(value) for key, value in counts.items()}


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Cluster regular-session news and attach multi-horizon labels from "
            "Alpaca SIP quote midpoints. No model is trained."
        )
    )
    parser.add_argument("--news", type=Path, default=DEFAULT_NEWS)
    parser.add_argument("--quotes-dir", type=Path, default=DEFAULT_QUOTES)
    parser.add_argument("--symbol", default="NVDA")
    parser.add_argument("--start-date", default="2025-10-01")
    parser.add_argument("--end-date", default="2026-05-01")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--cluster-minutes", type=int, default=5)
    parser.add_argument("--latency-minutes", type=int, default=1)
    parser.add_argument("--horizons", type=parse_horizons, default=parse_horizons("5,15,60"))
    parser.add_argument("--pre-vol-minutes", type=int, default=30)
    parser.add_argument("--min-pre-vol-observations", type=int, default=20)
    parser.add_argument("--z-cutoff", type=float, default=1.0)
    parser.add_argument("--spread-multiple", type=float, default=1.0)
    parser.add_argument("--max-quote-staleness-seconds", type=float, default=30.0)
    parser.add_argument("--include-indirect", action="store_true")
    parser.add_argument("--benchmark-quotes-dir", type=Path)
    parser.add_argument("--benchmark-symbol")
    parser.add_argument("--benchmark-beta", type=float, default=1.0)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--write-csv", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--force-rebuild-cache", action="store_true")
    parser.add_argument("--allow-missing-quote-days", action="store_true")
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not args.news.is_file():
        parser.error(f"News file does not exist: {args.news}")
    if not args.quotes_dir.is_dir():
        parser.error(f"Quote directory does not exist: {args.quotes_dir}")
    if args.cluster_minutes < 1 or args.latency_minutes < 0:
        parser.error("Cluster minutes must be positive and latency cannot be negative")
    if args.pre_vol_minutes < 2 or args.min_pre_vol_observations < 2:
        parser.error("Pre-event volatility settings must be at least two minutes/observations")
    if args.min_pre_vol_observations >= args.pre_vol_minutes:
        parser.error("Minimum pre-volatility observations must be below the window length")
    if args.z_cutoff <= 0 or args.spread_multiple < 0:
        parser.error("z-cutoff must be positive and spread-multiple cannot be negative")
    if args.max_quote_staleness_seconds <= 0:
        parser.error("max-quote-staleness-seconds must be positive")
    has_benchmark_dir = args.benchmark_quotes_dir is not None
    has_benchmark_symbol = bool(args.benchmark_symbol)
    if has_benchmark_dir != has_benchmark_symbol:
        parser.error(
            "--benchmark-quotes-dir and --benchmark-symbol must be provided together"
        )
    if has_benchmark_dir and not args.benchmark_quotes_dir.is_dir():
        parser.error(f"Benchmark quote directory does not exist: {args.benchmark_quotes_dir}")


def check_output_targets(
    output_dir: Path,
    write_csv: bool,
    overwrite: bool,
) -> dict[str, Path]:
    targets = {
        "events_parquet": output_dir / "news_price_events.parquet",
        "members_parquet": output_dir / "cluster_articles.parquet",
        "audit": output_dir / "build_audit.json",
    }
    if write_csv:
        targets["events_csv"] = output_dir / "news_price_events.csv"
        targets["members_csv"] = output_dir / "cluster_articles.csv"
    existing = [path for path in targets.values() if path.exists()]
    if existing and not overwrite:
        formatted = "\n".join(str(path) for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing outputs; pass --overwrite if intended:\n"
            + formatted
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return targets


def cache_audit(results: Iterable[object]) -> list[dict[str, object]]:
    return [asdict(result) for result in results]


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    validate_args(args, parser)
    targets = check_output_targets(args.output_dir, args.write_csv, args.overwrite)
    cache_dir = args.cache_dir or args.output_dir / ".minute_cache"

    relevance_types = {"direct", "indirect"} if args.include_indirect else {"direct"}
    print("filtering and clustering news", flush=True)
    articles, _schedule, news_audit = prepare_news(
        args.news,
        args.start_date,
        args.end_date,
        relevance_types,
    )
    clusters, memberships = build_fixed_clusters(
        articles,
        args.cluster_minutes,
        args.latency_minutes,
    )
    news_audit["event_clusters"] = len(clusters)

    required_utc_dates = sorted(
        pd.to_datetime(clusters.market_open_utc, utc=True).dt.strftime("%Y-%m-%d").unique()
    )
    print(
        f"materializing {len(required_utc_dates)} daily {args.symbol.upper()} minute caches",
        flush=True,
    )
    primary_paths, primary_cache_results, primary_missing = materialize_minute_cache(
        args.quotes_dir,
        cache_dir,
        args.symbol,
        required_utc_dates,
        force=args.force_rebuild_cache,
        progress=lambda message: print(message, flush=True),
    )
    primary_quotes = load_minute_cache(primary_paths, args.symbol)
    if primary_missing and not args.allow_missing_quote_days:
        raise FileNotFoundError(
            "Missing primary quote partitions for required UTC dates: "
            + ", ".join(primary_missing)
            + ". Pass --allow-missing-quote-days only if incomplete labels are intended."
        )

    benchmark_quotes = None
    benchmark_cache_results: list[object] = []
    benchmark_missing: list[str] = []
    if args.benchmark_quotes_dir is not None:
        print(
            f"materializing {len(required_utc_dates)} daily "
            f"{args.benchmark_symbol.upper()} minute caches",
            flush=True,
        )
        benchmark_paths, benchmark_cache_results, benchmark_missing = (
            materialize_minute_cache(
                args.benchmark_quotes_dir,
                cache_dir,
                args.benchmark_symbol,
                required_utc_dates,
                force=args.force_rebuild_cache,
                progress=lambda message: print(message, flush=True),
            )
        )
        benchmark_quotes = load_minute_cache(
            benchmark_paths, args.benchmark_symbol
        )
        if benchmark_missing and not args.allow_missing_quote_days:
            raise FileNotFoundError(
                "Missing benchmark quote partitions for required UTC dates: "
                + ", ".join(benchmark_missing)
                + ". Pass --allow-missing-quote-days only if incomplete labels are intended."
            )

    print("joining clusters to quote midpoints and constructing labels", flush=True)
    events = label_event_clusters(
        clusters=clusters,
        primary_quotes=primary_quotes,
        primary_symbol=args.symbol,
        horizons=args.horizons,
        pre_vol_minutes=args.pre_vol_minutes,
        minimum_pre_vol_observations=args.min_pre_vol_observations,
        z_cutoff=args.z_cutoff,
        spread_multiple=args.spread_multiple,
        max_quote_staleness_seconds=args.max_quote_staleness_seconds,
        benchmark_quotes=benchmark_quotes,
        benchmark_symbol=args.benchmark_symbol,
        benchmark_beta=args.benchmark_beta,
    )
    events = assign_chronological_split(
        events,
        args.train_fraction,
        args.validation_fraction,
    )

    integer_label_columns = [
        column
        for column in events.columns
        if column.startswith("label_code_") or column.startswith("barrier_code_")
    ]
    for column in integer_label_columns:
        events[column] = events[column].astype("Int8")

    print("writing merged datasets", flush=True)
    atomic_parquet(events, targets["events_parquet"])
    atomic_parquet(memberships, targets["members_parquet"])
    if args.write_csv:
        atomic_csv(events, targets["events_csv"])
        atomic_csv(memberships, targets["members_csv"])

    audit: dict[str, object] = {
        "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": "fixed_window_news_clusters_joined_to_minute_closing_sip_midpoints",
        "training_performed": False,
        "inputs": {
            "news": file_identity(args.news, include_sha256=True),
            "primary_quotes_dir": str(args.quotes_dir.resolve()),
            "benchmark_quotes_dir": (
                str(args.benchmark_quotes_dir.resolve())
                if args.benchmark_quotes_dir is not None
                else None
            ),
        },
        "settings": {
            "symbol": args.symbol.upper(),
            "start_date": args.start_date,
            "end_date": args.end_date,
            "relevance_types": sorted(relevance_types),
            "cluster_minutes": args.cluster_minutes,
            "latency_minutes": args.latency_minutes,
            "horizons_minutes": args.horizons,
            "pre_vol_minutes": args.pre_vol_minutes,
            "min_pre_vol_observations": args.min_pre_vol_observations,
            "z_cutoff": args.z_cutoff,
            "spread_multiple": args.spread_multiple,
            "max_quote_staleness_seconds": args.max_quote_staleness_seconds,
            "benchmark_symbol": (
                args.benchmark_symbol.upper() if args.benchmark_symbol else None
            ),
            "benchmark_beta": args.benchmark_beta,
            "train_fraction": args.train_fraction,
            "validation_fraction": args.validation_fraction,
            "allow_missing_quote_days": args.allow_missing_quote_days,
        },
        "news_audit": news_audit,
        "quote_cache": {
            "primary": cache_audit(primary_cache_results),
            "primary_missing_utc_dates": primary_missing,
            "benchmark": cache_audit(benchmark_cache_results),
            "benchmark_missing_utc_dates": benchmark_missing,
        },
        "outputs": {
            "event_rows": len(events),
            "membership_rows": len(memberships),
            "chronological_split_counts": json_counts(events.chronological_split),
            "paths": {name: str(path.resolve()) for name, path in targets.items()},
            "horizons": {
                f"{horizon}m": {
                    "label_counts": json_counts(events[f"label_{horizon}m"]),
                    "available": int(events[f"label_available_{horizon}m"].sum()),
                    "clean": int(events[f"clean_{horizon}m"].sum()),
                }
                for horizon in args.horizons
            },
        },
        "limitations": [
            "News publication timestamps have one-second rather than tick-level precision.",
            "Finnhub publication time may differ from the first instant a market participant saw the story.",
            "Only regular-session clusters are included; overnight news requires a separate close-to-open design.",
            (
                "Returns are raw NVDA returns and can contain market-wide movement."
                if benchmark_quotes is None
                else "Benchmark adjustment uses the configured fixed beta rather than a dynamically estimated beta."
            ),
            "A clean-horizon flag removes later news clusters but cannot remove unobserved information arrivals.",
        ],
    }
    temporary_audit = targets["audit"].with_name(targets["audit"].name + ".partial")
    temporary_audit.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_audit, targets["audit"])
    print(f"complete: {targets['events_parquet']}", flush=True)


if __name__ == "__main__":
    main()
