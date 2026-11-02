"""Shared utilities for the NVDA news-impact experiment."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_NEWS = ROOT / "news data pull/processed/all_classified_articles.csv"
DEFAULT_BARS = ROOT / "alpaca data pull/raw/sip/bars"
DEFAULT_QUOTES = ROOT / "alpaca data pull/raw/sip/quotes"
DEFAULT_OUTPUT = HERE / "output"


def parse_positive_ints(value: str) -> list[int]:
    """Parse a comma-separated collection of unique positive integers."""
    try:
        values = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    except ValueError as exc:
        raise ValueError(f"Expected comma-separated integers, received {value!r}") from exc
    if not values or values[0] < 1:
        raise ValueError("At least one positive integer is required")
    return values


def boolean_series(series: pd.Series) -> pd.Series:
    """Normalize common CSV boolean representations."""
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def stable_identifier(prefix: str, values: Iterable[object], length: int = 20) -> str:
    payload = "|".join(str(value) for value in values)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"


def json_list(values: Iterable[object]) -> str:
    cleaned = sorted({str(value).strip() for value in values if str(value).strip()})
    return json.dumps(cleaned, ensure_ascii=False, separators=(",", ":"))


def json_counts(series: pd.Series) -> dict[str, int]:
    counts = series.astype("string").fillna("missing").value_counts(dropna=False)
    return {str(key): int(value) for key, value in counts.items()}


def timestamps_to_ns(values: Iterable[object]) -> np.ndarray:
    """Return nanosecond integers independent of pandas' storage resolution."""
    return np.array([pd.Timestamp(value).value for value in values], dtype=np.int64)
