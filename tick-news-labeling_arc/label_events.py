"""Join event clusters to minute quote midpoints and construct outcome labels."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class PriceObservation:
    midpoint: float
    spread_bps: float
    bid_price: float
    ask_price: float
    last_quote_utc: pd.Timestamp | pd.NaT
    staleness_seconds: float
    valid_quote_updates: float


class MinuteQuoteLookup:
    """Fast as-of access to one symbol's cached minute-closing quote state."""

    def __init__(self, frame: pd.DataFrame, symbol: str):
        self.symbol = symbol.lower()
        required = {
            "minute_end_utc",
            f"{self.symbol}_midpoint",
            f"{self.symbol}_spread_bps",
            f"{self.symbol}_last_quote_utc",
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Minute quote data is missing columns: {sorted(missing)}")
        ordered = frame.sort_values("minute_end_utc").reset_index(drop=True)
        self.frame = ordered
        self.times_ns = np.array(
            [pd.Timestamp(value).value for value in ordered.minute_end_utc], dtype=np.int64
        )

    def observe(
        self,
        timestamp: pd.Timestamp,
        max_staleness_seconds: float,
    ) -> PriceObservation:
        if pd.isna(timestamp) or len(self.times_ns) == 0:
            return self._missing()
        timestamp = pd.Timestamp(timestamp)
        index = int(np.searchsorted(self.times_ns, timestamp.value, side="right") - 1)
        if index < 0:
            return self._missing()
        row = self.frame.iloc[index]
        last_quote = pd.Timestamp(row[f"{self.symbol}_last_quote_utc"])
        staleness = (timestamp - last_quote).total_seconds()
        if staleness < 0 or staleness > max_staleness_seconds:
            return self._missing(staleness)

        def value(name: str) -> float:
            raw = row.get(f"{self.symbol}_{name}", np.nan)
            return float(raw) if pd.notna(raw) else np.nan

        midpoint = value("midpoint")
        if not np.isfinite(midpoint) or midpoint <= 0:
            return self._missing(staleness)
        return PriceObservation(
            midpoint=midpoint,
            spread_bps=value("spread_bps"),
            bid_price=value("bid_price"),
            ask_price=value("ask_price"),
            last_quote_utc=last_quote,
            staleness_seconds=float(staleness),
            valid_quote_updates=value("valid_quote_updates"),
        )

    @staticmethod
    def _missing(staleness: float = np.nan) -> PriceObservation:
        return PriceObservation(
            midpoint=np.nan,
            spread_bps=np.nan,
            bid_price=np.nan,
            ask_price=np.nan,
            last_quote_utc=pd.NaT,
            staleness_seconds=float(staleness),
            valid_quote_updates=np.nan,
        )

    def pre_event_volatility(
        self,
        cluster_start: pd.Timestamp,
        market_open: pd.Timestamp,
        window_minutes: int,
        minimum_observations: int,
        max_staleness_seconds: float,
    ) -> tuple[float, int]:
        """Return standard deviation of contiguous one-minute pre-news log returns."""
        if len(self.times_ns) == 0:
            return np.nan, 0
        end = pd.Timestamp(cluster_start)
        start = max(
            pd.Timestamp(market_open),
            end - pd.Timedelta(minutes=window_minutes + 1),
        )
        left = int(np.searchsorted(self.times_ns, start.value, side="left"))
        right = int(np.searchsorted(self.times_ns, end.value, side="right"))
        sample = self.frame.iloc[left:right]
        if len(sample) < 2:
            return np.nan, 0
        times = pd.to_datetime(sample.minute_end_utc, utc=True)
        prices = sample[f"{self.symbol}_midpoint"].to_numpy(dtype=float)
        quote_times = pd.to_datetime(
            sample[f"{self.symbol}_last_quote_utc"], utc=True
        )
        freshness = (
            (times.reset_index(drop=True) - quote_times.reset_index(drop=True))
            .dt.total_seconds()
            .between(0, max_staleness_seconds)
            .to_numpy(dtype=bool)
        )
        log_prices = np.log(prices)
        returns = np.diff(log_prices)
        gaps = np.diff(np.array([value.value for value in times], dtype=np.int64)) / 1e9
        valid = (
            np.isfinite(returns)
            & (gaps > 0)
            & (gaps <= 90.0)
            & freshness[:-1]
            & freshness[1:]
        )
        usable = returns[valid]
        if len(usable) < minimum_observations:
            return np.nan, int(len(usable))
        return float(np.std(usable, ddof=1)), int(len(usable))

    def path(
        self,
        start_exclusive: pd.Timestamp,
        end_inclusive: pd.Timestamp,
    ) -> pd.DataFrame:
        left = int(
            np.searchsorted(self.times_ns, pd.Timestamp(start_exclusive).value, side="right")
        )
        right = int(
            np.searchsorted(self.times_ns, pd.Timestamp(end_inclusive).value, side="right")
        )
        return self.frame.iloc[left:right].copy()


def _observation_fields(prefix: str, observation: PriceObservation) -> dict[str, object]:
    return {
        f"{prefix}_midpoint": observation.midpoint,
        f"{prefix}_bid_price": observation.bid_price,
        f"{prefix}_ask_price": observation.ask_price,
        f"{prefix}_spread_bps": observation.spread_bps,
        f"{prefix}_last_quote_utc": observation.last_quote_utc,
        f"{prefix}_quote_staleness_seconds": observation.staleness_seconds,
        f"{prefix}_valid_quote_updates": observation.valid_quote_updates,
    }


def _count_later_clusters(
    starts: np.ndarray,
    lower: pd.Timestamp,
    upper: pd.Timestamp,
) -> int:
    lower_index = int(np.searchsorted(starts, pd.Timestamp(lower).value, side="left"))
    upper_index = int(np.searchsorted(starts, pd.Timestamp(upper).value, side="right"))
    return max(0, upper_index - lower_index)


def _label_from_return(value: float, threshold: float) -> tuple[object, object]:
    if not np.isfinite(value) or not np.isfinite(threshold):
        return pd.NA, pd.NA
    if value >= threshold:
        return "up", 1
    if value <= -threshold:
        return "down", -1
    return "neutral", 0


def _path_statistics(
    primary: MinuteQuoteLookup,
    benchmark: MinuteQuoteLookup | None,
    benchmark_beta: float,
    decision_time: pd.Timestamp,
    end_time: pd.Timestamp,
    entry_midpoint: float,
    benchmark_entry: float,
    threshold: float,
    max_staleness_seconds: float,
) -> dict[str, object]:
    if not np.isfinite(entry_midpoint) or not np.isfinite(threshold):
        return {
            "max_label_return": np.nan,
            "min_label_return": np.nan,
            "minutes_to_max": np.nan,
            "minutes_to_min": np.nan,
            "barrier_label": pd.NA,
            "barrier_code": pd.NA,
            "minutes_to_barrier": np.nan,
            "path_observations": 0,
        }

    path = primary.path(decision_time, end_time)
    values: list[tuple[pd.Timestamp, float]] = []
    for row in path.itertuples(index=False):
        timestamp = pd.Timestamp(row.minute_end_utc)
        primary_midpoint = float(getattr(row, f"{primary.symbol}_midpoint"))
        primary_quote_time = pd.Timestamp(
            getattr(row, f"{primary.symbol}_last_quote_utc")
        )
        primary_staleness = (timestamp - primary_quote_time).total_seconds()
        if (
            not np.isfinite(primary_midpoint)
            or primary_midpoint <= 0
            or primary_staleness < 0
            or primary_staleness > max_staleness_seconds
        ):
            continue
        value = float(np.log(primary_midpoint / entry_midpoint))
        if benchmark is not None:
            benchmark_observation = benchmark.observe(timestamp, max_staleness_seconds)
            if not (
                np.isfinite(benchmark_entry)
                and np.isfinite(benchmark_observation.midpoint)
                and benchmark_entry > 0
            ):
                continue
            value -= benchmark_beta * float(
                np.log(benchmark_observation.midpoint / benchmark_entry)
            )
        values.append((timestamp, value))

    if not values:
        return {
            "max_label_return": np.nan,
            "min_label_return": np.nan,
            "minutes_to_max": np.nan,
            "minutes_to_min": np.nan,
            "barrier_label": pd.NA,
            "barrier_code": pd.NA,
            "minutes_to_barrier": np.nan,
            "path_observations": 0,
        }

    timestamps = [item[0] for item in values]
    returns = np.array([item[1] for item in values], dtype=float)
    max_index = int(np.argmax(returns))
    min_index = int(np.argmin(returns))
    positive_hits = np.flatnonzero(returns >= threshold)
    negative_hits = np.flatnonzero(returns <= -threshold)

    first_positive = int(positive_hits[0]) if len(positive_hits) else None
    first_negative = int(negative_hits[0]) if len(negative_hits) else None
    if first_positive is None and first_negative is None:
        barrier_label: object = "neutral"
        barrier_code: object = 0
        barrier_index = None
    elif first_negative is None or (
        first_positive is not None and first_positive < first_negative
    ):
        barrier_label = "up"
        barrier_code = 1
        barrier_index = first_positive
    else:
        barrier_label = "down"
        barrier_code = -1
        barrier_index = first_negative

    def elapsed(index: int) -> float:
        return (timestamps[index] - decision_time).total_seconds() / 60.0

    return {
        "max_label_return": float(returns[max_index]),
        "min_label_return": float(returns[min_index]),
        "minutes_to_max": elapsed(max_index),
        "minutes_to_min": elapsed(min_index),
        "barrier_label": barrier_label,
        "barrier_code": barrier_code,
        "minutes_to_barrier": (
            elapsed(barrier_index) if barrier_index is not None else np.nan
        ),
        "path_observations": len(returns),
    }


def label_event_clusters(
    clusters: pd.DataFrame,
    primary_quotes: pd.DataFrame,
    primary_symbol: str,
    horizons: Iterable[int],
    pre_vol_minutes: int,
    minimum_pre_vol_observations: int,
    z_cutoff: float,
    spread_multiple: float,
    max_quote_staleness_seconds: float,
    benchmark_quotes: pd.DataFrame | None = None,
    benchmark_symbol: str | None = None,
    benchmark_beta: float = 1.0,
) -> pd.DataFrame:
    """Attach prices, returns, path statistics, labels, and news-overlap flags."""
    primary = MinuteQuoteLookup(primary_quotes, primary_symbol)
    benchmark = None
    if benchmark_quotes is not None and benchmark_symbol is not None:
        benchmark = MinuteQuoteLookup(benchmark_quotes, benchmark_symbol)

    horizon_values = sorted({int(value) for value in horizons})
    if not horizon_values or min(horizon_values) < 1:
        raise ValueError("At least one positive horizon is required")

    starts_by_session = {
        session: np.sort(np.array(
            [pd.Timestamp(value).value for value in group.cluster_start_utc],
            dtype=np.int64,
        ))
        for session, group in clusters.groupby("session_date", sort=False)
    }

    output_rows: list[dict[str, object]] = []
    for cluster in clusters.itertuples(index=False):
        row = cluster._asdict()
        decision_time = pd.Timestamp(cluster.decision_time_utc)
        market_close = pd.Timestamp(cluster.market_close_utc)
        entry = primary.observe(decision_time, max_quote_staleness_seconds)
        row.update(_observation_fields("decision", entry))

        pre_vol, pre_vol_count = primary.pre_event_volatility(
            pd.Timestamp(cluster.cluster_start_utc),
            pd.Timestamp(cluster.market_open_utc),
            pre_vol_minutes,
            minimum_pre_vol_observations,
            max_quote_staleness_seconds,
        )
        row["pre_event_volatility_1m"] = pre_vol
        row["pre_event_return_observations"] = pre_vol_count

        benchmark_entry = np.nan
        if benchmark is not None:
            benchmark_entry_observation = benchmark.observe(
                decision_time, max_quote_staleness_seconds
            )
            benchmark_entry = benchmark_entry_observation.midpoint
            row.update(
                _observation_fields(
                    f"decision_{benchmark.symbol}", benchmark_entry_observation
                )
            )

        starts = starts_by_session[str(cluster.session_date)]
        latency_count = _count_later_clusters(
            starts,
            pd.Timestamp(cluster.cluster_window_end_utc),
            decision_time,
        )
        row["new_clusters_during_latency"] = latency_count

        for horizon in horizon_values:
            suffix = f"{horizon}m"
            target_time = decision_time + pd.Timedelta(minutes=horizon)
            in_session = target_time <= market_close
            row[f"target_time_{suffix}_utc"] = target_time
            row[f"target_within_regular_session_{suffix}"] = bool(in_session)

            target = (
                primary.observe(target_time, max_quote_staleness_seconds)
                if in_session
                else PriceObservation(
                    midpoint=np.nan,
                    spread_bps=np.nan,
                    bid_price=np.nan,
                    ask_price=np.nan,
                    last_quote_utc=pd.NaT,
                    staleness_seconds=np.nan,
                    valid_quote_updates=np.nan,
                )
            )
            row.update(_observation_fields(f"target_{suffix}", target))

            raw_return = np.nan
            if np.isfinite(entry.midpoint) and np.isfinite(target.midpoint):
                raw_return = float(np.log(target.midpoint / entry.midpoint))
            row[f"raw_log_return_{suffix}"] = raw_return

            benchmark_return = np.nan
            if benchmark is not None and in_session:
                benchmark_target = benchmark.observe(
                    target_time, max_quote_staleness_seconds
                )
                row.update(
                    _observation_fields(
                        f"target_{suffix}_{benchmark.symbol}", benchmark_target
                    )
                )
                if np.isfinite(benchmark_entry) and np.isfinite(
                    benchmark_target.midpoint
                ):
                    benchmark_return = float(
                        np.log(benchmark_target.midpoint / benchmark_entry)
                    )
            row[f"benchmark_log_return_{suffix}"] = benchmark_return

            if benchmark is None:
                label_return = raw_return
            elif np.isfinite(raw_return) and np.isfinite(benchmark_return):
                label_return = raw_return - benchmark_beta * benchmark_return
            else:
                label_return = np.nan
            row[f"label_log_return_{suffix}"] = label_return
            row[f"volatility_normalized_return_{suffix}"] = (
                label_return / (pre_vol * sqrt(horizon))
                if np.isfinite(label_return) and np.isfinite(pre_vol) and pre_vol > 0
                else np.nan
            )

            volatility_threshold = (
                z_cutoff * pre_vol * sqrt(horizon) if np.isfinite(pre_vol) else np.nan
            )
            spread_threshold = (
                spread_multiple * entry.spread_bps / 10_000.0
                if np.isfinite(entry.spread_bps)
                else np.nan
            )
            threshold_candidates = [
                value
                for value in [volatility_threshold, spread_threshold]
                if np.isfinite(value)
            ]
            threshold = max(threshold_candidates) if threshold_candidates else np.nan
            if not np.isfinite(pre_vol):
                threshold = np.nan
            row[f"volatility_threshold_{suffix}"] = volatility_threshold
            row[f"spread_threshold_{suffix}"] = spread_threshold
            row[f"label_threshold_{suffix}"] = threshold
            label, code = _label_from_return(label_return, threshold)
            row[f"label_{suffix}"] = label
            row[f"label_code_{suffix}"] = code

            later_clusters = _count_later_clusters(
                starts,
                pd.Timestamp(cluster.cluster_window_end_utc),
                target_time,
            )
            row[f"new_clusters_during_{suffix}"] = later_clusters
            available = bool(
                in_session
                and np.isfinite(label_return)
                and np.isfinite(threshold)
            )
            row[f"label_available_{suffix}"] = available
            row[f"clean_{suffix}"] = bool(available and later_clusters == 0)

            path_stats = _path_statistics(
                primary,
                benchmark,
                benchmark_beta,
                decision_time,
                target_time,
                entry.midpoint,
                benchmark_entry,
                threshold,
                max_quote_staleness_seconds,
            ) if in_session else {
                "max_label_return": np.nan,
                "min_label_return": np.nan,
                "minutes_to_max": np.nan,
                "minutes_to_min": np.nan,
                "barrier_label": pd.NA,
                "barrier_code": pd.NA,
                "minutes_to_barrier": np.nan,
                "path_observations": 0,
            }
            for name, value in path_stats.items():
                row[f"{name}_{suffix}"] = value

        output_rows.append(row)

    result = pd.DataFrame(output_rows).sort_values("cluster_start_utc").reset_index(drop=True)
    result["return_basis"] = (
        f"{primary_symbol.upper()}_minus_{benchmark_beta:g}x_{benchmark_symbol.upper()}"
        if benchmark is not None and benchmark_symbol is not None
        else f"raw_{primary_symbol.upper()}"
    )
    return result
def assign_chronological_split(
    frame: pd.DataFrame,
    train_fraction: float,
    validation_fraction: float,
) -> pd.DataFrame:
    """Assign whole trading dates to train, validation, and test partitions."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between zero and one")
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must sum to less than one")

    result = frame.copy()
    dates = sorted(result.session_date.astype(str).unique())
    if len(dates) < 3:
        raise ValueError("At least three trading dates are required for chronological splits")
    train_count = max(1, int(np.floor(len(dates) * train_fraction)))
    validation_count = max(1, int(np.floor(len(dates) * validation_fraction)))
    if train_count + validation_count >= len(dates):
        validation_count = 1
        train_count = len(dates) - 2

    train_dates = set(dates[:train_count])
    validation_dates = set(dates[train_count:train_count + validation_count])
    result["chronological_split"] = np.select(
        [
            result.session_date.astype(str).isin(train_dates),
            result.session_date.astype(str).isin(validation_dates),
        ],
        ["train", "validation"],
        default="test",
    )
    return result


# End of module.
