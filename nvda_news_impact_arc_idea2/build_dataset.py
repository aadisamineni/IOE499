#!/usr/bin/env python3
"""Build NVDA news-reaction, continuation, and matched-control datasets."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from math import sqrt
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from sklearn.feature_extraction.text import HashingVectorizer

from common import (
    DEFAULT_NEWS,
    DEFAULT_OUTPUT,
    atomic_csv,
    atomic_json,
    atomic_parquet,
    boolean_series,
    file_sha256,
    json_counts,
    json_list,
    parse_positive_ints,
    stable_identifier,
    timestamps_to_ns,
)


REQUIRED_NEWS_COLUMNS = {
    "article_id",
    "headline",
    "summary",
    "source",
    "published_at_utc",
    "relevance_type",
    "event_category",
    "excluded",
    "event_representative",
}

EVENT_TAG_PATTERNS: dict[str, str] = {
    "earnings_guidance": r"\bearnings\b|\beps\b|\brevenue\b|\bguidance\b|\boutlook\b|\bforecast",
    "analyst_action": r"\banalyst|price target|\bupgrade|\bdowngrade|\brating\b",
    "export_china": r"export control|export restriction|\bchina\b|\blicen[cs]e\b|\bban\b",
    "partnership_contract": r"\bpartnership|\bpartner(?:ed|ship)?\b|\bcontract\b|\bagreement\b|\bdeal\b",
    "product_launch": r"\bunveil|\blaunch|\bintroduc|\breleas|\bblackwell\b|\brubin\b|\bgpu\b",
    "supply_manufacturing": r"\bsupply|\bshortage|\bproduction|\bfoundry|\btsmc\b|\bhbm\b|\bpackaging\b",
    "competition": r"\bamd\b|\bintel\b|custom chip|\basic\b|\bcompetitor",
    "ai_infrastructure": r"data cent|\binfrastructure\b|\bcapex\b|\bcloud\b|ai factory",
    "regulatory_legal": r"\bregulat|\bantitrust|\blawsuit|\bsubpoena|\bdoj\b|\bsec\b",
    "investment_acquisition": r"\binvest|\bstake\b|\bacqui|\bfunding\b|\bfinanc",
    "demand_orders": r"\bdemand\b|\border(?:s|ed)?\b|\bbacklog\b|\bsales\b",
    "leadership": r"\bjensen\b|chief executive|\bceo\b|\bexecutive\b",
}

POSITIVE_CUES = re.compile(
    r"\b(raise[sd]?|beat[se]?|growth|strong|record|surge[sd]?|win[sn]?|expand[sd]?|approval)\b",
    flags=re.IGNORECASE,
)
NEGATIVE_CUES = re.compile(
    r"\b(cut[st]?|miss(?:es|ed)?|weak|delay(?:s|ed)?|shortage|ban(?:ned)?|probe|lawsuit|decline[sd]?)\b",
    flags=re.IGNORECASE,
)


@dataclass
class SessionContext:
    market_period: str
    event_session_date: str
    outcome_session_date: str
    market_open_utc: pd.Timestamp
    market_close_utc: pd.Timestamp
    outcome_open_utc: pd.Timestamp
    outcome_close_utc: pd.Timestamp
    previous_close_utc: pd.Timestamp | pd.NaT
    analysis_time_utc: pd.Timestamp


def make_schedule(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    calendar = mcal.get_calendar("NASDAQ")
    schedule = calendar.schedule(
        start_date=(start - pd.Timedelta(days=10)).date(),
        end_date=(end + pd.Timedelta(days=10)).date(),
    ).reset_index()
    schedule = schedule.rename(columns={schedule.columns[0]: "session_index"})
    schedule["session_date"] = pd.to_datetime(schedule.session_index).dt.strftime(
        "%Y-%m-%d"
    )
    schedule["market_open"] = pd.to_datetime(schedule.market_open, utc=True)
    schedule["market_close"] = pd.to_datetime(schedule.market_close, utc=True)
    schedule["previous_close"] = schedule.market_close.shift(1)
    return schedule[
        ["session_date", "market_open", "market_close", "previous_close"]
    ].reset_index(drop=True)


class SessionMapper:
    """Map arbitrary news timestamps into tradable NASDAQ contexts."""

    def __init__(self, schedule: pd.DataFrame):
        self.schedule = schedule.reset_index(drop=True)
        self.by_date = {
            str(row.session_date): index
            for index, row in self.schedule.iterrows()
        }
        self.opens_ns = timestamps_to_ns(self.schedule.market_open)

    def _next_session_index(self, timestamp: pd.Timestamp) -> int | None:
        index = int(np.searchsorted(self.opens_ns, timestamp.value, side="left"))
        return index if index < len(self.schedule) else None

    def context(self, value: object) -> SessionContext | None:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        local = timestamp.tz_convert("America/New_York")
        local_date = local.strftime("%Y-%m-%d")
        minute_of_day = local.hour * 60 + local.minute
        same_index = self.by_date.get(local_date)

        period = "overnight"
        event_index: int | None = None
        outcome_index: int | None = None
        analysis_time = timestamp
        if same_index is not None:
            row = self.schedule.iloc[same_index]
            market_open = pd.Timestamp(row.market_open)
            market_close = pd.Timestamp(row.market_close)
            if market_open <= timestamp < market_close:
                period = "regular"
                event_index = same_index
                outcome_index = same_index
            elif 240 <= minute_of_day < 570 and timestamp < market_open:
                period = "premarket"
                event_index = same_index
                outcome_index = same_index
            elif market_close <= timestamp and minute_of_day < 1200:
                period = "afterhours"
                event_index = same_index
                outcome_index = same_index + 1
            else:
                outcome_index = self._next_session_index(timestamp)
        else:
            outcome_index = self._next_session_index(timestamp)

        if outcome_index is None or outcome_index >= len(self.schedule):
            return None
        if event_index is None:
            event_index = outcome_index
            analysis_time = pd.Timestamp(
                self.schedule.iloc[outcome_index].market_open
            )
        if period == "afterhours" and outcome_index >= len(self.schedule):
            return None

        event_row = self.schedule.iloc[event_index]
        outcome_row = self.schedule.iloc[outcome_index]
        return SessionContext(
            market_period=period,
            event_session_date=str(event_row.session_date),
            outcome_session_date=str(outcome_row.session_date),
            market_open_utc=pd.Timestamp(event_row.market_open),
            market_close_utc=pd.Timestamp(event_row.market_close),
            outcome_open_utc=pd.Timestamp(outcome_row.market_open),
            outcome_close_utc=pd.Timestamp(outcome_row.market_close),
            previous_close_utc=pd.Timestamp(outcome_row.previous_close),
            analysis_time_utc=analysis_time,
        )


def prepare_articles(
    news_path: Path,
    relevance_types: set[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    news = pd.read_csv(news_path, low_memory=False, keep_default_na=False)
    missing = REQUIRED_NEWS_COLUMNS.difference(news.columns)
    if missing:
        raise ValueError(f"News input is missing columns: {sorted(missing)}")
    published = pd.to_datetime(news.published_at_utc, utc=True, errors="coerce")
    valid_time = published.notna()
    excluded = boolean_series(news.excluded)
    representative = boolean_series(news.event_representative)
    relevance = news.relevance_type.astype(str).str.lower().isin(relevance_types)
    retained = news.loc[valid_time & ~excluded & representative & relevance].copy()
    retained["published_at_utc"] = published.loc[retained.index]
    retained["headline"] = retained.headline.fillna("").astype(str).str.strip()
    retained["summary"] = retained.summary.fillna("").astype(str).str.strip()
    retained["combined_text"] = retained.apply(
        lambda row: (
            f"{row.headline}. {row.summary}"
            if row.headline and row.summary
            else row.headline or row.summary
        ),
        axis=1,
    )
    retained = retained.loc[retained.combined_text.str.len().gt(0)].copy()
    retained = retained.sort_values(
        ["published_at_utc", "article_id"]
    ).reset_index(drop=True)
    audit = {
        "input_articles": len(news),
        "invalid_publication_timestamps": int((~valid_time).sum()),
        "excluded_articles": int(excluded.sum()),
        "nonrepresentative_articles": int((~representative).sum()),
        "outside_requested_relevance": int((~relevance).sum()),
        "retained_articles": len(retained),
        "relevance_types": sorted(relevance_types),
    }
    return retained, audit


def attach_contexts(
    articles: pd.DataFrame,
    mapper: SessionMapper,
) -> tuple[pd.DataFrame, int]:
    rows: list[dict[str, object]] = []
    unavailable = 0
    for article in articles.to_dict(orient="records"):
        context = mapper.context(article["published_at_utc"])
        if context is None:
            unavailable += 1
            continue
        rows.append({**article, **asdict(context)})
    return pd.DataFrame(rows), unavailable


def _single_event(row: pd.Series) -> dict[str, object]:
    duplicate_size = pd.to_numeric(
        pd.Series([row.get("duplicate_group_size", 1)]), errors="coerce"
    ).fillna(1).iloc[0]
    return {
        "event_id": stable_identifier(
            "event", [row.article_id, row.published_at_utc]
        ),
        "article_id": str(row.article_id),
        "article_ids_json": json_list([row.article_id]),
        "published_at_utc": row.published_at_utc,
        "last_article_utc": row.published_at_utc,
        "analysis_time_utc": row.analysis_time_utc,
        "market_period": row.market_period,
        "event_session_date": row.event_session_date,
        "outcome_session_date": row.outcome_session_date,
        "market_open_utc": row.market_open_utc,
        "market_close_utc": row.market_close_utc,
        "outcome_open_utc": row.outcome_open_utc,
        "outcome_close_utc": row.outcome_close_utc,
        "previous_close_utc": row.previous_close_utc,
        "headline": row.headline,
        "summary": row.summary,
        "combined_text": row.combined_text,
        "source": str(row.source),
        "sources_json": json_list([row.source]),
        "event_category": str(row.event_category or "uncategorized"),
        "event_categories_json": json_list([row.event_category]),
        "relevance_type": str(row.relevance_type),
        "article_count": 1,
        "duplicate_group_size_total": int(duplicate_size),
    }


def build_events(contextual_articles: pd.DataFrame) -> pd.DataFrame:
    """Keep tradable-time articles separate and batch closed-market information."""
    rows: list[dict[str, object]] = []
    live = contextual_articles.loc[
        ~contextual_articles.market_period.eq("overnight")
    ]
    rows.extend(_single_event(row) for _, row in live.iterrows())

    overnight = contextual_articles.loc[
        contextual_articles.market_period.eq("overnight")
    ]
    for outcome_date, group in overnight.groupby("outcome_session_date", sort=True):
        group = group.sort_values(["published_at_utc", "article_id"])
        first = group.iloc[0]
        categories = sorted(
            {
                str(value).strip()
                for value in group.event_category
                if str(value).strip()
            }
        )
        sources = sorted({str(value).strip() for value in group.source if str(value).strip()})
        duplicate_sizes = pd.to_numeric(
            group.get("duplicate_group_size", pd.Series(1, index=group.index)),
            errors="coerce",
        ).fillna(1)
        rows.append(
            {
                "event_id": stable_identifier(
                    "overnight",
                    [outcome_date, *group.article_id.astype(str).tolist()],
                ),
                "article_id": "",
                "article_ids_json": json_list(group.article_id),
                "published_at_utc": group.published_at_utc.min(),
                "last_article_utc": group.published_at_utc.max(),
                "analysis_time_utc": first.analysis_time_utc,
                "market_period": "overnight",
                "event_session_date": first.event_session_date,
                "outcome_session_date": outcome_date,
                "market_open_utc": first.market_open_utc,
                "market_close_utc": first.market_close_utc,
                "outcome_open_utc": first.outcome_open_utc,
                "outcome_close_utc": first.outcome_close_utc,
                "previous_close_utc": first.previous_close_utc,
                "headline": " | ".join(group.headline.astype(str)),
                "summary": "\n".join(group.summary.astype(str)),
                "combined_text": "\n".join(group.combined_text.astype(str)),
                "source": sources[0] if len(sources) == 1 else "multiple",
                "sources_json": json.dumps(sources, separators=(",", ":")),
                "event_category": categories[0] if len(categories) == 1 else "multiple",
                "event_categories_json": json.dumps(categories, separators=(",", ":")),
                "relevance_type": "direct",
                "article_count": len(group),
                "duplicate_group_size_total": int(duplicate_sizes.sum()),
            }
        )

    events = pd.DataFrame(rows).sort_values(
        ["analysis_time_utc", "published_at_utc", "event_id"]
    ).reset_index(drop=True)
    if events.event_id.duplicated().any():
        raise ValueError("Generated duplicate event identifiers")
    return events


def add_online_news_features(events: pd.DataFrame) -> pd.DataFrame:
    """Features use only events available strictly before the current event."""
    result = events.sort_values(
        ["analysis_time_utc", "published_at_utc", "event_id"]
    ).reset_index(drop=True).copy()
    times = pd.to_datetime(result.analysis_time_utc, utc=True)
    times_ns = timestamps_to_ns(times)
    for minutes in (15, 60, 1_440):
        values = np.zeros(len(result), dtype=int)
        width = pd.Timedelta(minutes=minutes).value
        for index, timestamp in enumerate(times_ns):
            left = int(np.searchsorted(times_ns, timestamp - width, side="left"))
            right = int(np.searchsorted(times_ns, timestamp, side="left"))
            values[index] = max(0, right - left)
        suffix = "24h" if minutes == 1_440 else f"{minutes}m"
        result[f"prior_news_count_{suffix}"] = values

    elapsed = np.full(len(result), np.nan)
    for index, timestamp in enumerate(times_ns):
        previous = int(np.searchsorted(times_ns, timestamp, side="left")) - 1
        if previous >= 0:
            elapsed[index] = (timestamp - times_ns[previous]) / 60_000_000_000
    result["minutes_since_prior_event"] = elapsed
    result["text_word_count"] = result.combined_text.str.split().str.len().astype(int)
    result["text_char_count"] = result.combined_text.str.len().astype(int)
    result["contains_number"] = result.combined_text.str.contains(
        r"\d", regex=True
    ).astype(int)
    result["contains_percent"] = result.combined_text.str.contains(
        r"%|percent", regex=True, case=False
    ).astype(int)
    result["contains_dollar"] = result.combined_text.str.contains(
        r"\$|\bdollars?\b", regex=True, case=False
    ).astype(int)
    tag_columns: list[str] = []
    for tag, pattern in EVENT_TAG_PATTERNS.items():
        column = f"tag_{tag}"
        result[column] = result.combined_text.str.contains(
            pattern, regex=True, case=False
        ).astype(int)
        tag_columns.append(column)
    result["semantic_tag_count"] = result[tag_columns].sum(axis=1).astype(int)
    result["positive_cue_count"] = result.combined_text.map(
        lambda value: len(POSITIVE_CUES.findall(str(value)))
    )
    result["negative_cue_count"] = result.combined_text.map(
        lambda value: len(NEGATIVE_CUES.findall(str(value)))
    )
    inferred_types: list[str] = []
    for row in result.itertuples(index=False):
        existing = str(row.event_category).strip()
        if existing and existing not in {"other_nvidia_development", "multiple"}:
            inferred_types.append(existing)
            continue
        matched = [
            tag
            for tag in EVENT_TAG_PATTERNS
            if int(getattr(row, f"tag_{tag}")) == 1
        ]
        inferred_types.append(matched[0] if matched else "other")
    result["fine_event_type"] = inferred_types

    vectorizer = HashingVectorizer(
        n_features=2**14,
        alternate_sign=False,
        norm="l2",
        stop_words="english",
        ngram_range=(1, 2),
    )
    matrix = vectorizer.transform(result.combined_text.fillna("").astype(str))
    max_similarity = np.zeros(len(result), dtype=float)
    mean_similarity = np.zeros(len(result), dtype=float)
    lookback_ns = pd.Timedelta(days=30).value
    for index, timestamp in enumerate(times_ns):
        left = int(np.searchsorted(times_ns, timestamp - lookback_ns, side="left"))
        right = int(np.searchsorted(times_ns, timestamp, side="left"))
        if right <= left:
            continue
        similarities = matrix[left:right].dot(matrix[index].T).toarray().ravel()
        if len(similarities):
            max_similarity[index] = float(np.max(similarities))
            mean_similarity[index] = float(np.mean(similarities))
    result["max_similarity_prior_30d"] = max_similarity
    result["mean_similarity_prior_30d"] = mean_similarity
    result["novelty_30d"] = 1.0 - max_similarity
    return result


class MarketLookup:
    def __init__(self, market: pd.DataFrame):
        required = {
            "bar_start_utc",
            "minute_end_utc",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "trade_count",
            "vwap",
            "price",
        }
        missing = required.difference(market.columns)
        if missing:
            raise ValueError(f"Market cache is missing columns: {sorted(missing)}")
        frame = market.copy()
        frame["bar_start_utc"] = pd.to_datetime(frame.bar_start_utc, utc=True)
        frame["minute_end_utc"] = pd.to_datetime(frame.minute_end_utc, utc=True)
        self.frame = frame.sort_values("minute_end_utc").reset_index(drop=True)
        self.ends_ns = timestamps_to_ns(self.frame.minute_end_utc)
        self.starts_ns = timestamps_to_ns(self.frame.bar_start_utc)

    def asof_index(
        self,
        timestamp: pd.Timestamp,
        max_staleness_seconds: float = 120.0,
    ) -> int | None:
        value = pd.Timestamp(timestamp).value
        index = int(np.searchsorted(self.ends_ns, value, side="right") - 1)
        if index < 0:
            return None
        staleness = (value - self.ends_ns[index]) / 1e9
        if staleness < 0 or staleness > max_staleness_seconds:
            return None
        return index

    def exact_start_index(self, timestamp: pd.Timestamp) -> int | None:
        value = pd.Timestamp(timestamp).value
        index = int(np.searchsorted(self.starts_ns, value, side="left"))
        if index >= len(self.starts_ns) or self.starts_ns[index] != value:
            return None
        return index

    def window_indices(
        self,
        start_exclusive: pd.Timestamp,
        end_inclusive: pd.Timestamp,
    ) -> tuple[int, int]:
        left = int(
            np.searchsorted(
                self.ends_ns, pd.Timestamp(start_exclusive).value, side="right"
            )
        )
        right = int(
            np.searchsorted(
                self.ends_ns, pd.Timestamp(end_inclusive).value, side="right"
            )
        )
        return left, right

    def window(
        self,
        start_exclusive: pd.Timestamp,
        end_inclusive: pd.Timestamp,
    ) -> pd.DataFrame:
        left, right = self.window_indices(start_exclusive, end_inclusive)
        return self.frame.iloc[left:right]


def _log_return(end_price: float, start_price: float) -> float:
    if not (
        np.isfinite(end_price)
        and np.isfinite(start_price)
        and end_price > 0
        and start_price > 0
    ):
        return np.nan
    return float(np.log(end_price / start_price))


def _price(row: pd.Series) -> float:
    value = float(row.get("price", np.nan))
    return value if np.isfinite(value) and value > 0 else float(row.close)


def _pre_event_features(
    event: pd.Series,
    lookup: MarketLookup,
    anchor_time: pd.Timestamp,
    anchor_index: int,
) -> dict[str, object]:
    anchor_row = lookup.frame.iloc[anchor_index]
    anchor_price = _price(anchor_row)
    result: dict[str, object] = {
        "anchor_price": anchor_price,
        "anchor_market_time_utc": anchor_row.minute_end_utc,
        "pre_spread_bps": float(anchor_row.get("spread_bps", np.nan)),
        "pre_quote_updates": float(anchor_row.get("valid_quote_updates", np.nan)),
        "pre_bid_size": float(anchor_row.get("bid_size", np.nan)),
        "pre_ask_size": float(anchor_row.get("ask_size", np.nan)),
        "pre_bar_volume": float(anchor_row.volume),
        "pre_bar_trade_count": float(anchor_row.trade_count),
    }
    for minutes in (1, 5, 15, 30):
        prior_index = lookup.asof_index(
            anchor_time - pd.Timedelta(minutes=minutes),
            max_staleness_seconds=120.0,
        )
        result[f"prior_return_{minutes}m"] = (
            _log_return(anchor_price, _price(lookup.frame.iloc[prior_index]))
            if prior_index is not None
            else np.nan
        )

    history_end = anchor_time
    history_start = anchor_time - pd.Timedelta(minutes=31)
    history = lookup.window(history_start, history_end)
    if len(history) < 10 and pd.notna(event.previous_close_utc):
        previous_close = pd.Timestamp(event.previous_close_utc)
        history = lookup.window(previous_close - pd.Timedelta(minutes=31), previous_close)
    prices = history.price.fillna(history.close).to_numpy(dtype=float)
    times = timestamps_to_ns(pd.to_datetime(history.minute_end_utc, utc=True))
    if len(prices) >= 2:
        returns = np.diff(np.log(prices))
        gaps = np.diff(times) / 60_000_000_000
        valid = np.isfinite(returns) & (gaps > 0) & (gaps <= 2)
        usable = returns[valid]
    else:
        usable = np.array([], dtype=float)
    result["pre_volatility_30m"] = (
        float(np.std(usable, ddof=1)) if len(usable) >= 8 else np.nan
    )
    result["pre_volatility_observations"] = int(len(usable))
    result["prior_volume_mean_30m"] = (
        float(history.volume.mean()) if len(history) else np.nan
    )
    result["prior_trade_count_mean_30m"] = (
        float(history.trade_count.mean()) if len(history) else np.nan
    )
    result["log_prior_volume_mean_30m"] = np.log1p(
        result["prior_volume_mean_30m"]
    )
    result["log_prior_trade_count_mean_30m"] = np.log1p(
        result["prior_trade_count_mean_30m"]
    )
    minutes_from_open = (
        anchor_time - pd.Timestamp(event.market_open_utc)
    ).total_seconds() / 60.0
    result["minutes_from_open"] = float(minutes_from_open)
    result["session_progress"] = float(minutes_from_open / 390.0)
    local = anchor_time.tz_convert("America/New_York")
    minute_of_day = local.hour * 60 + local.minute
    result["minute_of_day_sin"] = float(np.sin(2 * np.pi * minute_of_day / 1_440))
    result["minute_of_day_cos"] = float(np.cos(2 * np.pi * minute_of_day / 1_440))
    return result


def label_event(
    event: pd.Series,
    lookup: MarketLookup,
    horizons: Iterable[int],
    decision_delays: Iterable[int],
    z_cutoff: float,
    spread_multiple: float,
) -> dict[str, object]:
    result = event.to_dict()
    analysis_time = pd.Timestamp(event.analysis_time_utc)
    is_overnight = str(event.market_period) == "overnight"
    previous_close_price = np.nan
    feature_time = analysis_time
    feature_index: int | None = None

    if is_overnight:
        anchor_time = pd.Timestamp(event.outcome_open_utc)
        anchor_index = lookup.exact_start_index(anchor_time)
        if anchor_index is None:
            result["market_data_available"] = False
            return result
        anchor_price = float(lookup.frame.iloc[anchor_index].open)
        first_minute_end = anchor_time + pd.Timedelta(minutes=1)
        feature_time = anchor_time
        feature_index = lookup.asof_index(anchor_time, max_staleness_seconds=300.0)
        if pd.notna(event.previous_close_utc):
            previous_index = lookup.asof_index(
                pd.Timestamp(event.previous_close_utc), max_staleness_seconds=120.0
            )
            if previous_index is not None:
                previous_close_price = _price(lookup.frame.iloc[previous_index])
                if feature_index is None:
                    feature_index = previous_index
                    feature_time = pd.Timestamp(event.previous_close_utc)
    else:
        anchor_time = analysis_time
        anchor_index = lookup.asof_index(anchor_time, max_staleness_seconds=300.0)
        if anchor_index is None:
            result["market_data_available"] = False
            return result
        anchor_price = _price(lookup.frame.iloc[anchor_index])
        first_minute_end = analysis_time.floor("min") + pd.Timedelta(minutes=1)
        feature_index = anchor_index

    result["market_data_available"] = True
    if feature_index is None:
        result["market_data_available"] = False
        return result
    result.update(_pre_event_features(event, lookup, feature_time, feature_index))
    result["anchor_price"] = anchor_price
    result["overnight_gap_return"] = (
        _log_return(anchor_price, previous_close_price) if is_overnight else np.nan
    )
    result["previous_close_price"] = previous_close_price
    pre_volatility = float(result.get("pre_volatility_30m", np.nan))
    spread_bps = float(result.get("pre_spread_bps", np.nan))

    endpoint_prices: dict[int, float] = {}
    for horizon in horizons:
        endpoint = first_minute_end + pd.Timedelta(minutes=horizon - 1)
        endpoint_index = lookup.asof_index(endpoint, max_staleness_seconds=120.0)
        available = endpoint_index is not None
        end_price = (
            _price(lookup.frame.iloc[endpoint_index]) if available else np.nan
        )
        endpoint_prices[horizon] = end_price
        post_anchor_return = _log_return(end_price, anchor_price)
        total_return = (
            _log_return(end_price, previous_close_price)
            if is_overnight
            else post_anchor_return
        )
        reaction = total_return
        result[f"endpoint_{horizon}m_utc"] = endpoint
        result[f"label_available_{horizon}m"] = bool(available)
        result[f"reaction_return_{horizon}m"] = reaction
        result[f"absolute_reaction_return_{horizon}m"] = abs(reaction)
        result[f"post_anchor_return_{horizon}m"] = post_anchor_return
        result[f"total_return_including_gap_{horizon}m"] = total_return

        path_start = (
            anchor_time
            if is_overnight
            else lookup.frame.iloc[anchor_index].minute_end_utc
        )
        path = lookup.window(path_start, endpoint)
        if len(path):
            result[f"max_return_{horizon}m"] = float(
                np.max(np.log(path.high.to_numpy(dtype=float) / anchor_price))
            )
            result[f"min_return_{horizon}m"] = float(
                np.min(np.log(path.low.to_numpy(dtype=float) / anchor_price))
            )
            result[f"post_volume_{horizon}m"] = float(path.volume.sum())
            expected_volume = float(result.get("prior_volume_mean_30m", np.nan)) * len(path)
            result[f"volume_ratio_{horizon}m"] = (
                float(path.volume.sum() / expected_volume)
                if np.isfinite(expected_volume) and expected_volume > 0
                else np.nan
            )
        else:
            result[f"max_return_{horizon}m"] = np.nan
            result[f"min_return_{horizon}m"] = np.nan
            result[f"post_volume_{horizon}m"] = np.nan
            result[f"volume_ratio_{horizon}m"] = np.nan

        volatility_threshold = (
            z_cutoff * pre_volatility * sqrt(horizon)
            if np.isfinite(pre_volatility)
            else np.nan
        )
        spread_threshold = (
            spread_multiple * spread_bps / 10_000.0
            if np.isfinite(spread_bps)
            else 0.0
        )
        threshold = (
            max(volatility_threshold, spread_threshold)
            if np.isfinite(volatility_threshold)
            else np.nan
        )
        result[f"impact_threshold_{horizon}m"] = threshold
        result[f"standardized_absolute_return_{horizon}m"] = (
            abs(reaction) / threshold
            if np.isfinite(reaction) and np.isfinite(threshold) and threshold > 0
            else np.nan
        )
        result[f"impact_label_{horizon}m"] = (
            int(abs(reaction) >= threshold)
            if np.isfinite(reaction) and np.isfinite(threshold)
            else pd.NA
        )
        result[f"direction_label_{horizon}m"] = (
            int(reaction > 0) if np.isfinite(reaction) and reaction != 0 else pd.NA
        )
        result[f"material_direction_label_{horizon}m"] = (
            int(reaction > 0)
            if (
                np.isfinite(reaction)
                and np.isfinite(threshold)
                and abs(reaction) >= threshold
            )
            else pd.NA
        )

    for delay in decision_delays:
        decision_time = first_minute_end + pd.Timedelta(minutes=delay - 1)
        decision_index = lookup.asof_index(decision_time, max_staleness_seconds=120.0)
        decision_price = (
            _price(lookup.frame.iloc[decision_index])
            if decision_index is not None
            else np.nan
        )
        result[f"decision_time_{delay}m_utc"] = decision_time
        result[f"decision_price_{delay}m"] = decision_price
        result[f"initial_return_{delay}m"] = _log_return(decision_price, anchor_price)
        for horizon in horizons:
            if horizon <= delay:
                continue
            continuation = _log_return(endpoint_prices[horizon], decision_price)
            result[f"continuation_return_{delay}m_to_{horizon}m"] = continuation
            result[f"continuation_direction_{delay}m_to_{horizon}m"] = (
                int(continuation > 0)
                if np.isfinite(continuation) and continuation != 0
                else pd.NA
            )
            initial = float(result[f"initial_return_{delay}m"])
            result[f"continuation_same_sign_{delay}m_to_{horizon}m"] = (
                int(np.sign(initial) == np.sign(continuation))
                if np.isfinite(initial)
                and initial != 0
                and np.isfinite(continuation)
                and continuation != 0
                else pd.NA
            )
    return result


def add_news_contamination(
    events: pd.DataFrame,
    article_times: pd.Series,
    horizons: Iterable[int],
) -> pd.DataFrame:
    result = events.copy()
    times_ns = timestamps_to_ns(pd.to_datetime(article_times, utc=True).sort_values())
    for horizon in horizons:
        counts: list[int] = []
        for row in result.itertuples(index=False):
            lower = pd.Timestamp(row.analysis_time_utc).value
            endpoint = getattr(row, f"endpoint_{horizon}m_utc", pd.NaT)
            if pd.isna(endpoint):
                counts.append(0)
                continue
            upper = pd.Timestamp(endpoint).value
            left = int(np.searchsorted(times_ns, lower, side="right"))
            right = int(np.searchsorted(times_ns, upper, side="right"))
            counts.append(max(0, right - left))
        result[f"later_news_count_{horizon}m"] = counts
        result[f"clean_{horizon}m"] = np.array(counts) == 0
    return result


def assign_chronological_split(
    frame: pd.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> pd.DataFrame:
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("Training and validation fractions must be positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Training plus validation fractions must be below one")
    dates = sorted(frame.outcome_session_date.astype(str).unique())
    if len(dates) < 3:
        raise ValueError("At least three outcome sessions are required")
    train_end = max(1, min(len(dates) - 2, int(np.floor(len(dates) * train_fraction))))
    validation_end = max(
        train_end + 1,
        min(
            len(dates) - 1,
            int(np.floor(len(dates) * (train_fraction + validation_fraction))),
        ),
    )
    split_by_date = {
        date: (
            "train"
            if index < train_end
            else "validation"
            if index < validation_end
            else "test"
        )
        for index, date in enumerate(dates)
    }
    result = frame.copy()
    result["chronological_split"] = result.outcome_session_date.astype(str).map(
        split_by_date
    )
    return result


def build_controls(
    events: pd.DataFrame,
    article_times: pd.Series,
    schedule: pd.DataFrame,
    lookup: MarketLookup,
    horizons: list[int],
    decision_delays: list[int],
    z_cutoff: float,
    spread_multiple: float,
    interval_minutes: int,
    exclusion_minutes: int,
) -> pd.DataFrame:
    event_dates = set(events.outcome_session_date.astype(str))
    times_ns = timestamps_to_ns(pd.to_datetime(article_times, utc=True).sort_values())
    maximum_horizon = max(horizons)
    rows: list[dict[str, object]] = []
    for schedule_row in schedule.itertuples(index=False):
        session_date = str(schedule_row.session_date)
        if session_date not in event_dates:
            continue
        market_open = pd.Timestamp(schedule_row.market_open)
        market_close = pd.Timestamp(schedule_row.market_close)
        start = market_open + pd.Timedelta(minutes=30)
        stop = market_close - pd.Timedelta(minutes=maximum_horizon)
        for anchor_time in pd.date_range(
            start, stop, freq=f"{interval_minutes}min", inclusive="both"
        ):
            width = pd.Timedelta(minutes=exclusion_minutes).value
            left = int(np.searchsorted(times_ns, anchor_time.value - width, side="left"))
            right = int(np.searchsorted(times_ns, anchor_time.value + width, side="right"))
            if right > left:
                continue
            event = pd.Series(
                {
                    "event_id": stable_identifier("control", [anchor_time]),
                    "article_id": "",
                    "article_ids_json": "[]",
                    "published_at_utc": anchor_time,
                    "last_article_utc": anchor_time,
                    "analysis_time_utc": anchor_time,
                    "market_period": "control_regular",
                    "event_session_date": session_date,
                    "outcome_session_date": session_date,
                    "market_open_utc": market_open,
                    "market_close_utc": market_close,
                    "outcome_open_utc": market_open,
                    "outcome_close_utc": market_close,
                    "previous_close_utc": schedule_row.previous_close,
                    "headline": "",
                    "summary": "",
                    "combined_text": "",
                    "source": "none",
                    "sources_json": "[]",
                    "event_category": "no_news_control",
                    "event_categories_json": "[]",
                    "relevance_type": "none",
                    "article_count": 0,
                    "duplicate_group_size_total": 0,
                    "prior_news_count_15m": 0,
                    "prior_news_count_60m": 0,
                    "prior_news_count_24h": 0,
                    "minutes_since_prior_event": np.nan,
                    "text_word_count": 0,
                    "text_char_count": 0,
                    "contains_number": 0,
                    "contains_percent": 0,
                    "contains_dollar": 0,
                    "max_similarity_prior_30d": 0.0,
                    "mean_similarity_prior_30d": 0.0,
                    "novelty_30d": 1.0,
                }
            )
            rows.append(
                label_event(
                    event,
                    lookup,
                    horizons,
                    decision_delays,
                    z_cutoff,
                    spread_multiple,
                )
            )
    if not rows:
        return pd.DataFrame()
    controls = pd.DataFrame(rows)
    controls = controls.loc[controls.market_data_available.fillna(False)].copy()
    return assign_chronological_split(controls)


def event_study_summary(events: pd.DataFrame, horizons: Iterable[int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groupings = [
        ("overall", []),
        ("market_period", ["market_period"]),
        ("event_category", ["event_category"]),
        ("fine_event_type", ["fine_event_type"]),
        ("period_category", ["market_period", "event_category"]),
    ]
    for grouping_name, columns in groupings:
        grouped: Iterable[tuple[object, pd.DataFrame]]
        if columns:
            grouped = events.groupby(columns, dropna=False, sort=True)
        else:
            grouped = [("all", events)]
        for key, group in grouped:
            key_values = key if isinstance(key, tuple) else (key,)
            identifiers = {column: value for column, value in zip(columns, key_values)}
            for horizon in horizons:
                returns = pd.to_numeric(
                    group[f"reaction_return_{horizon}m"], errors="coerce"
                ).dropna()
                impacts = pd.to_numeric(
                    group[f"impact_label_{horizon}m"], errors="coerce"
                ).dropna()
                rows.append(
                    {
                        "grouping": grouping_name,
                        "horizon_minutes": horizon,
                        **identifiers,
                        "events": len(group),
                        "available_returns": len(returns),
                        "mean_return": float(returns.mean()) if len(returns) else np.nan,
                        "median_return": float(returns.median()) if len(returns) else np.nan,
                        "mean_absolute_return": (
                            float(returns.abs().mean()) if len(returns) else np.nan
                        ),
                        "positive_rate": (
                            float(returns.gt(0).mean()) if len(returns) else np.nan
                        ),
                        "impact_rate": (
                            float(impacts.mean()) if len(impacts) else np.nan
                        ),
                    }
                )
    return pd.DataFrame(rows)


def build_dataset(
    news_path: Path,
    market_path: Path,
    output_dir: Path,
    relevance_types: set[str],
    horizons: list[int],
    decision_delays: list[int],
    z_cutoff: float,
    spread_multiple: float,
    control_interval_minutes: int,
    control_exclusion_minutes: int,
    write_csv: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    articles, news_audit = prepare_articles(news_path, relevance_types)
    schedule = make_schedule(
        articles.published_at_utc.min(), articles.published_at_utc.max()
    )
    contextual, unmapped = attach_contexts(articles, SessionMapper(schedule))
    events = add_online_news_features(build_events(contextual))

    market = pd.read_parquet(market_path)
    lookup = MarketLookup(market)
    labeled_rows = [
        label_event(
            row,
            lookup,
            horizons,
            decision_delays,
            z_cutoff,
            spread_multiple,
        )
        for _, row in events.iterrows()
    ]
    events = pd.DataFrame(labeled_rows)
    events = add_news_contamination(events, articles.published_at_utc, horizons)
    events = assign_chronological_split(events)

    controls = build_controls(
        events,
        articles.published_at_utc,
        schedule,
        lookup,
        horizons,
        decision_delays,
        z_cutoff,
        spread_multiple,
        control_interval_minutes,
        control_exclusion_minutes,
    )
    summary = event_study_summary(events, horizons)

    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / "news_events.parquet"
    controls_path = output_dir / "no_news_controls.parquet"
    atomic_parquet(events, events_path)
    atomic_parquet(controls, controls_path)
    atomic_csv(summary, output_dir / "event_study_summary.csv")
    if write_csv:
        atomic_csv(events, output_dir / "news_events.csv")
        atomic_csv(controls, output_dir / "no_news_controls.csv")

    audit: dict[str, object] = {
        "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective": "Measure immediate and post-decision NVDA responses to Nvidia news.",
        "inputs": {
            "news": {
                "path": str(news_path.resolve()),
                "sha256": file_sha256(news_path),
            },
            "market": {
                "path": str(market_path.resolve()),
                "sha256": file_sha256(market_path),
            },
        },
        "settings": {
            "relevance_types": sorted(relevance_types),
            "horizons_minutes": horizons,
            "decision_delays_minutes": decision_delays,
            "z_cutoff": z_cutoff,
            "spread_multiple": spread_multiple,
            "control_interval_minutes": control_interval_minutes,
            "control_exclusion_minutes": control_exclusion_minutes,
            "overnight_policy": (
                "batch all closed-market articles by their next NASDAQ session; "
                "use the official opening price as the reaction anchor"
            ),
        },
        "news_audit": {
            **news_audit,
            "unmapped_articles": unmapped,
            "market_period_article_counts": json_counts(contextual.market_period),
        },
        "events": {
            "rows": len(events),
            "market_data_available": int(
                events.market_data_available.fillna(False).sum()
            ),
            "market_period_counts": json_counts(events.market_period),
            "split_counts": json_counts(events.chronological_split),
            "date_range": [
                str(events.outcome_session_date.min()),
                str(events.outcome_session_date.max()),
            ],
        },
        "controls": {
            "rows": len(controls),
            "split_counts": (
                json_counts(controls.chronological_split) if len(controls) else {}
            ),
        },
        "outputs": {
            "events": str(events_path.resolve()),
            "controls": str(controls_path.resolve()),
            "event_study": str((output_dir / "event_study_summary.csv").resolve()),
        },
        "limitations": [
            "Provider publication timestamps do not prove first tradable availability.",
            "Immediate reaction returns are event-study outcomes, not executable entries.",
            "Overnight batches combine all accumulated information before the next open.",
            "Later-news flags are diagnostic and are not known at the original decision time.",
        ],
    }
    atomic_json(audit, output_dir / "dataset_audit.json")
    return events, controls, audit


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--news", type=Path, default=DEFAULT_NEWS)
    parser.add_argument(
        "--market",
        type=Path,
        default=DEFAULT_OUTPUT / "market_minutes.parquet",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--include-indirect", action="store_true")
    parser.add_argument("--horizons", default="1,5,15,30,60")
    parser.add_argument("--decision-delays", default="1,5")
    parser.add_argument("--z-cutoff", type=float, default=1.0)
    parser.add_argument("--spread-multiple", type=float, default=1.0)
    parser.add_argument("--control-interval-minutes", type=int, default=5)
    parser.add_argument("--control-exclusion-minutes", type=int, default=60)
    parser.add_argument("--write-csv", action="store_true")
    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    if not args.news.is_file():
        parser.error(f"News file does not exist: {args.news}")
    if not args.market.is_file():
        parser.error(
            f"Market cache does not exist: {args.market}. Run market_data.py first."
        )
    try:
        horizons = parse_positive_ints(args.horizons)
        decisions = parse_positive_ints(args.decision_delays)
    except ValueError as exc:
        parser.error(str(exc))
    if args.z_cutoff <= 0 or args.spread_multiple < 0:
        parser.error("Threshold settings are invalid")
    if args.control_interval_minutes < 1 or args.control_exclusion_minutes < 0:
        parser.error("Control interval must be positive and exclusion cannot be negative")
    relevance_types = {"direct", "indirect"} if args.include_indirect else {"direct"}
    _, _, audit = build_dataset(
        news_path=args.news,
        market_path=args.market,
        output_dir=args.output_dir,
        relevance_types=relevance_types,
        horizons=horizons,
        decision_delays=decisions,
        z_cutoff=args.z_cutoff,
        spread_multiple=args.spread_multiple,
        control_interval_minutes=args.control_interval_minutes,
        control_exclusion_minutes=args.control_exclusion_minutes,
        write_csv=args.write_csv,
    )
    print(json.dumps({"status": "complete", **audit["events"]}, indent=2))


if __name__ == "__main__":
    main()
