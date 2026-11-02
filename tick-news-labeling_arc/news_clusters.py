"""Filter historical news and form fixed, non-chaining event clusters."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import pandas as pd
import pandas_market_calendars as mcal


REQUIRED_NEWS_COLUMNS = {
    "article_id",
    "headline",
    "summary",
    "source",
    "published_at_utc",
    "relevance_type",
    "event_category",
    "excluded",
    "duplicate_group_id",
    "event_representative",
}


@dataclass
class NewsAudit:
    input_articles: int
    invalid_publication_timestamps: int
    excluded_articles: int
    nonrepresentative_articles: int
    outside_requested_relevance: int
    outside_requested_dates: int
    outside_regular_session: int
    retained_articles: int
    event_clusters: int


def _boolean_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _article_text(row: pd.Series) -> str:
    headline = str(row.get("headline", "")).strip()
    summary = str(row.get("summary", "")).strip()
    if headline and summary:
        return f"{headline}. {summary}"
    return headline or summary


def _json_unique(values: Iterable[object]) -> str:
    cleaned = sorted({str(value).strip() for value in values if str(value).strip()})
    return json.dumps(cleaned, separators=(",", ":"))


def _cluster_identifier(group: pd.DataFrame) -> str:
    identity = "|".join(
        f"{row.article_id}:{row.published_at_utc}"
        for row in group.itertuples(index=False)
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    session = str(group.session_date.iloc[0]).replace("-", "")
    return f"news_{session}_{digest}"


def prepare_news(
    news_path: Path,
    start_date: str,
    end_date: str,
    relevance_types: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Filter articles and attach authoritative NASDAQ session boundaries."""
    news = pd.read_csv(news_path, low_memory=False, keep_default_na=False)
    missing = REQUIRED_NEWS_COLUMNS.difference(news.columns)
    if missing:
        raise ValueError(f"News input is missing columns: {sorted(missing)}")
    input_count = len(news)

    news["published_at_utc"] = pd.to_datetime(
        news.published_at_utc, utc=True, errors="coerce"
    )
    invalid_timestamp = news.published_at_utc.isna()
    excluded = _boolean_series(news.excluded)
    representative = _boolean_series(news.event_representative)
    relevance_ok = news.relevance_type.astype(str).str.lower().isin(relevance_types)

    start = pd.Timestamp(start_date, tz="UTC")
    end_exclusive = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)
    date_ok = news.published_at_utc.ge(start) & news.published_at_utc.lt(end_exclusive)

    base = news.loc[
        ~invalid_timestamp & ~excluded & representative & relevance_ok & date_ok
    ].copy()
    base["article_text"] = base.apply(_article_text, axis=1)
    base = base.loc[base.article_text.str.len().gt(0)].copy()
    base["local_date"] = (
        base.published_at_utc.dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    )

    calendar = mcal.get_calendar("NASDAQ")
    schedule = calendar.schedule(start_date=start_date, end_date=end_date).reset_index()
    schedule = schedule.rename(columns={schedule.columns[0]: "session_index"})
    schedule["session_date"] = pd.to_datetime(schedule.session_index).dt.strftime("%Y-%m-%d")
    schedule = schedule[["session_date", "market_open", "market_close"]]
    schedule["market_open"] = pd.to_datetime(schedule.market_open, utc=True)
    schedule["market_close"] = pd.to_datetime(schedule.market_close, utc=True)

    joined = base.merge(
        schedule, left_on="local_date", right_on="session_date", how="left", validate="many_to_one"
    )
    regular = (
        joined.market_open.notna()
        & joined.published_at_utc.ge(joined.market_open)
        & joined.published_at_utc.lt(joined.market_close)
    )
    retained = joined.loc[regular].copy()
    retained = retained.sort_values(
        ["session_date", "published_at_utc", "article_id"]
    ).reset_index(drop=True)

    audit = NewsAudit(
        input_articles=input_count,
        invalid_publication_timestamps=int(invalid_timestamp.sum()),
        excluded_articles=int(excluded.sum()),
        nonrepresentative_articles=int((~representative).sum()),
        outside_requested_relevance=int((~relevance_ok).sum()),
        outside_requested_dates=int((~date_ok & ~invalid_timestamp).sum()),
        outside_regular_session=int((~regular).sum()),
        retained_articles=len(retained),
        event_clusters=0,
    )
    return retained, schedule, asdict(audit)


def build_fixed_clusters(
    articles: pd.DataFrame,
    cluster_minutes: int,
    latency_minutes: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster articles in fixed windows anchored at the first unused article."""
    if cluster_minutes < 1:
        raise ValueError("cluster_minutes must be positive")
    if latency_minutes < 0:
        raise ValueError("latency_minutes cannot be negative")

    cluster_rows: list[dict[str, object]] = []
    membership_frames: list[pd.DataFrame] = []
    window = pd.Timedelta(minutes=cluster_minutes)
    latency = pd.Timedelta(minutes=latency_minutes)

    for session_date, session_articles in articles.groupby("session_date", sort=True):
        session_articles = session_articles.sort_values(
            ["published_at_utc", "article_id"]
        ).reset_index(drop=True)
        cursor = 0
        while cursor < len(session_articles):
            cluster_start = session_articles.at[cursor, "published_at_utc"]
            window_end = cluster_start + window
            stop = cursor
            while (
                stop < len(session_articles)
                and session_articles.at[stop, "published_at_utc"] < window_end
            ):
                stop += 1
            group = session_articles.iloc[cursor:stop].copy()
            cluster_id = _cluster_identifier(group)
            decision_time = window_end.ceil("min") + latency

            direct = group.relevance_type.astype(str).str.lower().eq("direct")
            cluster_rows.append(
                {
                    "news_cluster_id": cluster_id,
                    "session_date": session_date,
                    "market_open_utc": group.market_open.iloc[0],
                    "market_close_utc": group.market_close.iloc[0],
                    "cluster_start_utc": cluster_start,
                    "last_article_utc": group.published_at_utc.max(),
                    "cluster_window_end_utc": window_end,
                    "decision_time_utc": decision_time,
                    "article_count": len(group),
                    "direct_article_count": int(direct.sum()),
                    "indirect_article_count": int((~direct).sum()),
                    "article_ids_json": _json_unique(group.article_id),
                    "duplicate_group_ids_json": _json_unique(group.duplicate_group_id),
                    "sources_json": _json_unique(group.source),
                    "event_categories_json": _json_unique(group.event_category),
                    "companies_discussed_json": _json_unique(
                        group.get("companies_discussed", pd.Series(dtype=str))
                    ),
                    "combined_text": "\n".join(group.article_text.astype(str)),
                }
            )

            membership_columns = [
                column
                for column in [
                    "article_id",
                    "published_at_utc",
                    "headline",
                    "summary",
                    "source",
                    "relevance_type",
                    "event_category",
                    "duplicate_group_id",
                ]
                if column in group.columns
            ]
            members = group[membership_columns].copy()
            members.insert(0, "news_cluster_id", cluster_id)
            membership_frames.append(members)
            cursor = stop

    clusters = pd.DataFrame(cluster_rows)
    if clusters.empty:
        raise ValueError("No regular-session news clusters remain after filtering")
    clusters = clusters.sort_values("cluster_start_utc").reset_index(drop=True)
    if clusters.news_cluster_id.duplicated().any():
        raise ValueError("Generated duplicate news cluster identifiers")
    memberships = pd.concat(membership_frames, ignore_index=True)
    return clusters, memberships
