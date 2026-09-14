#!/usr/bin/env python3
"""Clean, classify, deduplicate, and event-group collected Finnhub news."""

from __future__ import annotations

import argparse
import difflib
import json
from collections import Counter, defaultdict, deque
from datetime import date
from pathlib import Path
from typing import Any

from common import (
    PROJECT_ROOT,
    atomic_write_json,
    epoch_to_utc,
    headline_tokens,
    load_config,
    matched_terms,
    normalize_text,
    normalize_url,
    parse_utc,
    read_jsonl,
    stable_id,
    term_present,
    utc_now_iso,
    write_csv,
)


MANIFEST_PATH = PROJECT_ROOT / "raw" / "request_manifest.jsonl"
TERMINAL_SUCCESS_STATUSES = {
    "success",
    "success_empty_unverified",
    "success_truncation_suspected",
}

OUTPUT_FIELDS = [
    "article_id",
    "queried_tickers",
    "companies_discussed",
    "headline",
    "summary",
    "source",
    "url",
    "published_at_utc",
    "retrieved_at_utc",
    "relevance_type",
    "event_category",
    "relevance_reason",
    "excluded",
    "exclusion_reason",
    "duplicate_group_id",
    "duplicate_group_size",
    "duplicate_of_article_id",
    "event_representative",
    "exact_duplicate_group_id",
    "raw_observation_count",
    "finnhub_related",
    "finnhub_category",
    "image_url",
    "normalized_url",
    "matched_relevance_terms",
]

EVENT_BLOCK_STOPWORDS = {
    "ai",
    "chip",
    "chips",
    "company",
    "companies",
    "market",
    "markets",
    "new",
    "news",
    "nvidia",
    "nvda",
    "says",
    "shares",
    "stock",
    "stocks",
    "tech",
    "technology",
}


class UnionFind:
    def __init__(self, count: int) -> None:
        self.parent = list(range(count))
        self.rank = [0] * count

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


def latest_windows() -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(MANIFEST_PATH):
        if row.get("record_type") == "request_window" and row.get("window_id"):
            latest[str(row["window_id"])] = row
    return latest


def load_observations(config: dict[str, Any]) -> list[dict[str, Any]]:
    start = date.fromisoformat(config["collection"]["start_date"])
    end = date.fromisoformat(config["collection"]["end_date"])
    observations: list[dict[str, Any]] = []
    for row in latest_windows().values():
        if row.get("status") not in TERMINAL_SUCCESS_STATUSES:
            continue
        raw_file = row.get("raw_file")
        if not raw_file:
            continue
        path = PROJECT_ROOT / str(raw_file)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            envelope = json.load(handle)
        if not envelope.get("terminal_window"):
            continue
        body = envelope.get("response")
        if not isinstance(body, list):
            continue
        ticker = str(envelope.get("request", {}).get("symbol") or row.get("ticker") or "")
        retrieved_at = str(envelope.get("retrieved_at_utc") or row.get("retrieved_at_utc") or "")
        for article in body:
            if not isinstance(article, dict):
                continue
            published_at = epoch_to_utc(article.get("datetime"))
            if published_at:
                published_date = parse_utc(published_at).date()
                if published_date < start or published_date > end:
                    continue
            observations.append(
                {
                    "article_id": str(article.get("id") or ""),
                    "queried_ticker": ticker,
                    "headline": str(article.get("headline") or "").strip(),
                    "summary": str(article.get("summary") or "").strip(),
                    "source": str(article.get("source") or "").strip(),
                    "url": str(article.get("url") or "").strip(),
                    "normalized_url": normalize_url(str(article.get("url") or "")),
                    "published_at_utc": published_at,
                    "retrieved_at_utc": retrieved_at,
                    "finnhub_related": str(article.get("related") or "").strip(),
                    "finnhub_category": str(article.get("category") or "").strip(),
                    "image_url": str(article.get("image") or "").strip(),
                }
            )
    return observations


def quality_key(observation: dict[str, Any]) -> tuple[Any, ...]:
    populated = sum(bool(observation.get(field)) for field in observation)
    return (
        populated,
        len(str(observation.get("summary") or "")),
        len(str(observation.get("headline") or "")),
    )


def exact_deduplicate(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not observations:
        return []
    union_find = UnionFind(len(observations))
    by_id: dict[str, int] = {}
    by_url: dict[str, int] = {}
    for index, observation in enumerate(observations):
        article_id = observation["article_id"]
        normalized_url = observation["normalized_url"]
        if article_id:
            if article_id in by_id:
                union_find.union(index, by_id[article_id])
            else:
                by_id[article_id] = index
        if normalized_url:
            if normalized_url in by_url:
                union_find.union(index, by_url[normalized_url])
            else:
                by_url[normalized_url] = index

    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, observation in enumerate(observations):
        groups[union_find.find(index)].append(observation)

    articles: list[dict[str, Any]] = []
    for group in groups.values():
        representative = max(group, key=quality_key).copy()
        ids = sorted({row["article_id"] for row in group if row["article_id"]})
        urls = sorted({row["normalized_url"] for row in group if row["normalized_url"]})
        representative["article_id"] = ids[0] if ids else ""
        representative["queried_tickers"] = "|".join(
            sorted({row["queried_ticker"] for row in group if row["queried_ticker"]})
        )
        representative["retrieved_at_utc"] = min(
            (row["retrieved_at_utc"] for row in group if row["retrieved_at_utc"]),
            default="",
        )
        representative["finnhub_related"] = "|".join(
            sorted(
                {
                    item.strip()
                    for row in group
                    for item in row["finnhub_related"].replace(";", ",").split(",")
                    if item.strip()
                }
            )
        )
        representative["raw_observation_count"] = len(group)
        representative["exact_duplicate_group_id"] = stable_id(
            "exact", ids + urls or [representative["headline"], representative["published_at_utc"]]
        )
        representative.pop("queried_ticker", None)
        articles.append(representative)
    return articles


def identify_companies(text: str, finnhub_related: str, aliases: dict[str, list[str]]) -> str:
    normalized = normalize_text(text)
    identified = {
        item.strip().upper()
        for item in finnhub_related.split("|")
        if item.strip()
    }
    for ticker, names in aliases.items():
        if any(term_present(normalized, name) for name in names):
            identified.add(ticker)
    return "|".join(sorted(identified))


def classify_article(article: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    relevance = config["relevance"]
    headline_normalized = normalize_text(article["headline"])
    combined_normalized = normalize_text(f"{article['headline']} {article['summary']}")
    direct_hits = matched_terms(combined_normalized, relevance["direct_keywords"])
    headline_direct_hits = matched_terms(headline_normalized, relevance["direct_keywords"])

    exclusion_reasons: list[str] = []
    exclusions = relevance["exclusions"]
    if matched_terms(headline_normalized, exclusions["earnings"]):
        exclusion_reasons.append("earnings_report_preview_or_recap")
    if matched_terms(headline_normalized, exclusions["recommendation_listicle"]):
        exclusion_reasons.append("generic_recommendation_or_listicle")
    if matched_terms(headline_normalized, exclusions["technical_analysis"]):
        exclusion_reasons.append("technical_analysis_only")
    price_hits = matched_terms(headline_normalized, exclusions["past_price_movement"])
    material_hits = matched_terms(combined_normalized, exclusions["material_underlying_terms"])
    if price_hits and not material_hits:
        exclusion_reasons.append("past_price_movement_without_new_information")

    event_category = ""
    category_hits: list[str] = []
    for category in relevance["event_categories"]:
        hits = matched_terms(combined_normalized, category["terms"])
        if len(hits) > len(category_hits):
            event_category = category["name"]
            category_hits = hits

    if direct_hits:
        relevance_type = "direct"
        if not event_category:
            event_category = "other_nvidia_development"
        relevance_reason = (
            "Headline or Finnhub summary explicitly mentions Nvidia/NVDA"
            + (f" in the {event_category} context" if event_category else "")
            + "."
        )
        relevance_hits = direct_hits + category_hits
        if not headline_direct_hits and not category_hits:
            exclusion_reasons.append("summary_only_mention_without_material_nvidia_event")
    else:
        relevance_type = "not_relevant"
        relevance_reason = (
            "No direct Nvidia/NVDA mention or configured two-part indirect connection was found."
        )
        relevance_hits = []
        for rule in relevance["indirect_rules"]:
            if rule.get("required_term_groups"):
                group_hits = [
                    matched_terms(combined_normalized, group)
                    for group in rule["required_term_groups"]
                ]
                rule_matches = all(group_hits)
                rule_hits = [term for hits in group_hits for term in hits]
            else:
                topic_hits = matched_terms(combined_normalized, rule["topic_terms"])
                connection_hits = matched_terms(combined_normalized, rule["connection_terms"])
                rule_matches = bool(topic_hits and connection_hits)
                rule_hits = topic_hits + connection_hits
            if rule_matches:
                relevance_type = "indirect"
                event_category = rule["category"]
                relevance_reason = rule["reason"]
                relevance_hits = rule_hits
                break
        if relevance_type == "not_relevant":
            exclusion_reasons.append("insufficient_nvidia_connection")

    article["companies_discussed"] = identify_companies(
        combined_normalized,
        article["finnhub_related"],
        relevance["entity_aliases"],
    )
    article["relevance_type"] = relevance_type
    article["event_category"] = event_category
    article["relevance_reason"] = relevance_reason
    article["matched_relevance_terms"] = "|".join(dict.fromkeys(relevance_hits))
    article["excluded"] = bool(exclusion_reasons)
    article["exclusion_reason"] = ";".join(dict.fromkeys(exclusion_reasons))
    return article


def headline_similarity(left: str, right: str) -> tuple[float, float]:
    normalized_left = normalize_text(left)
    normalized_right = normalize_text(right)
    sequence = difflib.SequenceMatcher(None, normalized_left, normalized_right).ratio()
    tokens_left = headline_tokens(left)
    tokens_right = headline_tokens(right)
    union = tokens_left | tokens_right
    jaccard = len(tokens_left & tokens_right) / len(union) if union else 0.0
    return sequence, jaccard


def canonical_article_key(article: dict[str, Any]) -> str:
    return (
        article.get("article_id")
        or article.get("normalized_url")
        or f"{article.get('published_at_utc')}|{normalize_text(article.get('headline', ''))}"
    )


def event_group(articles: list[dict[str, Any]], config: dict[str, Any]) -> None:
    if not articles:
        return
    similarity = config["relevance"]["similarity"]
    sequence_threshold = float(similarity["headline_sequence_threshold"])
    jaccard_threshold = float(similarity["headline_token_jaccard_threshold"])
    event_window_seconds = int(similarity["event_window_hours"]) * 3600
    minimum_shared = int(similarity["minimum_shared_tokens"])
    union_find = UnionFind(len(articles))
    token_sets = [headline_tokens(article["headline"]) for article in articles]
    blocking_token_sets = [tokens - EVENT_BLOCK_STOPWORDS for tokens in token_sets]
    article_times = [
        parse_utc(article["published_at_utc"]) if article.get("published_at_utc") else None
        for article in articles
    ]
    token_index: dict[str, deque[int]] = defaultdict(deque)

    ordered = sorted(
        (
            index
            for index, article in enumerate(articles)
            if article["relevance_type"] in {"direct", "indirect"}
            and article_times[index] is not None
        ),
        key=lambda index: articles[index].get("published_at_utc") or "9999",
    )
    for index in ordered:
        article = articles[index]
        tokens = token_sets[index]
        blocking_tokens = blocking_token_sets[index]
        current_time = article_times[index]
        candidate_counts: Counter[int] = Counter()
        for token in blocking_tokens:
            queue = token_index[token]
            while queue and (
                current_time - article_times[queue[0]]
            ).total_seconds() > event_window_seconds:
                queue.popleft()
            candidate_counts.update(queue)
        for candidate, blocking_shared_count in candidate_counts.items():
            if blocking_shared_count < minimum_shared:
                continue
            other = articles[candidate]
            shared = tokens & token_sets[candidate]
            if len(shared) < minimum_shared:
                continue
            other_time = article_times[candidate]
            if abs((current_time - other_time).total_seconds()) > event_window_seconds:
                continue
            sequence, jaccard = headline_similarity(article["headline"], other["headline"])
            if sequence >= sequence_threshold or jaccard >= jaccard_threshold:
                union_find.union(index, candidate)
        for token in blocking_tokens:
            token_index[token].append(index)

    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(articles)):
        groups[union_find.find(index)].append(index)

    for members in groups.values():
        member_keys = sorted(canonical_article_key(articles[index]) for index in members)
        group_id = stable_id("event", member_keys)
        representative_index = min(
            members,
            key=lambda index: (
                articles[index]["excluded"],
                articles[index]["relevance_type"] == "not_relevant",
                articles[index].get("published_at_utc") or "9999",
                -len(articles[index].get("summary") or ""),
                canonical_article_key(articles[index]),
            ),
        )
        representative = articles[representative_index]
        representative_id = representative.get("article_id") or canonical_article_key(representative)
        for index in members:
            article = articles[index]
            article["duplicate_group_id"] = group_id
            article["duplicate_group_size"] = len(members)
            article["event_representative"] = index == representative_index
            article["duplicate_of_article_id"] = "" if index == representative_index else representative_id
            if (
                index != representative_index
                and not article["excluded"]
                and article["relevance_type"] in {"direct", "indirect"}
            ):
                article["excluded"] = True
                reasons = [reason for reason in article["exclusion_reason"].split(";") if reason]
                reasons.append("duplicate_event_coverage")
                article["exclusion_reason"] = ";".join(dict.fromkeys(reasons))


def serialize_rows(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for article in articles:
        row = article.copy()
        row["excluded"] = "true" if article["excluded"] else "false"
        row["event_representative"] = (
            "true" if article.get("event_representative") else "false"
        )
        rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Path to config JSON (defaults to project config.json)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if not MANIFEST_PATH.exists():
        raise SystemExit("No raw/request_manifest.jsonl found. Run collect_news.py first.")

    observations = load_observations(config)
    articles = exact_deduplicate(observations)
    articles = [classify_article(article, config) for article in articles]
    event_group(articles, config)
    articles.sort(key=lambda row: (row.get("published_at_utc") or "", canonical_article_key(row)))
    serialized = serialize_rows(articles)
    signals = [
        row
        for row in serialized
        if row["excluded"] == "false" and row["relevance_type"] in {"direct", "indirect"}
    ]

    processed = PROJECT_ROOT / "processed"
    write_csv(processed / "all_classified_articles.csv", serialized, OUTPUT_FIELDS)
    write_csv(processed / "signal_articles.csv", signals, OUTPUT_FIELDS)
    summary = {
        "generated_at_utc": utc_now_iso(),
        "raw_observations": len(observations),
        "unique_articles_after_id_url_deduplication": len(articles),
        "exact_duplicate_observations_removed": len(observations) - len(articles),
        "signal_articles_after_exclusions_and_event_grouping": len(signals),
        "retained_direct": sum(row["relevance_type"] == "direct" for row in signals),
        "retained_indirect": sum(row["relevance_type"] == "indirect" for row in signals),
        "sentiment_classification": "not_performed",
        "note": (
            "Relevance uses configured headline/summary/entity rules. Sentiment is deliberately "
            "not inferred and no confidence score is generated."
        ),
    }
    atomic_write_json(processed / "processing_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
