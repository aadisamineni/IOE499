"""Offline temporal matching of existing stock flags to existing news records."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import html
from importlib.metadata import version
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

import pandas as pd
import pandas_market_calendars as mcal

ROOT = Path(__file__).resolve().parents[1]
NEWS_ROOT = ROOT / "news data pull"
STOCK = ROOT / "data/processed/NVDA_daily_features_2025-10-01_2026-05-31.csv"
NEWS = NEWS_ROOT / "processed/all_classified_articles.csv"
TZ = "America/New_York"
FLAG_MAP = {"1pct_up": ("price_move_1pct", "1"),
            "1pct_down": ("price_move_1pct", "-1"),
            "5pct_up": ("price_move_5pct", "1"),
            "5pct_down": ("price_move_5pct", "-1")}
ORIGINAL_FLAGS = ["price_move_1pct", "price_move_5pct"]
FLAG_COLUMNS = [f"shift_{key}" for key in FLAG_MAP]
TIMING = "documented_adjusted_close_to_close_previous_NASDAQ_session_close"
REVISION_FIELDS = ["updated_at_utc", "modified_at_utc", "revised_at_utc"]
EVENT_COLUMNS = ["event_date", "ticker"] + ORIGINAL_FLAGS + FLAG_COLUMNS + [
    "shift_categories", "previous_session_date", "cutoff_utc", "cutoff_new_york",
    "interval_end_utc", "timing_assumption", "provisional_cutoff"]
SUMMARY_COLUMNS = EVENT_COLUMNS + [
    f"{prefix}_{window}" for window in ["24h", "72h", "7d"]
    for prefix in ["article_count", "collection_window_verified"]] + [
    "coverage_issues", "incomplete_lookback_at_news_start", "extends_past_news_collection_end",
    "known_collection_gap", "potential_truncation", "zero_match_interpretation",
    "historical_availability_uncertain_count", "possible_revision_or_identity_conflict_count",
    "missing_article_fields_count", "provider_source_completeness_unverified"]


def read_csv(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def iso(value):
    return value.isoformat() if value is not None and not pd.isna(value) else ""


def parse_timestamp(value):
    """Require an actual time and explicit UTC offset; never assume a timezone."""
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
    except (ValueError, TypeError, OverflowError):
        return None, "invalid_timestamp"


def window_bounds(cutoff):
    # 24/72 hours are elapsed time. Seven calendar days preserves the NY wall time
    # across DST and can span 167 or 169 hours.
    return {"24h": cutoff - pd.Timedelta(hours=24),
            "72h": cutoff - pd.Timedelta(hours=72),
            "7d": (cutoff.tz_convert(TZ) - pd.DateOffset(days=7)).tz_convert("UTC")}


def membership(published, cutoff, bounds=None):
    bounds = window_bounds(cutoff) if bounds is None else bounds
    if not bounds["7d"] <= published < cutoff:
        return None
    hours = (cutoff - published).total_seconds() / 3600
    return {"hours_before_cutoff": hours, "in_24h": int(published >= bounds["24h"]),
            "in_72h": int(published >= bounds["72h"]), "in_7d": 1,
            "lookback_bucket": "0–24 hours" if hours <= 24 else
                               "more than 24–72 hours" if hours <= 72 else
                               "more than 72 hours–7 calendar days"}


def calendar_events(stock, start, end):
    schedule = mcal.get_calendar("NASDAQ").schedule(
        start_date=pd.Timestamp(start) - pd.Timedelta(days=20), end_date=end)
    dates = pd.to_datetime(stock.trading_date, format="%Y-%m-%d", errors="coerce")
    bad = stock.loc[dates.isna()].copy()
    if len(bad):
        bad["review_reason"] = "missing_or_invalid_stock_date"
    valid = stock.loc[dates.notna()].copy()
    valid = valid.loc[valid.trading_date.between(start, end)].copy()
    if valid.trading_date.duplicated().any():
        raise ValueError("Duplicate stock dates; cannot define a unique event without review")
    for column in ORIGINAL_FLAGS:
        if not valid[column].isin(["-1", "0", "1"]).all():
            raise ValueError(f"Unexpected/missing encoding in {column}; refusing to reinterpret")
    if not valid.ticker.eq("NVDA").all():
        raise ValueError("Stock input contains a non-NVDA ticker")
    for category, (column, value) in FLAG_MAP.items():
        valid[f"shift_{category}"] = valid[column].eq(value).astype(int)
    flagged = valid.loc[valid[FLAG_COLUMNS].any(axis=1)].sort_values("trading_date")
    events = []
    for row in flagged.to_dict("records"):
        day = pd.Timestamp(row["trading_date"])
        if day not in schedule.index:
            raise ValueError(f"Flagged date is not a NASDAQ session: {day}")
        position = schedule.index.get_loc(day)
        previous = schedule.iloc[position - 1]
        cutoff = previous.market_close
        event = {"event_date": row["trading_date"], "ticker": row["ticker"],
                 **{key: row[key] for key in ORIGINAL_FLAGS + FLAG_COLUMNS},
                 "shift_categories": "|".join(k for k in FLAG_MAP if row[f"shift_{k}"]),
                 "previous_session_date": str(schedule.index[position - 1].date()),
                 "cutoff_utc": iso(cutoff), "cutoff_new_york": iso(cutoff.tz_convert(TZ)),
                 "interval_end_utc": iso(schedule.loc[day, "market_close"]),
                 "timing_assumption": TIMING, "provisional_cutoff": 0}
        events.append(event)
    expected = schedule.index[schedule.index >= pd.Timestamp(start)]
    missing_dates = expected.difference(pd.DatetimeIndex(pd.to_datetime(valid.trading_date)))
    return events, valid, bad, missing_dates.strftime("%Y-%m-%d").tolist()


def manifest_rows():
    latest = {}
    path = NEWS_ROOT / "raw/request_manifest.jsonl"
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row.get("record_type") == "request_window" and row.get("window_id"):
            latest[row["window_id"]] = row
    return list(latest.values())


def audit_raw_versions(rows):
    """Audit content variants collapsed by upstream ID/URL deduplication.

    Different snapshots retrieved later do not prove WHEN a publisher revised an
    article. Mark these conflicts as uncertain; never infer a revision time.
    """
    versions = defaultdict(set)
    raw_paths = []
    fields = set()
    for request in rows:
        if request.get("status") not in {"success", "success_empty_unverified",
                                          "success_truncation_suspected"}:
            continue
        path = NEWS_ROOT / request["raw_file"]
        if not path.exists():
            continue
        envelope = json.loads(path.read_text())
        if not envelope.get("terminal_window"):
            continue
        raw_paths.append(path)
        for article in envelope.get("response", []):
            fields.update(article)
            content = (str(article.get("headline", "")).strip(),
                       str(article.get("summary", "")).strip(), str(article.get("datetime", "")))
            for key in ("id:" + str(article.get("id", "")), "url:" + str(article.get("url", ""))):
                if key not in {"id:", "url:"}:
                    versions[key].add(content)
    return versions, raw_paths, sorted(fields)


def prepare_articles(news, versions):
    # Keep existing relevance classifications. Do not apply upstream content or
    # event-group exclusions: those would discard distinct coverage.
    selected = news.loc[news.relevance_type.isin(["direct", "indirect"])].copy()
    omitted = news.loc[~news.relevance_type.isin(["direct", "indirect"])].copy()
    omitted["matching_exclusion_reason"] = "existing_relevance_not_direct_or_indirect"
    duplicates = selected.loc[selected.duplicated(keep="first")].copy()
    duplicates["matching_exclusion_reason"] = "exact_duplicate_record"
    selected = selected.drop_duplicates().copy()
    articles, review, excluded = [], [], []
    for row in selected.to_dict("records"):
        row["article_match_key"] = hashlib.sha256(
            json.dumps(row, sort_keys=True).encode()).hexdigest()[:24]
        published, error = parse_timestamp(row.get("published_at_utc", ""))
        if error:
            entry = {**row, "review_field": "published_at_utc", "review_reason": error}
            review.append(entry)
            excluded.append({**row, "matching_exclusion_reason": error})
            continue
        row["publication_timestamp_utc"] = iso(published)
        row["publication_timestamp_new_york"] = iso(published.tz_convert(TZ))
        row["_published"] = published
        row["_retrieved"], retrieval_error = parse_timestamp(row.get("retrieved_at_utc", ""))
        if retrieval_error:
            review.append({**{k: v for k, v in row.items() if not k.startswith("_")},
                           "review_field": "retrieved_at_utc", "review_reason": retrieval_error})
        row["raw_content_variant_count"] = max(
            len(versions.get("id:" + row.get("article_id", ""), set())),
            len(versions.get("url:" + row.get("url", ""), set())), 1)
        row["possible_revision_or_identity_conflict"] = int(row["raw_content_variant_count"] > 1)
        row["_revisions"] = []
        row["_revision_ambiguous"] = False
        for field in REVISION_FIELDS + ["first_available_at_utc"]:
            if row.get(field):
                parsed, problem = parse_timestamp(row[field])
                if problem:
                    review.append({**{k: v for k, v in row.items() if not k.startswith("_")},
                                   "review_field": field, "review_reason": problem})
                    row["_revision_ambiguous"] = True
                else:
                    row["_revisions"].append((field, parsed))
        row["known_revision_timestamp_present"] = int(any(row.get(f) for f in REVISION_FIELDS))
        row["missing_article_id"] = int(not row.get("article_id"))
        row["missing_headline"] = int(not row.get("headline"))
        row["missing_source"] = int(not row.get("source"))
        row["missing_url"] = int(not row.get("url"))
        articles.append(row)
    return articles, review, pd.concat([omitted, duplicates, pd.DataFrame(excluded)], ignore_index=True)


def revision_exclusion(article, cutoff):
    if article["_revision_ambiguous"]:
        return "ambiguous_revision_or_availability_timestamp"
    for field, stamp in article["_revisions"]:
        if stamp >= cutoff:
            return "known_post_cutoff_revision_no_earlier_version" if field in REVISION_FIELDS else \
                   "known_availability_at_or_after_cutoff"
    return ""


def coverage_table(requests, start, end, configured):
    states = defaultdict(set)
    for request in requests:
        status = request.get("status", "unknown")
        if status.startswith("split_"):
            continue
        for day in pd.date_range(request["from_date"], request["to_date"]):
            state = status
            if status.startswith("success") and not (NEWS_ROOT / request.get("raw_file", "")).is_file():
                state = "missing_raw_response_file"
            states[(request["ticker"], str(day.date()))].add(state)
    rows = []
    for day in pd.date_range(pd.Timestamp(start) - pd.Timedelta(days=10), end):
        day_string = str(day.date())
        for ticker in configured["tickers"]:
            if not configured["start_date"] <= day_string <= configured["end_date"]:
                state = "outside_collected_date_range"
            else:
                found = states.get((ticker, day_string), set())
                problems = found - {"success"}
                state = "|".join(sorted(problems)) if problems else "success" if found else "no_terminal_request"
            rows.append({"coverage_date_utc": day_string, "ticker": ticker,
                         "coverage_state": state})
    return pd.DataFrame(rows)


def window_coverage(start, cutoff, coverage):
    # Collector explicitly filters UTC publication dates, so these are UTC day
    # bounds, not invented timestamps on date-only articles.
    days = coverage.loc[coverage.coverage_date_utc.between(
        str(start.date()), str((cutoff - pd.Timedelta(nanoseconds=1)).date()))]
    problems = days.loc[days.coverage_state.ne("success")]
    return problems


def match_events(events, articles, coverage, configured):
    matches, summaries, pair_exclusions = [], [], []
    ordered = sorted(articles, key=lambda a: (a["_published"], a["article_match_key"]))
    for event in events:
        cutoff = pd.Timestamp(event["cutoff_utc"])
        bounds = window_bounds(cutoff)
        current = []
        for article in ordered:
            flags = membership(article["_published"], cutoff, bounds)
            if flags is None:
                continue
            reason = revision_exclusion(article, cutoff)
            if reason:
                pair_exclusions.append({**event, "article_match_key": article["article_match_key"],
                                        "article_id": article["article_id"],
                                        "publication_timestamp_utc": iso(article["_published"]),
                                        "exclusion_reason": reason})
                continue
            snapshot = article["_retrieved"]
            uncertain = (snapshot is None or snapshot >= cutoff or
                         article["possible_revision_or_identity_conflict"] == 1)
            record = {**{k: v for k, v in article.items() if not k.startswith("_")},
                      **event, **flags, "historical_availability_uncertain": int(uncertain),
                      "availability_note": "retrospective_snapshot_no_historical_version_guarantee"
                      if uncertain else "snapshot_observed_before_cutoff",
                      "known_post_cutoff_revision": 0}
            matches.append(record)
            current.append(record)
        summary = event.copy()
        all_problems = []
        for label, beginning in bounds.items():
            summary[f"article_count_{label}"] = sum(r[f"in_{label}"] for r in current)
            problems = window_coverage(beginning, cutoff, coverage)
            summary[f"collection_window_verified_{label}"] = int(problems.empty)
            all_problems.extend(problems.to_dict("records"))
        unique_problems = {(r["coverage_date_utc"], r["ticker"], r["coverage_state"])
                           for r in all_problems}
        compact = defaultdict(set)
        for day, ticker, state in unique_problems:
            compact[(day, state)].add(ticker)
        summary["coverage_issues"] = json.dumps([
            {"date_utc": day, "state": state, "tickers": sorted(tickers)}
            for (day, state), tickers in sorted(compact.items())], separators=(",", ":"))
        first = pd.Timestamp(configured["start_date"], tz="UTC")
        last = pd.Timestamp(configured["end_date"], tz="UTC") + pd.Timedelta(days=1)
        summary["incomplete_lookback_at_news_start"] = int(bounds["7d"] < first)
        summary["extends_past_news_collection_end"] = int(cutoff > last)
        summary["known_collection_gap"] = int(any(
            state in {"outside_collected_date_range", "no_terminal_request", "missing_raw_response_file"}
            or "failed" in state or "unavailable" in state for _, _, state in unique_problems))
        summary["potential_truncation"] = int(any("truncation" in state for _, _, state in unique_problems))
        summary["zero_match_interpretation"] = ("has_matches" if current else
            "zero_matches_with_collection_gap_or_uncertainty" if unique_problems else
            "zero_eligible_matches_in_verified_collection_windows")
        summary["historical_availability_uncertain_count"] = sum(
            r["historical_availability_uncertain"] for r in current)
        summary["possible_revision_or_identity_conflict_count"] = sum(
            r["possible_revision_or_identity_conflict"] for r in current)
        summary["missing_article_fields_count"] = sum(any(r[f"missing_{k}"] for k in
            ["article_id", "headline", "source", "url"]) for r in current)
        summary["provider_source_completeness_unverified"] = 1
        summaries.append(summary)
    match_columns = list(dict.fromkeys(EVENT_COLUMNS + [
        key for article in articles[:1] for key in article if not key.startswith("_")
    ] + ["article_match_key", "article_id", "headline", "source", "url", "event_category",
         "publication_timestamp_utc", "publication_timestamp_new_york", "hours_before_cutoff",
         "lookback_bucket", "in_24h", "in_72h", "in_7d", "historical_availability_uncertain",
         "availability_note", "known_post_cutoff_revision"]))
    return (pd.DataFrame(summaries, columns=SUMMARY_COLUMNS),
            pd.DataFrame(matches).reindex(columns=match_columns), pair_exclusions)


def validate_results(summary, matches, stock):
    flagged = stock.loc[stock[FLAG_COLUMNS].any(axis=1)].set_index("trading_date")
    assert set(summary.event_date) == set(flagged.index), "Missing or extra flagged events"
    assert summary.event_date.is_unique
    for event in summary.to_dict("records"):
        for key in ORIGINAL_FLAGS + FLAG_COLUMNS:
            assert event[key] == flagged.loc[event["event_date"], key], f"Changed indicator {key}"
    assert not matches.duplicated(["event_date", "article_match_key"]).any()
    for row in matches.to_dict("records"):
        pub, cutoff = pd.Timestamp(row["publication_timestamp_utc"]), pd.Timestamp(row["cutoff_utc"])
        expected = membership(pub, cutoff)
        assert expected is not None and pub < cutoff
        assert row["in_24h"] <= row["in_72h"] <= row["in_7d"]
        for key, value in expected.items():
            assert row[key] == value
        for key in ORIGINAL_FLAGS + FLAG_COLUMNS:
            assert row[key] == flagged.loc[row["event_date"], key]
    for event in summary.to_dict("records"):
        linked = matches.loc[matches.event_date.eq(event["event_date"])]
        for window in ["24h", "72h", "7d"]:
            assert event[f"article_count_{window}"] == linked[f"in_{window}"].sum()
        assert event["article_count_24h"] <= event["article_count_72h"] <= event["article_count_7d"]
    return {"all_flagged_days_present": True, "original_indicators_unchanged": True,
            "strictly_pre_cutoff": True, "unique_event_article_pairs": True,
            "window_membership_and_counts_consistent": True}


def write_html(path, summary, matches, metadata):
    escape = lambda v: html.escape(str(v), quote=True)
    content = ["<!doctype html><html lang='en'><meta charset='utf-8'>",
               "<title>NVDA preceding news matches</title><style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 20px}summary{cursor:pointer}li{margin:14px 0}small{color:#555}.warning{background:#fff3d0;padding:12px}a{color:#164aa0}</style>",
               "<h1>NVDA shifts and preceding news</h1>",
               "<p>Temporal associations only; no causal inference. Existing stock flags are unchanged.</p>",
               f"<p class='warning'>Requested stock dates: {escape(metadata['requested_start'])}–{escape(metadata['requested_end'])}. "
               "Available flags start October 1; news collection ends May 1. "
               "News was retrieved retrospectively: publication times do not guarantee historical versions.</p>",
               f"<p>{len(summary)} flagged days; {len(matches)} event–article pairs. "
               "All listed articles are strictly before the prior session close. "
               "Times show explicit UTC and New York offsets.</p>"]
    groups = {day: group for day, group in matches.groupby("event_date")}
    for event in summary.to_dict("records"):
        content += [f"<section><h2>{escape(event['event_date'])} — {escape(event['shift_categories'])}</h2>",
                    f"<p>Cutoff: {escape(event['cutoff_utc'])} / {escape(event['cutoff_new_york'])}. "
                    f"Articles: 24h={event['article_count_24h']}, 72h={event['article_count_72h']}, 7d={event['article_count_7d']}.</p>"]
        if event["coverage_issues"] != "[]":
            content.append(f"<details class='warning'><summary>Incomplete or uncertain collection coverage</summary><pre>{escape(event['coverage_issues'])}</pre></details>")
        linked = groups.get(event["event_date"])
        if linked is None:
            content.append(f"<p>No matching articles. {escape(event['zero_match_interpretation'])}</p></section>")
            continue
        content.append("<details><summary>Show preceding headlines</summary><ol>")
        for row in linked.sort_values("publication_timestamp_utc", ascending=False).to_dict("records"):
            title = escape(row["headline"] or "[Missing headline]")
            url = row.get("url", "")
            if urlsplit(url).scheme in {"http", "https"}:
                title = f"<a href='{escape(url)}' rel='noopener noreferrer'>{title}</a>"
            content.append(f"<li>{title}<br><small>{escape(row['publication_timestamp_utc'])} / "
                f"{escape(row['publication_timestamp_new_york'])} · {escape(row['source'])} · "
                f"ID {escape(row['article_id'])} · {escape(row['lookback_bucket'])} · "
                f"{escape(row['event_category'])} · Historical availability uncertain: "
                f"{row['historical_availability_uncertain']}</small></li>")
        content.append("</ol></details></section>")
    content.append("</html>")
    path.write_text("\n".join(content))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2025-09-15")
    parser.add_argument("--end", default="2026-05-15")
    args = parser.parse_args()
    if pd.Timestamp(args.start) > pd.Timestamp(args.end):
        raise ValueError("Start must precede end")
    out = ROOT / f"data/matched/{args.start}_{args.end}"
    out.mkdir(parents=True, exist_ok=True)
    source_paths = [STOCK, NEWS, NEWS_ROOT / "config.json",
                    NEWS_ROOT / "raw/request_manifest.jsonl", ROOT / "NvidiaDatapull.py"]
    hashes_before = {str(p.relative_to(ROOT)): digest(p) for p in source_paths}
    stock, news = read_csv(STOCK), read_csv(NEWS)
    events, scoped_stock, invalid_stock, missing_dates = calendar_events(stock, args.start, args.end)
    requests = manifest_rows()
    versions, raw_paths, raw_fields = audit_raw_versions(requests)
    configured = json.loads((NEWS_ROOT / "config.json").read_text())["collection"]
    articles, timestamp_review, exclusions = prepare_articles(news, versions)
    coverage = coverage_table(requests, args.start, args.end, configured)
    summary, matches, pair_exclusions = match_events(events, articles, coverage, configured)
    checks = validate_results(summary, matches, scoped_stock)
    hashes_after = {str(p.relative_to(ROOT)): digest(p) for p in source_paths}
    assert hashes_before == hashes_after, "Source files changed during matching"
    matched_keys = set(matches.article_match_key)
    unlinked = [{**{k: v for k, v in a.items() if not k.startswith("_")},
                 "matching_exclusion_reason": "no_eligible_event_window_or_pair_excluded"}
                for a in articles if a["article_match_key"] not in matched_keys]
    exclusions = pd.concat([exclusions, pd.DataFrame(unlinked)], ignore_index=True)
    metadata = {
        "generated_at_utc": iso(pd.Timestamp.now(tz="UTC")),
        "requested_start": args.start, "requested_end": args.end,
        "stock_path": str(STOCK.relative_to(ROOT)), "news_path": str(NEWS.relative_to(ROOT)),
        "source_sha256": hashes_before, "source_files_unchanged": True,
        "package_versions": {p: version(p) for p in ["pandas", "pandas_market_calendars"]},
        "source_stock_date_range": [stock.trading_date.min(), stock.trading_date.max()],
        "source_news_publication_range_utc": [news.published_at_utc.min(), news.published_at_utc.max()],
        "timing": TIMING, "flag_encoding": "Two original -1/0/+1 columns encode four directions; direction views are exact equality checks, not recalculated returns.",
        "indicator_mapping": FLAG_MAP,
        "category_event_counts": {key: int(summary[f"shift_{key}"].sum()) for key in FLAG_MAP},
        "category_pair_counts": {key: int(matches[f"shift_{key}"].sum()) for key in FLAG_MAP},
        "overlap_1pct_5pct_up": int((summary.shift_1pct_up & summary.shift_5pct_up).sum()),
        "overlap_1pct_5pct_down": int((summary.shift_1pct_down & summary.shift_5pct_down).sum()),
        "flagged_event_count": len(summary), "event_article_pair_count": len(matches),
        "unique_matched_articles": len(matched_keys), "input_article_count": len(news),
        "selected_existing_direct_indirect_articles": int(news.relevance_type.isin(["direct", "indirect"]).sum()),
        "unmatched_event_days": summary.loc[summary.article_count_7d.eq(0), "event_date"].tolist(),
        "missing_stock_sessions": missing_dates, "invalid_stock_date_rows": len(invalid_stock),
        "timestamp_review_rows": len(timestamp_review), "pair_exclusion_count": len(pair_exclusions),
        "article_exclusion_counts": exclusions.matching_exclusion_reason.value_counts().to_dict(),
        "events_with_coverage_issues": int(summary.collection_window_verified_7d.eq(0).sum()),
        "events_with_known_collection_gaps": int(summary.known_collection_gap.sum()),
        "events_with_potential_truncation": int(summary.potential_truncation.sum()),
        "raw_terminal_response_files_audited": len(raw_paths), "raw_article_fields": raw_fields,
        "articles_with_raw_content_variants": sum(a["possible_revision_or_identity_conflict"] for a in articles),
        "known_revision_timestamp_articles": sum(a["known_revision_timestamp_present"] for a in articles),
        "historical_availability_uncertain_pairs": int(matches.historical_availability_uncertain.sum()),
        "limitations": ["No original stock flags exist for September 15–30; warm-up prices are not used to invent flags.",
            "News was collected only October 1–May 1, by UTC publication date.",
            "Finnhub snapshots lack revision timestamps and historical article versions; retrieval time is not publication time.",
            "Upstream ID/URL deduplication can collapse content variants; raw variant counts flag potential revisions/identity conflicts, not proven update times.",
            "Existing direct/indirect classifications are reused, including previously excluded content and distinct same-event coverage; no new labels.",
            "Verified collection windows do not establish complete publisher coverage or historical availability."],
        "validation": checks,
    }
    summary.to_csv(out / "event_summary.csv", index=False)
    matches.to_csv(out / "event_article_matches.csv", index=False)
    for category in FLAG_MAP:
        matches.loc[matches[f"shift_{category}"].eq(1)].to_csv(out / f"{category}_articles.csv", index=False)
    pd.DataFrame(timestamp_review, columns=list(news.columns) +
        ["article_match_key", "review_field", "review_reason"]).to_csv(out / "timestamp_review.csv", index=False)
    exclusions.to_csv(out / "excluded_articles.csv", index=False)
    revision_rows = [{k: v for k, v in a.items() if not k.startswith("_")}
                     for a in articles if a["possible_revision_or_identity_conflict"] or
                     a["known_revision_timestamp_present"] or a["_revision_ambiguous"]]
    pd.DataFrame(revision_rows, columns=list(news.columns) + ["article_match_key",
        "raw_content_variant_count", "possible_revision_or_identity_conflict",
        "known_revision_timestamp_present"]).to_csv(out / "article_revision_review.csv", index=False)
    pd.DataFrame(pair_exclusions, columns=EVENT_COLUMNS +
        ["article_match_key", "article_id", "publication_timestamp_utc", "exclusion_reason"]).to_csv(
        out / "excluded_event_article_pairs.csv", index=False)
    pd.DataFrame({"missing_stock_session": missing_dates}).to_csv(out / "missing_stock_sessions.csv", index=False)
    invalid_stock.reindex(columns=list(stock.columns) + ["review_reason"]).to_csv(out / "stock_date_review.csv", index=False)
    coverage.to_csv(out / "collection_coverage_by_day.csv", index=False)
    (out / "validation_report.json").write_text(json.dumps(metadata, indent=2) + "\n")
    write_html(out / "report.html", summary, matches, metadata)
    print(json.dumps({"output_directory": str(out), **{k: metadata[k] for k in [
        "category_event_counts", "category_pair_counts", "flagged_event_count", "event_article_pair_count",
        "unique_matched_articles", "unmatched_event_days", "missing_stock_sessions", "article_exclusion_counts",
        "timestamp_review_rows", "pair_exclusion_count", "events_with_coverage_issues",
        "articles_with_raw_content_variants", "validation"]}}, indent=2))
    examples = matches.sort_values("hours_before_cutoff").groupby("event_date").head(1).head(5)
    print("\nExamples:\n" + examples[["event_date", "shift_categories", "article_id", "headline",
          "publication_timestamp_utc", "cutoff_utc", "hours_before_cutoff"]].to_string(index=False))


if __name__ == "__main__":
    main()
