# April hourly move browser

Open report.html locally. No server or network is needed to browse it; source article links require internet.

Rerun from the repository root:

```sh
.venv/bin/python hourly_models/april_news_timing/organized/build_report.py
```

Uses pandas from the root requirements and prior April timing outputs. Original source files and previous reports are preserved. This script imports the earlier assistant annotations and extends content review to all 14 qualifying April hours. No API or new model training is involved.

## Scope and interpretation

All valid full-hour April 2026 NVDA regular-session bars with absolute open-to-close return >=1%. Final half-hour bars and overnight gaps are excluded. The upstream audit includes 126 valid full hours; 14 qualify. Each article window is [start-1h,start), [start,end), or [end,end+1h). There are 131 event–article links from 115 distinct article IDs. Existing exact-copy filtering is retained. Retain direct and indirect news; do not overwrite existing relevance classifications.

These are hourly observations, not precisely detected jumps. Same-day qualifying hours link to one another. They may form a shared episode; do not sum their returns or treat them as independent evidence. Charts show hourly OHLC candles including the final half-hour; they do not interpolate a tick path. Blue highlights the selected hour.

## Primary sections

One provisional primary section is assigned per event–article pair:

1. Potentially relevant development: concrete development with plausible Nvidia/industry connection, not proof of novelty, materiality or causation.
2. Development + price reporting: concrete development plus explicit existing NVDA price movement.
3. Price reporting: explicit NVDA movement without a clear substantive additional development.
4. Background / opinion: general analysis, other-stock performance, incidental mentions, or clear older-performance/announcement discussion without a timely connection.
5. Unclear: vague/incomplete content or identified timestamp/version concerns that prevent confident placement.

Authored assessments and overrides are in build_report.py and the imported prior review. Every match preserves headline, summary, source, timestamps, existing relevance, explanation and historical-version warning. These are assistant judgments, not human consensus. Freshness remains unverified even in development sections; first release timestamps and causal connections are not established. Existing provider timestamps describe saved retrospective records; live pages can change.

Existing story_group identifiers come from duplicate_group_id in the original news dataset, with article ID fallback. They are heuristic groupings, not a verified count of underlying announcements. Related articles remain individually visible. Relevance sections can split an existing group; no distinct coverage is deleted.

## Using the report

Use date navigation to select a move, inspect its chart, and select All/Before/During/After. Each of the five sections shows article cards with a plain-language reason. Expand the saved summary to audit the classification. Counts inside sections follow the timing filter. The month overview counts links, not unique articles. Each link occupies exactly one category, unlike the older overlapping-label report.

## Files and validation

- report.html: interactive local browser report, all 14 moves.
- classified_matches.csv: 131 pairs with primary categories, explanations, supporting source text and timing.
- event_category_counts.csv: event × phase × category counts, including zero cells, and existing story-group counts.
- month_category_counts.csv: summed link counts by phase/category.
- metadata.json: source hashes, totals, provenance and checks.

Checks assert complete event representation, exact annotation coverage, unique event–article pairs, valid categories, reconciled counts and unchanged source hashes. All selected windows have successful collection records; provider coverage and original article versions are not guaranteed. A zero is zero collected articles, not proof of absence of news. This is descriptive selection on large observed moves; predictive tests require non-move hours and independent evaluation.
