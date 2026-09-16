# NVDA News-and-Market-Movement Research Project Report

Generated from the repository contents and saved outputs on 2026-09-16.

## Executive summary

This project is building a reproducible research dataset to study whether news published before a trading session is temporally associated with large subsequent Nvidia (NVDA) stock-price movements. It is not yet a prediction model, trading strategy, sentiment engine, or portfolio-management system.

The work combines:

1. Daily NVDA market data from Yahoo Finance.
2. Derived technical and movement indicators calculated using NASDAQ trading sessions.
3. Company-news metadata from Finnhub for NVDA and eight related technology companies.
4. Deterministic relevance classification of the news.
5. Strict time-window matching between prior news and already-observed NVDA movement events.
6. Audit files and tests intended to make missing coverage, timestamp problems, revisions, and timing assumptions visible.

The current matching run identifies 103 flagged NVDA sessions and 31,807 event/article links. These are associations only; the outputs do not establish that an article caused a price movement.

## Research question and unit of analysis

The implicit research question is: “What relevant news appeared before days on which NVDA experienced a daily adjusted-close movement of at least 1% or 5%, and what categories of news surrounded those events?”

The stock event is a trading session whose adjusted close changed from the previous NASDAQ session by at least a configured threshold. The project preserves the original signed indicators:

- `price_move_1pct`: -1, 0, or +1.
- `price_move_5pct`: -1, 0, or +1.

The matcher decodes these into four views—1% up, 1% down, 5% up, and 5% down—without recalculating or altering the source flags. A 5% event is also a 1% event in the same direction, so category counts overlap.

The fundamental observation in the final matched dataset is an event–article pair. One article can be linked to multiple events, and multiple distinct articles can describe the same underlying event.

## End-to-end workflow

```text
Yahoo Finance daily prices ──> validation ──> engineered NVDA features
                                                   │
Finnhub company news ──> raw request audit ──> relevance classification
                                                   │
                         previous-session cutoff + lookback windows
                                                   │
                         event/article matches + HTML/CSV/JSON audits
```

### 1. Market-data acquisition and feature engineering

`NvidiaDatapull.py` downloads NVDA daily history for October 1, 2025 through May 31, 2026, using an exclusive June 1 end date. It separately requests a warm-up period from July 3 through September 30 so rolling features and one-session lags are available on the first requested output date.

The source fields include OHLC prices, adjusted close, volume, dividends, and stock splits. The script retains provider values and does not synthesize missing rows, apply another adjustment, or reconstruct pre-split trade prices.

It then calculates:

- one-session adjusted-close return;
- open-to-close return;
- five-session adjusted-close return;
- 20-session return volatility;
- 20-session average volume;
- inclusive ±1% and ±5% movement flags;
- one-session lag versions of each derived feature.

The lagged fields are intended for premarket use. Same-day close, volume, high/low, and unlagged derived fields are only known after that session ends.

Validation checks trading-session completeness against the packaged NASDAQ calendar, including holidays and early closes. It also checks dates, nulls, finite values, positive prices, nonnegative volume, and OHLC bounds. Corporate actions and returns of at least 10% are retained for review rather than silently removed.

### 2. News collection

The `news data pull` pipeline queries Finnhub’s company-news endpoint for the period October 1, 2025 through May 1, 2026. NVDA is the primary ticker; AMD, TSM, MU, AVGO, MSFT, AMZN, GOOGL, and META provide contextual coverage for competitors, suppliers, cloud companies, and other AI-infrastructure participants.

The collector uses seven-day request windows, retries transient failures and rate limiting, records every request in a JSONL manifest, saves raw response envelopes, and adaptively splits windows when a response reaches the configurable truncation threshold. It also performs endpoint-access preflight checks at both ends of the date range. The API key is read from an environment variable and is not intended to be stored in configuration or logs.

### 3. News classification and filtering

`process_news.py` deduplicates upstream records by article ID/URL and applies deterministic headline, summary, entity, topic, and exclusion rules. It labels relevant records as `direct` or `indirect` and keeps the original classification fields and event categories.

The rules focus on Nvidia-relevant economic or industry mechanisms, including export controls, AI infrastructure spending, manufacturing/packaging/HBM, competing accelerators, partnerships and contracts, supply constraints, regulation, and product announcements. Earnings, listicles/recommendations, technical analysis, and articles merely describing a past price move are excluded according to configured phrase rules.

Sentiment is explicitly not performed. No confidence score, full-text interpretation, causal label, or trading signal is invented.

### 4. Temporal matching

`matching/match_news.py` reads the existing stock flags and classified news offline. For an event on trading date `t`, the article cutoff is the previous NASDAQ session’s official close. This is consistent with a close-to-close movement label: news must be published before the close that ends the prior price interval.

Each article must satisfy `window_start <= publication_time < cutoff`. Three lookbacks are recorded:

- 24 elapsed hours;
- 72 elapsed hours;
- 7 calendar days using New York wall-clock time.

The calendar-day window correctly spans 167 or 169 elapsed hours around daylight-saving transitions. Exact 24- and 72-hour boundaries belong to the shorter bucket, and the event cutoff itself is excluded.

Publication timestamps must include an actual time and explicit UTC offset. Missing, date-only, invalid, or timezone-ambiguous records are sent to review rather than assigned an assumed timezone.

The matcher also checks revision and availability fields. A known post-cutoff revision or post-cutoff first-availability timestamp excludes that event/article pair. Current data contains no known revision timestamps. Because the collection was made retrospectively and lacks historical article versions, matches are marked as historically uncertain.

## Current results

The generated validation report records:

| Measure | Result |
|---|---:|
| NVDA stock coverage | 2025-10-01 to 2026-05-29 |
| News publication coverage | 2025-10-01 to 2026-05-01 UTC |
| Flagged event days | 103 |
| 1% up events | 58 |
| 1% down events | 45 |
| 5% up events | 4 |
| 5% down events | 1 |
| Classified direct/indirect articles selected for matching | 9,740 |
| Unique matched articles | 9,669 |
| Event/article pairs | 31,807 |
| Articles excluded as not direct/indirect | 43,333 |
| Events with coverage issues | 19 |
| Events with known collection gaps | 9 |
| Events with potential truncation nearby | 10 |
| Raw terminal response files audited | 520 |
| Articles with possible raw content variants | 66 |
| Known revision-timestamp articles | 0 |

The one- and five-percent categories overlap: all four 5% up events are included among 1% up events, and the single 5% down event is included among 1% down events. Therefore, filtered category files must not be added together to estimate unique events or unique articles.

Three requested event dates—May 13, May 14, and May 15, 2026—have no available stock flags because the source stock file ends at May 29 but the requested matching period extends beyond the available flagged period and no qualifying flags exist there. Twelve September 15–30 sessions are reported as missing because the warm-up data is deliberately not used to invent original movement flags.

## Outputs and reproducibility

Important outputs are:

- `data/raw/`: downloaded stock data, warm-up data, and metadata.
- `data/processed/`: NVDA source fields plus engineered features.
- `news data pull/raw/`: request manifest, preflight results, raw API responses, and errors.
- `news data pull/processed/`: classified article tables and processing summary.
- `news data pull/reports/`: coverage, exclusions, request-window, and sample reports.
- `data/matched/2025-09-15_2026-05-15/`: event summaries, article matches, filtered views, HTML report, review files, and validation JSON.

The matcher hashes its primary inputs before and after execution and confirms that they were unchanged. The validation output also records package versions and source hashes. The test suite contains synthetic tests for feature math, warm-up behavior, missing sessions, inclusive thresholds, timestamp parsing, DST, holidays, early closes, duplicate handling, revisions, overlaps, and zero-match coverage gaps.

Typical commands are:

```bash
python3 NvidiaDatapull.py
python3 "news data pull/scripts/run_pipeline.py"
python3 matching/match_news.py --start 2025-09-15 --end 2026-05-15
python3 -m unittest discover -s tests -v
```

## What this project does not yet do

It does not train or evaluate a predictive model, create a future target, calculate abnormal returns relative to a benchmark, estimate statistical significance, infer sentiment, read guaranteed full article text, prove historical article availability, or make investment decisions. The association results also do not control for market-wide news, earnings effects that were excluded by rule, confounding events, article prominence, publication-source quality, or multiple testing.

## Key limitations and interpretation risks

1. Finnhub provides metadata and summaries, not guaranteed full article text, so relevance rules operate on limited text.
2. Successful API requests do not prove complete publisher coverage. Three minimum windows were flagged as potentially truncated, and NVDA truncation warnings occur around February 26 and March 17–18, 2026.
3. The data was retrieved in September 2026. Retrieval time is not historical publication availability; all 31,807 pairs are therefore marked uncertain.
4. Upstream ID/URL deduplication can collapse article-content variants. The project flags 66 possible variants but cannot reconstruct earlier versions or determine revision times.
5. The collected news range is defined by UTC dates, while event windows are based on exchange-local session closes. Boundary coverage therefore needs the explicit coverage flags in the output.
6. Relevance classification is deterministic but rule-based. It may miss relevant stories, include borderline stories, and inherit upstream classifications.
7. The stock data is a retrospectively adjusted provider snapshot, not a point-in-time historical archive. Provider revisions can affect reproducibility across future reruns.

## Recommended next research steps

The strongest next step is a separate analysis layer that aggregates matched articles by event and category, compares event days with non-event control days, and reports effect sizes with uncertainty intervals. Before modeling, extend stock and news collection to a common complete period, resolve or quantify flagged coverage gaps, define a point-in-time article-availability policy, and decide whether overlapping thresholds should be modeled hierarchically or as mutually exclusive labels.

Only after those decisions should lagged news features be used in an out-of-sample prediction experiment. Any model evaluation should use chronological splits, avoid same-day information leakage, include market/sector controls, and compare against simple return and no-news baselines.

