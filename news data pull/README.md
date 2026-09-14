# Nvidia-relevant Finnhub news pipeline

This project collects and audits Finnhub company-news metadata for a news-driven Nvidia research dataset. It intentionally stops before sentiment, trading, or portfolio-rebalancing logic.

All project code, configuration, documentation, raw responses, processed files, and reports live inside this `news data pull` folder. Every script resolves paths relative to its own file, so the space in the folder name is supported.

## Scope

- Inclusive publication-date range: **2025-10-01 through 2026-05-01**.
- Primary query: `NVDA`.
- Context queries: `AMD`, `TSM`, `MU`, `AVGO`, `MSFT`, `AMZN`, `GOOGL`, and `META`.
- The queried ticker and the companies discussed in an article are stored separately. A ticker association from Finnhub is not treated as proof of material relevance.
- Classification uses only Finnhub-supplied headlines, summaries, URLs, sources, related-symbol metadata, and timestamps. Finnhub company news does not promise full article text.

## Documentation and access check

Checked on 2026-09-14:

- Finnhub's official company-news definition requires `symbol`, `from`, and `to`. Its response schema lists `category`, `datetime`, `headline`, `id`, `image`, `related`, `source`, `summary`, and `url`: <https://finnhub.io/docs/api/company-news> and the [official OpenAPI schema](https://github.com/Finnhub-Stock-API/finnhub-go/blob/master/api/openapi.yaml).
- Finnhub's current [pricing page](https://finnhub.io/pricing) advertises one year of company news and 60 calls/minute on Free; paid fundamental tiers advertise three or twenty years depending on billing tier.
- A live authenticated probe returned non-empty NVDA results for October 1–2, 2025. The normal run repeats probes at both ends of the configured range and saves the responses under `raw/preflight/`. This verifies account/endpoint access, not completeness of Finnhub's source coverage.
- The OpenAPI schema does not state a company-news result cap. The collector therefore treats a configurable high count as potential truncation, recursively splits multi-day windows, and flags a high-count one-day window rather than declaring it complete.

## Setup

Python 3.10 or newer is required. There are no third-party runtime dependencies.

```bash
cd "news data pull"
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Make the key available only through the environment. Do not put it in `config.json`, source code, command output, or committed files.

```bash
export FINNHUB_API_KEY="your-key"
```

To check historical access without beginning the full collection:

```bash
python3 scripts/collect_news.py --preflight-only
```

Run the full pipeline:

```bash
python3 scripts/run_pipeline.py
```

Clear the shell variable afterward if desired:

```bash
unset FINNHUB_API_KEY
```

Because paths are derived from the script location, commands also work from elsewhere:

```bash
python3 "/absolute/path/to/news data pull/scripts/run_pipeline.py"
```

## Individual stages and resume behavior

```bash
python3 scripts/collect_news.py
python3 scripts/process_news.py
python3 scripts/generate_report.py
python3 scripts/validate_outputs.py
```

The collector writes a JSONL request manifest and one raw JSON envelope per request. Completed terminal windows are skipped on rerun. Failed windows are retried, while adaptive parent windows marked for splitting resume through their child windows. To retry only selected configured tickers:

```bash
python3 scripts/collect_news.py --skip-preflight --tickers NVDA AMD
```

Requests are sequential, paced at 1.1 seconds by default, and retry transient network failures, HTTP 429, and selected 5xx responses with exponential backoff. Authentication tokens are used only to construct the request in memory; authenticated URLs are never logged or saved.

## Folder layout

```text
news data pull/
├── config.json
├── requirements.txt
├── README.md
├── scripts/
│   ├── common.py
│   ├── collect_news.py
│   ├── process_news.py
│   ├── generate_report.py
│   ├── validate_outputs.py
│   └── run_pipeline.py
├── raw/
│   ├── request_manifest.jsonl
│   ├── preflight_report.json
│   ├── preflight/
│   ├── responses/
│   └── errors/
├── processed/
│   ├── all_classified_articles.csv
│   ├── signal_articles.csv
│   └── processing_summary.json
├── reports/
│   ├── coverage_report.md
│   ├── coverage_report.json
│   ├── monthly_counts.csv
│   ├── request_windows.csv
│   ├── exclusion_counts.csv
│   ├── unavailable_periods.csv
│   ├── sample_retained_articles.csv
│   └── sample_retained_articles.md
└── tests/
    └── test_pipeline.py
```

## Classification and audit rules

The rules are transparent and editable in `config.json`.

1. Exact observations are deduplicated by Finnhub article ID and normalized URL. Their queried tickers are unioned.
2. Relevance classification is performed independently of exclusions. Direct items explicitly mention `Nvidia` or `NVDA`. Indirect items must match both a subject rule and a concrete Nvidia-business connection rule.
3. A generic semiconductor article is not retained merely because Finnhub returned it for a semiconductor ticker.
4. Earnings coverage, generic recommendations/listicles, technical-analysis-only items, and unexplained retrospective price-movement items are preserved in `all_classified_articles.csv` with exclusion reasons but omitted from `signal_articles.csv`.
5. Similar headlines within the configured event window are grouped. One representative remains in the signal file; repeated event coverage is retained in the audit file with `duplicate_event_coverage` and the representative ID.
6. Relevance reasons list the rule-based rationale. No confidence or sentiment score is generated. Sentiment should be implemented later as an independent stage if the project requires it.

The main output fields include:

`article_id`, `queried_tickers`, `companies_discussed`, `headline`, `summary`, `source`, `url`, `published_at_utc`, `retrieved_at_utc`, `relevance_type`, `event_category`, `relevance_reason`, `excluded`, `exclusion_reason`, and `duplicate_group_id`.

## Coverage interpretation

`published_at_utc` is converted from Finnhub's Unix publication timestamp. `retrieved_at_utc` is recorded separately and means only when this pipeline retrieved the API response. It must not be interpreted as the historical time at which the article became available.

The report distinguishes successful windows from:

- failed or plan/auth-unavailable windows;
- interrupted windows with no terminal request;
- one-day windows with potential truncation; and
- HTTP 200 empty arrays, which remain labeled `empty_response_unverified` rather than silently becoming “days without news.”

Finally, collecting associated company tickers can still miss broad export-policy or regulatory news that Finnhub does not associate with any configured company. That limitation cannot be repaired using the company-news endpoint alone.

## Tests

```bash
python3 -m unittest discover -s tests -v
```
