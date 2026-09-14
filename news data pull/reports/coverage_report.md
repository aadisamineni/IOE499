# Finnhub company-news coverage report

Generated at: 2026-09-14T16:01:27Z

## Scope

- Requested period: 2025-10-01 through 2026-05-01, inclusive.
- Tickers: NVDA, AMD, TSM, MU, AVGO, MSFT, AMZN, GOOGL, META.
- NVDA is the primary dataset; the other tickers are contextual association queries.

## Access and request coverage

- Historical endpoint access verified at both ends: True.
- Fully verified terminal-window coverage: False.
- Successful terminal windows: 520.
- Failed terminal windows: 0.
- Unavailable terminal windows: 0.
- Empty 200 responses kept as unverified: 0.
- Potentially truncated minimum windows: 3.

See `request_windows.csv` for every latest request-window state and `unavailable_periods.csv` for failed, unavailable, empty-unverified, potentially truncated, or not-yet-collected dates.

## Article audit summary

- Raw ticker observations: 80647.
- Unique articles after ID/URL deduplication: 53073.
- Exact duplicate observations removed: 27574.
- Multi-article event groups: 308.
- Event-duplicate articles suppressed: 257.
- Retained direct signals: 5996.
- Retained indirect signals: 755.

## Interpretation limitations

- Finnhub company news supplies headlines, summaries, URLs, sources, related symbols, and metadata—not guaranteed full article text.
- `retrieved_at_utc` records this collection run. It does not establish when an article was historically available to a trading system.
- A response associated with a queried ticker is not treated as proof that the article discusses or materially affects that company.
- Related-ticker collection can miss broad policy news that Finnhub does not associate with one of the configured companies.
- Empty HTTP 200 arrays remain explicitly labeled unverified rather than being interpreted as zero-news proof.
- Relevance and exclusions are deterministic configuration rules. No sentiment or confidence score is invented.
- This project creates a news dataset only; it contains no trading or portfolio-rebalancing logic.

## Official documentation checked

- Company-news endpoint and response schema: https://finnhub.io/docs/api/company-news
- Current plan history and request limits: https://finnhub.io/pricing
- Official OpenAPI schema: https://github.com/Finnhub-Stock-API/finnhub-go/blob/master/api/openapi.yaml
