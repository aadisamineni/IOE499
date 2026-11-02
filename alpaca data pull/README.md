# Alpaca NVDA high-frequency data pull

This folder contains a resumable downloader for the highest-resolution NVDA
market data available through the configured Alpaca account from **October 1,
2025 through May 1, 2026 inclusive**.

The default run requests:

- every historical trade tick;
- every historical quote tick;
- one-minute raw bars, Alpaca's smallest historical bar interval;
- opening and closing auction records; and
- corporate actions with `data_quality=all`.

The script probes historical SIP access first and falls back to IEX if needed.
When available, it also requests the BOATS feed for overnight trading. SIP is
preferred over IEX because SIP consolidates all U.S. exchanges while IEX is a
subset.

## Credentials

Credentials are read from the standard `APCA_API_KEY_ID` and
`APCA_API_SECRET_KEY` environment variables when present. Otherwise the script
reads the local files:

- `../Alpaca_api_key/Public_key.end`
- `../Alpaca_api_key/Private_key.env`

The entire credential directory is Git-ignored. Keys are sent only as HTTPS
headers and are never printed, stored in request manifests, or copied into this
folder.

## Run and resume

From the repository root:

```sh
.venv/bin/python "alpaca data pull/pull_alpaca_nvda.py"
```

The downloader partitions bulk endpoints by UTC day, requests up to 10,000
records per API page, and saves each completed file as gzip-compressed JSONL.
Every record preserves Alpaca's fields and adds `_symbol`, `_kind`, and `_feed`
provenance fields. Temporary page chunks in `.state/` make interrupted downloads
resumable; rerunning the same command skips completed files. Four daily
partitions are downloaded concurrently behind a shared 190-request-per-minute
limiter, below the configured account's 200-request limit.

Useful checks:

```sh
# Verify credentials and available feeds without downloading the date range
.venv/bin/python "alpaca data pull/pull_alpaca_nvda.py" --probe-only

# Exercise one UTC day
.venv/bin/python "alpaca data pull/pull_alpaca_nvda.py" --max-days 1
```

Bulk files are placed under `raw/<feed>/<kind>/` and intentionally ignored by
Git. `metadata/request_manifest.jsonl` records each completed file's query
window, rows, pages, compressed size, SHA-256 digest, and timestamp.
`metadata/summary.json` aggregates coverage and counts without containing
credentials.

## Timestamp and session scope

Trades and quotes retain Alpaca's nanosecond RFC-3339 timestamps. Files are
partitioned by UTC date, not exchange trading date. SIP data can contain regular
and extended-hours records. BOATS, when authorized, is stored separately so it
is not silently mixed with consolidated SIP data.

Alpaca documentation:

- [Historical trades](https://docs.alpaca.markets/us/v1.4.2/reference/stocktradesingle-1)
- [Historical quotes](https://docs.alpaca.markets/us/reference/stockquotes-1)
- [Historical one-minute bars](https://docs.alpaca.markets/us/reference/stockbarsingle-1)
- [Historical auctions](https://docs.alpaca.markets/us/reference/stockauctions-1)
- [Corporate actions](https://docs.alpaca.markets/us/reference/corporateactions-1)
