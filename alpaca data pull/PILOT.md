# October 2025 tick-data pilot

Use the project-local environment and AVS downloader (imported from AVS commit 5ffb51f). Dependencies are in requirements.txt in this directory; requests is also pinned in the root requirements.txt.

Run from the repository root:

```sh
.venv/bin/python "alpaca data pull/pull_alpaca_nvda.py" --start 2025-10-01 --end 2025-10-31 --feeds sip --no-overnight --kinds bars trades quotes corporate_actions --workers 4 --requests-per-minute 90
```

Dates are inclusive UTC daily partitions. SIP includes extended-hours records; regular-session filtering belongs in the subsequent analysis. The separate BOATS feed is disabled. The request rate is capped at 90/minute to leave capacity for a concurrent collection on the same account; coordinate with AVS if his collection is still running.

Configure APCA_API_KEY_ID and APCA_API_SECRET_KEY locally, or use the ignored credential files documented in README.md. Never commit or paste secrets into chat.

Raw gzip JSONL files are saved beneath `alpaca data pull/raw/sip/` (corporate actions under `raw/reference/`). Download metadata is generated beneath `alpaca data pull/metadata/`. The downloader resumes completed partitions/pages. Existing AVS download metadata was deliberately not imported: local metadata must describe actual local downloads.

October was selected as the first full calendar month of the requested data, before examining event-study outcomes. Verify news collection coverage and completed trade/quote partitions before selecting eligible events. This month is a feasibility pilot, not a confirmatory thesis result. No statistical test has been run yet.
