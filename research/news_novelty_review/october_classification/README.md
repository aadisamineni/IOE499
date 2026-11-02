# October classification ledger

This directory records actual review progress, not a completed corpus classification.

## Current scope

- Full saved collection: 53,073 records, preserved in `full_collection_inventory.csv` with explicit pending status.
- October 2025: 5,232 records.
- October records previously marked direct/indirect: 1,014, including upstream excluded and nonrepresentative records.
- Provisional content pass complete: all 1,014 October direct/indirect records, including excluded and nonrepresentative rows.
- Remaining October direct/indirect content assessments: 0. Web verification is incomplete; unresolved labels remain unresolved.
- Upstream relevance rejects: 920 of 4,218 assessed, covering October 1–7 UTC; 3,298 remain pending. Upstream labels are preserved, with separate relevance-reconsideration flags.
- Combined October content assessments: **1,934 of 5,232 (36.96%)**. This is content-review coverage, not completed novelty verification.
- Records with linked external evidence: 385 across both subsets. A source link may only corroborate an event and is not proof of first public disclosure.

The assistant assessed saved headlines and summaries for IDs 1–104, and available summaries or excerpts for subsequent batches. These are provisional content assessments, not a full-text audit. External checking focused on identifiable developments and their publication history. It did not independently retrieve every full article or search the publication history of all opinion and price-roundup pieces. No automatic word rules or bulk LLM predictions are represented as manual judgments. Empty or inadequate text remains uncertain. No stock-return datasets were loaded in this classification batch.

All judgments are provisional assistant assessments, not human consensus or validated targets. No original news/filtering files were changed. The first 50-article random pilot remains separate; these rows form a chronological batch and are not a representative random estimate of the whole collection.

## Files

- `october_all_articles.csv`: **primary combined ledger**, all 5,232 October records, including 3,298 pending.
- `october_review_progress.json`: **primary overall progress**, covering both original direct/indirect records and reviewed relevance rejects.
- `reject_report.html`: readable reject-audit report with evidence links and caveats.
- `reject_review_ledger.csv`: all 4,218 upstream rejects with reviewed/pending status; `reviewed_rejects.csv` contains only explicit assessments.
- `relevance_reconsideration.csv`: 146 flagged rejects, including broad contextual candidates and clear NVIDIA links. These are not all certified relevance errors.
- `all_sources.csv` and `all_event_groups.csv`: combined evidence and explicitly linked developments. Earliest saved article is not guaranteed first disclosure.
- `independent_reject_review.csv`: separate independent-review sheet for rejects; existing judgments are preserved.
- `reject_audit_findings.md`: concrete missed-link and timestamp examples.
- `report.html`: readable 1,014-article report with source links and caveats.
- `reviewed_articles.csv`: the 1,014 provisional assessments.
- `october_review_ledger.csv`: all 1,014 October direct/indirect records with assessment status.
- `full_collection_inventory.csv`: all 53,073 records with review status; preserves excluded and out-of-scope items.
- `sources.csv`: source URL, displayed time/precision, observation and check date.
- `progress.json`: counts for the original direct/indirect subset only; use `october_review_progress.json` for overall progress.
- `independent_review.csv`: source text plus blank fields for an independent reviewer; reruns preserve this sheet.

## Interpreting the fields

`content_type` and `novelty_status` answer different questions. Price reporting may contain new facts. Opinion can itself be new commentary. Neither label proves that an article contains no information.

- `earlier_disclosure_found`: a located earlier disclosure covers the identified development; no incremental fact was established in the available sampled text. Does not certify that unavailable full text adds nothing.
- `release_time_supported`: a company/investor/publisher-issued release time matches the saved article time. Does not guarantee it was the first public disclosure or material for NVDA.
- `new_detail_candidate`: reporting may add a concrete detail to an older announcement; verify that detail rather than rejecting the whole story as old.
- `candidate_timing_unresolved`: specific report, speech, interview or announcement, but original publication chronology is insufficiently established.
- `revision_history_unresolved`: current source is a corrected version; cannot safely assume later text existed at the earlier saved time.
- `followup_reporting`: explicit attribution to another outlet's reporting; underlying source history unresolved.
- `not_established`: content assessed, but novelty is not verified. This includes commentary and peripheral items and must not be interpreted as false/old.
- `timestamp_conflict`: saved timestamp conflicts with a located source/version; resolve provenance before matching price windows.
- `unreviewed`: no substantive review done. Never use as a model target.

Every row keeps `first_release_proven=false` and `training_ready=false`. Those fields are deliberate: timing evidence is necessary but not sufficient for historical availability or model eligibility.

## Research queues

The files prefixed `all_` combine both reviewed subsets: 388 announcement/new-detail candidates, 186 earlier-disclosure records, 22 timestamp/revision records, 1,302 novelty-unverified records, and 36 follow-ups. These queues partition the 1,934 assessed records; they are not training labels. The files below without `all_` retain the original direct/indirect subset for compatibility:

- `announcement_candidates.csv`: 170 candidate records (12 release times supported, 133 timing unresolved, 25 possible new details). These are articles, not 170 independent events.
- `earlier_disclosure_articles.csv`: 112 articles with an earlier located disclosure for the identified development.
- `timestamp_and_revision_review.csv`: 11 records needing timestamp or historical-version resolution.
- `content_only_unverified.csv`: 706 records with novelty not established. Do not use these as negative targets.
- `followup_articles.csv`: 15 follow-ups with underlying chronology still requiring attention.
- `event_groups.csv`: groups explicitly linked developments; earliest saved article is not necessarily first public disclosure. Groups may need further consolidation.

Some recap links apply only to a specific development within a mixed article. No queue establishes price causality or trading value.

## Findings worth preserving

1. AMD-Cohere expansion reports on October 1 repeat a September release.
2. Nscale's October 1 saved funding release appears 2h38m after the time on the company page.
3. The Microsoft-Nebius agreement dates to September, but October 2 reporting adds potentially new information. Treat the agreement and its newly disclosed details separately.
4. SHI's company page points to a corrected/replaced release appearing after saved October 2 records. Preserve the revision warning rather than using current text as historical fact.
5. Phaidra funding, a research-publisher release and SuperX's server release have matching externally supported release minutes. They differ greatly in substantive NVDA relevance.
6. Fujitsu expansion and UAE delay reporting are specific developments with unresolved original-release minutes. An older underlying deal does not make new delay reporting old.

## Continue the review

The direct/indirect provisional content pass now reaches ID 1014 (October 31). Prioritize unresolved announcement chronology and the upstream relevance rejects next. Before assigning labels, read the saved text. For each concrete development, search company/regulatory releases and earlier reporting, preserve evidence dates and timezone precision, and check existing local coverage. Keep absolute first release unknown unless historical evidence supports a stronger conclusion. Do not choose cases based on observed returns.

Continue upstream-reject reading at **R921, October 8 at 00:15:56 UTC**. The first seven UTC days have a provisional content assessment across both relevance subsets. This does not mean every story has been searched on the web or fully verified. Audit the remaining rejects and unresolved evidence before claiming complete event coverage. The full October novelty verification and full collection review are unfinished; `october_review_progress.json` explicitly records that.

Assessments are rendered by `../classify_october.py`. Add explicit reviewed annotations there and regenerate from repository root:

```sh
.venv/bin/python research/news_novelty_review/classify_october.py
.venv/bin/python research/news_novelty_review/review_october_rejects.py
```

Run both scripts in that order: the second builds the combined inventory and overall outputs. Reject assessments live in `../reject_batches/*.json`; the renderer performs no automatic novelty inference. `relevance_audit=no_direct_nvda_link_established` means unconfirmed linkage, not a verified irrelevant label. Do not overwrite source relevance flags using these provisional judgments.

Do not overwrite independent human judgments. Do not move to model fitting using provisional first-release candidates without availability, revision and price-quality checks. Tick/minute-cache matching is a separate subsequent step.
