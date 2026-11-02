# AVS review and next experiment

Reviewed 2026-10-26 against fetched origin/AVS commit c99cb7a. Current working branch and existing work were preserved. This is a design review and a news-only preprocessing reproduction, not a rerun of the ARC training job.

## Evidence and scope

The new code is in `tick-news-labeling_arc/`: daily SIP quotes become minute-closing midpoints, news becomes fixed five-minute clusters, and forward 5/15/60-minute returns become volatility/spread-adjusted labels. The trainer uses clustered headline/summary text, TF-IDF and balanced logistic regression with chronological train/validation/test splits.

The new ARC build audit, event Parquet and model outputs are not committed and were not found locally. The reported poor accuracy therefore cannot yet be independently diagnosed numerically. Older hourly metrics belong to a different experiment and must not be presented as results of this tick-data run.

Running the latest AVS `prepare_news` and `build_fixed_clusters` functions against the local classified news CSV with defaults produced:

| Measure | Count |
|---|---:|
| Input unique news records | 53,073 |
| Direct representative articles surviving regular-session filtering | 2,038 |
| Five-minute news clusters before price/clean filters | 1,595 |
| Trading dates containing clusters | 147 |
| Publication range | 2025-10-01 through 2026-05-01 |
| First-article-to-decision delay | 6.00–6.98 minutes |

These counts are conditional on the local news snapshot. They do not establish the ARC run's actual sample or quote coverage. The default date range is seven months, even if collected quotes span a year. With these 147 dates, the builder's default split would allocate 102 training, 22 validation and 23 test dates before subsequent label filtering.

## Likely limitations to investigate

1. **Effective sample size:** gigabytes of ticks improve measurement precision, but training observations are news clusters. At most 1,595 clusters survive the news stage here, with fewer eligible clean labels and dependence between nearby stories. The trainer permits up to 5,000 text features.
2. **Timing:** a five-minute collection window plus rounding and one-minute latency means the first prediction occurs six to seven minutes after the first article. The five-minute target measures continuation after that delay. Weak continuation does not imply no immediate news reaction.
3. **Selection:** defaults exclude overnight news, indirect news and upstream excluded stories. Existing upstream rules exclude earnings. Check whether that scope matches the research question; treat overnight announcements as a separate design.
4. **Predictability versus relevance:** a positive business development need not cause an immediate positive return. Expectations, prior reporting and market movements matter. TF-IDF captures wording without directly measuring surprise or novelty.
5. **Baseline gaps:** the new ARC trainer compares text only with a class-prior model. Add market-only and market-plus-text comparisons before attributing value to language.
6. **Benchmark:** QQQ/SOXX adjustment is optional and absent by default. Without it, the labels include broad market movement. Benchmark-adjusted labels still do not establish causality.
7. **Retrospective selection:** `clean_h` uses subsequent news arrivals through the horizon. It is useful for a retrospective subset, but is unavailable at prediction time. Evaluate all eligible events for a deployable forecasting claim, with clean events as a secondary analysis.
8. **Availability:** publication timestamps are retrospective metadata. Confirm first availability and revisions where possible. Inspect deduplication representatives for later copies before interpreting minute-level effects.

## Recommended experiment sequence

### 1. Audit before another full preprocessing run

Collect the actual ARC files:

- `output/build_audit.json`
- `tfidf_output/model_metrics.json`
- `tfidf_output/validation_metrics.csv`
- `tfidf_output/confusion_matrix.csv`
- `tfidf_output/test_predictions.csv`
- `output/news_price_events.parquet` if available

Report retained events/dates/classes per split, quote staleness, missing-label causes, class-prior performance, per-class recall, macro-F1, balanced accuracy and log loss. A three-class model's raw accuracy has no universal 33% baseline. Add paired trading-day bootstrap intervals. Do not treat adjacent rows as independent trials.

### 2. Check whether the outcome contains a measurable news response

Use the compact minute caches, avoiding another pass over all raw quotes. Audit approximately 50 stories without looking at future returns: new information versus recap, relevant entity, original timing, event type and duplicate status. Plot benchmark-adjusted return paths around first publication, including the pre-publication interval and current decision time. Compare absolute reactions against matched non-news times with similar time of day and prior volatility. Report association rather than causation.

An earlier-publication event study and a delayed prediction experiment answer different questions. Shortening latency requires an explicit plausible availability assumption and new labels. Do not select the most profitable horizon on the test set.

### 3. Run one controlled semantic-feature comparison

Choose one primary target in advance. Suggested primary question: does news improve prediction of subsequent movement magnitude beyond market conditions? Retain direction as secondary if needed.

Compare identical eligible rows and chronological folds:

1. Class-prior baseline.
2. Market controls: trailing return/volatility, spread, time of day and article count, all available by the decision time.
3. Market controls plus existing TF-IDF.
4. Market controls plus a small semantic feature set: event category, entity, financial sentiment, novelty and concrete announced changes.

Use frozen embeddings or fixed-schema extraction rather than training a language model from scratch. Validate semantic fields against human annotations first. FinBERT is a financial sentiment model, not a stock-return predictor: https://arxiv.org/abs/1908.10063.

Keep vocabulary, scaling, feature selection and tuning within development folds. Group by trading date, address recurring event episodes and purge overlapping outcome windows where needed. Reserve a later period for final evaluation because the existing test has already informed project decisions. An improvement should repeat across development periods and improve the preselected metric relative to market-only controls, with uncertainty reported.

### 4. Bound the RAG pilot

RAG retrieves evidence and supplies it to a generator; its original evidence concerns knowledge-intensive NLP, not reliable price forecasting: https://arxiv.org/abs/2005.11401.

For a research assistant, index news text plus event IDs, publication/availability times, entities, categories and source links. Keep tick/minute data in Parquet and calculate price statistics through ordinary code. Retrieve relevant stories and join measured historical outcomes to produce sourced comparisons. Clearly separate historical observations from explanations and avoid causal language.

For forecasting, retrieve only material available before the current decision. A historical analogue's outcome is usable only if that horizon ended before the decision. Apply cutoff filters before ranking. Exclude future document revisions and avoid test outcomes entering features. A language model may also remember historical events from pretraining, so historical backtests need an explicit contamination limitation and preferably a future-period evaluation.

A few thousand news documents are a manageable retrieval pilot; the 80 GB quote archive need not be embedded or sent to a generator. For scale, 10,000 vectors of 768 float32 values occupy about 31 MB before index/text overhead. Exact API cost depends on chosen models and token counts and was not estimated here.

Start with 30–50 reviewed questions and compare simple keyword retrieval against semantic or hybrid retrieval. Evaluate retrieval relevance, citation correctness, unsupported claims and numerical agreement with computed returns. Add historical-analogue forecasting only after a no-generation retrieval baseline shows incremental value.

## Project decision

Recommended main deliverable: a reproducible study of whether news event meaning and novelty improve measured post-news response prediction over market controls. A sourced event-research assistant is a feasible companion deliverable. If semantic features show no repeatable incremental signal, report that finding and its timing/sample limitations; adding RAG does not resolve a missing predictive relationship.
