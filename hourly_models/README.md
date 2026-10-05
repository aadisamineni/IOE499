# Hourly NVDA models: fixed 1% threshold

Run from the repository root:

```sh
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python hourly_models/run_models.py
.venv/bin/python -m unittest discover -s tests -p test_hourly_models.py -v
```

Open `output/report.html` for tables and graphs. The script runs offline using
AVS's unchanged hourly CSV and metadata copied from commit
`c024dae052f7b5fca064e67e8242b054b8d005a6`. It does not overwrite earlier daily
experiments or original news. Primary input SHA-256 hashes are recorded and
checked before/after execution.

## Observation and timestamp alignment

One observation is a full 60-minute regular-session bar. The target return is
`close/open - 1` for that bar; the logistic outcome is **absolute return >= 0.01**.
Ridge predicts the absolute return itself, without thresholding its target.
Hourly provider OHLC is preserved on its provider adjustment basis; comparisons
are within the same bar, not between adjusted daily and hourly price levels.

The provider timestamps identify bar starts. A 10:30 bar's outcome ends at
11:30, and all news must be strictly before 10:30. Predictor windows are
`[start - 1 hour, start)` and `[start - 24 hours, start)`. Explicit source
publication timestamps are normalized to UTC; NASDAQ calendar times account
for New York DST and early closes. For example, news at 10:15 can enter the
10:30 prediction, never the 9:30 prediction. Overnight news can enter the next
open's lookback, but the target does not include overnight price changes.
The 24-hour window is not a full-weekend window.

All 166 final short bars are excluded from outcomes, including early-close
sessions. Eleven missing-price rows remain in the source; current invalid prices
or missing prior controls exclude the affected observation, not trigger filling
or interpolation. The preceding 20 complete-duration bars form volatility and
volume averages; any missing data in those windows propagates. The previous bar
return/volume can come from a completed final half-hour bar, explicitly a lag
control rather than an hourly outcome. Initial rows without 20 bars of history
are excluded. Zero news within verified collection windows remains a valid zero.

Only direct/indirect classified news is used; original content exclusions are
not reapplied. Counts mean distinct input article records containing each term,
not repeated token mentions. Article revisions at/after the cutoff are excluded
if explicit revision fields exist; historical versions remain uncertain in the
current retrospectively collected dataset. Collection state must be successful
for every covered UTC date and all configured tickers. Missing/truncated/empty
or out-of-range collection windows are excluded rather than treated as no news.

## Models and evaluation

Both models compare the same samples and three feature sets:

1. Market-only: previous completed return, absolute return, 20-full-bar sample
   return volatility, log prior volume, log 20-bar average volume, minutes since
   session open, and time since previous bar.
2. Market plus log1p article counts in the preceding 1h/24h windows.
3. The above plus log1p article counts containing ai, chip, earnings,
   infrastructure, demand, china, and hit in each window.

All scaling is fitted on training observations only. Logistic regression uses
L2 regularization, fixed C=0.1, and a fixed 0.5 classification probability cutoff.
Ridge uses fixed alpha=10 and predictions are clipped at zero, with raw
predictions retained. No hyperparameter or cutoff tuning on test results.
The user's **1% return threshold** is distinct from the **0.5 probability cutoff**.

Earliest 70% of eligible trading days train; the rest test. Enforce a 24-hour
separation between the last training label end and first test prediction to
avoid shared news windows. No additional embargo rows are needed in this run
because the boundary falls across a larger natural coverage/weekend gap.
Checks ensure no trading day or exact article ID crosses the train/test boundary.
Within-partition repeated articles are allowed. The baseline logistic probability
is the training positive rate; regression baselines are training mean and median.

A paired trading-day bootstrap (2,000 draws, seed 42) assesses test error
*differences* with fixed fitted models. It accounts for within-day dependence,
not all multi-day dependence or feature-selection uncertainty. Intervals are
exploratory, not causal claims or a substitute for a fresh future dataset.

## Run results

- 796 usable hours: 550 training hours over 94 days; 246 test hours over 41 days.
- Training: October 6, 2025–February 25, 2026. Test: March 2–May 1, 2026.
- Test large moves: 35/246 (14.2%).
- Logistic AUC: market-only 0.7110; plus counts 0.7133; plus words 0.7114.
- Logistic Brier score: market-only 0.11446; plus words 0.11553 (higher is worse).
- At p>=0.5, all fitted models have zero true positives. High accuracy mostly
  reflects the large number of small-movement hours. The constant baseline has
  85.8% accuracy while detecting no large moves.
- Ridge MAE: market-only 0.3617 percentage points; plus words 0.3632 pp.
  The training-median baseline is better by MAE at 0.3440 pp.
- Ridge R²: market-only 0.0640; plus words 0.0337. Market-only improves RMSE over
  the training-mean baseline, but explanatory power is limited.
- News-minus-market bootstrap intervals span zero for both Brier and MAE.

**Conclusion:** market history/time-of-day provides some ranking information for
large hourly moves in this split. These news counts/words do not demonstrate
incremental predictive value. No causal inference. These outcomes differ from
prior up/down-only models and their accuracy should not be compared directly.
The historical sample and word choices have already been explored; testing is
chronological but not a pristine prospective evaluation.

## Files

- `output/hourly_dataset_and_predictions.csv`: observations, features, targets,
  split membership and predictions. Train predictions are in-sample.
- `output/test_predictions.csv`: later-date predictions only.
- `output/hourly_article_links.csv`: exact publication timestamps, cutoffs and
  1h/24h membership for auditing alignment.
- `output/bar_audit.csv`: every source bar and nonexclusive exclusion reasons.
- `output/timestamp_review.csv`: rejected ambiguous publication timestamps.
- `output/coefficients.csv`: coefficients per training standard deviation and
  logistic odds ratios; not causal effects or significance tests.
- `output/metrics.json`: versions, source hashes, settings, metrics, confusion
  matrices, uncertainty and validation checks.
- `output/model_comparison.png`, `output/report.html`: standalone plot and report.
