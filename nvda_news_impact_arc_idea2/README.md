# NVDA news impact — ARC idea 2

This is a separate NVDA-only experiment. It does not modify or overwrite the
existing `tick-news-labeling_arc` workflow.

For an automated coding/research agent, read `AGENTS.md` first. It contains the
canonical HPC commands, leakage invariants, artifact checklist, and verification
steps. `BASELINE_RESULTS.md` records the completed quote-level benchmark and
clearly identifies the remaining transformer run.

The central change is to distinguish three questions that the earlier TF-IDF
classifier mixed together:

1. **Reaction:** how NVDA moves immediately after reported news;
2. **impact:** whether the move is unusually large relative to NVDA's own
   pre-event volatility; and
3. **continuation:** whether the move continues or reverses after a tradable
   one- or five-minute decision delay.

All inputs are Nvidia news and NVDA market data. No ETF, index, or other stock
is required.

## Data policy

- Direct Nvidia event representatives are the default news population.
- Premarket, regular-session, after-hours, overnight, weekend, and holiday news
  are retained.
- News released while NVDA has extended-hours bars is evaluated at its reported
  time.
- Closed-market information is batched into the next NASDAQ opening information
  set. This avoids assigning many weekend articles the same opening return as if
  they were independent observations.
- Immediate returns begin with the last completed minute before an article.
- Continuation targets begin after one or five minutes and use only information
  available by that decision.
- Later news is flagged for attribution analysis but is not used to choose the
  live prediction population.
- Every train/validation/test split is chronological and keeps complete outcome
  dates together.

## Outputs

The default output directory is `nvda_news_impact_arc_idea2/output/` and is
Git-ignored because it contains generated Parquet files and fitted models.

Important files are:

- `market_minutes.parquet`: compact NVDA minute bars, optionally enriched with
  SIP quote midpoints and spreads;
- `market_cache_audit.json`: per-partition row counts and provenance;
- `news_events.parquet`: news metadata, pre-event predictors, immediate returns,
  continuation returns, impact thresholds, and contamination diagnostics;
- `no_news_controls.parquet`: ordinary NVDA intervals without nearby news;
- `event_study_summary.csv`: reaction summaries by horizon, session, and event
  category;
- `transformer_features.parquet`: optional frozen FinBERT probabilities and
  embeddings;
- `models_output/model_comparison.csv`: held-out model results;
- `models_output/test_predictions.csv`: auditable event-level predictions;
- `models_output/walk_forward_metrics.csv`: hyperparameter-selection folds;
- `models_output/matched_control_comparison.csv`: news events matched to
  same-date/time/state no-news intervals;
- `models_output/model_metrics.json`: complete methodology and results; and
- `models_output/models/*.joblib`: fitted development-period models.

## Quick local baseline

This uses the small minute-bar source and does not process the 453 million raw
quotes:

```bash
python -m pip install -r nvda_news_impact_arc_idea2/requirements.txt
python nvda_news_impact_arc_idea2/run_nvda_news_impact_arc_idea2.py
```

The run is restartable because every daily market partition has its own Parquet
cache and source-file identity metadata.

## Full HPC run

Install the transformer extras once:

```bash
python -m pip install -r nvda_news_impact_arc_idea2/requirements-transformers.txt
```

Then run the full workflow with parallel SIP quote aggregation and GPU FinBERT:

```bash
python nvda_news_impact_arc_idea2/run_nvda_news_impact_arc_idea2.py \
  --include-quotes \
  --include-transformers \
  --separate-period-models \
  --workers 16 \
  --device cuda
```

`run_hpc_arc_idea2.slurm` is a starting Slurm job file. Adjust its partition,
GPU, memory, wall-time, and environment activation lines to match the cluster.

Stages can also be submitted independently:

```bash
python nvda_news_impact_arc_idea2/run_nvda_news_impact_arc_idea2.py \
  --stage market --include-quotes --workers 32

python nvda_news_impact_arc_idea2/run_nvda_news_impact_arc_idea2.py \
  --stage dataset

python nvda_news_impact_arc_idea2/run_nvda_news_impact_arc_idea2.py \
  --stage transformers --device cuda --finbert-batch-size 128

python nvda_news_impact_arc_idea2/run_nvda_news_impact_arc_idea2.py \
  --stage models --bootstrap-iterations 2000
```

The pooled run includes market period as a feature. With
`--separate-period-models`, independent model directories are also produced for
regular, premarket, after-hours, and overnight observations. This prevents a
large opening-gap event from being treated as statistically interchangeable
with an ordinary intraday headline.

## Models and comparisons

The trainer evaluates these feature sets when their inputs are available:

1. NVDA market state only;
2. structured news only;
3. NVDA market state plus structured news;
4. TF-IDF only;
5. market state, structured news, and TF-IDF;
6. frozen FinBERT sentiment;
7. reduced frozen FinBERT embeddings; and
8. the complete combined model.

After the frozen models have been evaluated, optional chronological fine-tuning
is available for a small number of predeclared binary tasks. For example:

```bash
python nvda_news_impact_arc_idea2/fine_tune_finbert.py \
  --task material_direction_15m \
  --market-periods regular \
  --batch-size 32
```

This resets FinBERT's classification head, uses the earliest 70% of outcome
dates for fitting, the next 15% for early stopping, and the final 15% once for
testing. Fine-tuning should not replace the frozen-feature baselines and should
only be pursued for targets that show stable signal.

Regularized logistic regression is used for impact/direction tasks and ridge
regression for continuous returns. More complex models should only be added if
the news features first improve on the market-only baseline out of sample.

The default task list includes 5-, 15-, and 60-minute impact, 15-minute
direction, direction among material 15-minute events, 1-to-15 and 5-to-60
minute continuation, and continuous 15-/60-minute reaction and continuation
returns.

## Testing

```bash
python -m unittest discover -s nvda_news_impact_arc_idea2/tests -v
```

The tests use synthetic data and never scan the raw Alpaca collection.

## Interpretation

Immediate-reaction targets are event-study outcomes, not executable pre-news
trades. The continuation tasks are the tradable experiments. A useful result is
an out-of-sample improvement over the NVDA market-only model, not a visually
impressive in-sample coefficient or isolated accuracy number.
