# Agent instructions: NVDA news impact ARC idea 2

These instructions apply to every file under `nvda_news_impact_arc_idea2/`.

## Objective and scope

This is an NVDA-only research pipeline. It joins Nvidia-relevant news to NVDA
SIP bars and quotes, measures immediate price impact, and tests whether news
predicts post-publication continuation or reversal.

- Do not use anything under the former repository `temp/` directory.
- Do not introduce QQQ, SPY, SOXX, or other-company returns unless the user
  explicitly changes the scope.
- Do not modify `tick-news-labeling_arc/`; it is the comparison experiment.
- Keep generated data and models under this folder's ignored `output/` path.
- Never load the repository's 840-million-row combined Alpaca CSV. Stream the
  partitioned files under `alpaca data pull/raw/sip/` through `market_data.py`.

## Authoritative inputs

Run from the repository root. Defaults resolve to:

```text
news data pull/processed/all_classified_articles.csv
alpaca data pull/raw/sip/bars/NVDA_bars_YYYY-MM-DD.jsonl.gz
alpaca data pull/raw/sip/quotes/NVDA_quotes_YYYY-MM-DD.jsonl.gz
```

The current collection covers October 1, 2025 through May 1, 2026. News rows
must be non-excluded event representatives. Direct Nvidia relevance is the
default; indirect news is opt-in.

## Environment setup

Core pipeline:

```bash
python -m pip install -r nvda_news_impact_arc_idea2/requirements.txt
```

Frozen FinBERT and fine-tuning stages:

```bash
python -m pip install -r nvda_news_impact_arc_idea2/requirements-transformers.txt
```

Use the cluster's normal virtual environment or module system. The provided
Slurm file assumes `.venv/bin/activate`; edit that line if the cluster uses a
different environment.

## Canonical execution

The complete HPC run is:

```bash
python nvda_news_impact_arc_idea2/run_nvda_news_impact_arc_idea2.py \
  --include-quotes \
  --include-transformers \
  --separate-period-models \
  --workers 16 \
  --device cuda \
  --bootstrap-iterations 2000
```

Or submit the editable job template:

```bash
sbatch nvda_news_impact_arc_idea2/run_hpc_arc_idea2.slurm
```

The main runner executes market caching, event construction, optional frozen
transformer extraction, model fitting, and report generation. Stages are
restartable and can be invoked separately with `--stage market`, `dataset`,
`transformers`, `models`, or `report`.

## Exact manual stage order

Use these commands when diagnosing or resuming a stage:

```bash
python nvda_news_impact_arc_idea2/market_data.py \
  --include-quotes --workers 16

python nvda_news_impact_arc_idea2/build_dataset.py

python nvda_news_impact_arc_idea2/extract_transformer_features.py \
  --device cuda --batch-size 128

python nvda_news_impact_arc_idea2/train_models.py \
  --transformer-features \
  nvda_news_impact_arc_idea2/output/transformer_features.parquet \
  --bootstrap-iterations 2000

python nvda_news_impact_arc_idea2/generate_report.py
```

To fit a period-specific model manually:

```bash
python nvda_news_impact_arc_idea2/train_models.py \
  --market-periods regular \
  --transformer-features \
  nvda_news_impact_arc_idea2/output/transformer_features.parquet \
  --output-dir nvda_news_impact_arc_idea2/output/models_output_regular
```

Optional return-supervised fine-tuning should be run only after frozen features
have been compared with the simpler models:

```bash
python nvda_news_impact_arc_idea2/fine_tune_finbert.py \
  --task material_direction_15m \
  --market-periods regular \
  --batch-size 32 \
  --output-dir \
  nvda_news_impact_arc_idea2/output/finetuned/material_direction_15m_regular
```

## Leakage and timing invariants

Do not weaken these rules:

1. Split complete `outcome_session_date` blocks chronologically.
2. Fit TF-IDF vocabulary, scalers, imputers, PCA, and estimators inside each
   training fold.
3. Immediate-reaction models may use only information available before the
   article. They must not use the initial post-news return or opening gap.
4. Continuation models may use the initial one- or five-minute response because
   their decision occurs after that response.
5. Closed-market articles are batched by the next NASDAQ session. The opening
   gap belongs to their reaction outcome, not their pre-news predictors.
6. `clean_<horizon>` and `later_news_count_<horizon>` are diagnostics. Do not
   select a live prediction sample using knowledge that later news did not
   arrive.
7. Provider publication timestamps are used as supplied and must not be
   silently shifted to improve results.
8. A language model processing historical text must see only the supplied
   headline/summary. Prefer a frozen checkpoint whose training cutoff predates
   October 2025.

## Model comparison policy

Every news model must be compared with the corresponding NVDA market-only
model on the same chronological test rows.

- Impact classification: ROC AUC, average precision, Brier score, and log loss.
- Direction classification: ROC AUC and balanced accuracy.
- Continuous returns: RMSE, MAE, Spearman correlation, and prediction-tail
  spread.
- Uncertainty: resample complete outcome dates, never individual events.

Do not claim news has incremental predictive value when the paired day-block
interval versus the market-only model includes zero. Immediate-reaction
results are event-study associations, not executable trading returns.

## Expected artifacts

Generated outputs are intentionally Git-ignored and must be rebuilt after a
fresh clone or cherry-pick:

```text
output/market_minutes.parquet
output/market_cache_audit.json
output/news_events.parquet
output/no_news_controls.parquet
output/dataset_audit.json
output/event_study_summary.csv
output/transformer_features.parquet                 # optional
output/models_output/model_metrics.json
output/models_output/model_comparison.csv
output/models_output/test_predictions.csv
output/models_output/walk_forward_metrics.csv
output/models_output/top_features.csv
output/models_output/matched_control_comparison.csv
output/research_report.md
```

Before interpreting results, require `status: complete` in all applicable audit
and metrics JSON files.

## Verification

Run the project-specific tests:

```bash
python -m unittest discover -s nvda_news_impact_arc_idea2/tests -v
```

Then confirm the original ARC workflow remains compatible:

```bash
python -m unittest discover -s tick-news-labeling_arc/tests -v
```

The broader repository currently has two unrelated failures in
`NvidiaDatapull.py` because `boolcod` is undefined. Do not edit that user-owned
file as part of this project unless specifically asked.

## Start-here files

1. Read this file completely.
2. Read `README.md` for design and CLI explanations.
3. Read `BASELINE_RESULTS.md` for the last completed non-transformer benchmark.
4. Use `run_nvda_news_impact_arc_idea2.py --help` before changing defaults.
