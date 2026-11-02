# Logistic regression on preceding-news words and phrases

Run from the project root:

```sh
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python News_word_frequency/run_logistic_regression.py
.venv/bin/python News_word_frequency/plot_logistic_regression.py
.venv/bin/python -m unittest discover -s tests -p test_logistic_news.py -v
```

The root requirements pin the installed environment, including scikit-learn,
scipy, numpy and pandas. No new source data is downloaded. The requested AVS
logistic-regression implementation was not found in local files or the fetched
AVS branch; this is a documented standalone analysis, not an exact reproduction.

## Question and sample

Predict existing 1%-up (1) versus 1%-down (0) flags, conditional on a flagged
session. Neutral days are absent: this does not predict whether a large movement
will happen, its magnitude, or volatility. Use 92 events with verified 24-hour
collection windows, and only news strictly before the prior session close.
Headline/summary extraction reuses distinctive_terms.py. Terms count once per day.
Inputs remain unchanged and are checked with SHA-256 hashes.

## Two separate fits

1. **Chronological evaluation:** earliest 75% of eligible days train; latest 25%
   test. Rebuild candidates from articles rather than loading the full-sample
   selected vocabulary. Require at least five training days of support, remove
   constants, and select the top 20 up-associated and top 20 down-associated
   candidates using training labels only. Binary presence features are unscaled.
   L2 logistic regression uses fixed C=0.1, lbfgs, tolerance 1e-8 and max_iter=5000;
   there is no hyperparameter search or test-guided tuning. Predict up at p>=0.5.
   Training labels must end strictly before the first test cutoff (no extra rows
   needed removal for this split). Exact article identity cannot cross the split.
2. **Descriptive inspected-word fit:** fit the previously displayed full-sample
   top-20 lists on all 92 days. These terms were selected using these same labels,
   so coefficients are descriptive and cannot be treated as independently
   validated. No test performance is attributed to this second fit.

Three expanding-window diagnostics inside the training period reselect words
within each fold, use a one-event gap, and enforce the same label cutoff and
article-identity separation. These diagnostics do not tune the final model.
They use equally sized blocks of events, not equally long calendar periods.

See the official [LogisticRegression documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
and [time-aware validation documentation](https://scikit-learn.org/stable/modules/cross_validation.html).

## Results and interpretation

Training: 69 events, October 6–March 10. Test: 23 events, March 12–April 30,
2026 (13 up, 10 down). The fixed training-frequency baseline predicts up for all
23 test events and uses the training up fraction as its probability.

- Logistic test accuracy: 52.2% (12/23); baseline: 56.5% (13/23).
- Logistic test balanced accuracy: 51.9%; baseline: 50.0%.
- Logistic test AUC: 0.531; baseline: 0.500.
- Logistic test log loss: 0.695; baseline: 0.685 (lower is better).
- Training accuracy: 97.1%. The gap to test accuracy indicates poor generalization.

Coefficients are conditional changes in log odds of an up event. `exp(coefficient)`
is the odds ratio for a term's presence, holding other included terms fixed.
Odds ratios are not probability ratios. `phi_correlation_with_up` is a separate
unadjusted Pearson correlation between binary presence and binary direction on
the stated fitting sample. Negative values favor down; positive favor up.

In the descriptive full-sample fit, hit has phi=-0.315 and coefficient=-0.263
(odds ratio 0.769). This is an association, not evidence of prediction or causality.
No conventional p-values are supplied for this regularized, feature-selected
analysis; correlated terms can make individual coefficients unstable.

The full dataset has already been explored in this project. Although the code
keeps later test labels out of fitting and automated selection, the split is an
exploratory temporal check, not an untouched prospective holdout. Small sample
size, thousands of candidate terms, repeated news episodes, retrospective
article versions and incomplete coverage limit conclusions.

## Outputs

- `graphs.html`: AVS-style scatterplots and layered timelines plus logistic
  coefficient, ROC, confusion-matrix, accuracy and test-probability graphs.
  Standalone PNGs are under `plots/`. The six illustrative terms are bullish,
  scale ai, known, deals, hit and securities, chosen from the earlier full-sample
  lists; these graphs are descriptive. Frequency here counts distinct articles
  containing a term, unlike AVS's raw token-occurrence counts. Returns use the
  existing strict temporal matches rather than AVS's calendar-date join.
  Red lines on scatterplots are ordinary linear fits, not logistic curves.
- `scatter_source_data.csv`, `scatter_correlations.csv`: exact day-level plot
  data and Pearson correlations for all 92 eligible flagged days.

- `report.html`: metrics and both coefficient tables.
- `metrics.json`: settings, versions, input hashes, date ranges, confusion
  matrices, baselines, convergence checks and temporal diagnostics.
- `training_coefficients.csv`: train-only selected terms, coefficients, odds
  ratios, correlations and support counts.
- `inspected_words_descriptive_coefficients.csv`: the previously viewed terms,
  fitted on the full sample, explicitly descriptive only.
- `test_predictions.csv`: dated probabilities, predictions, actual flags and
  baseline probabilities for every test event.
- `split_audit.csv`: all included events and their train/test/embargo designation.
- `model_feature_matrix.csv`: binary train-selected vocabulary by event day.
