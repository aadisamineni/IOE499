# Completed quote-level baseline

This file records the completed non-transformer benchmark so a new agent can
distinguish existing evidence from work that still needs to run on the HPC
cluster.

## Completed processing

- 452,785,851 raw NVDA SIP quote rows streamed.
- 139,268 NVDA minute bars materialized.
- 139,115 minute rows use a fresh quote midpoint; 153 fall back to bar close.
- Median observed minute-ending spread: approximately 1.10 basis points.
- 5,996 direct Nvidia source articles retained.
- 4,695 modeled information events created.
- 4,677 events have valid market anchors.
- Event periods: 2,038 regular, 1,879 premarket, 636 after-hours, and 142
  closed-market/overnight batches.
- 533 strict no-news control intervals constructed.

The generated artifacts remain in `output/` in the originating workspace but
are Git-ignored. A different session must regenerate them from the raw inputs.

## Held-out pooled baseline

The last 15% of eligible outcome dates is held out. Hyperparameters are chosen
using expanding-window folds inside the earlier development period.

| Task | Best held-out result | Market-only comparison | Interpretation |
| --- | ---: | ---: | --- |
| 5-minute impact | market-only ROC AUC 0.585 | 0.585 | News did not add incrementally. |
| 15-minute impact | market-only ROC AUC 0.614 | 0.614 | NVDA state predicts magnitude. |
| 60-minute impact | market-only ROC AUC 0.678 | 0.678 | Strongest magnitude baseline. |
| 15-minute direction | structured-news ROC AUC 0.501 | market-only 0.470 | Near chance; paired interval includes zero. |
| Material-event 15-minute direction | structured-news ROC AUC 0.548 | market-only 0.492 | Tentative; paired interval includes zero. |
| 1-to-15-minute continuation | market-only ROC AUC 0.485 | 0.485 | No useful continuation signal. |
| 5-to-60-minute continuation | market-only ROC AUC 0.513 | 0.513 | No useful continuation signal. |

Regular-session-only material-direction modeling was somewhat stronger:
market plus structured news reached ROC AUC 0.593 versus 0.514 market-only, but
the paired day-block interval still crossed zero. Treat this as a hypothesis
for FinBERT, not a confirmed result.

## Matched no-news event study

Regular-session news events were matched to no-news NVDA intervals using time
of day, pre-event volatility, and prior return.

| Horizon | Mean news minus control absolute return | Day-block 95% interval |
| --- | ---: | ---: |
| 5 minutes | +0.000278, or +2.78 bps | [+1.70, +3.83] bps |
| 15 minutes | +0.000254, or +2.54 bps | [+0.45, +4.56] bps |
| 60 minutes | -0.000127, or -1.27 bps | [-6.30, +4.12] bps |

The current evidence therefore supports **short-horizon news impact/magnitude**
more strongly than signed direction or tradable continuation.

## Remaining HPC work

The frozen FinBERT and fine-tuning code is implemented but has not been run in
the originating local environment. Run the full Slurm job after installing
`requirements-transformers.txt`. The job will:

1. reuse or rebuild quote caches;
2. extract frozen FinBERT probabilities and embeddings;
3. refit pooled and separate-period models with those features;
4. regenerate the report; and
5. fine-tune FinBERT on regular-session material-event direction with an
   untouched chronological test block.

Do not skip the frozen-feature comparison and report only the fine-tuned result.
