#!/bin/bash
#SBATCH --job-name=nvda-news-idea2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=nvda_news_impact_arc_idea2/slurm-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate

python nvda_news_impact_arc_idea2/run_nvda_news_impact_arc_idea2.py \
  --include-quotes \
  --include-transformers \
  --separate-period-models \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --device cuda

# Fine-tune only the most defensible text-sensitive target after all frozen
# baselines are written. It retains its own untouched chronological test block.
python nvda_news_impact_arc_idea2/fine_tune_finbert.py \
  --task material_direction_15m \
  --market-periods regular \
  --batch-size 32 \
  --output-dir nvda_news_impact_arc_idea2/output/finetuned/material_direction_15m_regular
