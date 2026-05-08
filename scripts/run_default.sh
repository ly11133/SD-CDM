#!/usr/bin/env bash
set -euo pipefail

python -m sd_cdm.train \
  --config configs/default.yaml \
  --interactions data/processed/interactions.csv \
  --q-matrix data/processed/q_matrix.csv \
  --output-dir outputs/default
