#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"

if [[ ! -f data/processed/interactions.csv || ! -f data/processed/q_matrix.csv ]]; then
  echo "Processed data not found. Generating a small dummy dataset under data/processed/."
  "$PYTHON_BIN" scripts/generate_dummy_data.py --output-dir data/processed
fi

"$PYTHON_BIN" -m sd_cdm.train \
  --config configs/default.yaml \
  --interactions data/processed/interactions.csv \
  --q-matrix data/processed/q_matrix.csv \
  --output-dir outputs/default
