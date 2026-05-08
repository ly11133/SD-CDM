# Project Structure

This repository follows a compact research-code layout for cognitive diagnosis experiments.

```text
SD-CDM/
├── configs/              # YAML experiment configurations
├── data/
│   ├── raw/              # Original datasets, not tracked by Git
│   └── processed/        # Normalized CSV files, not tracked by Git
├── docs/                 # Notes for users and reviewers
├── outputs/              # Training logs, metrics, checkpoints
├── scripts/              # Data preparation and convenience commands
├── src/sd_cdm/           # Reusable SD-CDM package
└── tests/                # Minimal correctness checks
```

## Data Flow

1. Put the original response logs and Q-matrix under `data/raw/`.
2. Convert them into normalized CSV files with `scripts/prepare_data.py`.
3. Train SD-CDM with `python -m sd_cdm.train`.
4. Review metrics and checkpoints under `outputs/`.
