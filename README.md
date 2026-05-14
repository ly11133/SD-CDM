# SD-CDM

Official implementation scaffold for **Strategy-Aware Dual-Channel Cognitive Diagnosis for Student Modeling**.

SD-CDM models student cognition with two diagnostic channels and a strategy-aware concept attention mechanism:

- **BiKD**: bidirectional knowledge diagnosis with positive mastery and negative misconception states.
- **CAM**: cognitive attention machine for student-specific concept activation.
- **Prediction**: strategy-weighted concept evidence with item difficulty and trap intensity.

This repository is prepared for paper submission and reproducibility. Experimental numbers in the manuscript should be regenerated with the final cleaned datasets before public release.

## Repository Structure

```text
SD-CDM/
  configs/                 # Experiment configuration files
  data/                    # Dataset format documentation and local data folders
  scripts/                 # Data preparation helpers
  src/sd_cdm/              # Model, data loading, training, evaluation
  tests/                   # Minimal smoke tests
  outputs/                 # Local experiment outputs, ignored by git
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Data Format

Interactions are expected as a CSV file with at least:

```text
student_id,exercise_id,label
```

The Q-matrix is expected as a CSV file where each row corresponds to one exercise and each concept column is binary:

```text
exercise_id,c1,c2,c3,...
```

See `data/README.md` for details.

## Quick Start

```bash
python scripts/generate_dummy_data.py --output-dir data/processed

python -m sd_cdm.train \
  --config configs/default.yaml \
  --interactions data/processed/interactions.csv \
  --q-matrix data/processed/q_matrix.csv \
  --output-dir outputs/default
```

The training command writes the following local artifacts under `outputs/default/`:

- `best_model.pt`
- `history.csv`
- `metrics.json`
- `mappings.json`

The default config is intentionally small so the public smoke run finishes
quickly. Use a separate experiment config for full paper-scale runs.

You can also run the default script:

```bash
bash scripts/run_default.sh
```

Run smoke tests with:

```bash
python -m pytest tests
```

## Model Variants

Use `--variant` to run the implemented ablations:

```bash
python -m sd_cdm.train \
  --config configs/default.yaml \
  --interactions data/processed/interactions.csv \
  --q-matrix data/processed/q_matrix.csv \
  --output-dir outputs/base \
  --variant base
```

Available variants are `full`, `base`, `wo-bikd`, and `wo-cam`. The default is `full`.

## Reproducibility Notes

- Raw datasets are not committed to this repository.
- Keep dataset preprocessing scripts deterministic.
- Store final fold splits, random seeds, and evaluation logs under `outputs/` during experiments.
- Before public release, verify that all reported manuscript results are reproduced by the committed code and documented commands.
- Real third-party datasets must be downloaded by users and converted to the format in `data/README.md`; do not upload private or redistributable-restricted raw data.

## Citation

```bibtex
@article{lu2026sdcdm,
  title   = {Strategy-Aware Dual-Channel Cognitive Diagnosis for Student Modeling},
  author  = {Su, Yu and Lu, Yu and Lu, Junyu and Yang, Baoyi and Li, Hongqing and Huang, Zhenya},
  journal = {Expert Systems with Applications},
  year    = {2026},
  note    = {Manuscript under preparation}
}
```

Replace this BibTeX entry after the article is formally published.

## License

This project currently uses the MIT License. Confirm the license with all authors before making the repository public.
