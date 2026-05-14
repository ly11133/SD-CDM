# Reproducibility Guide

This repository provides a clean public implementation and a dummy-data smoke
pipeline. The default config is intentionally small for quick validation. It
does not include third-party raw datasets or complete paper-scale 5-fold by
5-seed experiment outputs.

## Data Format

Prepare two CSV files:

```text
data/processed/interactions.csv
data/processed/q_matrix.csv
```

`interactions.csv` must contain:

```text
student_id,exercise_id,label
```

`q_matrix.csv` must contain one exercise key column followed by binary concept
columns:

```text
exercise_id,c1,c2,c3,...
```

Every exercise in `interactions.csv` must appear in `q_matrix.csv`, and every
Q-matrix row must include at least one required concept.

## Smoke Run

Generate synthetic data and run the default model:

```bash
python scripts/generate_dummy_data.py --output-dir data/processed

python -m sd_cdm.train \
  --config configs/default.yaml \
  --interactions data/processed/interactions.csv \
  --q-matrix data/processed/q_matrix.csv \
  --output-dir outputs/default
```

Or run:

```bash
bash scripts/run_default.sh
```

## Output Files

The training command writes:

- `best_model.pt`: best validation checkpoint.
- `history.csv`: epoch-level training loss and validation metrics.
- `metrics.json`: final test metrics, dataset sizes, split sizes, and variant.
- `mappings.json`: student, exercise, and concept mappings used by the run.

## Metrics

The default trainer reports AUC, ACC, RMSE, and DOA on the test split. DOA is
computed from learned positive mastery profiles and concept-level empirical
student ordering.

## Paper-Scale Reproduction

For manuscript numbers, record and report:

- Dataset source, license, preprocessing rules, and filtering thresholds.
- Train, validation, and test split protocol.
- Random seeds or fold identifiers.
- Hyperparameters used for SD-CDM and baselines.
- Mean and standard deviation over folds.
- Statistical tests against the strongest baseline.
- Hardware and software environment.
