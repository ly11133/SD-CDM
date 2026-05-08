# Data

Raw datasets should be placed in `data/raw/` and converted into the normalized files under `data/processed/`.

## Interaction File

Required columns:

| Column | Description |
|---|---|
| `student_id` | Original or remapped student identifier |
| `exercise_id` | Original or remapped exercise identifier |
| `label` | Binary response label, where 1 means correct and 0 means incorrect |

Optional columns such as timestamp, dataset split, or response time can be retained but are ignored by the default trainer.

## Q-Matrix File

The default loader expects one row per exercise:

| Column | Description |
|---|---|
| `exercise_id` | Exercise identifier matching the interaction file |
| `c1...cK` | Binary concept indicators |

The first column is treated as the exercise key by default. All remaining columns are treated as concept indicators.

## Releasing Data

Do not commit raw third-party datasets unless the license explicitly allows redistribution. For publication, provide download links and preprocessing commands instead.
