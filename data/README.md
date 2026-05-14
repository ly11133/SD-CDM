# Data

Raw datasets should be placed in `data/raw/` and converted into the normalized files under `data/processed/`.

The repository intentionally keeps both folders ignored by git except for
`.gitkeep`. Do not commit private datasets or third-party datasets whose license
does not allow redistribution.

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

Each exercise row must contain at least one `1` across the concept columns.

## Dummy Data

For a local smoke test, generate synthetic files with:

```bash
python scripts/generate_dummy_data.py --output-dir data/processed
```

These files are only for checking that installation, training, and tests run
end to end. They are not used for paper results.

## Releasing Data

Do not commit raw third-party datasets unless the license explicitly allows redistribution. For publication, provide download links and preprocessing commands instead.
