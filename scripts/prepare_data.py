"""Validate and normalize response logs and Q-matrix files for SD-CDM."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sd_cdm.data import build_data_bundle
from sd_cdm.utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SD-CDM CSV files.")
    parser.add_argument("--interactions", type=Path, required=True)
    parser.add_argument("--q-matrix", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--student-col", default="student_id")
    parser.add_argument("--exercise-col", default="exercise_id")
    parser.add_argument("--label-col", default="label")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle = build_data_bundle(
        interactions_path=args.interactions,
        q_matrix_path=args.q_matrix,
        student_col=args.student_col,
        exercise_col=args.exercise_col,
        label_col=args.label_col,
    )

    interactions = pd.DataFrame(
        {
            "student_id": bundle.interactions["student_idx"],
            "exercise_id": bundle.interactions["exercise_idx"],
            "label": bundle.interactions["label"],
        }
    )
    q_matrix = pd.DataFrame(bundle.q_matrix, columns=bundle.concept_names)
    q_matrix.insert(0, "exercise_id", range(bundle.num_exercises))

    interactions.to_csv(args.output_dir / "interactions.csv", index=False)
    q_matrix.to_csv(args.output_dir / "q_matrix.csv", index=False)
    save_json(
        {
            "num_students": bundle.num_students,
            "num_exercises": bundle.num_exercises,
            "num_concepts": bundle.num_concepts,
            "student_map": {str(k): int(v) for k, v in bundle.student_map.items()},
            "exercise_map": {str(k): int(v) for k, v in bundle.exercise_map.items()},
            "concept_names": bundle.concept_names,
        },
        args.output_dir / "metadata.json",
    )
    print(f"Prepared files under {args.output_dir}")


if __name__ == "__main__":
    main()
