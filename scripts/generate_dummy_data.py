"""Generate a small synthetic dataset for smoke-testing the SD-CDM pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dummy SD-CDM CSV files.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--num-students", type=int, default=30)
    parser.add_argument("--num-exercises", type=int, default=15)
    parser.add_argument("--num-concepts", type=int, default=4)
    parser.add_argument("--interactions", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    q_rows = []
    for exercise_idx in range(args.num_exercises):
        concepts = rng.choice(args.num_concepts, size=rng.integers(1, min(3, args.num_concepts) + 1), replace=False)
        row = {"exercise_id": f"e{exercise_idx}"}
        for concept_idx in range(args.num_concepts):
            row[f"c{concept_idx + 1}"] = int(concept_idx in concepts)
        q_rows.append(row)
    q_df = pd.DataFrame(q_rows)

    mastery = rng.beta(2.0, 2.0, size=(args.num_students, args.num_concepts))
    difficulty = rng.normal(0.0, 0.5, size=args.num_exercises)
    q_matrix = q_df[[f"c{i + 1}" for i in range(args.num_concepts)]].to_numpy(float)

    rows = []
    for _ in range(args.interactions):
        student_idx = int(rng.integers(args.num_students))
        exercise_idx = int(rng.integers(args.num_exercises))
        required = q_matrix[exercise_idx] > 0
        ability = mastery[student_idx, required].mean()
        logit = 3.0 * (ability - 0.5) - difficulty[exercise_idx]
        prob = 1.0 / (1.0 + np.exp(-logit))
        label = int(rng.random() < prob)
        rows.append({"student_id": f"s{student_idx}", "exercise_id": f"e{exercise_idx}", "label": label})

    pd.DataFrame(rows).to_csv(args.output_dir / "interactions.csv", index=False)
    q_df.to_csv(args.output_dir / "q_matrix.csv", index=False)
    print(f"Wrote dummy data to {args.output_dir}")


if __name__ == "__main__":
    main()
