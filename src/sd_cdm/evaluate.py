"""Evaluation metrics for cognitive diagnosis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score


def binary_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(float)
    y_pred = np.asarray(y_pred).astype(float)
    y_label = (y_pred >= 0.5).astype(int)
    metrics = {
        "acc": float(accuracy_score(y_true, y_label)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }
    metrics["auc"] = float(roc_auc_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan")
    return metrics


def degree_of_agreement(interactions: pd.DataFrame, q_matrix: np.ndarray, h_pos: np.ndarray) -> float:
    """Compute concept-level Degree of Agreement (DOA).

    DOA compares empirical student ordering on each concept with the ordering
    implied by learned positive mastery values.
    """

    required = {"student_idx", "exercise_idx", "label"}
    missing = required - set(interactions.columns)
    if missing:
        raise ValueError(f"Interactions are missing indexed columns: {sorted(missing)}")

    concept_scores: list[float] = []
    for concept_idx in range(q_matrix.shape[1]):
        concept_exercises = np.flatnonzero(q_matrix[:, concept_idx] > 0)
        if concept_exercises.size == 0:
            continue

        subset = interactions[interactions["exercise_idx"].isin(concept_exercises)]
        if subset.empty:
            continue

        rates = subset.groupby("student_idx")["label"].mean()
        students = rates.index.to_numpy(dtype=int)
        if len(students) < 2:
            continue

        agree = 0
        total = 0
        empirical = rates.to_numpy(dtype=float)
        mastery = h_pos[students, concept_idx]
        for i in range(len(students)):
            for j in range(i + 1, len(students)):
                if empirical[i] == empirical[j]:
                    continue
                total += 1
                empirical_order = empirical[i] > empirical[j]
                mastery_order = mastery[i] > mastery[j]
                if empirical_order == mastery_order:
                    agree += 1
        if total > 0:
            concept_scores.append(agree / total)

    return float(np.mean(concept_scores)) if concept_scores else float("nan")
