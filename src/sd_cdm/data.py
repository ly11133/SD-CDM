"""Data utilities for SD-CDM experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class DataBundle:
    interactions: pd.DataFrame
    q_matrix: np.ndarray
    student2idx: Dict[str, int]
    exercise2idx: Dict[str, int]
    concept_names: list[str]

    @property
    def num_students(self) -> int:
        return len(self.student2idx)

    @property
    def num_exercises(self) -> int:
        return len(self.exercise2idx)

    @property
    def num_concepts(self) -> int:
        return len(self.concept_names)

    @property
    def student_map(self) -> Dict[str, int]:
        return self.student2idx

    @property
    def exercise_map(self) -> Dict[str, int]:
        return self.exercise2idx


class InteractionDataset(Dataset):
    """Torch dataset returning student, exercise, Q-row, and label tensors."""

    def __init__(self, interactions: pd.DataFrame, q_matrix: np.ndarray):
        required = {"student_idx", "exercise_idx", "label"}
        missing = required - set(interactions.columns)
        if missing:
            raise ValueError(f"Interactions are missing indexed columns: {sorted(missing)}")

        self.student_ids = interactions["student_idx"].to_numpy(dtype=np.int64)
        self.exercise_ids = interactions["exercise_idx"].to_numpy(dtype=np.int64)
        self.labels = interactions["label"].to_numpy(dtype=np.float32)
        self.q_matrix = q_matrix.astype(np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        exercise_id = self.exercise_ids[index]
        return (
            torch.tensor(self.student_ids[index], dtype=torch.long),
            torch.tensor(exercise_id, dtype=torch.long),
            torch.tensor(self.q_matrix[exercise_id], dtype=torch.float32),
            torch.tensor(self.labels[index], dtype=torch.float32),
        )


def load_interactions(path: str | Path, student_col: str, exercise_col: str, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {student_col, exercise_col, label_col} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in interactions file: {sorted(missing)}")

    df = df.rename(columns={student_col: "student_id", exercise_col: "exercise_id", label_col: "label"})
    df = df[["student_id", "exercise_id", "label"]].copy()
    df["student_id"] = df["student_id"].astype(str)
    df["exercise_id"] = df["exercise_id"].astype(str)
    df["label"] = df["label"].astype(float)

    invalid_labels = sorted(set(df["label"].unique()) - {0.0, 1.0})
    if invalid_labels:
        raise ValueError(f"Labels must be binary 0/1; found {invalid_labels[:10]}")
    return df


def load_q_matrix(path: str | Path, exercise_key_col: str = "exercise_id") -> Tuple[pd.DataFrame, list[str]]:
    q_df = pd.read_csv(path)
    if exercise_key_col not in q_df.columns:
        raise ValueError(f"Q-matrix must contain exercise key column: {exercise_key_col}")

    concept_cols = [col for col in q_df.columns if col != exercise_key_col]
    if not concept_cols:
        raise ValueError("Q-matrix must contain at least one concept column")

    q_df = q_df.copy()
    q_df[exercise_key_col] = q_df[exercise_key_col].astype(str)
    if q_df[exercise_key_col].duplicated().any():
        dupes = q_df.loc[q_df[exercise_key_col].duplicated(), exercise_key_col].head(10).tolist()
        raise ValueError(f"Q-matrix contains duplicate exercise IDs: {dupes}")

    q_df[concept_cols] = q_df[concept_cols].astype(float)
    if not q_df[concept_cols].isin([0.0, 1.0]).all().all():
        raise ValueError("Q-matrix concept columns must be binary 0/1")
    if (q_df[concept_cols].sum(axis=1) == 0).any():
        raise ValueError("Each Q-matrix row must require at least one concept")
    return q_df, concept_cols


def build_data_bundle(
    interactions_path: str | Path,
    q_matrix_path: str | Path,
    student_col: str = "student_id",
    exercise_col: str = "exercise_id",
    label_col: str = "label",
    exercise_key_col: str = "exercise_id",
) -> DataBundle:
    interactions = load_interactions(interactions_path, student_col, exercise_col, label_col)
    q_df, concept_names = load_q_matrix(q_matrix_path, exercise_key_col)

    missing_exercises = sorted(set(interactions["exercise_id"]) - set(q_df[exercise_key_col]))
    if missing_exercises:
        raise ValueError(f"Interactions contain exercises missing from Q-matrix: {missing_exercises[:10]}")

    student_ids = sorted(interactions["student_id"].unique())
    exercise_ids = sorted(q_df[exercise_key_col].unique())
    student2idx = {sid: idx for idx, sid in enumerate(student_ids)}
    exercise2idx = {eid: idx for idx, eid in enumerate(exercise_ids)}

    interactions["student_idx"] = interactions["student_id"].map(student2idx).astype(int)
    interactions["exercise_idx"] = interactions["exercise_id"].map(exercise2idx).astype(int)

    q_df["_exercise_idx"] = q_df[exercise_key_col].map(exercise2idx).astype(int)
    q_df = q_df.sort_values("_exercise_idx")
    q_matrix = q_df[concept_names].to_numpy(dtype=np.float32)
    return DataBundle(interactions, q_matrix, student2idx, exercise2idx, concept_names)
