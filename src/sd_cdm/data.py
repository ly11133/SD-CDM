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


class InteractionDataset(Dataset):
    """Torch dataset returning student, exercise, Q-row, and label tensors."""

    def __init__(self, interactions: pd.DataFrame, q_matrix: np.ndarray):
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
    df["label"] = df["label"].astype(float)
    return df[["student_id", "exercise_id", "label"]].copy()


def load_q_matrix(path: str | Path, exercise_key_col: str = "exercise_id") -> Tuple[pd.DataFrame, list[str]]:
    q_df = pd.read_csv(path)
    if exercise_key_col not in q_df.columns:
        raise ValueError(f"Q-matrix must contain exercise key column: {exercise_key_col}")
    concept_cols = [col for col in q_df.columns if col != exercise_key_col]
    if not concept_cols:
        raise ValueError("Q-matrix must contain at least one concept column")
    q_df[concept_cols] = q_df[concept_cols].astype(float)
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

    student_ids = sorted(interactions["student_id"].astype(str).unique())
    exercise_ids = sorted(q_df[exercise_key_col].astype(str).unique())
    student2idx = {sid: idx for idx, sid in enumerate(student_ids)}
    exercise2idx = {eid: idx for idx, eid in enumerate(exercise_ids)}

    interactions["student_idx"] = interactions["student_id"].astype(str).map(student2idx)
    interactions["exercise_idx"] = interactions["exercise_id"].astype(str).map(exercise2idx)
    if interactions["exercise_idx"].isna().any():
        missing = interactions.loc[interactions["exercise_idx"].isna(), "exercise_id"].unique()[:10]
        raise ValueError(f"Interactions contain exercises missing from Q-matrix: {missing}")

    q_df["_exercise_idx"] = q_df[exercise_key_col].astype(str).map(exercise2idx)
    q_df = q_df.sort_values("_exercise_idx")
    q_matrix = q_df[concept_names].to_numpy(dtype=np.float32)
    return DataBundle(interactions, q_matrix, student2idx, exercise2idx, concept_names)
