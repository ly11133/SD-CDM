"""Core implementation of SD-CDM.

The implementation follows the manuscript modules:
BiKD for positive mastery and negative misconception, CAM for strategy-aware
attention, and a binary prediction objective with semantic regularization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualChannelEmbedding(nn.Module):
    """Student-level positive mastery and negative misconception embeddings."""

    def __init__(self, num_students: int, num_concepts: int):
        super().__init__()
        self.e_pos = nn.Embedding(num_students, num_concepts)
        self.e_neg = nn.Embedding(num_students, num_concepts)
        nn.init.xavier_uniform_(self.e_pos.weight)
        nn.init.xavier_uniform_(self.e_neg.weight)

    def forward(self, student_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h_pos = torch.sigmoid(self.e_pos(student_ids))
        h_neg = torch.sigmoid(self.e_neg(student_ids))
        return h_pos, h_neg


class ItemParameters(nn.Module):
    """Exercise difficulty and trap intensity parameters."""

    def __init__(self, num_exercises: int):
        super().__init__()
        self.difficulty = nn.Embedding(num_exercises, 1)
        self.trap_raw = nn.Embedding(num_exercises, 1)
        nn.init.xavier_uniform_(self.difficulty.weight)
        nn.init.uniform_(self.trap_raw.weight, a=-2.0, b=-1.0)

    def forward(self, exercise_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        d_j = self.difficulty(exercise_ids)
        mu_j = torch.sigmoid(self.trap_raw(exercise_ids))
        return d_j, mu_j


class StrategyEmbedding(nn.Module):
    """Student habit, exercise, and concept embeddings used by CAM."""

    def __init__(self, num_students: int, num_exercises: int, num_concepts: int, dim: int):
        super().__init__()
        self.e_habit = nn.Embedding(num_students, dim)
        self.e_exercise = nn.Embedding(num_exercises, dim)
        self.e_concept = nn.Embedding(num_concepts, dim)
        nn.init.xavier_uniform_(self.e_habit.weight)
        nn.init.xavier_uniform_(self.e_exercise.weight)
        nn.init.xavier_uniform_(self.e_concept.weight)

    def forward(self, student_ids: torch.Tensor, exercise_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        v_habit = self.e_habit(student_ids)
        e_j = self.e_exercise(exercise_ids)
        c_k = self.e_concept.weight.unsqueeze(0).expand(student_ids.size(0), -1, -1)
        return v_habit, e_j, c_k


class CognitiveAttentionMachine(nn.Module):
    """Personalized concept attention over the concepts required by an exercise."""

    def __init__(self, dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.w1 = nn.Linear(dim * 3, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        v_habit: torch.Tensor,
        e_j: torch.Tensor,
        c_k: torch.Tensor,
        q_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_concepts, dim = c_k.shape
        v_expand = v_habit.unsqueeze(1).expand(batch_size, num_concepts, dim)
        e_expand = e_j.unsqueeze(1).expand(batch_size, num_concepts, dim)
        z = torch.cat([v_expand, e_expand, c_k], dim=-1)
        logits = self.w2(F.relu(self.w1(z))).squeeze(-1)
        logits = logits.masked_fill(q_mask <= 0, torch.finfo(logits.dtype).min)
        alpha = F.softmax(logits, dim=-1)
        return torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)


def compute_bidirectional_evidence(
    h_pos: torch.Tensor,
    h_neg: torch.Tensor,
    mu_j: torch.Tensor,
    q_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute concept-level evidence h_pos - mu_j * h_neg over required concepts."""

    return (h_pos - mu_j * h_neg) * q_mask


def margin_regularization(
    h_pos: torch.Tensor,
    h_neg: torch.Tensor,
    q_mask: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 0.2,
) -> torch.Tensor:
    """Regularize correct responses so positive evidence exceeds misconception evidence."""

    correct = labels > 0.5
    if correct.sum() == 0:
        return torch.zeros((), device=h_pos.device)

    denom = q_mask.sum(dim=-1).clamp(min=1)
    pos_avg = (h_pos * q_mask).sum(dim=-1) / denom
    neg_avg = (h_neg * q_mask).sum(dim=-1) / denom
    margin = F.relu(neg_avg - pos_avg + epsilon)
    return (margin * correct.float()).sum() / correct.sum().clamp(min=1)


class SDCDM(nn.Module):
    """Strategy-Aware Dual-Channel Cognitive Diagnosis Model."""

    def __init__(
        self,
        num_students: int,
        num_exercises: int,
        num_concepts: int,
        emb_dim: int = 64,
        cam_hidden: int = 256,
        epsilon: float = 0.2,
        lambda_reg: float = 0.1,
    ):
        super().__init__()
        self.epsilon = epsilon
        self.lambda_reg = lambda_reg
        self.dual_channel = DualChannelEmbedding(num_students, num_concepts)
        self.item_params = ItemParameters(num_exercises)
        self.strategy = StrategyEmbedding(num_students, num_exercises, num_concepts, emb_dim)
        self.cam = CognitiveAttentionMachine(emb_dim, cam_hidden)

    def forward(
        self,
        student_ids: torch.Tensor,
        exercise_ids: torch.Tensor,
        q_matrix: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        q_matrix = q_matrix.float()
        h_pos, h_neg = self.dual_channel(student_ids)
        d_j, mu_j = self.item_params(exercise_ids)
        evidence = compute_bidirectional_evidence(h_pos, h_neg, mu_j, q_matrix)
        v_habit, e_j, c_k = self.strategy(student_ids, exercise_ids)
        alpha = self.cam(v_habit, e_j, c_k, q_matrix)
        score = (alpha * evidence).sum(dim=-1) - d_j.squeeze(-1)
        pred = torch.sigmoid(score)

        loss = None
        if labels is not None:
            labels = labels.float()
            pred_loss = F.binary_cross_entropy(pred.clamp(1e-7, 1 - 1e-7), labels)
            reg_loss = margin_regularization(h_pos, h_neg, q_matrix, labels, self.epsilon)
            loss = pred_loss + self.lambda_reg * reg_loss

        return {
            "pred": pred,
            "score": score,
            "h_pos": h_pos,
            "h_neg": h_neg,
            "mu_j": mu_j.squeeze(-1),
            "d_j": d_j.squeeze(-1),
            "v_habit": v_habit,
            "alpha": alpha,
            "loss": loss,
        }

    @torch.no_grad()
    def get_student_profile(self, student_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        h_pos, h_neg = self.dual_channel(student_ids)
        v_habit = self.strategy.e_habit(student_ids)
        return {"h_pos": h_pos, "h_neg": h_neg, "v_habit": v_habit}

    @torch.no_grad()
    def get_item_profile(self, exercise_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        d_j, mu_j = self.item_params(exercise_ids)
        return {"d_j": d_j.squeeze(-1), "mu_j": mu_j.squeeze(-1)}


@dataclass
class TrainerConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-5
    device: str = "cpu"


class SDCDMTrainer:
    """Minimal trainer wrapper used by command-line experiments."""

    def __init__(self, model: SDCDM, config: TrainerConfig):
        self.model = model.to(config.device)
        self.device = config.device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    def train_step(
        self,
        student_ids: torch.Tensor,
        exercise_ids: torch.Tensor,
        q_matrix: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        self.model.train()
        student_ids = student_ids.to(self.device)
        exercise_ids = exercise_ids.to(self.device)
        q_matrix = q_matrix.to(self.device)
        labels = labels.to(self.device)

        out = self.model(student_ids, exercise_ids, q_matrix, labels)
        self.optimizer.zero_grad()
        out["loss"].backward()
        self.optimizer.step()
        return float(out["loss"].detach().cpu())

    @torch.no_grad()
    def predict(self, student_ids: torch.Tensor, exercise_ids: torch.Tensor, q_matrix: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        out = self.model(student_ids.to(self.device), exercise_ids.to(self.device), q_matrix.to(self.device))
        return out["pred"].cpu()
