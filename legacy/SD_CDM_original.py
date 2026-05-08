# coding: utf-8
"""
SD-CDM: Strategy-Aware Dual-Channel Cognitive Diagnosis Model
=============================================================================
Core implementation covering all innovations described in the paper:

  BiKD  — Bidirectional Knowledge Diagnostic Module (h_pos, h_neg, μ_j, d_j)
  CAM   — Cognitive Attention Machine (v_habit → α_ijk)
  Loss  — BCE + margin-based semantic regularization + L2 weight decay

Key outputs (Section 3.7):
  · h_pos, h_neg    — dual-channel knowledge states  [I × K]
  · μ_j             — trap intensity coefficients      [J]
  · d_j             — item difficulty                  [J]
  · v_habit         — strategy habit vectors           [I × D]
  · α_ijk           — per-interaction attention weights
  · P_ij            — predicted correctness probability

Reference equations map to code comments as Eq.(1) – Eq.(12).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional


# ===========================================================================
# Eq.(1) – Dual-Channel Embedding & Bounded Trait Initialization
#  hipos = σ(xi^T E_pos),   hineg = σ(xi^T E_neg)
# ===========================================================================
class DualChannelEmbedding(nn.Module):
    """
    Maps each student to two K-dimensional bounded vectors in (0, 1)
    representing positive mastery and negative misconception.
    """
    def __init__(self, num_students: int, num_concepts: int):
        super().__init__()
        # E_pos, E_neg ∈ R^{I × K}   — trainable positive/negative trait matrices
        self.E_pos = nn.Embedding(num_students, num_concepts)
        self.E_neg = nn.Embedding(num_students, num_concepts)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.E_pos.weight)
        nn.init.xavier_uniform_(self.E_neg.weight)

    def forward(self, stu_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            stu_ids: [B] student indices
        Returns:
            h_pos: [B, K]  positive mastery  ∈ (0, 1)  — hipos in Eq.(1)
            h_neg: [B, K]  negative misconception ∈ (0, 1) — hineg in Eq.(1)
        """
        h_pos = torch.sigmoid(self.E_pos(stu_ids))   # Eq.(1) left
        h_neg = torch.sigmoid(self.E_neg(stu_ids))   # Eq.(1) right
        return h_pos, h_neg


# ===========================================================================
# Eq.(2) – Item Difficulty & Trap Intensity
#  μ_j = σ(m_j),   m_j ~ U(−2.0, −1.0) initially
# ===========================================================================
class ItemParameters(nn.Module):
    """
    Learns scalar difficulty d_j and trap intensity coefficient μ_j per exercise.
    μ_j is initialised with a negative uniform prior so it starts in (0.12, 0.27),
    preventing early saturation of the sigmoid in the negative channel.
    """
    def __init__(self, num_exercises: int):
        super().__init__()
        # trainable scalar difficulty  d_j ∈ R
        self.diff = nn.Embedding(num_exercises, 1)
        nn.init.xavier_uniform_(self.diff.weight)

        # trainable raw trap parameter  m_j ∈ R,  initialised U(−2, −1)
        self.m_raw = nn.Embedding(num_exercises, 1)
        nn.init.uniform_(self.m_raw.weight, a=-2.0, b=-1.0)

    def forward(self, exer_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            d_j: [B, 1]    difficulty (no activation — raw logit space)
            μ_j: [B, 1]    trap intensity ∈ (0, 1)   —  Eq.(2)
        """
        d_j = self.diff(exer_ids)                       # [B, 1]
        mu_j = torch.sigmoid(self.m_raw(exer_ids))     # [B, 1]  Eq.(2)
        return d_j, mu_j


# ===========================================================================
# Eq.(3) – Bidirectional Interaction Mechanism  (BiKD core)
#  g_ijk = h_pos_ik − μ_j · h_neg_ik,   for k ∈ C_j
# ===========================================================================
def compute_gi(
    h_pos: torch.Tensor,
    h_neg: torch.Tensor,
    mu_j: torch.Tensor,
    q_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Concept-level bidirectional contribution before CAM aggregation.

    Args:
        h_pos:  [B, K]  positive mastery
        h_neg:  [B, K]  negative misconception
        mu_j:   [B, 1]  trap intensity coefficient
        q_mask: [B, K]  binary mask — 1 if concept k is required by exercise e_j

    Returns:
        gi: [B, K]  g_ijk values (masked: 0 for non-required concepts)
            gi[b, k] = h_pos[b,k] − μ_j[b] · h_neg[b,k]   when q_mask[b,k]=1
                       0                                    otherwise
    Eq.(3)
    """
    penalty = mu_j * h_neg                         # [B, K]  μ_j h_neg_ik
    gi = (h_pos - penalty) * q_mask                # [B, K]  Eq.(3)
    return gi


# ===========================================================================
# Eq.(4) – Latent Strategy Habit Vector Modeling
#  v_habit = xi^T E_habit,   e_j = yj^T E_ex,   c_k = zk^T E_con
# ===========================================================================
class StrategyEmbedding(nn.Module):
    """
    Dense embeddings for habit-driven concept attention.
    """
    def __init__(self, num_students: int, num_exercises: int,
                 num_concepts: int, dim: int = 64):
        super().__init__()
        self.dim = dim
        self.E_habit = nn.Embedding(num_students, dim)     # I × D
        self.E_ex    = nn.Embedding(num_exercises, dim)     # J × D
        self.E_con   = nn.Embedding(num_concepts, dim)      # K × D

        nn.init.xavier_uniform_(self.E_habit.weight)
        nn.init.xavier_uniform_(self.E_ex.weight)
        nn.init.xavier_uniform_(self.E_con.weight)

    def forward(self, stu_ids: torch.Tensor, exer_ids: torch.Tensor, con_ids: torch.Tensor):
        """
        Args:
            stu_ids:  [B]      student indices
            exer_ids: [B]      exercise indices
            con_ids:  [B, K]   concept indices (0..K-1 for required concepts,
                               padded with 0 elsewhere; masked out downstream)
        Returns:
            v_habit:  [B, D]      — Eq.(4) left
            e_j:      [B, D]      — Eq.(4) middle
            C_k:      [B, K, D]  — Eq.(4) right (all concepts, for CAM scoring)
        """
        v_habit = self.E_habit(stu_ids)                    # [B, D]
        e_j     = self.E_ex(exer_ids)                       # [B, D]
        C_k     = self.E_con.weight.unsqueeze(0)            # [1, K, D] → broadcast to [B, K, D]
        return v_habit, e_j, C_k


# ===========================================================================
# Eqs.(5)–(7) – Cognitive Attention Machine (CAM)
#  z_ijk = [v_habit_i || e_j || c_k]                                Eq.(5)
#  a_ijk = w2^T · ReLU(W1 · z_ijk + b1) + b2                        Eq.(6)
#  α_ijk = softmax_{k'∈C_j}(exp(a_ijk'))                             Eq.(7)
# ===========================================================================
class CAM(nn.Module):
    """
    Strategy-aware concept attention: given a student's habit vector and the
    exercise context, produce a personalised attention distribution over the
    required concepts.
    """
    def __init__(self, dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        # z_ijk  is 3D-dimensional  (concatenation of 3 D-dim vectors)
        self.W1 = nn.Linear(dim * 3, hidden_dim)           # H × 3D
        self.W2 = nn.Linear(hidden_dim, 1)                 # H → 1

    def forward(self, v_habit: torch.Tensor, e_j: torch.Tensor,
                C_k: torch.Tensor, q_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            v_habit: [B, D]     strategy habit vector
            e_j:     [B, D]     exercise embedding
            C_k:     [B, K, D]  concept embeddings (all K concepts)
            q_mask:  [B, K]     Q-matrix mask: 1 where concept k required

        Returns:
            alpha:   [B, K]     attention weights α_ijk (0 where q_mask=0)
        """
        B, K, D = C_k.shape

        # Eq.(5):  z_ijk = [v_habit || e_j || c_k]
        # Expand v_habit and e_j to [B, K, D]
        v_exp = v_habit.unsqueeze(1).expand(B, K, D)       # [B, K, D]
        e_exp = e_j.unsqueeze(1).expand(B, K, D)           # [B, K, D]
        z = torch.cat([v_exp, e_exp, C_k], dim=-1)          # [B, K, 3D]  Eq.(5)

        # Eq.(6):  a_ijk = w2^T · ReLU(W1 · z_ijk + b1) + b2
        a = self.W2(F.relu(self.W1(z))).squeeze(-1)         # [B, K]      Eq.(6)

        # Eq.(7):  softmax exclusively within C_j
        a_masked = a.masked_fill(q_mask == 0, float('-inf'))
        alpha = F.softmax(a_masked, dim=-1)                 # [B, K]      Eq.(7)
        alpha = torch.nan_to_num(alpha, nan=0.0)            # handle |C_j|=0 edge case
        return alpha


# ===========================================================================
# Eq.(8)–(9) – Strategy-Weighted Prediction Fusion → Probability
#  S_ij = Σ_{k∈C_j} α_ijk · g_ijk − d_j                          Eq.(8)
#  P_ij = 1 / (1 + exp(−S_ij))                                    Eq.(9)
# ===========================================================================
def compute_prediction(alpha: torch.Tensor, gi: torch.Tensor,
                       d_j: torch.Tensor) -> torch.Tensor:
    """
    Args:
        alpha: [B, K]  CAM attention weights
        gi:    [B, K]  BiKD concept-level contributions (already Q-masked)
        d_j:   [B, 1]  item difficulty
    Returns:
        P_ij:  [B]     predicted correctness probability ∈ (0, 1)
    """
    S_ij = (alpha * gi).sum(dim=-1) - d_j.squeeze(-1)       # [B]  Eq.(8)
    P_ij = torch.sigmoid(S_ij)                               # [B]  Eq.(9)
    return P_ij


# ===========================================================================
# Eq.(11) – Margin-based Semantic Regularization
#  L_reg = Σ_{(ui,ej)∈R+} max(0, mean(h_neg) − mean(h_pos) + ε) / |C_j|
# ===========================================================================
def margin_regularization(
    h_pos: torch.Tensor,
    h_neg: torch.Tensor,
    q_mask: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 0.2,
) -> torch.Tensor:
    """
    For each *correctly answered* interaction, enforce that the average
    positive mastery exceeds the average negative misconception by at
    least ε on the concepts required by the exercise.

    Args:
        h_pos:   [B, K]  positive mastery
        h_neg:   [B, K]  negative misconception
        q_mask:  [B, K]  Q-matrix mask
        labels:  [B]     observed response ri_j ∈ {0, 1}
        epsilon: margin hyperparameter
    Returns:
        L_reg: scalar
    """
    # only apply to correct responses  R+
    correct_mask = (labels > 0.5)                            # [B]

    if correct_mask.sum() == 0:
        return torch.tensor(0.0, device=h_pos.device)

    # number of required concepts per exercise, used as normaliser
    n_concepts = q_mask.sum(dim=-1).clamp(min=1)             # [B]

    # average positive / negative activation over C_j
    pos_avg = (h_pos * q_mask).sum(dim=-1) / n_concepts      # [B]
    neg_avg = (h_neg * q_mask).sum(dim=-1) / n_concepts      # [B]

    # hinge: max(0, neg_avg − pos_avg + ε)
    margin = F.relu(neg_avg - pos_avg + epsilon)              # [B]

    L_reg = (margin * correct_mask.float()).sum() / correct_mask.sum().clamp(min=1)
    return L_reg


# ===========================================================================
# Eq.(10),(12) – Full SD-CDM Model
#  L = L_pred + λ · L_reg + λ_Θ · ||Θ||²
# ===========================================================================
class SDCDM(nn.Module):
    """
    Strategy-Aware Dual-Channel Cognitive Diagnosis Model.

    Parameters
    ----------
    num_students  : I
    num_exercises : J
    num_concepts  : K
    emb_dim       : D  (default 64, as reported in the paper)
    cam_hidden    : H  (default 256)
    epsilon       : margin for Eq.(11)       (default 0.2)
    lambda_reg    : λ  for Eq.(12)           (default 0.1)
    weight_decay  : λ_Θ for L2 in Eq.(12)    (default 1e-5, applied via optimizer)

    Usage
    -----
    model = SDCDM(I, J, K)
    outputs = model(stu_ids, exer_ids, q_matrix)
    # outputs.dict  → everything defined in Section 3.7
    loss = outputs.loss
    """

    def __init__(self, num_students: int, num_exercises: int,
                 num_concepts: int, emb_dim: int = 64,
                 cam_hidden: int = 256, epsilon: float = 0.2,
                 lambda_reg: float = 0.1):
        super().__init__()

        self.epsilon = epsilon
        self.lambda_reg = lambda_reg

        # --- Modules ---
        self.dual_channel = DualChannelEmbedding(num_students, num_concepts)
        self.item_params  = ItemParameters(num_exercises)
        self.strategy     = StrategyEmbedding(num_students, num_exercises,
                                              num_concepts, emb_dim)
        self.cam          = CAM(dim=emb_dim, hidden_dim=cam_hidden)

    def forward(self, stu_ids: torch.Tensor, exer_ids: torch.Tensor,
                q_matrix: torch.Tensor, labels: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        """
        Args:
            stu_ids:  [B]      student indices
            exer_ids: [B]      exercise indices
            q_matrix: [B, K]   Q-matrix rows (binary mask per exercise)
            labels:   [B]      observed ri_j ∈ {0, 1}  (required for loss)

        Returns dict with keys:
            P_ij     [B]     predicted probability           Eq.(9)
            h_pos    [B, K]  positive mastery                Eq.(1)
            h_neg    [B, K]  negative misconception          Eq.(1)
            mu_j     [B, 1]  trap intensity coefficient       Eq.(2)
            d_j      [B, 1]  item difficulty
            v_habit  [B, D]  strategy habit vector           Eq.(4)
            alpha    [B, K]  CAM attention weights           Eq.(7)
            loss     scalar  total loss (None if labels not given)
        """
        # ---- Stage 1: BiKD - Dual-Channel Knowledge Initialisation ----
        h_pos, h_neg = self.dual_channel(stu_ids)            # Eq.(1)
        d_j, mu_j    = self.item_params(exer_ids)            # Eq.(2)
        gi           = compute_gi(h_pos, h_neg, mu_j, q_matrix)  # Eq.(3)

        # ---- Stage 2: CAM - Strategy-Aware Concept Activation ----
        v_habit, e_j, C_k = self.strategy(stu_ids, exer_ids, None)  # Eq.(4)
        alpha = self.cam(v_habit, e_j, C_k, q_matrix)                # Eqs.(5-7)

        # ---- Stage 3: Prediction ----
        P_ij = compute_prediction(alpha, gi, d_j)           # Eqs.(8-9)

        # ---- Loss computation ----
        loss = None
        if labels is not None:
            # BCE prediction loss                                       Eq.(10)
            L_pred = F.binary_cross_entropy(
                P_ij.clamp(1e-7, 1 - 1e-7), labels.float()
            )
            # margin semantic regularisation                            Eq.(11)
            L_reg = margin_regularization(h_pos, h_neg, q_matrix,
                                          labels, self.epsilon)
            # total objective                                            Eq.(12)
            loss = L_pred + self.lambda_reg * L_reg

        return {
            'P_ij':    P_ij,
            'h_pos':   h_pos,
            'h_neg':   h_neg,
            'mu_j':    mu_j.squeeze(-1),
            'd_j':     d_j.squeeze(-1),
            'v_habit': v_habit,
            'alpha':   alpha,
            'loss':    loss,
        }

    # ------------------------------------------------------------------
    # Convenience: extract per-student, per-exercise diagnostic profiles
    # after training (used for RQ3–RQ5 analyses, Sections 4.4–4.6)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_diagnostic_profile(self, stu_ids: torch.Tensor
                               ) -> Dict[str, torch.Tensor]:
        """
        Returns full-concept diagnostic profiles for a set of students.
        (No exercise context needed — these are the learned traits.)
        """
        h_pos, h_neg = self.dual_channel(stu_ids)
        v_habit = self.strategy.E_habit(stu_ids)
        return {'h_pos': h_pos, 'h_neg': h_neg, 'v_habit': v_habit}

    @torch.no_grad()
    def get_item_profile(self, exer_ids: torch.Tensor
                         ) -> Dict[str, torch.Tensor]:
        """Returns item-level diagnostic parameters."""
        d_j, mu_j = self.item_params(exer_ids)
        return {'d_j': d_j.squeeze(-1), 'mu_j': mu_j.squeeze(-1)}


# ===========================================================================
# Training loop (cf. Section 4.1.3 — Adam, lr=0.001, batch_size=256,
#   early stopping when val AUC doesn't improve for 10 epochs)
# ===========================================================================
class SDCDMTrainer:
    """
    Simple trainer matching the paper's training protocol.

    Example
    -------
    trainer = SDCDMTrainer(model, lr=0.001, weight_decay=1e-5)
    for epoch in range(max_epochs):
        for batch in train_loader:
            trainer.train_step(batch)
        val_auc = trainer.evaluate(val_loader)
    """

    def __init__(self, model: SDCDM, lr: float = 0.001,
                 weight_decay: float = 1e-5, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

    def train_step(self, stu_ids: torch.Tensor, exer_ids: torch.Tensor,
                   q_matrix: torch.Tensor, labels: torch.Tensor) -> float:
        """Single batch forward + backward. Returns loss value."""
        self.model.train()

        stu_ids, exer_ids = stu_ids.to(self.device), exer_ids.to(self.device)
        q_matrix = q_matrix.to(self.device)
        labels   = labels.to(self.device)

        out = self.model(stu_ids, exer_ids, q_matrix, labels)

        self.optimizer.zero_grad()
        out['loss'].backward()
        self.optimizer.step()

        return out['loss'].item()

    @torch.no_grad()
    def predict(self, stu_ids: torch.Tensor, exer_ids: torch.Tensor,
                q_matrix: torch.Tensor) -> torch.Tensor:
        """Prediction only (no loss)."""
        self.model.eval()
        stu_ids, exer_ids = stu_ids.to(self.device), exer_ids.to(self.device)
        q_matrix = q_matrix.to(self.device)
        out = self.model(stu_ids, exer_ids, q_matrix)
        return out['P_ij'].cpu()

    @torch.no_grad()
    def get_diagnostics(self, stu_ids: torch.Tensor, exer_ids: torch.Tensor,
                        q_matrix: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extracts all interpretable variables for post-hoc analysis
        (Sections 4.4–4.6), as described in Section 4.1.4.

        Returns CPU tensors:
            P_ij, h_pos, h_neg, mu_j, d_j, v_habit, alpha
        """
        self.model.eval()
        stu_ids, exer_ids = stu_ids.to(self.device), exer_ids.to(self.device)
        q_matrix = q_matrix.to(self.device)

        out = self.model(stu_ids, exer_ids, q_matrix)
        return {k: v.cpu() for k, v in out.items() if k != 'loss'}
