import numpy as np
import pandas as pd
import torch

from sd_cdm.evaluate import degree_of_agreement
from sd_cdm.model import SDCDM


def test_model_forward_shapes():
    model = SDCDM(num_students=4, num_exercises=5, num_concepts=6, emb_dim=8, cam_hidden=16)
    student = torch.tensor([0, 1, 2])
    exercise = torch.tensor([0, 2, 4])
    q_vector = torch.tensor(
        [
            [1, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 1],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([1.0, 0.0, 1.0])

    outputs = model(student, exercise, q_vector, labels=labels)

    assert outputs["pred"].shape == (3,)
    assert outputs["alpha"].shape == (3, 6)
    assert outputs["h_pos"].shape == (3, 6)
    assert outputs["h_neg"].shape == (3, 6)
    assert torch.allclose((outputs["alpha"] * q_vector).sum(dim=1), torch.ones(3), atol=1e-4)
    assert torch.isfinite(outputs["loss"])


def test_uniform_attention_single_concept_item():
    model = SDCDM(num_students=2, num_exercises=2, num_concepts=3, emb_dim=4, cam_hidden=8, use_cam=False)
    outputs = model(
        torch.tensor([0]),
        torch.tensor([0]),
        torch.tensor([[0.0, 1.0, 0.0]]),
        labels=torch.tensor([1.0]),
    )
    assert torch.allclose(outputs["alpha"], torch.tensor([[0.0, 1.0, 0.0]]))


def test_doa_perfect_ordering():
    interactions = pd.DataFrame(
        {
            "student_idx": [0, 0, 1, 1],
            "exercise_idx": [0, 1, 0, 1],
            "label": [1.0, 1.0, 0.0, 0.0],
        }
    )
    q_matrix = np.array([[1.0], [1.0]], dtype=np.float32)
    h_pos = np.array([[0.9], [0.1]], dtype=np.float32)
    assert degree_of_agreement(interactions, q_matrix, h_pos) == 1.0
