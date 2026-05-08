import torch

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

    assert outputs["prob"].shape == (3,)
    assert outputs["alpha"].shape == (3, 6)
    assert torch.allclose((outputs["alpha"] * q_vector).sum(dim=1), torch.ones(3), atol=1e-5)
    assert torch.isfinite(outputs["loss"])
