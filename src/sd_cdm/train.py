"""Command-line training entry point for SD-CDM."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .data import InteractionDataset, build_data_bundle
from .evaluate import binary_metrics, degree_of_agreement
from .model import SDCDM, SDCDMTrainer, TrainerConfig
from .utils import resolve_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SD-CDM on response logs and a Q-matrix.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--interactions", type=Path, required=True)
    parser.add_argument("--q-matrix", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/run"))
    parser.add_argument("--variant", choices=["full", "base", "wo-bikd", "wo-cam"], default="full")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def split_indices(n_items: int, val_ratio: float, test_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_items)
    n_test = int(round(n_items * test_ratio))
    n_val = int(round(n_items * val_ratio))
    test_idx = indices[:n_test]
    val_idx = indices[n_test : n_test + n_val]
    train_idx = indices[n_test + n_val :]
    return train_idx, val_idx, test_idx


def make_loader(dataset: InteractionDataset, indices: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(Subset(dataset, indices.tolist()), batch_size=batch_size, shuffle=shuffle)


def variant_flags(name: str) -> tuple[bool, bool]:
    if name == "base":
        return False, False
    if name == "wo-bikd":
        return False, True
    if name == "wo-cam":
        return True, False
    return True, True


def validation_score(metrics: dict[str, float]) -> float:
    auc = metrics.get("auc", float("nan"))
    if np.isfinite(auc):
        return auc
    acc = metrics.get("acc", float("nan"))
    rmse = metrics.get("rmse", float("nan"))
    if np.isfinite(acc) and np.isfinite(rmse):
        return acc - rmse
    if np.isfinite(acc):
        return acc
    return -np.inf


@torch.no_grad()
def evaluate(model: SDCDM, loader: DataLoader, device: str, interactions: pd.DataFrame | None = None, q_matrix: np.ndarray | None = None) -> dict[str, float]:
    model.eval()
    labels_list: list[np.ndarray] = []
    probs_list: list[np.ndarray] = []
    for student, exercise, q_vector, label in loader:
        outputs = model(student.to(device), exercise.to(device), q_vector.to(device), labels=label.to(device))
        labels_list.append(label.detach().cpu().numpy())
        probs_list.append(outputs["pred"].detach().cpu().numpy())

    y_true = np.concatenate(labels_list)
    y_pred = np.concatenate(probs_list)
    metrics = binary_metrics(y_true, y_pred)
    if interactions is not None and q_matrix is not None:
        student_ids = torch.arange(model.strategy.e_habit.num_embeddings, device=device)
        profile = model.get_diagnostic_profile(student_ids)
        metrics["doa"] = degree_of_agreement(interactions, q_matrix, profile["h_pos"].detach().cpu().numpy())
    return metrics


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 2026))
    set_seed(seed)
    device = resolve_device(cfg.get("device", "auto"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = cfg["data"]
    bundle = build_data_bundle(
        interactions_path=args.interactions,
        q_matrix_path=args.q_matrix,
        student_col=data_cfg.get("student_col", "student_id"),
        exercise_col=data_cfg.get("exercise_col", "exercise_id"),
        label_col=data_cfg.get("label_col", "label"),
        exercise_key_col=data_cfg.get("exercise_key_col", "exercise_id"),
    )
    dataset = InteractionDataset(bundle.interactions, bundle.q_matrix)
    train_idx, val_idx, test_idx = split_indices(
        len(dataset),
        float(cfg["training"]["val_ratio"]),
        float(cfg["training"]["test_ratio"]),
        seed,
    )

    batch_size = int(cfg["training"]["batch_size"])
    train_loader = make_loader(dataset, train_idx, batch_size, shuffle=True)
    val_loader = make_loader(dataset, val_idx, batch_size, shuffle=False)
    test_loader = make_loader(dataset, test_idx, batch_size, shuffle=False)

    model_cfg = cfg["model"]
    use_dual_channel, use_cam = variant_flags(args.variant)
    model = SDCDM(
        num_students=bundle.num_students,
        num_exercises=bundle.num_exercises,
        num_concepts=bundle.num_concepts,
        emb_dim=int(model_cfg["emb_dim"]),
        cam_hidden=int(model_cfg["cam_hidden"]),
        epsilon=float(model_cfg.get("epsilon", 0.2)),
        lambda_reg=float(model_cfg.get("lambda_reg", 0.1)),
        use_dual_channel=use_dual_channel,
        use_cam=use_cam,
    )

    trainer = SDCDMTrainer(
        model,
        TrainerConfig(
            lr=float(cfg["training"]["lr"]),
            weight_decay=float(cfg["training"]["weight_decay"]),
            device=device,
        ),
    )

    best_val_score = -np.inf
    best_path = args.output_dir / "best_model.pt"
    patience = int(cfg["training"].get("patience", 10))
    stale_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in tqdm(range(1, int(cfg["training"]["epochs"]) + 1), desc="Training"):
        losses = []
        for student, exercise, q_vector, label in train_loader:
            loss_value = trainer.train_step(student, exercise, q_vector, label)
            losses.append(loss_value)

        val_metrics = evaluate(model, val_loader, device)
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(record)

        val_score = validation_score(val_metrics)
        if not best_path.exists() or val_score > best_val_score:
            best_val_score = val_score
            stale_epochs = 0
            torch.save(model.state_dict(), best_path)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))

    test_interactions = bundle.interactions.iloc[test_idx].copy()
    test_metrics = evaluate(model, test_loader, device, test_interactions, bundle.q_matrix)

    pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
    save_json(
        {
            "variant": args.variant,
            "test_metrics": test_metrics,
            "num_students": bundle.num_students,
            "num_exercises": bundle.num_exercises,
            "num_concepts": bundle.num_concepts,
            "splits": {
                "train": int(len(train_idx)),
                "val": int(len(val_idx)),
                "test": int(len(test_idx)),
            },
        },
        args.output_dir / "metrics.json",
    )
    save_json(
        {
            "student_map": {str(k): int(v) for k, v in bundle.student_map.items()},
            "exercise_map": {str(k): int(v) for k, v in bundle.exercise_map.items()},
            "concept_names": bundle.concept_names,
        },
        args.output_dir / "mappings.json",
    )
    print(test_metrics)


if __name__ == "__main__":
    main()
