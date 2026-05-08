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
from .evaluate import binary_metrics
from .model import SDCDM, SDCDMTrainer, TrainerConfig
from .utils import resolve_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SD-CDM on response logs and a Q-matrix.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--interactions", type=Path, required=True)
    parser.add_argument("--q-matrix", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/run"))
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


@torch.no_grad()
def evaluate(model: SDCDM, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    labels_list = []
    probs_list = []
    for batch in loader:
        student = batch["student"].to(device)
        exercise = batch["exercise"].to(device)
        q_vector = batch["q_vector"].to(device)
        label = batch["label"].to(device)
        outputs = model(student, exercise, q_vector, labels=label)
        labels_list.append(label.detach().cpu().numpy())
        probs_list.append(outputs["prob"].detach().cpu().numpy())

    y_true = np.concatenate(labels_list)
    y_pred = np.concatenate(probs_list)
    return binary_metrics(y_true, y_pred)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 2026)))
    device = resolve_device(cfg.get("device", "auto"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = cfg["data"]
    bundle = build_data_bundle(
        interactions_path=args.interactions,
        q_matrix_path=args.q_matrix,
        student_col=data_cfg["student_col"],
        exercise_col=data_cfg["exercise_col"],
        label_col=data_cfg["label_col"],
    )
    dataset = InteractionDataset(bundle)
    train_idx, val_idx, test_idx = split_indices(
        len(dataset),
        float(cfg["training"]["val_ratio"]),
        float(cfg["training"]["test_ratio"]),
        int(cfg.get("seed", 2026)),
    )

    train_loader = make_loader(dataset, train_idx, int(cfg["training"]["batch_size"]), shuffle=True)
    val_loader = make_loader(dataset, val_idx, int(cfg["training"]["batch_size"]), shuffle=False)
    test_loader = make_loader(dataset, test_idx, int(cfg["training"]["batch_size"]), shuffle=False)

    model_cfg = cfg["model"]
    model = SDCDM(
        num_students=bundle.num_students,
        num_exercises=bundle.num_exercises,
        num_concepts=bundle.num_concepts,
        emb_dim=int(model_cfg["emb_dim"]),
        cam_hidden=int(model_cfg["cam_hidden"]),
        dropout=float(model_cfg["dropout"]),
    ).to(device)

    trainer = SDCDMTrainer(
        model,
        TrainerConfig(
            lr=float(cfg["training"]["lr"]),
            weight_decay=float(cfg["training"]["weight_decay"]),
            margin_lambda=float(cfg["training"]["margin_lambda"]),
        ),
    )

    best_val_auc = -np.inf
    best_path = args.output_dir / "best_model.pt"
    history: list[dict[str, float]] = []

    for epoch in tqdm(range(1, int(cfg["training"]["epochs"]) + 1), desc="Training"):
        model.train()
        losses = []
        for batch in train_loader:
            loss_value = trainer.train_step(
                batch["student"].to(device),
                batch["exercise"].to(device),
                batch["q_vector"].to(device),
                batch["label"].to(device),
            )
            losses.append(loss_value)

        val_metrics = evaluate(model, val_loader, device)
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(record)

        val_auc = val_metrics.get("auc", float("nan"))
        if np.isfinite(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), best_path)

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
    test_metrics = evaluate(model, test_loader, device)

    pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
    save_json(
        {
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
