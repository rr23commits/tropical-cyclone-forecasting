#!/usr/bin/env python3
"""Train and evaluate the frozen Genesis GridSat CNN--GRU experiment once."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.genesis_gridsat_gru import (
    DEFAULT_CANDIDATE_MANIFEST, DEFAULT_CROP_ROOT, DEFAULT_STATE_DIR,
    GenesisCNNGRU, GenesisGridSatSamples, GenesisImageDataset, split_genesis_samples,
    load_genesis_samples,
)
from src.gridsat_gru import ImageScaler
from src.gru import set_seed


def _paths(samples: GenesisGridSatSamples) -> tuple[Path, ...]:
    return tuple(path for history in samples.image_paths for path in history)


def _loader(samples: GenesisGridSatSamples, scaler: ImageScaler, batch_size: int, shuffle: bool = False) -> DataLoader:
    return DataLoader(GenesisImageDataset(samples, scaler), batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _loss(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = count = 0
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            total += float(nn.functional.binary_cross_entropy_with_logits(logits, labels.to(device), reduction="sum").item())
            count += len(labels)
    return total / count


def _predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels, logits = [], []
    with torch.no_grad():
        for images, batch_labels in loader:
            labels.append(batch_labels.numpy())
            logits.append(model(images.to(device)).cpu().numpy())
    return np.concatenate(labels), np.concatenate(logits)


def binary_metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, float | None]:
    """Compute threshold metrics and tie-aware ROC-AUC without another dependency."""
    predictions = (logits >= 0).astype(int)
    truth = labels.astype(int)
    true_positive = int(((predictions == 1) & (truth == 1)).sum())
    false_positive = int(((predictions == 1) & (truth == 0)).sum())
    false_negative = int(((predictions == 0) & (truth == 1)).sum())
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    auc: float | None = None
    if truth.min() != truth.max():
        order = np.argsort(logits)
        ranks = np.empty(len(logits), dtype=float)
        start = 0
        while start < len(logits):
            end = start + 1
            while end < len(logits) and logits[order[end]] == logits[order[start]]:
                end += 1
            ranks[order[start:end]] = (start + end + 1) / 2
            start = end
        positives = int(truth.sum())
        negatives = len(truth) - positives
        auc = float((ranks[truth == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))
    return {"accuracy": float((predictions == truth).mean()), "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0, "roc_auc": auc}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_CANDIDATE_MANIFEST)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--crop-root", type=Path, default=DEFAULT_CROP_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("results/genesis_cnn_gru"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        parser.error("--epochs, --patience, and --batch-size must be positive")

    set_seed(args.seed)
    groups = split_genesis_samples(load_genesis_samples(args.manifest, args.state_dir, args.crop_root))
    if any(not len(groups[split].candidates) for split in groups):
        raise ValueError("complete Genesis histories must cover train, validation, and test")
    scaler = ImageScaler.fit(_paths(groups["train"]))
    train_loader = _loader(groups["train"], scaler, args.batch_size, shuffle=True)
    validation_loader = _loader(groups["validation"], scaler, args.batch_size)
    device = torch.device(args.device)
    model = GenesisCNNGRU().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_state, best_loss, best_epoch, stale = copy.deepcopy(model.state_dict()), float("inf"), 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(model(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
        validation_loss = _loss(model, validation_loader, device)
        if validation_loss < best_loss:
            best_state, best_loss, best_epoch, stale = copy.deepcopy(model.state_dict()), validation_loss, epoch, 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(best_state)
    # Test data is read only after the validation-selected checkpoint is final.
    labels, logits = _predict(model, _loader(groups["test"], scaler, args.batch_size), device)
    counts = {split: {"total": len(group.candidates), "negative": int((group.candidates.label_24h == 0).sum()), "positive": int((group.candidates.label_24h == 1).sum())} for split, group in groups.items()}
    report = {"seed": args.seed, "device": str(device), "epochs_trained": best_epoch,
              "validation_binary_cross_entropy": best_loss, "class_counts": counts,
              "test": binary_metrics(labels, logits)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "image_mean": scaler.mean, "image_scale": scaler.scale,
                "seed": args.seed, "epochs_trained": best_epoch}, args.output_dir / "model.pt")
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
