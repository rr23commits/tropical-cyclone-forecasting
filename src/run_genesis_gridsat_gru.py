#!/usr/bin/env python3
"""Train and evaluate the frozen Genesis GridSat CNN--GRU experiment once."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import replace
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
from src.gridsat_gru import ImageScaler, load_crop
from src.gru import set_seed


def _paths(samples: GenesisGridSatSamples) -> tuple[Path, ...]:
    return tuple(path for history in samples.image_paths for path in history)


def materialize_crops(
    groups: dict[str, GenesisGridSatSamples], crop_root: Path, cache_dir: Path,
    progress_every: int = 100, log: Callable[[str], None] = print,
) -> dict[str, GenesisGridSatSamples]:
    """Copy each required Drive crop once, then point all histories at local copies."""
    sources = tuple(dict.fromkeys(path for group in groups.values() for path in _paths(group)))
    cache_dir.mkdir(parents=True, exist_ok=True)
    copied = reused = 0
    for number, source in enumerate(sources, 1):
        target = cache_dir / source.relative_to(crop_root)
        if target.exists() and target.stat().st_size == source.stat().st_size:
            reused += 1
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied += 1
        if number % progress_every == 0 or number == len(sources):
            log(f"progress: {number}/{len(sources)} crops ({copied} copied, {reused} cached)")
    return {
        split: replace(group, image_paths=tuple(
            tuple(cache_dir / path.relative_to(crop_root) for path in history)
            for history in group.image_paths
        ))
        for split, group in groups.items()
    }


def fit_image_scaler(crop_paths: tuple[Path, ...], progress_every: int = 100, log: Callable[[str], None] = print) -> ImageScaler:
    """Match ImageScaler.fit while making the one-time local scan observable."""
    if not crop_paths:
        raise ValueError("cannot fit an image scaler without training crops")
    total = squared_total = 0.0
    count = 0
    for number, path in enumerate(crop_paths, 1):
        values = load_crop(path)
        valid = values[np.isfinite(values)]
        total += float(valid.sum(dtype=np.float64))
        squared_total += float(np.square(valid, dtype=np.float64).sum())
        count += len(valid)
        if number % progress_every == 0 or number == len(crop_paths):
            log(f"progress: {number}/{len(crop_paths)} training crops")
    if not count:
        raise ValueError("training GridSat crops contain no finite IR pixels")
    mean = total / count
    variance = max(squared_total / count - mean ** 2, 0.0)
    return ImageScaler(mean, max(float(np.sqrt(variance)), 1.0))


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
    parser.add_argument("--cache-dir", type=Path, default=Path(tempfile.gettempdir()) / "genesis_gridsat_cnn_gru_crops")
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

    print("[1/7] Loading manifest...", flush=True)
    set_seed(args.seed)
    groups = split_genesis_samples(load_genesis_samples(args.manifest, args.state_dir, args.crop_root))
    print("[2/7] Building complete candidate dataset...", flush=True)
    if any(not len(groups[split].candidates) for split in groups):
        raise ValueError("complete Genesis histories must cover train, validation, and test")
    for split, group in groups.items():
        print(f"{split}: {len(group.candidates)} candidates", flush=True)
    print("[3/7] Preparing local/cached crop data...", flush=True)
    groups = materialize_crops(groups, args.crop_root, args.cache_dir)
    print("[4/7] Fitting image scaler...", flush=True)
    scaler = fit_image_scaler(_paths(groups["train"]))
    train_loader = _loader(groups["train"], scaler, args.batch_size, shuffle=True)
    validation_loader = _loader(groups["validation"], scaler, args.batch_size)
    device = torch.device(args.device)
    print("[5/7] Initializing CNN-GRU...", flush=True)
    model = GenesisCNNGRU().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_state, best_loss, best_epoch, stale = copy.deepcopy(model.state_dict()), float("inf"), 0, 0
    print("[6/7] Training...", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_total = train_count = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(model(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            train_total += float(loss.item()) * len(labels)
            train_count += len(labels)
        validation_loss = _loss(model, validation_loader, device)
        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_total / train_count:.6f} | val_loss={validation_loss:.6f}", flush=True)
        if validation_loss < best_loss:
            best_state, best_loss, best_epoch, stale = copy.deepcopy(model.state_dict()), validation_loss, epoch, 0
            print(f"Best validation checkpoint saved at epoch {epoch}.", flush=True)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping at epoch {epoch}.", flush=True)
                break
    model.load_state_dict(best_state)
    print(f"Best validation checkpoint restored from epoch {best_epoch}.", flush=True)
    # Test data is read only after the validation-selected checkpoint is final.
    print("[7/7] Evaluating test set...", flush=True)
    labels, logits = _predict(model, _loader(groups["test"], scaler, args.batch_size), device)
    counts = {split: {"total": len(group.candidates), "negative": int((group.candidates.label_24h == 0).sum()), "positive": int((group.candidates.label_24h == 1).sum())} for split, group in groups.items()}
    report = {"seed": args.seed, "device": str(device), "epochs_trained": best_epoch,
              "validation_binary_cross_entropy": best_loss, "class_counts": counts,
              "test": binary_metrics(labels, logits)}
    for name, value in report["test"].items():
        print(f"{name}={value if value is None else f'{value:.6f}'}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "image_mean": scaler.mean, "image_scale": scaler.scale,
                "seed": args.seed, "epochs_trained": best_epoch}, args.output_dir / "model.pt")
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
