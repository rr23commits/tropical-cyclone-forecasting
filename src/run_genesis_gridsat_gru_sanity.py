#!/usr/bin/env python3
"""Run one read-only forward/backward pass on a tiny frozen Genesis subset."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from src.genesis_gridsat_gru import GenesisCNNGRU, load_genesis_images, load_genesis_samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("results/genesis_candidate_manifest.csv"))
    parser.add_argument("--state-dir", type=Path, default=Path("genesis_gridsat_state"))
    parser.add_argument("--crop-root", type=Path, default=Path("genesis_gridsat_crops"))
    parser.add_argument("--limit", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.limit <= 4:
        parser.error("--limit must be between 1 and 4")

    samples = load_genesis_samples(args.manifest, args.state_dir, args.crop_root)
    images = load_genesis_images(samples, args.limit)
    labels = torch.tensor(samples.candidates.label_24h.iloc[:len(images)].to_numpy(), dtype=torch.float32)
    logits = GenesisCNNGRU()(images)
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()
    print(f"candidates={len(samples.candidates)} images={tuple(images.shape)} labels={tuple(labels.shape)} logits={tuple(logits.shape)} loss={loss.item():.6f}")


if __name__ == "__main__":
    main()
