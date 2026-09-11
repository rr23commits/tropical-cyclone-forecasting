# Phase 3 — Compact GRU

## Goal

Compare one small sequence model with the Phase 2 baselines on identical held-out storm origins.

## Implementation

`src/gru.py` trains direct +6/+24/+48-hour one-layer GRUs over the existing five-state, nine-feature sequences. It predicts north/east displacement and wind change. Training-only feature/target scaling, fixed seed 42, validation width selection (16 or 32), and validation early stopping prevent test-set selection.

`python3 -m src.run_gru <csv>` reruns the three baselines and GRU on the same test rows, then writes `results/phase3_gru_comparison.csv`.

## Verification

Unit tests cover scaler isolation, output shape, held-out prediction, and existing Phase 1/2 rules. The real-data comparison uses the fixed 1980–2015/2016–2019/2020–2025 split.

## Limitation

This is one compact GRU experiment, not a final architecture comparison. It introduces no environmental data or other sequence-model family.
