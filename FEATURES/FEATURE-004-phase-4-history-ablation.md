# Phase 4 — History-Length Ablation

## Goal

Test whether 6, 12, 24, 36, or 48 hours of best-track history best supports the compact GRU at each forecast horizon.

## Implementation

`src/run_history_ablation.py` trains the unchanged Phase 3 GRU for 2, 3, 5, 7, and 9 input states. Each model trains on its own eligible training windows. For fair selection and testing, validation and test sets are restricted to the exact common storm issue/target times across all five history lengths.

## Verification

Tests cover generic history construction and common-origin alignment. The run writes `results/phase4_history_ablation.csv` with validation and held-out metrics, selected architecture details, and the validation-track selection flag.

## Result

Lowest common-validation track error selects 48h history for +6h, 36h for +24h, and 48h for +48h. A decision is still needed on using per-horizon histories or one uniform history before an ERA5 ablation.
