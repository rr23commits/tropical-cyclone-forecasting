# Phase 1 — IBTrACS Pipeline

## Goal

Provide reproducible, leakage-safe historical windows and targets for the audited North Atlantic IBTrACS subset.

## Implementation

`src/ibtracs.py` loads and filters the official CSV, constructs +6/+24/+48-hour direct samples, and assigns whole-storm splits. `src/audit_phase1.py` reports the resulting storm and origin counts.

## Verification

`tests/test_ibtracs.py` covers loading, filtering, contiguous windows, future-label separation, and storm-atomic chronological splits.

## Limitations

No scaling, baseline, neural model, ERA5 feature, or evaluation metric is implemented in Phase 1.
