# Bug 006: Paired comparison point estimate did not match its interval estimator

## Problem

`paired_gru_ridge()` displayed an origin-weighted GRU--Ridge difference while its 95% interval bootstrapped equal-weighted per-storm differences.

## Root cause

The paired rows were averaged directly for the displayed delta, but bootstrap draws were built from `groupby("SID").mean()` values.

## Fix

The report now displays storm-macro GRU, Ridge, and delta means, matching the bootstrap estimator. A regression test uses unequal origin counts per storm to distinguish the two estimators.

## Verification

Focused report tests and the regenerated paired table pass. This changes reporting only; no prediction, split, or model training changed.
