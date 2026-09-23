# Feature 011: Track-only evaluation finalization

## Goal

Make the frozen track-only experiment auditable for a B.Tech report without broadening its scientific scope.

## Implementation

`report_eda` creates source eligibility/missingness, split/horizon, training-only distribution, and illustrative training-storm outputs. `run_gru_seed_robustness` reports fixed seeds 42/43/44 without test-set selection. `report_evaluation` exports directional residual summaries/plots and consistent storm-macro paired estimates. `write_provenance` records source and result hashes, environment, revision, and fixed protocol.

## Verification

The source replay regenerated held-out errors, evaluation tables/plots, EDA, seed results, and provenance. Focused tests cover the paired estimator, residual fields, EDA missingness, provenance hashing, and equal-weight seed aggregation.

## Limits

The project remains a track-only direct-forecast panel. It does not add ARIMA, stationarity testing, environmental inputs, or new models.
