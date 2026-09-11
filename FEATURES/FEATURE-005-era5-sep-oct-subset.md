# ERA5 September--October subset

## Goal

Make the scalar ERA5 ablation feasible for the B.Tech project without weakening its storm-wise chronological evaluation.

## Scope and flow

`manifest_from_origins()` retains September and October from the unchanged Phase 4 origin union. The CLI reads or creates `data/raw/era5/v1/request_manifest_sep_oct.csv`, acquires its 264 specs through the canonical lock/journal flow with a bounded four-worker recovery queue, and passes the same seasonal origin subset to feature extraction.

## Implementation

The old 705-file `request_manifest.csv`, all NetCDF files, and `request_jobs.json` remain untouched. The new manifest has 1,339 dates, 88 months, and three products per month. Off-season cached files are not selected.

## Verification

The generated manifest was checked for September/October-only dates, 1980--2025 years, 88 months, and 264 specs. The focused ERA5 suite passed 27 tests without CDS access, including bounded concurrent recovery and no duplicate submission on restart.

## Limitation

Results apply to peak/late-season North Atlantic cases, not every seasonal condition.
