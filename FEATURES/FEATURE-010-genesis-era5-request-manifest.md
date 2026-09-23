# Feature 010: TCC genesis ERA5 request manifest

## Goal and scope

Generate and validate a durable, inspectable CDS request plan for the frozen 1,276-row basin-resolved TCC cohort, then acquire and validate its raw CDS files. Crop and feature extraction remain out of scope.

## Execution flow

`src.manifest_genesis_era5` selects the 1,276 non-`UNKNOWN` rows through `src.preflight_genesis.cohort()`. It verifies the exact nine consecutive 3-hour timestamps per issue row, joins all required `(tcc_track_id,time)` positions from the existing TCC position audit, and confirms the issue-time position matches the frozen candidate row. Repeated history references deduplicate to unique track/time crop centers. The module groups positions by track×UTC day and constructs one pressure-level and one SST payload per group.

## Request and predictor contract

- Pressure dataset: `reanalysis-era5-pressure-levels`; variables `u_component_of_wind`, `v_component_of_wind`, `temperature`, `specific_humidity`, `vertical_velocity`; requested levels 200, 500, 700, and 850 hPa.
- Surface dataset: `reanalysis-era5-single-levels`; variable `sea_surface_temperature`.
- Frozen predictors: vorticity from u/v@850; u/v shear at 850–200 hPa; RH from temperature/specific humidity@700; vertical velocity@500; SST at surface.
- The CDS pressure variable×level selection is Cartesian: 20 combinations are requested, of which seven pressure variable-level pairs are needed by the predictors. The 13 additional returned combinations are explicit in the manifest and volume estimate.
- Each geographic area covers the per-batch union of all centered 5° patches, rounded outward to the 0.25° grid. Longitudes use the shortest circular interval; west>east denotes dateline crossing. Requested times are only the exact TCC history hours, even though the source products are hourly.
- Each crop center is snapped to the nearest native 0.25° ERA5 grid point; the exact center and 21×21 grid bounds are recorded per track/time. This keeps a symmetric 5° patch on the native grid, with at most 0.125° center displacement.

## Durability and validation

Every payload has a deterministic request key, canonical payload and SHA-256 fingerprint. The JSON manifest stores exact crop centers and a fingerprint over frozen labels/splits and row fields, and uses atomic replacement. `--validate-only` regenerates the expected manifest from the source cohort and TCC positions and fails on any mismatch. The generator still has no CDS client or submission call. `src.acquire_era5 --genesis-manifest <manifest>` provides the separate raw-file adapter: it preserves the exact payload/key, reuses the existing journal/retry/resume safeguards, and validates returned files before atomic promotion. On resume, no-ID non-DNS ambiguities are exact-payload-matched against one bounded read-only CDS history snapshot before any new POST; only exactly one CDS match is persisted/recovered. A snapshot read/network failure leaves all candidates unresolved and continues normal acquisition without a POST. Zero or multiple matches remain unresolved and are never resubmitted. A positively identified pre-CDS DNS failure may be retried with the old attempt preserved in `submission_history`; 502/reset/timeout never use that retry path. Duplicate job IDs are rejected. Crop and feature extraction remain later work.

For new Genesis submissions, the adapter creates a fresh Requests session per POST and uses 10-second connect / 90-second read timeouts with `retry_max=1`. New exceptions and missing job IDs remain fail-closed; current-run ambiguities are unresolved and not resubmitted. On a later run, non-DNS ambiguities are exact-payload reconciled first, while only proven DNS-resolution failures are retryable. Phase 4 submission and persisted-job recovery settings are unchanged. Focused tests cover the matching cardinalities, safe DNS-only retry/history, unchanged validated/submitted recovery, and duplicate ID protection. The earlier live transport attempt failed before DNS resolution; this recovery-gap fix was not run against acquisition state.

The bounded live test added `--limit-new N`, whose selection runs under the durable lock and excludes all journaled requests and existing targets. At concurrency 2, a cap of 20 selected the next 20 untouched manifest requests. The first, `genesis_sst_100207_19840616.nc`, failed before CDS hostname resolution and was journaled ambiguous without a job ID or output. The runner stopped immediately: 0/20 completed, and the other 19 were not touched. This run observed no connection reset, HTTP 502, or 30-second timeout. The manifest hash remained unchanged; no feature extraction followed.

## Current manifest and verification

`results/genesis_era5_request_manifest.json` has 1,396 track/day groups and 2,792 requests (1,396 pressure, 1,396 SST). The frozen 11,484 row/time references deduplicate to 6,676 track/time positions over 6,438 distinct UTC timestamps (1982-03-13 03Z through 2018-11-20 03Z). There are 1,396 distinct regional areas; 22 cross the dateline. Area heights are 5–14.75° and widths 5–16.5°.

The float32 uncompressed value-array estimate for requested CDS payloads is 598,544,352 bytes (570.82 MiB), including Cartesian extras. The estimated final 5° crops contain 94,211,712 bytes (89.85 MiB) before metadata. These estimates are not compressed transfer sizes.

Two manifest-generation tests cover overlapping-history deduplication, dateline geometry, predictor/level fidelity, and frozen-cohort mutation rejection. Adapter tests cover manifest loading, deterministic identities, pressure/SST separation, dateline payload preservation, failed-download restart, and ambiguous resume that preserves the journal entry while continuing later requests. `make test` passes 66 tests. Manifest generation and validate-only both pass against the local frozen cohort/position inputs.

## Live smoke test

The smoke test submitted only TCC 100015 on 1982-03-13. Pressure job `71eda6c8-92dc-4f26-ac11-e31ed5366e52` produced `genesis_pressure_100015_19820313.nc` (535,572 bytes); SST job `2968a18f-1a93-4866-8189-7b72d5c77f4a` produced `genesis_sst_100015_19820313.nc` (33,551 bytes). Both are journaled as `validated`; each contains the seven manifest timestamps and requested area, with the expected variables, units, and levels. A competing invocation was rejected by the canonical lock, and the original owner completed both jobs without duplication.

The full resume was run at concurrency 2 with the manifest and payloads unchanged. It recovered both previously persisted job IDs, then validated 88 additional requests before the next fresh submission returned HTTP 502 without a job ID. The fail-closed guard stopped the runner; no request was resubmitted. The journal now has 120 validated jobs and three unchanged/recorded ambiguous requests: `genesis_pressure_100048_19820726.nc` (connection reset, fingerprint `c156d74455e6152e03744ca8a9b766df1b452e033890719433d8b210c6fe83af`), `genesis_sst_100049_19820728.nc` (read timeout, fingerprint `dcd089eabfe912696db904cb16e18f4d8840d69ef2926d2e818c31e001394dd7`), and `genesis_pressure_100207_19840616.nc` (HTTP 502, fingerprint `ffe3bd813d1a0660b7676fe5904660b9642349ff3c85937459c113e70238fd6f`). No job IDs remain in `submitted` state. Raw storage contains 120 promoted NetCDF files totaling 20,144,560 bytes; there are no `.part` files. The latest runner status is 120 validated, zero known-job failures, three ambiguous, and 2,669 requests not yet validated. Future resumes continue skipping all three no-ID ambiguous records; reconcile them only with authoritative CDS evidence, and do not automatically resubmit.

## Next

When CDS connectivity is restored and acquisition is authorized, resume with the existing manifest and durable state at concurrency 2: `python3 -m src.acquire_era5 --genesis-manifest results/genesis_era5_request_manifest.json --raw-dir data/raw/era5/v1 --concurrency 2`. Do not begin crop/extraction until all manifest keys have validated raw outputs. Extraction remains outstanding.
