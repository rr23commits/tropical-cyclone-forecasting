# Feature 009: metadata-only LPS–TCC crosswalk audit

## Goal and scope

Determine whether the South Asian ERA5 LPS catalogue can enrich a strict subset of the existing 1,276-row TCC manifest. The audit must leave the TCC cohort, labels, splits, genesis preflight, and track-only experiment unchanged. It does not download the catalogue, derive unavailable ERA5 fields, or train a model.

## Contract and implementation

`src.audit_lps_tcc_crosswalk` reuses `src.audit_genesis.load_tcc()` and `src.preflight_genesis.cohort()`. For every accepted manifest row it recovers the nine source=1 positions keyed by `(tcc_track_id, UTC timestamp)`. Reused history positions are deduplicated before matching. LPS matches use the same exact UTC hour and only `lat_detected`/`lon_detected`; rows without an observed/detected position-source marker, marked posterior/interpolated/projected/smoothed/final, or lacking detected coordinates are excluded. A candidate must be within 1 degree great-circle distance (111.195 km), be the only LPS detection within range, and not be within range of another unique TCC position. Ambiguous candidates are recorded but not assigned.

The tool audits availability of only `mean_vort_850_detectedcentre` and `rh700_mean_pct_detectedcentre`. It does not use LPS category, intensity, or track history as model inputs or targets. TCC `label_24h`, `label_48h`, and `split` remain copied from the manifest. On a successful run it emits row-level and unique-position CSVs, a JSON summary, and a human-readable report. A missing catalogue emits a blocked-input JSON/Markdown status only.

## Current result and limitations

The local Parquet is 291,600,496 bytes, 368,925 rows, 276 columns, dated 1940-05-17 through 2025-12-03. All eight required columns exist. The file has 2,980 track IDs; time and track ID are non-null. Observed rows have detected-center coordinates and 850-hPa vorticity; interpolated rows have these fields null. The 700-hPa RH field is null on all 368,925 rows.

The 6,438 exact TCC timestamps are represented at 3,120 timestamps (48.46%); 3,318 have no catalogue row. A lazy Polars projection/time filter yielded 4,741 rows (464,157-byte temporary CSV) for the existing CLI, because this Python environment lacks a pandas Parquet engine. The unchanged matching criteria produced 1,275 rows with 0/9 matches, one with 5/9, and zero with 9/9: strict retained subset 0/1,276 (0%). The only partial row is TCC 103109 (NI, 2013, test split, negative 24h label); it matched five timestamps to LPS track 6846. There were zero ambiguous positions. Accepted distances (n=5): median 94.074 km, mean 89.764 km, p90 108.141 km, max 110.268 km. All five accepted positions have vorticity; none has RH. Every basin, split, and 24h label group has 0% full histories. Verdict: **C — crosswalk fails; abandon this catalogue for TCC enrichment**.

The catalogue's `row_id` repeats across different times but is unique within each time. The audit's validation now checks `(time,row_id)`, matching the identity key already used by reverse-claim detection. This does not change the 1-degree threshold, exact-time matching, ambiguity rejection, or any scientific criteria. Full row-level outputs are in `results/lps_tcc_crosswalk/`.

## Verification

Five focused tests and `make test` pass (59 total). No TCC cohort, labels, splits, genesis preflight, or track-only experiment were changed. No unavailable RH values were imputed.
