# BUG-004: LPS row IDs are reused across timestamps

## Problem

The crosswalk loader rejected the local LPS catalogue before matching because its observed rows reuse `row_id` values at different UTC times.

## Discovery and expected behavior

The local v5.4.2 Parquet has 368,925 rows. In the exact-time slice required by the frozen TCC histories, 3,893 rows are observed and 147 observed rows repeat a `row_id` used at another time. No `(time,row_id)` pair is duplicated. Matching and reverse-claim tracking are already scoped by timestamp, so distinct hourly positions must remain valid.

## Root cause and fix

`_detected_rows()` required globally unique `row_id`, stricter than the actual catalogue key. Validation now requires non-null `row_id` and uniqueness of `(time,row_id)`. No match threshold, candidate selection, or ambiguity rule changed.

## Verification and limitations

A focused test verifies reuse across different times is allowed and same-time duplicate keys still fail. Five crosswalk tests and `make test` pass (59 tests). The crosswalk then completed: zero full histories, one 5/9 partial, zero ambiguous positions. RH is null throughout the catalogue. Full audit output: `results/lps_tcc_crosswalk/`.
