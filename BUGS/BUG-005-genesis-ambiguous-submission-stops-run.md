# BUG-005: Genesis resume skipped safe ambiguity recovery

## Problem

A Genesis CDS submission that raised before returning a job ID was correctly journaled as ambiguous, but Genesis resume skipped every no-ID entry before reaching the existing exact-payload CDS job matcher. Proven pre-CDS DNS failures were also not retried, leaving most of the manifest stranded.

## Discovery and expected behavior

The controlled transport run encountered an immediate DNS failure on `genesis_sst_100207_19840616.nc`; no CDS job ID or raw file was produced. Later state review found 2,568 no-ID entries, of which 2,561 were confirmed DNS-resolution failures and seven were genuinely delivery-unknown (HTTP 502, read timeout, or connection reset). Genesis must retry only the confirmed pre-CDS class, and attempt exact payload reconciliation for the seven unknowns before any new POST. Phase 4 behavior must remain unchanged.

## Execution flow and root cause

`acquire()` persisted canonical payload/fingerprint before `retrieve()`. The Genesis resume branch skipped all existing no-ID entries before `_matching_ambiguous_jobs()`, despite an exact-payload recovery path already existing for the shared Phase 4 flow. The no-ID guard also did not distinguish positively identified pre-CDS DNS failures from errors where delivery could have occurred.

## Fix

Before any new Genesis POST, run `_matching_ambiguous_jobs()` for each existing no-ID non-DNS entry using its exact persisted dataset, canonical payload, and fingerprint. Persist and recover only a unique match; zero or multiple matches stay unresolved and are never resubmitted. Retry only recorded failures positively identifying DNS resolution failure for `cds.climate.copernicus.eu`, preserving the old attempt in `submission_history`. Existing validated files and persisted job IDs retain their reuse/recovery paths. Reject duplicate job IDs. Phase 4 behavior is unchanged.

## Verification and limitations

Focused mocked tests cover unique/zero/multiple matching, 502/timeout/reset remaining unresolved, DNS-only retry and history preservation, no POST before ambiguity lookup, validated and persisted-job reuse, and duplicate job-ID rejection. `make test` passes all 80 tests; `git diff --check` is clean. No live acquisition or existing journal/data modification is part of this fix.
