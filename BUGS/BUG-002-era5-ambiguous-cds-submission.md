# ERA5 ambiguous CDS submission after HTTP 502

## Problem

`sst_198511.nc` stopped at `HTTP 502 Server Error: Bad Gateway` from the CDS execution endpoint. No CDS request ID was returned, so a retry could create a duplicate job if the server accepted the request before the gateway response was lost.

## Investigation

- The journal entry is intentionally preserved as `{"submission_error": "502 ..."}`; it has no `request_id`, no final file, and no partial file.
- CDS `get_jobs(limit=100, sortby="-created")` returned 99 account jobs. Each of its 31 `reanalysis-era5-single-levels` jobs was fetched read-only and compared with the exact existing manifest payload for `sst_198511.nc`.
- No request payload matched. The job list's newest entry was created at `2026-09-10T21:12:49.297708`.

## Conclusion

There is no server-recorded matching job in the current account history, which is strong evidence that the 502 occurred before CDS accepted this request. It is not a distributed-systems proof that acceptance was impossible: a client receives a definite submission result only when CDS returns a monitor URL/request ID.

## Fix

The runner now persists the canonical JSON payload, SHA-256 fingerprint, and CDS dataset before calling `retrieve()`. On a later `submission_error` without `jobs`, it paginates `client.client.get_jobs()` and compares each server-recorded collection and canonical request payload.

- Exactly one match: atomically attach that returned CDS ID to the existing journal entry, then use the existing recovery flow.
- Zero or multiple matches: remain blocked and never call `retrieve()`.

This safely recovers accepted requests whose response was lost. The installed CDS client submits only the request payload and offers no idempotency-key parameter; therefore no client-only change can safely auto-resubmit a zero-match 502 while guaranteeing no duplicate CDS job. That case needs a CDS-supported idempotency key or an explicit human-authorized resubmission after operational confirmation.

## One-time operator resolution

After the account-history investigation found no matching job, an operator explicitly authorized a single replacement for this legacy entry. Under the acquisition lock, the original record was retained verbatim as `sst_198511.nc.legacy_502_operator_archive`; only then did the existing `acquire()` path regenerate and persist the current canonical payload/fingerprint before submission. CDS returned job `a1d556f5-df45-460e-a592-86a21eb44f13`. Its 3,173,019-byte asset passed the existing NetCDF validation and was atomically promoted to `sst_198511.nc`. No general legacy-recovery workflow was added.

## Verification

Focused tests cover submission-attempt persistence, exact recovery, zero/multiple/unrelated matches, pagination, payload canonicalization, and unchanged successful-job download behavior. `KMP_DUPLICATE_LIB_OK=TRUE python3 -m unittest discover -s tests -p 'test_era5.py' -v` passed 21 tests; the full unit suite passed 36. No CDS request was submitted, downloaded, deleted, or journaled while implementing this fix.
