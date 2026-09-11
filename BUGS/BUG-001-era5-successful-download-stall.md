# ERA5 successful-job download stall

## Problem

CDS job `9299a2b2-83ae-4cc9-aff5-956646ede33b` reached `successful`, but its job-specific temporary file remained zero bytes and the runner remained inside the CDS client download call.

## Root cause

The client transport timeout/retry settings do not impose a total deadline on `Remote.download()`. The first signal-based deadline was cooperative and cannot preempt arbitrary blocked transport code. The attempted forked-worker replacement crashed with SIGSEGV (`-11`) on macOS because it inherited the initialized CDS/HTTP client; Python documents `fork` as unsafe on macOS. The observed November attempt ultimately exited with an `IncompleteRead` after a partial download; it was not a permanently active CDS job.

## Fix and verification

`Remote.download()` is no longer used. Once CDS reports success, the parent obtains the result asset URL and expected byte count, then system `curl` resumes the persistent job-specific `.part` file with `--continue-at -`, three retries, and 900-second transfer/retry limits. `_recover_job()` retries the same asset at most three times; a byte-count check and NetCDF validation precede promotion. Exhaustion records `download_error`, returns a blocked result, and the CLI exits non-zero without later requests. Focused tests cover curl resume, retained partials after failure, promotion, exhausted failure journaling, and same-job recovery without replacement submission.

## Controlled verification

On 2026-09-11, the existing successful CDS job `ad689dd9-00e7-4e14-a4ad-fcf187331f38` was recovered as the only requested acquisition item. Curl resumed its 90,238,720-byte partial to the CDS-reported 137,340,043 bytes. Validation passed for the exact September manifest (100 timestamps, expected pressure levels, variables, units, and 0.25° grid), then promotion completed in 5.711 seconds of the final recovery invocation. The journal retained the same request ID and changed status to `validated`; no CDS submission occurred.
