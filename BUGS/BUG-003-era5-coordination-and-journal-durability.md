# ERA5 coordination and journal durability

## Problem

The ERA5 lock and journal followed `--raw-dir`, allowing different output directories to submit duplicate CDS jobs. Journal writes also directly overwrote JSON, risking a corrupt or lost job ID after interruption.

## Fix

`src.acquire_era5` now resolves one project-canonical state directory, `data/raw/era5/v1`, from the repository. It owns both `acquire.lock` and `request_jobs.json`; `--raw-dir` only selects where validated NetCDF files are stored.

Journal writes now create a same-directory temporary file, write and `fsync` the complete JSON, preserve the existing journal mode when it exists, atomically replace the journal, then attempt a directory `fsync`. A failed write removes only its temporary file and leaves the prior journal intact.

## Verification

Focused tests verify shared coordination across two output directories, atomic replacement, preserved JSON after replacement failure, latest attempt/job-ID persistence, BUG-002 exact recovery, fail-closed zero/multiple matching, and a four-worker queue whose coordinator serializes journal writes. The ERA5 suite passed 27 tests. No CDS request was submitted.

## Remaining limitation

The canonical state directory is local to one repository checkout and filesystem. Independent checkouts/machines still require an external shared state service for cross-host coordination.
