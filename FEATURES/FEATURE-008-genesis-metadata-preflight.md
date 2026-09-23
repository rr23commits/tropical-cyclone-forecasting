# Feature 008: basin-resolved genesis external-data preflight

## Goal and scope

Verify only the accepted 1,276 basin-resolved genesis candidates before any external-data acquisition. Do not change labels, cohort, or the frozen track-only experiment. Archive an explicit NO-GO whenever a verified source cannot be acquired within the stated storage/access constraints.

## Implementation and flow

`src.preflight_genesis` requires exactly 1,276 non-`UNKNOWN` rows and nine stored 3-hour input times per row. It paginates the official NOAA GridSat S3 object index—not frame data—for exact filenames and byte sizes, then reports missing timestamps and dependent rows. ERA5 is assessed for five predictors at every stored historical TCC position: vorticity@850 (derived from u/v), 850–200-hPa shear, RH@700 (derived from temperature/specific humidity), omega@500, and SST.

## Verification and current limitation

The focused checker test and `make test` pass. The resumed S3 audit completed: 6,438/6,438 required objects are listed, no timestamp/channel gaps exist, all 1,276 rows have nine listed frames, and exact source bytes are 247,888,386,134. The 4 MiB annual listing cache and JSON preflight are metadata only. GridSat is deliberately frozen as NO-GO because 28 GiB free disk cannot support a 230.86-GiB transfer under the no-full-archive constraint.

The ERA5-only full-cohort follow-up remains a NO-GO: the frozen history includes 3-hour inputs, while the pressure-level ARCO time-series store is 6-hourly. The previously inspected hourly global-chunk store is not a feasible full-cohort route. A later limited proof used the official geo-chunked pressure-level and single-level ARCO stores, with the existing CDS bearer credential and xarray/Zarr. Of 1,276 accepted rows, 384 issue times coincide exactly with the pressure store's 6-hour grid; no interpolation was attempted. For the first eligible candidate (TCC track 100017, 1982-03-15 18Z, -12.2/54.6), the nearest pressure grid was -12.25/54.5 at the exact source time. Metadata confirms `u`, `v`, `r`, and `w` plus `sst`, with 0.25-degree grid spacing. However, the first required value read (`u` at 850 hPa) timed out after 20.78 seconds. Its existing Zarr chunk has a 2,548,510-byte Content-Length, and xarray/Zarr silently yielded its NaN fill value rather than a valid value. This is a fail-closed NO-GO: no other variable values were requested, no training feature table was written, and no alternate client or fallback was tried. The audit is `results/genesis_era5_poc_audit.json`; 105,187 bytes of completed metadata/coordinate responses were counted in the instrumented attempts, while partial bytes from the timed-out request and cumulative bytes across all exploratory attempts are unknown. No ERA5 feature tensor exists and no model was trained.

The next remote-subset proof of concept also stopped safely. NCEI THREDDS did not return OPeNDAP metadata within its request limit, so an NCSS data subset was not requested. The corresponding official S3 object advertises byte ranges but is an opaque HDF5/NetCDF file; ranges cannot safely select the requested grid window. No image bytes were transferred.
