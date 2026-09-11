# ERA5 Acquisition Plan — Phase 5

> **Frozen historical plan.** ERA5/CDS is not part of the completed track-only experiment. Do not run, resume, or extend this plan without a separately approved future scope.

## Decision

Use official Copernicus CDS ERA5 **hourly reanalysis** data, not a global download. The audited Phase 4 selections require 48h history for +6h and +48h, and 36h for +24h. Build the ERA5 request manifest from the union of their issue times: 2,472 unique dates from 1980-07-18 through 2025-10-31, at 00:00, 06:00, 12:00, and 18:00 UTC only.

## Required variables

Download three reanalysis products, all on the same regional grid:

| CDS dataset | CDS variable(s) | Pressure level(s) | Derived Phase 5 feature |
|---|---|---:|---|
| `reanalysis-era5-pressure-levels` | `u_component_of_wind`, `v_component_of_wind` | 850, 700, 500, 200 hPa | `steering_u`, `steering_v` = arithmetic mean of 850/700/500 hPa winds; `vertical_shear` = magnitude of the 200–850 hPa wind-vector difference |
| `reanalysis-era5-pressure-levels` | `relative_humidity` | 700 hPa | `relative_humidity_700` |
| `reanalysis-era5-single-levels` | `sea_surface_temperature` | surface | `sea_surface_temperature` |

This produces five environmental input values per issue time. Preserve source units (`m s-1`, `%`, and K) until the existing training-only scaler is applied.

## Spatial and temporal scope

- **Region:** CDS area `[55, -110, 5, -5]` (`North, West, South, East`) on the catalogue dataset's supported 0.25° regular latitude/longitude grid. The audited valid positions span 7.2–51.9°N and 105.1–6.0°W; the box supplies a 2–5° margin.
- **Time:** only the 2,472 manifest dates and the four exact IBTrACS issue hours. No February or March request is needed because none of the selected Phase 4 forecast origins occurs then.
- **Why cropped data is valid:** CDS supports regional `area` subsetting; the current catalogue schema does not accept an explicit `grid` request key. ERA5 is hourly; issue times exactly match requested ERA5 analysis times, so no temporal interpolation is needed.

The catalogue products deliver the requested regional subset on their supported 0.25° regular grid. Phase 5 uses only scalar environmental features sampled at each cyclone issue location, so explicit 1° regridding is unnecessary. This does not change the scientific experiment: predictors remain consistently sampled at the cyclone location and exact issue time.

## Format, files, and expected size

Request `data_format: "netcdf"` / NetCDF4 and `download_format: "unarchived"`, one month per file per product. NetCDF is preferred because Phase 5 will need coordinate/time-aware access; GRIB would require `cfgrib`/ecCodes without adding value here.

Keep raw files outside Git, for example:

```text
data/raw/era5/v1/
  pressure_wind_YYYYMM.nc       # u/v at 200, 500, 700, 850 hPa
  humidity_700_YYYYMM.nc        # relative humidity at 700 hPa
  sst_YYYYMM.nc                 # sea-surface temperature
  request_manifest.csv          # date, times, CDS request version, area, resolution
```

There are at most 235 active year-months and therefore at most 705 files. Actual CDS output size must be recorded after download; no request may add an unsupported `grid` parameter.

## CDS acquisition specification

1. Register/login at CDS, accept the licences for both datasets, and configure `cdsapi` with the personal CDS key.
2. Generate the manifest from the existing Phase 4 selected histories (steps 9, 7, 9), deduplicated by UTC issue date. Group it by `YYYY-MM`; use only its listed days.
3. For every active month, submit these three requests with `product_type: ["reanalysis"]`, `time: ["00:00", "06:00", "12:00", "18:00"]`, and `area: [55, -110, 5, -5]`. Do not send a `grid` parameter; CDS returns its supported 0.25° regular grid:

```python
# pressure_wind_YYYYMM.nc
client.retrieve("reanalysis-era5-pressure-levels", {
    "product_type": ["reanalysis"],
    "variable": ["u_component_of_wind", "v_component_of_wind"],
    "pressure_level": ["200", "500", "700", "850"],
    "year": ["YYYY"], "month": ["MM"], "day": active_days,
    "time": ["00:00", "06:00", "12:00", "18:00"],
    "area": [55, -110, 5, -5],
    "data_format": "netcdf", "download_format": "unarchived",
}, "pressure_wind_YYYYMM.nc")
```

Use the same request with `variable: ["relative_humidity"]`, `pressure_level: ["700"]` for humidity, and dataset `reanalysis-era5-single-levels` with `variable: ["sea_surface_temperature"]` for SST. Before the full run, submit **one active month** and verify variable names, units, coordinate order, four timestamps/day, and area before continuing.

The current CDS guidance documents pressure-level and single-level ERA5 as hourly, global regular grids; it supports NetCDF output and regional `area` subsetting. See [CDS pressure-level ERA5](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=overview), [CDS single-level ERA5](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview), and [CDS API guidance](https://cds.climate.copernicus.eu/en/how-to-api).

## Phase 5 alignment and leakage rules

For an IBTrACS sample at origin `(SID, issue_time, issue_lat, issue_lon)`:

1. Require an ERA5 record at **exactly** `issue_time`; do not use time interpolation or a later ERA5 timestamp.
2. Spatially sample the nearest 0.25° ERA5 grid point to **issue_lat/issue_lon only**. Store `era5_time`, `era5_grid_lat`, and `era5_grid_lon` in the extracted feature table.
3. Derive the five values above from that sampled state. Never use target time, target latitude/longitude, future storm state, or a future ERA5 field.
4. If any required ERA5 value is absent (notably SST near land), mark the origin unavailable. Do not impute it. Compare history-only and history+ERA5 on their identical surviving origin intersection.

This is a retrospective reanalysis ablation, not an operational real-time forecast: ERA5 analysis fields are restricted by timestamp, but reanalysis itself benefits from retrospective data assimilation.

## Dependencies and acceptance checks

Install only when acquisition is approved: `cdsapi`, `xarray`, `netCDF4` (or `h5netcdf`). No `cfgrib` is needed with NetCDF output. Before Phase 5 implementation, verify every NetCDF file has the requested variables/levels, UTC timestamps, region, 0.25° regular grid, units, and no duplicate `(time, lat, lon)` states. Then create a compact extracted issue-feature table keyed by `SID` and `issue_time` before touching the GRU pipeline.

## Exact data to obtain

Obtain the three monthly NetCDF file series above for every active manifest month from 1980-07 through 2025-10, restricted to the manifest's issue days and four six-hour times and area `[55, -110, 5, -5]`, on the CDS catalogue's supported 0.25° regular grid. Also retain the generated request manifest and CDS retrieval metadata. These are the only ERA5 inputs needed before Phase 5 can begin.
