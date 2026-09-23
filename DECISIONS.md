# Decisions

## Phase 1 IBTrACS subset

- **Decision:** Use IBTrACS v04r01 North Atlantic records from 1980–2025 with `TRACK_TYPE == main`, `USA_AGENCY == hurdat_atl`, exact six-hour UTC timestamps, complete USA position/wind values, and eligible tropical/subtropical statuses.
- **Why:** The audit found this is the most consistent source and avoids mixing incompatible wind averaging conventions or interpolated non-six-hour records.
- **Consequence:** Missing states are excluded per forecast window, and pressure remains optional.

## Direct displacement targets

- **Decision:** Build a separate direct sample set for each horizon, with north/east displacement and wind change targets.
- **Why:** This prevents recursive error accumulation and keeps target construction aligned with the design.
- **Consequence:** All future values are labels only; feature windows end at the issue time.

## Split assignment

- **Decision:** Assign a whole storm to one split from its first recorded season.
- **Why:** A storm must not contribute correlated windows to more than one split.
- **Consequence:** The Phase 2 split-year boundaries are recorded below and must be reused for comparable results.

## Baseline evaluation split

- **Decision:** Use 1980–2015 for training, 2016–2019 for validation, and 2020–2025 for held-out testing in Phase 2.
- **Why:** It provides a chronological, recent multi-season test set and a validation period for ridge regularization without using test storms.
- **Consequence:** Phase 2 metrics are comparable only under this fixed split; changing it requires rerunning every baseline.

## Ridge baseline

- **Decision:** Use a NumPy multi-output ridge regression over flattened historical features, selecting its regularization value on validation data.
- **Why:** It is a transparent regularized lagged-regression baseline without adding an ML framework dependency.
- **Consequence:** Feature and target standardization are fitted on training rows only; the test split remains untouched until final scoring.

## First neural experiment

- **Decision:** Compare a one-layer GRU with hidden widths 16 and 32, fixed seed 42, Adam (`lr=0.001`, `weight_decay=0.0001`), batch size 64, maximum 100 epochs, and patience 15.
- **Why:** This is a bounded sequence-model experiment appropriate for five input states. Width and checkpoint are selected with validation MSE only; no test row enters model selection.
- **Consequence:** This bounded GRU is the frozen neural architecture for the final track-only experiment. It improves wind metrics and +24/+48-hour track metrics, while Ridge remains slightly better for +6-hour track error.

## History-length ablation

- **Decision:** Compare 6h/12h/24h/36h/48h histories with the unchanged Phase 3 GRU. Select history length by lowest Haversine track error on a validation set aligned to common origins across all lengths.
- **Why:** Longer histories otherwise exclude more storm origins, so comparing each length on its own validation/test set would confound history with case difficulty.
- **Consequence:** Validation selects 48h for +6h, 36h for +24h, and 48h for +48h. The final primary comparison deliberately uses one uniform 24-hour history instead, for a clear consistent system.

## ERA5 catalogue spatial resolution

- **Decision:** Request CDS ERA5 pressure- and single-level regional subsets at the catalogue's supported 0.25° regular latitude/longitude resolution, without a `grid` parameter, then sample the nearest grid point to each issue location.
- **Why:** The current CDS schema rejects `grid`; Phase 5 uses scalar location/time-aligned environmental predictors rather than gridded model inputs, so explicit 1° regridding has no scientific role.
- **Consequence:** The five planned environmental features, issue times, region, and leakage safeguards are unchanged; acquisition size increases relative to the former unsupported 1° plan.

## ERA5 completed-asset transfer

- **Decision:** Transfer CDS completed-result assets with system `curl`, preserving the job-specific partial file using `--continue-at -`, bounded retries, and 900-second limits.
- **Why:** Curl provides a process-bounded transfer with native HTTP resume while keeping CDS submission, polling, result metadata, and job identity under `cdsapi` and the durable journal.
- **Alternatives considered:** `Remote.download()` does not preserve resumable partials; the standard-library spawned worker previously replaced it but is no longer required.
- **Consequence:** The host must provide curl. Byte-count verification, NetCDF validation, atomic promotion, and no-replacement-job recovery policy remain unchanged.

## ERA5 ambiguous CDS submissions

- **Decision:** Persist a canonical CDS request payload, SHA-256 fingerprint, and dataset before submission. On an ambiguous response with no returned job ID, recover only one exact server-recorded job found through paginated CDS job history; otherwise remain blocked.
- **Why:** A 502 can arrive before or after CDS accepts a request. Exact payload matching can safely recover a lost successful response without inventing an idempotency mechanism or risking a duplicate request.
- **Consequence:** A zero- or multiple-match entry never resubmits automatically. Legacy ambiguous entries without stored payload metadata remain blocked for human diagnosis.

## Genesis ambiguous-request resume behavior

- **Superseded on 2026-09-14:** The initial Genesis rule left every existing no-ID entry untouched and skipped CDS history lookup. This bypassed the already-existing exact-payload matcher and stranded both recoverable CDS jobs and proven pre-CDS DNS failures.

- **Decision:** Before any new Genesis POST on resume, exact-payload-match each existing no-ID non-DNS ambiguity against paginated CDS job history. Persist/recover only exactly one matching job; zero or multiple matches remain unresolved and are never resubmitted. Automatically retry only an entry whose recorded error positively identifies CDS hostname DNS resolution failure before delivery; preserve the previous attempt under `submission_history`. Reject duplicate CDS job IDs. **Why:** recover accepted work safely without treating ambiguous delivery as a retryable failure. **Scope:** Genesis manifest requests only; Phase 4 behavior is unchanged.

- **Decision:** Build one bounded, read-only CDS history snapshot per Genesis ambiguity-reconciliation run and locally match every delivery-unknown entry against it. A history/metadata read failure leaves all candidates unresolved and continues normal resume. **Why:** independently traversing full account history for each ambiguity delayed startup without improving the proof standard. **Consequence:** each listed job metadata record is fetched once per run; pagination is capped fail-closed, and no request is POSTed because reconciliation cannot complete.

- **Decision:** For each new Genesis CDS POST, use a fresh Requests session with a 10-second connect timeout, 90-second read timeout, and exactly one client attempt (`retry_max=1`). Treat an exception or absent job ID as ambiguous and never retry automatically. **Why:** prevent stale pooled connections and allow longer server responses without risking duplicate CDS jobs. **Scope/tradeoff:** only new Genesis submissions use these options; Phase 4 and persisted-job recovery keep existing settings. This reduces transport ambiguity but cannot recover a request accepted by CDS when the response is lost.

- **Decision:** A bounded Genesis run may use `--limit-new N`; under the canonical acquisition lock it selects only the first N manifest requests with neither a journal entry nor a raw target. **Why:** a controlled live run must not resume old jobs or touch journaled ambiguities while enforcing an exact request cap. A later no-ID ambiguity is journaled and reported, while the remaining selected specs continue.
- **Consequence:** A Genesis run may still complete with unresolved requests; 502, connection reset, timeout, and zero/multiple exact matches are never automatically resubmitted. Only a unique exact match receives a persisted job ID. The legacy Phase 4 path retains its existing recovery behavior.

## ERA5 coordination and journal durability

- **Decision:** Keep one repository-canonical ERA5 state directory for the process lock and request journal; use `--raw-dir` only for final data files. Persist the journal with write/fsync/atomic-replace.
- **Why:** Output-directory selection must not create independent submission histories, and a journal is the provenance authority only if interrupted writes cannot expose partial JSON.
- **Consequence:** Different output directories share CDS job IDs and cannot submit the same request independently. Coordination remains local to a repository checkout.

## ERA5 bounded acquisition queue

- **Decision:** Recover/submit the reduced ERA5 manifest in batches of at most four jobs. A single lock-owning coordinator is the only journal writer; workers use separate CDS clients and return state for coordinator persistence.
- **Why:** CDS polling and resilient asset transfers are slow enough that serial execution wastes available network capacity. Keeping submission identity and journal writes centralized retains the fail-closed duplicate-prevention model.
- **Consequence:** A failed job is journaled independently and does not reset other active jobs. The queue has a conservative default of four and is configurable by `--concurrency`.

## ERA5 September--October subset

- **Decision:** Restrict the ERA5 environmental ablation to every September and October in the existing 1980--2025 Phase 4 origin union: 88 months and 264 product files.
- **Why:** It retains 6,354 origins across 335 storms, including 247/33/55 train/validation/test storms, while covering the climatological Atlantic peak and late season at practical B.Tech scale.
- **Alternatives considered:** September only is cheaper but leaves only 25 validation and 38 test storms; the 705-file all-season plan is unnecessary for the scalar-feature ablation. Retaining arbitrary already-downloaded months would make split-season coverage uneven.
- **Consequence:** The CLI uses the separately named September--October manifest and filters feature rows to the same months. Cached off-season files remain on disk but cannot enter this experiment.

## Final project experiment: track only

- **Decision:** Treat the five-state (24-hour) IBTrACS track-only experiment as the project deliverable. Forecast direct north/east displacement and wind change at +6, +12, +24, and +48 hours, comparing persistence, constant motion, ridge, and a compact GRU.
- **Why:** It produces a complete, reproducible result with local data and a held-out chronological whole-storm test set. It meets the planned track-only ablation before adding any data-acquisition risk.
- **Consequence:** The Phase 4 per-horizon history-length result remains a useful ablation, but the final primary comparison fixes history to 24 hours for a clear, consistent system.

## Freeze ERA5/CDS and defer external environmental data

- **Decision:** Freeze the existing ERA5/CDS pipeline and omit the environmental/satellite ablations from the current delivery. Do not resume acquisition or add another ingestion path.
- **Why:** CDS recovery has consumed substantial effort without a complete feature table. No TropiCycloneNet subset is available locally, and its documented storm-centered grids and label conventions do not directly match the existing IBTrACS samples or planned ERA5 variables.
- **Consequence:** The report must state that environmental and TCIR extensions are future work, rather than presenting an unverified cross-dataset comparison.

## Freeze the track-only architecture for error analysis

- **Decision:** Keep the final five-state GRU, candidate widths, training budget, features, target definitions, and chronological splits unchanged while analysing errors.
- **Why:** Error findings are useful only if they describe the reported model rather than a moving architecture.
- **Consequence:** The analysis recreates the selected model deterministically and exports held-out rows. It reports storm-macro, 2,000-resample bootstrap intervals because six-hour windows within one storm are correlated.

## Static historical research-replay frontend

- **Decision:** Build the frontend as a dependency-free static page that projects the existing held-out error rows onto an SVG map; export a compact JSON payload from those rows rather than adding a backend or running models in the UI.
- **Why:** `held_out_errors.csv` already includes true IBTrACS issue/target states and every frozen model prediction at +6/+12/+24/+48 hours. A static export is sufficient for historical replay and prevents a UI from implying live operational inference.
- **Consequence:** The UI labels itself RESEARCH REPLAY / NON-OPERATIONAL. Its export supplements result rows with the corresponding real IBTrACS `NAME`, `BASIN`, and `USA_ATCF_ID` only for display. A bundled Natural Earth-derived country-boundary file provides an offline basemap; environmental overlays and other absent experiment fields remain unavailable.

## TCIR feasibility for the frozen held-out period

- **Decision:** Do not start a TCIR image ablation against the current 2020--2025 held-out storms using the documented TCIR release.
- **Why:** TCIR documents Atlantic labels sourced from HURDAT2 for 2003--2016, while this experiment's test storms are 2020--2025. The official HDF5 metadata key does not publish an explicit ID/timestamp schema, and the dataset is distributed as combined regional HDF5 files, so exact linkage and storage cannot be confirmed from its public metadata page alone.
- **Consequence:** Expected usable held-out matches are zero; no dataset was downloaded and no utility is needed. A future image study requires a source with held-out-period coverage or a separately approved evaluation design, plus an explicit audit of storm identifiers, exact UTC timestamps, image availability, and storage before data ingestion.

## HURSAT-B1 v07b feasibility audit

- **Decision:** Do not treat HURSAT archive-index matches as usable satellite samples or alter the frozen evaluation. Consider only a separately scoped 2020--2024 matched-subset pilot after frame-level availability is verified.
- **Why:** NOAA's v07b storm-archive index matches 80/94 2020--2024 held-out storm IDs (1,327/1,586 issue origins), but it does not expose each archive's frame timestamps or per-channel validity. 2025 is outside the release. The matched storm archives total 7.50 GB compressed, before extraction.
- **Consequence:** Exact usable issue-time/sample count is currently unknown. Match with IBTrACS SID's 13-digit serial plus exact UTC frame time, and use only imagery at or before each issue time. The current audit downloaded no imagery and leaves the track-only experiment, splits, results, evaluation, and frontend unchanged.

## Separate TCC genesis feasibility audit

- **Initial decision (superseded below):** Link TCC directly to any coincident IBTrACS position within 1,000 km, ranking candidate SIDs by overlap count then nearest distance.
- **Why it was revised:** Direct positional matching attached nearby but unrelated IBTrACS storms to many TCC tracks and produced 142 multi-SID cases. The public merged table has a stronger shared track number between TCC `source=1` records and their `source=2` IB-derived continuation; source=2 begins three hours after the TCC endpoint for all 1,420 shared IDs.
- **Consequence:** Do not use the initial position-only TCC-to-IB mapping for labels or basins. Preserve the earlier manifest as historical audit output; regenerate with the shared-ID bridge and safeguards below.

## TCC linkage-resolution audit

- **Decision:** Join source=1 TCC tracks to source=2 continuations by shared `number`, validating the three-hour seam and ≤1,000-km endpoint displacement. Map each continuation to official IBTrACS by exact-time positions, ranking by matched fixes then nearest distance; leave exact score ties and single-fix matches unresolved. Use track-constant `devflag=0` only as evidence of lifetime nondevelopment, never as an input feature. Require full TCC follow-up for unlinked negatives and adequate IBTrACS wind coverage for linked negatives. Exclude `basin=UNKNOWN` from the recommended basin-resolved experiment.
- **Why:** [TCC documentation](https://journals.ametsoc.org/view/journals/atot/28/8/2010jtecha1522_1.xml) describes recording developing/nondeveloping track status, and all 1,250 `devflag=1` tracks in this release have a same-number source=2 continuation. This shared ID avoids selecting an input from its future genesis location; later positions are only used to resolve label metadata. The merged CSV does not preserve basin for TCC-only tracks. An older [v01r01 TCC export](https://ams.confex.com/ams/92Annual/webprogram/Handout/Paper201370/AMS_annual_2012_pdf.pdf) lists `initial_ba`, but covers 1982–2009 and has no documented crosswalk into this merged file's `number` for the full period.
- **Consequence:** The 56,514 unmatched tracks are all `devflag=0`, so they support negatives when their TCC tracks cover the full target window; their basins remain unknown. Keep them out of basin-conditioned or basin-stratified samples unless the original basin field can be crosswalked consistently for the full period. The basin-resolved cohort has 1,276 issue times (839/437 +24h positive/negative). The audit manifest retains unknown-basin candidates separately. This work does not change the frozen track-only experiment.

## Genesis external-data preflight gate

- **Decision:** Use the official NOAA GridSat S3 object index, not the timed-out THREDDS catalog, as the availability authority. Retain a resumable annual metadata cache before considering any imagery transfer.
- **Why:** The S3 listing establishes exact key presence and object bytes for all 6,438 required timestamps. The completed audit found 6,438/6,438 frames and 1,276/1,276 complete candidate histories, totaling 247,888,386,134 source bytes.
- **Consequence:** The next machine must have a documented transfer budget before extraction. The proposed compact output is nine 96x96 `float16` IR-window patches per candidate, centred at each historical TCC position; it is 211,673,088 bytes before metadata. This environment has 28 GiB free, so it must not retrieve 230.86 GiB of source objects. ERA5’s availability does not override this transfer gate. No labels, cohort, or frozen track-only artifacts change.

## Genesis GridSat and ERA5 acquisition gate

- **Decision:** Freeze GridSat as a NO-GO and do not construct an ERA5-only feature dataset from a source that cannot preserve the frozen 3-hour timestamps compactly.
- **Why:** NOAA's exact GridSat index proves all 6,438 required frames exist, but their 247,888,386,134-byte source transfer exceeds the workstation budget. Official CDS pressure-level ARCO is 6-hourly. The inspected public hourly ERA5 ARCO distribution stores each pressure variable as a full global/all-level chunk per hour (117,354,372 bytes for representative u wind), so extracting six raw fields for 6,438 required timestamps would be multi-terabyte transfer rather than point-level acquisition.
- **Alternatives considered:** Remote GridSat subset services timed out or could not identify an HDF5 spatial subset; CDS ARCO loses every intervening 3-hour step; public hourly ARCO is the same ERA5 product but has unsuitable chunking. None may be silently treated as compact point access.
- **Consequence:** The 1,276-row cohort, labels, and splits are preserved but no genesis environmental tensor is emitted. Revisit only with hourly, location-chunked ERA5 access or an approved, bounded CDS point/time retrieval plan. This supersedes the prior conditional ERA5 acquisition path for genesis only; it does not alter the historical frozen track-only ERA5 code or experiment.
- **POC update:** The official geo-chunked ARCO route was tested for one exact 6-hour-aligned cohort row. Metadata and the nearest grid coordinate were available, but the first required `u@850` chunk GET timed out after 20.78 seconds; xarray/Zarr returned a NaN fill value. The 2,548,510-byte chunk exists, but its field value was not reliably retrieved. Stop without retrying through a different client or generating a partial feature table. This does not reopen the full-cohort route: 892 of the 1,276 rows are not aligned to the pressure store's 6-hour grid, and a single point read failed. Audit: `results/genesis_era5_poc_audit.json`.

## LPS–TCC crosswalk audit contract

- **Decision:** Keep the audit isolated from genesis/preflight logic. Join source=1 TCC positions to LPS rows only at the same UTC time and within 1 degree great-circle distance, using `lat_detected`/`lon_detected`. Match a unique candidate only when the LPS detection is not also claimed by another unique TCC `(track,time)` position; report all candidate counts and ambiguity rather than breaking ties. Overlapping history windows reuse a deduplicated physical TCC position.
- **Why:** The LPS catalogue has no TCC IDs, and its final/published centers may be projected or interpolated. A conservative detected-center match preserves the TCC candidate, target, and split semantics for any retained regional subset. LPS intensity/category fields are not inputs or labels.
- **Consequence:** The local v5.4.2 catalogue covers 3,120/6,438 required TCC timestamps. The unchanged 1-degree crosswalk yields 1,275 zero-match rows, one 5/9 partial row, zero full matches, zero ambiguities, and five accepted positions on one LPS track. All basin, split, and label groups therefore have 0% full matches; strict retained size is zero. The RH feature is null in all 368,925 catalogue rows, so it cannot enrich this cohort. Verdict: **C — crosswalk fails; abandon this catalogue for TCC enrichment**. The Parquet row ID repeats across times; validation now checks its already-used `(time,row_id)` identity rather than requiring global uniqueness. No matching or scientific criteria changed.

## TCC genesis ERA5 request-manifest gate

- **Decision:** Reopen only the separate 1,276-row TCC genesis ERA5 acquisition path for a metadata-only CDS request manifest. Keep the Phase 5 North Atlantic ERA5 pipeline and frozen track-only experiment unchanged. Do not submit requests in this stage.
- **Why:** The accepted cohort's frozen histories require 3-hour timestamps, so the hourly ERA5 pressure-level CDS product and the separate hourly single-level SST product retain every requested input time while allowing regional `area` selection. The failed ARCO point read does not establish that this CDS route is unavailable.
- **Request contract:** Batch by unique TCC track×UTC day. Each pressure request contains all required pressure source variables and the full required level set; SST is separate. Dynamic boxes union that batch's TCC-centered 5° patches and round outward on the native 0.25° grid. Use the smallest circular longitude interval so dateline boxes stay regional. Deduplicate shared `(track,time)` history positions. Crop centers snap to the nearest native grid point (≤0.125° shift) so each saved patch is exactly 21×21 cells; exact centers and bounds are retained in the manifest.
- **Predictor preservation:** Keep vorticity@850 derived from u/v@850, 850–200-hPa u/v shear, RH@700 derived from temperature/specific humidity@700, omega@500, and SST. CDS pressure variable×level selection is Cartesian; the manifest requests 20 combinations to include the seven required pairs and explicitly accounts for 13 extra combinations.
- **Durability:** Canonical payloads use the existing ERA5 SHA-256 fingerprint helper. Deterministic request keys, payloads, crop centers, and a frozen cohort/labels/splits fingerprint are stored in an atomically written manifest. Any later submitter must use the existing lock, journal, ambiguous-submission recovery, and retry/resume path, and may submit only after manifest inspection. The generator itself has no CDS client or submission call.
- **Consequence:** The current manifest has 1,396 track/day batches and 2,792 requests. Float32 uncompressed request arrays estimate to 570.82 MiB including Cartesian extras; cropped predictor arrays estimate to 89.85 MiB. These are array-size estimates, not compressed CDS transfer estimates. This initially covered manifest generation/review only; a later user authorization reopened raw acquisition. It does not reopen GridSat.

## TCC genesis manifest acquisition adapter

- **Decision:** Extend `FileSpec` with an optional exact manifest payload/dataset/key form and route it through the existing `acquire()` coordinator. Keep the legacy generated Phase-4 `FileSpec` path unchanged.
- **Why:** The durable safety mechanisms already operate on a file identity and a canonical payload. A small adapter preserves the approved JSON request exactly while reusing the process lock, atomic journal, ambiguous-submission recovery, bounded concurrency, curl resume, byte-count check, and promotion path.
- **Validation:** Manifest loading fails before constructing a CDS client if any key, date, fingerprint, dataset, variable list, pressure level list, or output format differs from the approved contract. Download validation accepts the equivalent signed or 0–360 longitude representation CDS may emit for dateline areas, but requires the exact circular 0.25° interval, times, variables, units, and levels.
- **Consequence:** `--genesis-manifest` acquires/validates raw files only. It does not implement crop or feature extraction. No CDS request is authorized or submitted by this implementation work.
