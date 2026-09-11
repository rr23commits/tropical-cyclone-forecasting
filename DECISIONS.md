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
