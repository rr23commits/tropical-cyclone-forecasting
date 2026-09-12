# Feature 007: Static research-replay frontend

## Goal and scope

Present the frozen held-out track-only experiment as an interactive historical IBTrACS replay. It is explicitly RESEARCH REPLAY / NON-OPERATIONAL and performs no inference.

## Flow

`make frontend-data` runs `src.export_replay` over the existing held-out error and summary CSVs. `frontend/index.html`, `style.css`, and `app.js` load the resulting local JSON through a standard static server.

## Implementation

The exporter augments each held-out SID with its real IBTrACS name, basin, and ATCF ID. The SVG map uses a bundled Natural Earth-derived boundary topology for actual coastlines/country boundaries, supports pan/zoom/reset-fit, and displays real observed track points plus archived model predictions/actual targets at +6/+12/+24/+48 hours. The slider, scrubber, and playback advance through every exported historical timestamp, while leads appear only when that exact issue time has an archived prediction. The selected Persistence, Constant Motion, Ridge, or GRU model drives forecast paths, table, chart, errors, labels, and legend. The model comparison uses the existing aggregate +24-hour held-out summary.

Each sidebar now has its normal collapse button while open and a separate screen-edge reopen tab while closed. A closed panel translates fully off-screen, leaving no panel content or background visible; the two panel states do not affect one another.

The traditional top header is intentionally absent. The map is a center workspace whose left/right margins match open panel widths and animate to zero independently on collapse. Identity is a small left-panel lockup; replay mode, held-out scope, and the non-operational warning are consolidated in the bottom strip.

The initial app container contains no visible loading copy and is atomically replaced by the workstation after local JSON/topology loading. The right-side collapse control is outside the scrollable panel, anchored to its inner edge, so it remains visible before collapse; its matching viewport-edge tab reopens the panel.

## Files

- `src/export_replay.py`
- `frontend/index.html`, `frontend/style.css`, `frontend/app.js`, `frontend/data/replay.json`, `frontend/data/countries-110m.json`
- `Makefile`
- `tests/test_export_replay.py`

## Verification

`make frontend-data` exported 107 storms. `make test` passed 47 tests, `node --check frontend/app.js` passed, and a local `python3 -m http.server` served the page, replay JSON, and local boundary data successfully.

## Limitations

The retained result files do not include storm names, pressure, environmental fields, ensembles, or a geographic basemap, so the UI deliberately does not claim or synthesize them. The map is a local latitude/longitude projection of the available replay geometry.
