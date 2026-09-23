# Monthly Activity research frontend

## Goal

Present the existing monthly storm-activity experiment as a compact research page without changing the experiment or Spatial Trajectory page.

## Implementation

- `src/export_monthly_frontend.py` reads the existing monthly series, decomposition, forecasts, and metrics and writes strict JSON.
- `frontend/monthly.html`, `monthly.css`, and `monthly.js` render the actual series, selectable model forecasts, month-of-year seasonality, decomposition, model comparison, and methodology.
- Missing decomposition edge values remain null in the payload; no scientific values are estimated in the frontend.
- The main chart compresses 1980–2019 and gives most width to the 2020–2025 test window. SVG hit areas provide crosshair/date/value tooltips; bars, decomposition, and metric rows respond to hover, while model selection and metric clicks update the selected forecast view.
- The Spatial Trajectory template has one direct link to `monthly.html`; no replay controls or map rendering were redesigned.

## Verification

The exporter, strict JSON load, both JavaScript syntax checks, diff whitespace check, and full repository suite (108 tests) pass. Browser rendering was not smoke-tested because browser automation was unavailable in this environment.

## Limitations

The page is static and depends on the generated JSON being refreshed when monthly artifacts change. It does not provide live inference.
