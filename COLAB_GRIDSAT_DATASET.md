# GridSat dataset extraction (Colab)

The extractor uses the frozen IBTrACS filtering, storm split (`<=2015`, `2016-19`, `2020-25`) and direct-horizon origin builders. It saves no targets. It groups all pending origins by exact issue time, downloads one temporary global file, writes 201x201 float16 IR crops, then deletes the source file.

Only crops with `valid_fraction >= 0.90` are saved. Origins below that threshold are not imputed; they remain in `manifest.csv` with `status=excluded`, their measured `valid_fraction`, and `exclusion_reason`. Saved rows have `status=saved` and a crop path.

```bash
!pip -q install netCDF4
!PYTHONPATH=. python -u -m src.build_gridsat_dataset --ibtracs /content/ibtracs.NA.list.v04r01.csv --output /content/gridsat_dataset --max-source-files 100
```

Run the same command repeatedly to resume. Set `--max-source-files 0` only when sufficient Colab time and network quota are available. Copy `/content/gridsat_dataset` to Drive between sessions; it contains `origins.csv`, append-only `manifest.csv`, crops, and `summary.json`.
