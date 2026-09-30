# GridSat POC in Google Colab

This runs exactly the fixed 11-origin POC. It does not train a model or alter
the completed forecasting outputs. Each global GridSat source file is streamed
to Colab temporary storage, cropped to 201x201 `irwin_cdr` cells, and deleted.
Only `results/gridsat_poc/crops/`, `manifest.csv`, and `summary.json` remain.

1. Start a new Colab notebook with a standard CPU runtime.
2. Run:

```bash
!git clone https://github.com/rr23commits/tropical-cyclone-forecasting.git
%cd tropical-cyclone-forecasting
!pip -q install netCDF4
```

3. Upload the project input file `ibtracs.NA.list.v04r01.csv` to `/content`.
   It is not stored in the public repository.
4. Run exactly:

```bash
!PYTHONPATH=. python -u -m src.gridsat_poc \
  --ibtracs /content/ibtracs.NA.list.v04r01.csv \
  --output results/gridsat_poc
```

If an origin fails, execution stops and writes `results/gridsat_poc/FAILED.json`.
Do not rerun into an existing output directory; choose a new empty directory or
delete the failed POC output deliberately.
