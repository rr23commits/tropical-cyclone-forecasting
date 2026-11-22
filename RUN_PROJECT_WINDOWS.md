# TROP-CYC: Windows launch instructions

## Easiest option

1. Extract the complete ZIP archive.
2. Open the extracted `tropical-cyclone-forecasting-main` folder in VS Code.
3. Double-click `START_TROP_CYC.bat` from File Explorer, or run `.\START_TROP_CYC.bat` in the VS Code PowerShell terminal.
4. The script creates `.venv` if needed, installs `requirements.txt`, regenerates the spatial and monthly browser data, starts the local web server, and opens all three pages.

The launcher uses the existing experiment outputs. It does **not** retrain GARCH, VAR, PatchTST, or the other forecasting models.

## Three pages

Once the server is running, these addresses work on the same computer:

- Spatial Trajectory: http://127.0.0.1:8000/index.html
- Monthly Activity: http://127.0.0.1:8000/monthly.html
- Genesis Imagery: http://127.0.0.1:8000/genesis.html

Important: `127.0.0.1` and `localhost` are local addresses. They only work on the computer where the Python server is running. Do not open these links from a different device or before the server starts.

## Manual commands (PowerShell)

Run these from the folder containing `requirements.txt`:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.export_replay
.\.venv\Scripts\python.exe -m src.export_monthly_frontend
.\.venv\Scripts\python.exe -m http.server 8000 --directory frontend
```

Keep that terminal open, then visit the three addresses above in your browser.

If PowerShell says port 8000 is already in use, stop the old server with Ctrl+C in its terminal, or use another port (for example 8001) and change the URLs to port 8001.

## Monthly model dashboard

The Monthly Activity page includes the following models when their forecast and test metrics are present in the supplied results:

- Seasonal naive, ARIMA, SARIMA, Additive Holt-Winters
- XGBoost, LSTM, GRU, Prophet, AutoTS, XGBoost + Prophet ensemble
- ARCH/GARCH, VAR, Transformer (PatchTST)

The benchmark table displays held-out test MAE, RMSE, MASE, and WAPE. The page uses the supplied saved experiment outputs; opening the page does not retrain models.

The ARCH/GARCH, VAR, and PatchTST outputs are already included in `results/monthly_activity_ts/` and are exported into `frontend/data/monthly.json` by `src.export_monthly_frontend`.

## Geographic outlines

The Spatial Trajectory page can display storm tracks without the optional country-boundary file. If you want country outlines, download `countries-110m.json` into `frontend/data/` using the optional command in the earlier project instructions.
