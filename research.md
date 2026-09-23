# Cyclone Forecasting — Research Notes

## 1. Problem

Tropical-cyclone forecasting estimates how an already identified cyclone will evolve: mainly its future track and intensity, and sometimes pressure. It is a Time Series problem because the forecast depends on ordered past observations, lagged motion and intensity changes, and multiple forecast horizons. It is an AI/ML problem because cyclone evolution depends on nonlinear relationships between storm history and its atmospheric and oceanic environment. The project is meaningful for both courses if it focuses on a manageable short-range forecasting task rather than trying to build an operational weather system.

## 2. Papers

### TCBench (Gomez et al., 2026)

- **Dataset:** Global IBTrACS observations, with outputs from dynamical and neural weather models.
- **Inputs:** An existing cyclone's initial position and intensity, plus forecast fields/model outputs for benchmark evaluation.
- **Outputs:** Future cyclone track and intensity.
- **Model:** Benchmark and evaluation framework covering TIGGE, AIFS, Pangu-Weather, FourCastNet v2 and GenCast.
- **Forecast horizon:** Short to medium range, approximately 1–5 days.
- **Metrics:** Deterministic and probabilistic storm-following track and intensity metrics.
- **Main takeaway:** Neural weather models can forecast tracks skillfully, while intensity remains harder and often needs post-processing.
- **Relevance to our project:** Supports forecasting an existing storm and emphasizes consistent, storm-following evaluation. Our project can use the same principle with much smaller models.

### Jiang et al. (2023), *Transformer-based tropical cyclone track and intensity forecasting*

- **Dataset:** CMA best-track data for the Northwest Pacific, 1980–2021.
- **Inputs:** Historical cyclone position, intensity and derived historical features.
- **Outputs:** Latitude, longitude, maximum sustained wind and minimum sea-level pressure.
- **Model:** Transformer with temporal self-attention and feature preprocessing.
- **Forecast horizon:** Short-range, including lead times up to about 24 hours.
- **Metrics:** Regression errors for position, wind and pressure, compared with LSTM and GRU models.
- **Main takeaway:** A Transformer can model temporal dependencies and performed better than the recurrent models tested in that study.
- **Relevance to our project:** Confirms that joint track-and-intensity prediction is a valid multivariate time-series task. It does not establish that a Transformer is necessary for a small undergraduate dataset.

### Tong et al. (2022), *Short-term prediction of the intensity and track of tropical cyclone via ConvLSTM model*

- **Dataset:** Northwest Pacific cyclone data with track and intensity records.
- **Inputs:** Historical track and intensity variables.
- **Outputs:** Future track and intensity.
- **Model:** ConvLSTM, compared with standard LSTM and other forecast guidance.
- **Forecast horizon:** 6 hours and longer short-term leads.
- **Metrics:** Track and intensity forecast errors.
- **Main takeaway:** Combining temporal modelling with feature interactions can improve on a plain LSTM, but the problem remains difficult at longer leads.
- **Relevance to our project:** Provides a precedent for comparing a simple recurrent baseline with a more expressive model, without requiring satellite imagery.

### Lin et al. (2025), *Enhancing tropical cyclone track and intensity predictions with the OWZP-Transformer model*

- **Dataset:** CMA best-track data and ERA5 reanalysis for the Northwest Pacific, 1980–2023.
- **Inputs:** Position, wind, pressure, translation speed/direction, intensity change, ERA5 steering flow, wind shear, humidity and a structural Okubo–Weiss–Zeta variable.
- **Outputs:** Next-6-hour latitude, longitude, intensity and pressure; 24-hour future intensity change.
- **Model:** Transformer with physically motivated predictor groups and ablation experiments.
- **Forecast horizon:** 6-hour state prediction and 24-hour intensity-change prediction.
- **Metrics:** MAE, RMSE, \(R^2\), signed bias, Haversine track distance and rapid-intensification detection scores.
- **Main takeaway:** Environmental variables helped intensity prediction in this experiment but did not necessarily improve track prediction; predictor-group ablations were informative.
- **Relevance to our project:** Motivates a future environmental study, but it is not part of the completed track-only experiment.

### Huang et al. (2024), *Global Tropical Cyclone Intensity Forecasting with Multi-modal Multi-scale Causal Autoregressive Model*

- **Dataset:** A global satellite-and-ERA5 dataset (SETCD) aligned with best-track data.
- **Inputs:** Historical intensity, environmental fields and satellite observations.
- **Outputs:** Future cyclone intensity.
- **Model:** Multimodal, multiscale causal autoregressive deep-learning model.
- **Forecast horizon:** Multi-step intensity forecasting.
- **Metrics:** Intensity regression errors over different lead times.
- **Main takeaway:** Satellite and environmental information can support intensity forecasting, but the data pipeline and model are substantially more complex.
- **Relevance to our project:** Satellite fusion is a possible later extension, not a necessary part of the core semester project.

## 3. Datasets

### IBTrACS

NOAA's International Best Track Archive for Climate Stewardship is a global collection of agency best tracks. It provides storm identifiers, timestamps, position, wind, pressure, basin and some derived motion variables.

It is the best core dataset for this project because it is public, long-running and directly represents observed cyclone evolution. Important issues are missing values, provisional records, agency-specific wind averaging periods and duplicated/interpolated temporal records. Wind columns should not be mixed without checking their measurement conventions. Data should be split by whole storm, not by random rows.

### ERA5

ERA5 is an hourly global reanalysis with atmospheric pressure-level and surface variables. It can provide environmental predictors such as steering winds, vertical wind shear, humidity and sea-surface temperature around a storm.

It is useful for an ablation or extension after the best-track pipeline works. The main issues are download/storage cost, spatial and temporal matching, retrospective availability, possible intensity biases at coarse resolution and leakage if future-time or future-track information is used.

### Satellite datasets

TCIR provides centred infrared, water-vapour, visible and passive-microwave cyclone images. TC PRIMED provides a much larger global collection of satellite observations and ERA5 environmental files.

Satellite data can add information about cyclone structure and intensity, but introduces image preprocessing, missing-channel, sampling and image-to-future-label alignment problems. TC PRIMED is also very large. Satellite imagery is not implemented in the completed experiment.

## 4. What Existing Research Tells Us

- Track forecasting from historical observations and modern weather-model fields is already well studied; using AI for cyclone prediction alone is not a research gap.
- Joint prediction of track, wind and pressure with LSTM, GRU, ConvLSTM, TCN and Transformer models is common.
- Track forecasts are generally easier than intensity forecasts.
- Rapid intensification, rapid weakening, recurving tracks and land interaction remain difficult.
- Environmental variables can improve intensity forecasts but may add noise to track forecasts.
- Modern neural weather models perform relatively well on track while intensity remains less reliable.
- Recent papers increasingly use ablation studies and physically motivated predictors rather than only changing neural architectures.
- Common mistakes to avoid are random row-level splits, leakage from future observations, inconsistent wind definitions, weak persistence baselines, and reporting only average errors without horizon or storm-level breakdowns.
- A useful undergraduate contribution is a reproducible comparison of historical information and environmental information under unseen-storm evaluation, not a claim of inventing cyclone AI.

## 5. What This Means for Our Project

Our current direction is to:

- forecast an already identified cyclone;
- focus primarily on future track and maximum wind;
- use historical best-track data as the core input;
- treat environmental information as future work because ERA5/CDS is frozen;
- compare learned models against simple persistence, motion and statistical baselines;
- evaluate performance on storms that were not used for training.

The completed experiment fixes these choices: North Atlantic HURDAT-aligned IBTrACS, five six-hour history states, +6/+12/+24/+48-hour direct forecasts, a compact GRU plus three baselines, and whole-storm chronological splits. `design.md` records the final scope.

## Final submission status

The final reported result is the track-only experiment only. It evaluates Persistence, Constant Motion, Ridge, and a seed-42 compact GRU on identical held-out North Atlantic storm origins from 2020--2025. Mean errors are origin-weighted; the report additionally provides storm-macro means, storm-clustered bootstrap intervals, and a paired whole-storm GRU--Ridge comparison. No random row split, environmental feature, satellite image, or externally generated forecast enters these results.

The separate Genesis ERA5 acquisition is incomplete (899/2,792 validated requests). Its complete subset has no chronological test candidates, so it is future work and must not be presented as an environmental ablation or a contribution to the reported forecast skill.

## 6. Questions to Resolve

- Which basin offers the best balance of data quality, relevance and manageable scope?
- Should pressure be a primary output or a secondary analysis target?
- Which agency/source and wind averaging convention should be used consistently?
- What history length and forecast horizons are feasible after inspecting the data?
- Which ERA5 variables can be extracted without excessive storage or leakage risk?
- Is direct multihorizon prediction preferable to recursive forecasting for the final experiment?
- Which simple statistical baseline is strongest and easiest to reproduce?
- How should missing labels and storms with short records be handled?
- What compute budget is available for recurrent, convolutional or Transformer comparisons?
- Is satellite imagery realistically achievable after the core experiment?

## 7. References

- [Gomez et al. (2026), *TCBench: A Benchmark for Tropical Cyclone Track and Intensity Forecasting at the Global Scale*](https://arxiv.org/abs/2601.23268)
- [Jiang et al. (2023), *Transformer-based tropical cyclone track and intensity forecasting*](https://www.sciencedirect.com/science/article/pii/S0167610523001435)
- [Tong et al. (2022), *Short-term prediction of the intensity and track of tropical cyclone via ConvLSTM model*](https://www.sciencedirect.com/science/article/pii/S0167610522001301)
- [Lin et al. (2025), *Enhancing tropical cyclone track and intensity predictions with the OWZP-Transformer model*](https://www.nature.com/articles/s44387-025-00037-3)
- [Huang et al. (2024), *Global Tropical Cyclone Intensity Forecasting with Multi-modal Multi-scale Causal Autoregressive Model*](https://arxiv.org/abs/2402.13270)
- [NOAA, International Best Track Archive for Climate Stewardship (IBTrACS)](https://www.ncei.noaa.gov/products/international-best-track-archive)
- [NOAA, IBTrACS v04r01 column documentation](https://www.ncei.noaa.gov/sites/default/files/2025-09/IBTrACS_v04r01_column_documentation.pdf)
- [ECMWF/Copernicus, ERA5 hourly data on pressure levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels)
- [TCIR satellite cyclone dataset](https://www.csie.ntu.edu.tw/~htlin/program/TCIR/)
- [NOAA/CIRA, TC PRIMED dataset](https://rammb-data.cira.colostate.edu/tcprimed/)
