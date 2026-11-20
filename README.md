
# Retrieval-Augmented Air Quality Forecasting

## Overview

An independent research project exploring air-quality
time-series forecasting using historical environmental
measurements and retrieval-augmented machine learning.

## Dataset

Beijing Multi-Site Air Quality Dataset (UCI).

- 12 monitoring stations
- 420,768 hourly observations
- March 2013 – February 2017

## Current Progress

- Data loading and validation
- Missing-value analysis
- Exploratory data analysis
- Seasonal PM2.5 analysis
- Correlation analysis with wind speed

## Implemented Research

1. Chronological train/validation/test splits
2. Persistence and Ridge regression validation baselines
3. Gradient boosting and historical kNN retrieval experiment runner
4. Ridge with retrieved residuals
5. Per-season and per-station evaluation exports
6. Optional zero-shot Chronos-Bolt baseline (requires extra package and model download)

## Installation

Python 3.12 is recommended.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Dataset Source

https://archive.ics.uci.edu/dataset/501/beijing

Download and extract the original station CSV files
under data/raw/.

## Status

Research in progress. Prior notebook validation results: persistence MAE 9.785, RMSE 19.059; Ridge MAE 9.677, RMSE 18.274 (µg/m³). Retrieval and boosting results must be measured by running the pipeline; do not infer improvements from implementation alone.

Run `python -m pytest -q`, then `python -m src.run_experiments` from the project root. New comparison results are written to `results/`. Test period is not used for model selection.

The `reports/` directory is intentionally omitted from the repository; application reports can be submitted separately as PDF.