
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

## Planned Research

1. Chronological train/validation/test splits
2. Baseline forecasting models
3. Time-series foundation models
4. Historical-pattern retrieval
5. Retrieval-augmented forecasting
6. Evaluation across pollution regimes

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

Research in progress. Forecasting results have not
yet been established.