# plcmf

MIMO-GRU forecasting with PLCMF-based feature selection across 6 electricity-load
regions and 3 time periods.

## Project structure

```
MIMO-GRU-PLCMF.py    Main Python script: feature-masked MIMO-GRU forecast
plcmf_analysis.r     R script: PLCMF / CCF / Granger / ADF analysis pipeline
output_analysis/     Figures and CSVs produced by the R script
pyproject.toml       uv project definition (Python 3.13)
```

## Requirements

- [uv](https://docs.astral.sh/uv/) (Python package manager)
- Python 3.13 (managed automatically by uv)
- Data CSVs (see [Data](#data))

## Setup

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
# or: brew install uv

# 2. Create the virtual environment and install dependencies
uv sync
```

This creates a `.venv/` and installs `numpy`, `pandas`, `matplotlib`,
`scikit-learn`, and `tensorflow` exactly as pinned in `uv.lock`.

## Data

Place the following CSVs in the repository root. Each file must contain a
`date`, `time`, and load columns named `region-1` ... `region-6` (or
`region_1` ... `region_6`):

| Period   | Training        | Testing        |
|----------|-----------------|----------------|
| Period 1 | `trainingpagi.csv`   | `testingpagi.csv`   |
| Period 2 | `trainingsiang.csv`  | `testingsiang.csv`  |
| Period 3 | `trainingmalam.csv`  | `testingmalam.csv`  |

## Running the forecast

```bash
uv run python MIMO-GRU-PLCMF.py
```

The script runs the `SW`, `PACF`, and `PLCMF` configurations for all three
periods and writes results to `hasil_MIMO_GRU_jurnal/`:

- `forecast_<config>_<period>_<region>.csv` — rolling one-step-ahead forecasts
- `all_metrics.csv` — RMSE / MAPE / SMAPE per config, period, region, and split
- `plot_test_<config>.png` — actual vs. forecast plots

## Optional: R analysis

Run `plcmf_analysis.r` in R/RStudio to reproduce the feature-selection analysis
(ADF, CCF, PLCMF with BH-FDR, Granger causality) that produces the inputs used
to define the feature masks in `MIMO-GRU-PLCMF.py`. It writes its output to
`output_analysis/`.

## License

See [LICENSE](LICENSE).
