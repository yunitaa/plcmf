# plcmf

MIMO-GRU forecasting with PLCMF-based feature selection across 6 electricity-load
regions and 3 time periods.

The repository is split into two pipelines:

1. **`plcmf_analysis.r`** — R analysis pipeline that derives the input features
   (ADF stationarity, CCF, PLCMF with BH-FDR correction, Granger causality).
2. **`MIMO-GRU-PLCMF.py`** — Python forecast pipeline that trains feature-masked
   MIMO-GRU models (configurations `SW`, `PACF`, `PLCMF`) and produces the
   forecast accuracy metrics.

## Requirements

- [uv](https://docs.astral.sh/uv/) (Python package manager)
- Python 3.13 (managed automatically by uv)
- Data CSVs (see [Data](#data))
- R (for the R analysis pipeline)

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

> **Data availability disclaimer:** The load datasets used in this study are
> proprietary to PT PLN (Persero) and are subject to copyright/confidentiality
> restrictions. They are therefore **not** included in this repository and
> cannot be redistributed. All code, methodology, and output files are fully
> provided; to reproduce the results, supply your own data in the format below.
> For a data-free smoke test of the R pipeline, `plcmf_analysis.r` includes a
> `generate_synthetic_data()` function (see "OPSI A" in the script).

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
periods and writes every result to **`hasil_MIMO_GRU_jurnal/`**.

## Repository structure & where to find results

```
plcmf/
├── MIMO-GRU-PLCMF.py            # Python forecast pipeline
├── plcmf_analysis.r             # R feature-selection analysis pipeline
├── pyproject.toml / uv.lock     # uv project definition & locked dependencies
├── hasil_MIMO_GRU_jurnal/       # ← Python pipeline output (generated)
└── output_analysis/             # ← R pipeline output (generated)
```

### Results from the forecast pipeline → `hasil_MIMO_GRU_jurnal/`

This directory is created by `MIMO-GRU-PLCMF.py`:

| File pattern | Content |
|--------------|---------|
| `all_metrics.csv` | **Main numerical results table.** RMSE, MAPE, SMAPE for every combination of configuration (`SW`/`PACF`/`PLCMF`) × period (`period_1`–`period_3`) × region (`region_1`–`region_6`) × split (`train`/`test`) — the source for the reported forecast accuracy values. |
| `forecast_<config>_<period>_<region>.csv` | Rolling one-step-ahead test forecasts vs. actuals (240 steps) for each of 54 config × period × region combinations. |
| `plot_test_<config>.png` | Actual vs. forecast plots, one figure per configuration (3×6 panel grid of period × region). |

The final console summary (per-configuration/per-period mean MAPE, SMAPE, RMSE
and RMSE win-counts) mirrors the rows of `all_metrics.csv`.

### Results from the R analysis pipeline → `output_analysis/`

This directory is created by `plcmf_analysis.r`. Committed outputs currently
present, plus files regenerated on each run:

**Numerical results (CSV)**

| File | Content |
|------|---------|
| `plcmf_values_period_1..3.csv` | PLCMF partial-correlation values per lag per region pair (`PLCMF_Value`, `Sig_PLCMF`, `Bound`) |
| `granger_values_period_1..3.csv` | Granger causality p-values per region pair (older export; the current script also writes `granger_aic_period_*.csv` with the lag order `p*`) |
| `gru_features_period_1..3.csv` | Final recommended GRU input features per region: own PACF lags + cross-region PLCMF/Granger-significant lags |
| `adf_test_period_*.csv` | ADF stationarity test statistic, p-value, and selected lag per region |
| `adf_summary_table2a.csv` | ADF results across all periods (paper Table 2a) |
| `pstar_distribution_table2b.csv`, `pstar_detail.csv` | Granger optimal-lag-order `p*` distribution (paper Table 2b) |
| `bh_fdr_summary_period_*.csv`, `bh_fdr_detail_period_*.csv` | BH-FDR multiple-testing correction results |
| `bh_fdr_all_periods_table3.csv` | BH-FDR sensitivity summary across periods (paper Table 3) |

**Figures (PNG)**

| File | Content |
|------|---------|
| `ccf_period_*.png` | Multi-lag cross-correlation matrix (ACF on the diagonal) |
| `plcmf_heatmap_period_*.png`, `plcmf_lines_period_*.png` | PLCMF partial-correlation heatmaps and per-element vs. lag plots |
| `bh_fdr_period_*.png` | Significant elements before vs. after BH-FDR per lag |
| `granger_period_*.png`, `granger_aic_period_*.png` | Granger causality p-value map and significance map |

## R analysis

Run `plcmf_analysis.r` in R/RStudio to reproduce the feature-selection analysis
that derives the feature masks hard-coded in `MIMO-GRU-PLCMF.py`. Edit
`BASE_DIR` at the entry point (line ~1448) to point at your data CSVs before
running. All output is written to `output_analysis/`.

## License

See [LICENSE](LICENSE).
