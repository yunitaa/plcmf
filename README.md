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
> cannot be redistributed. To reproduce the results, supply your own data in
> the format below.
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
periods and prints the forecast accuracy metrics to the console.

## Repository structure & where to find results

```
plcmf/
├── MIMO-GRU-PLCMF.py            # Python forecast pipeline (deposited)
├── plcmf_analysis.r             # R feature-selection analysis pipeline (deposited)
├── pyproject.toml / uv.lock     # uv project definition & locked dependencies (deposited)
└── output_analysis/             # R pipeline outputs (deposited)
```

### Deposited vs. restricted materials

| Category | Items | Status |
|----------|-------|--------|
| **Deposited** | `MIMO-GRU-PLCMF.py`, `plcmf_analysis.r`, `pyproject.toml`/`uv.lock` | Included in this repository/release |
| **Deposited** | `output_analysis/` (R pipeline CSVs and figures) | Included in this repository/release |
| **Deposited** | `output_analysis/all_metrics.csv` — aggregate forecast error metrics (RMSE/MAPE/SMAPE per configuration × period × region × window), compiled from the forecast runs; **contains no load values** | Included in this repository/release |
| **Restricted** | PT PLN (Persero) load datasets (`training*.csv`, `testing*.csv`) | Not redistributable; not included |

### Results from the R analysis pipeline → `output_analysis/` (deposited)

This directory **is included in this repository**. It was produced by
`plcmf_analysis.r`; a complete run of the script reproduces it:

**Numerical results (CSV)**

| File | Content |
|------|---------|
| `plcmf_values_period_1..3.csv` | PLCMF partial-correlation values per lag per region pair (`PLCMF_Value`, `Sig_PLCMF`, `Bound`) |
| `granger_aic_period_1..3.csv` | Granger causality p-values per region pair with the AIC-selected optimal lag order `p*` |
| `granger_values_period_1..3.csv` | Granger causality p-values per region pair (from an earlier run) |
| `gru_features_period_1..3.csv` | Final recommended GRU input features per region: own PACF lags + cross-region PLCMF/Granger-significant lags |
| `adf_test_period_*.csv` | ADF stationarity test statistic, p-value, and selected lag per region |
| `adf_summary_table3.csv` | ADF results across all periods (paper Table 3) |
| `pstar_distribution_table5.csv`, `pstar_detail.csv` | Granger optimal-lag-order `p*` distribution (paper Table 5) |
| `bh_fdr_summary_period_*.csv`, `bh_fdr_detail_period_*.csv` | BH-FDR multiple-testing correction results |
| `bh_fdr_all_periods_table4.csv` | BH-FDR sensitivity summary across periods (paper Table 4) |

**Figures (PNG)**

| File | Content |
|------|---------|
| `ccf_period_*.png` | Multi-lag cross-correlation matrix (ACF on the diagonal) |
| `plcmf_heatmap_period_*.png`, `plcmf_lines_period_*.png` | PLCMF partial-correlation heatmaps and per-element vs. lag plots |
| `bh_fdr_period_*.png` | Significant elements before vs. after BH-FDR per lag |
| `granger_period_*.png`, `granger_aic_period_*.png` | Granger causality p-value map and significance map |

### Mapping results to the manuscript

Every derived numerical result and figure reported in the manuscript can be
traced to a **deposited** file in this repository (`output_analysis/`):

| Manuscript item | Location | Status |
|-----------------|----------|--------|
| ADF stationarity results per period (Table 3) | `output_analysis/adf_test_period_*.csv`, `output_analysis/adf_summary_table3.csv` | Deposited |
| Granger causality optimal lag order `p*` (Table 5) | `output_analysis/pstar_distribution_table5.csv`, `output_analysis/pstar_detail.csv` | Deposited |
| Granger causality p-values per region pair | `output_analysis/granger_aic_period_*.csv` (legacy: `granger_values_period_*.csv`) | Deposited |
| PLCMF partial-correlation values per lag | `output_analysis/plcmf_values_period_*.csv` | Deposited |
| BH-FDR multiple-testing correction (Table 4) | `output_analysis/bh_fdr_summary_period_*.csv`, `output_analysis/bh_fdr_all_periods_table4.csv` | Deposited |
| CCF / PLCMF / Granger / BH-FDR figures | `output_analysis/ccf_period_*.png`, `plcmf_heatmap_period_*.png`, `plcmf_lines_period_*.png`, `granger_aic_period_*.png`, `bh_fdr_period_*.png` | Deposited |
| Forecast accuracy per configuration (Tables 8–9) | `output_analysis/all_metrics.csv` | Deposited |

## R analysis

Run `plcmf_analysis.r` in R/RStudio to reproduce the feature-selection analysis
that derives the feature masks hard-coded in `MIMO-GRU-PLCMF.py`. Edit
`BASE_DIR` at the entry point (line ~1448) to point at your data CSVs before
running. All output is written to `output_analysis/`.

## License

See [LICENSE](LICENSE).
