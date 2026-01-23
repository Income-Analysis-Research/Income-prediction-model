# Project Structure

**Income Forecasting Research Project - Clean Structure**

```
newresearchproj/
├── backend/                      # FastAPI web server
│   └── app_improved.py          # Main API with multi-year forecasting (1102 lines)
│
├── frontend/                     # Web interface
│   └── index.html               # Single-page app with ZIP search (1463 lines)
│
├── src/                         # Core source code
│   ├── data/                    # Data processing scripts
│   │   ├── build_us_panel_dataset.py              # Panel data generation (291 lines)
│   │   └── build_us_multiyear_forecasting_dataset.py  # Multi-year datasets (282 lines)
│   │
│   └── training/                # Model training scripts
│       ├── train_us_hybrid.py                     # Legacy hybrid model (522 lines)
│       ├── train_bayesian_mcmc.py                 # Bayesian MCMC (R²=0.77)
│       └── train_us_multiyear_forecast_models.py  # Multi-year forecasting (260 lines)
│
├── datasets/                    # Processed datasets
│   ├── us_panel_dataset.csv    # 110,430 records, 113.9 MB (2018-2022)
│   ├── us_forecasting_1yr.csv  # 82,596 pairs, 16.5 MB
│   ├── us_forecasting_2yr.csv  # 55,027 pairs, 11.0 MB
│   ├── us_forecasting_3yr.csv  # 27,487 pairs, 5.5 MB
│   └── parquet/                # Optimized format (10-100x faster loading)
│
├── models/                      # Trained models
│   ├── us_forecast/            # Legacy single-year models
│   │   ├── stat_model.pkl      # Ridge regression (R²=0.50)
│   │   ├── ml_model.pkl        # RandomForest (R²=0.73)
│   │   └── hybrid_model.pkl    # Stacking (R²=0.69)
│   │
│   ├── us_forecast_multiyear/  # Multi-year forecasting models
│   │   ├── rf_model_1yr.pkl    # 1-year horizon (R²=0.76, RMSE=$35k)
│   │   ├── rf_model_2yr.pkl    # 2-year horizon (R²=0.80, RMSE=$32k)
│   │   ├── rf_model_3yr.pkl    # 3-year horizon (R²=0.90, RMSE=$23k)
│   │   └── *.metadata.json     # Performance metrics + feature importance
│   │
│   ├── us_bayesian/            # Bayesian MCMC models
│   │   ├── bayesian_mcmc_trace.nc  # Posterior samples (R²=0.77)
│   │   └── metadata.json
│   │
│   └── india/                  # India district models
│
├── scripts/                     # Utility scripts
│   ├── download_us_irs_zip_data.py  # IRS data downloader
│   ├── convert_to_parquet.py        # Parquet converter (10x speedup)
│   └── analyze_*.py                 # Data analysis tools
│
├── docs/                        # Documentation
│   └── SOURCES.md              # Data source references
│
├── reports/                     # Generated reports & plots
│
├── terminal_forecast.py         # CLI forecasting tool (267 lines)
│
├── README.md                    # Main documentation (663 lines)
├── PROJECT_STATUS.md            # Development log (1619+ lines)
├── DATASETS.md                  # Dataset specifications
├── requirements.txt             # Python dependencies
└── .gitignore                  # Git exclusions

```

## Key Files Overview

### Backend & API
- **app_improved.py**: Complete FastAPI server with 4 model loaders:
  - `USModelLoader`: Single-year RF/XGB/GB models
  - `BayesianModelLoader`: Hierarchical MCMC with state effects
  - `IndiaModelLoader`: District-level models
  - `ForecastModelLoader`: Multi-year forecasting (1yr/2yr/3yr)

### Data Pipeline
1. **Panel Generation** (`build_us_panel_dataset.py`):
   - Input: Raw IRS ZIP data (2018-2022)
   - Output: 110k records with 15 engineered features
   - Coverage: 27,769 ZIPs across 50 states

2. **Multi-Year Datasets** (`build_us_multiyear_forecasting_dataset.py`):
   - Creates temporal pairs: t → t+n (n=1,2,3)
   - Time-based train/test splits (chronological)
   - 165k+ total transition pairs

### Model Training
1. **Multi-Year Models** (`train_us_multiyear_forecast_models.py`):
   - RandomForest (200 trees, max_depth=20)
   - Separate model per horizon (no extrapolation)
   - Saves models + metadata + feature importance

2. **Bayesian MCMC** (`train_bayesian_mcmc.py`):
   - PyMC hierarchical model with state random effects
   - Calibrated 95% credible intervals
   - NUTS sampler (2000 samples, 1000 tune)

### Frontend & Tools
- **index.html**: Full web interface with:
  - Base year selector (2019-2022)
  - Target year with confidence indicators
  - Scrollable ZIP list (27k+ codes with search)
  - Real-time predictions via `/forecast/us`

- **terminal_forecast.py**: Command-line forecasting:
  - Interactive prompts for ZIP + target year
  - Loads all 3 models at startup
  - Shows actual vs predicted when available

## Model Performance Summary

| Model | Type | Test R² | RMSE | Training Samples | Notes |
|-------|------|---------|------|------------------|-------|
| **Multi-Year 2yr** | RandomForest | **0.80** | **$32k** | 27,509 | Best performer |
| Bayesian MCMC | Hierarchical | 0.77 | $34k | 4,065 | With uncertainty |
| Multi-Year 1yr | RandomForest | 0.76 | $35k | 55,073 | High confidence |
| Legacy ML | RandomForest | 0.73 | $45k | 8,126 | Original |
| Hybrid | Stacking | 0.69 | - | 8,126 | RF+Ridge |
| Statistical | Ridge | 0.50 | - | 8,126 | Baseline |

## Data Coverage

- **Geographic**: All 50 US states + DC
- **Temporal**: 2018-2022 (5 years)
- **Granularity**: ZIP code level
- **Total ZIPs**: 27,769 unique codes
- **Panel Records**: 110,430 ZIP-year pairs
- **Forecast Horizon**: Up to 3 years ahead

## Development Timeline

1. **Phase 1**: Single-year models (R²=0.73)
2. **Phase 2**: ZIP coverage bug fix (4k→27k ZIPs)
3. **Phase 3**: Bayesian MCMC implementation (R²=0.77)
4. **Phase 4**: Multi-year forecasting system (R²=0.80) ← Current
5. **Phase 5**: Web interface + ZIP search ← Current

## Next Steps

- [ ] Commit multi-year system to git
- [ ] Add model documentation notebooks
- [ ] Performance benchmarking suite
- [ ] Extended forecasting (4-5 year horizons)
- [ ] Regional aggregation (county/state level)
- [ ] Uncertainty quantification for multi-year models
