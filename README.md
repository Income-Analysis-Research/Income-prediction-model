# Hierarchical Bayesian Panel Regression — Regional Income Prediction

Estimates average ZIP-code income across the United States (2011–2022) using a **Hierarchical Bayesian panel regression** estimated via **MCMC (NUTS)**. The model decomposes income into source shares (wages, capital gains, dividends, etc.), controls for state-level heterogeneity and annual macro trends, and returns full posterior distributions for every parameter.

---

## Model

$$Income_{z,t} = \beta X_{z,t} + u_{\text{state}} + \gamma_t + \epsilon_{z,t}$$

| Symbol | Meaning |
|--------|---------|
| $z$ | ZIP code |
| $t$ | Year (2011–2022) |
| $X_{z,t}$ | Income composition shares of AGI: wages, business, capital gains, interest, dividends, unemployment, social security |
| $u_{\text{state}}$ | Hierarchical state random intercept ~ N(0, σ²_state) |
| $\gamma_t$ | Year fixed effect capturing inflation and macro trends |
| $\epsilon_{z,t}$ | Observation noise ~ N(0, σ²_obs) |

**Response:** `log(1 + AVG_INCOME)` — income follows a log-normal distribution (Stukalenko 2005).

**Estimated via:** NUTS (No-U-Turn Sampler), 2 chains, 2000 draws, target_accept = 0.95.

---

## Key Results

| Metric | Value |
|--------|-------|
| R² (posterior predictive) | **0.71** |
| RMSE (log scale) | 0.2316 |
| MAE (log scale) | 0.1742 |
| Max R-hat (convergence) | **1.005** ✓ |

### Beta Coefficients

| Income Share | β mean | Direction |
|---|---|---|
| capital_gains | +0.111 | ↑ High-income ZIPs are investment-driven |
| dividends | +0.097 | ↑ Wealth-derived income |
| interest | +0.034 | ↑ |
| wages | −0.103 | ↓ Wage-dependent ZIPs = lower income |
| unemployment | −0.117 | ↓ Strongest negative predictor |
| social_security | −0.114 | ↓ Retirement / low-income areas |
| business | −0.033 | ↓ Slight negative |

### Top States by Random Effect (u_state)
DC (0.385) > CT (0.364) > NJ (0.327) > MA (0.260) > NY (0.145)

### Year Effects (γ_t)
Sharp upward jump in **2020** (+0.318) driven by COVID stimulus. 2021–2022 remain elevated. Pre-2019 years are negative, consistent with real income growth trends.

---

## Repository Structure

```
.
├── hierarchical_bayesian_panel.py   # Main model script (run this)
├── zip_income_panel_structural_2011_2022.csv  # Dataset (IRS SOI, 334K rows)
├── requirements.txt                 # Python dependencies
│
├── references/                      # Research papers informing the model
│   ├── estimation-methods-of-inequality-of-the-population-incomes.pdf  (Stukalenko 2005)
│   ├── kalivoshko2020.pdf           (Kalivoshko et al. 2020 — year effects / inflation)
│   ├── sherri2021.pdf               (Sherri et al. 2021 — multi-chain MCMC strategy)
│   └── zheng2009.pdf                (Zheng et al. 2009 — multiple chains reduce autocorrelation)
│
└── results/                         # Generated after running the model
    ├── trace.nc                     # Full MCMC posterior (load directly with arviz)
    ├── beta_coefficients.csv
    ├── state_effects.csv
    ├── year_effects.csv
    ├── summary_global.csv
    ├── beta_forest.png
    ├── state_effects.png
    ├── year_effects.png
    ├── trace_plot.png
    ├── posterior_predictive.png
    └── prior_vs_posterior.png
```

---

## Setup

### Requirements
- Python 3.12+
- Windows: **MinGW-w64 g++** recommended for PyTensor's C++ backend (4–10× faster)
  - Install via: `winget install BrechtSanders.WinLibs.POSIX.UCRT`
  - Or the model auto-falls back to pure Python (slower but correct)

### 1. Create virtual environment

```bash
python -m venv .venv
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Run the model

```bash
python hierarchical_bayesian_panel.py
```

Expected runtime: **15–30 minutes** (3,000 ZIPs × 12 years × 2 MCMC chains).  
All outputs are saved to `results/`.

---

## Predict Specific ZIP (2023+)

After running the next-year model script once, you can reuse saved artifacts and predict for a specific ZIP without retraining.

### 1. Generate inference artifacts

```bash
python hierarchical_bayesian_panel_next_year.py
```

This creates:
- `results/trace.nc`
- `results/model_metadata.npz`
- `results/latest_zip_features.csv`
- `results/training_sampled_zips.csv`

### 2. Predict one ZIP from CLI

```bash
python zip_income_predictor.py --zip 10001 --year 2023
```

You can also run it without `--zip` and it will prompt for input.

To enforce that the ZIP must be inside the 3,000 ZIP training subsample:

```bash
python zip_income_predictor.py --zip 10001 --year 2023 --strict-training-zip
```

### 3. Run Streamlit app (ZIP input UI)

```bash
streamlit run streamlit_app.py
```

In the app, enter ZIP and forecast year (for example 2023), then click **Predict**.
Use the **Strict training ZIP scope** toggle in the sidebar to allow only training-subsample ZIPs.

---

## Load Pre-computed Results (skip re-running MCMC)

If `results/trace.nc` is present, load the full posterior directly:

```python
import arviz as az
import pandas as pd

# Load posterior
trace = az.from_netcdf("results/trace.nc")

# Beta coefficients
beta_df = az.summary(trace, var_names=["beta"], round_to=4)
beta_df.index = [
    "share_wages", "share_business", "share_capital_gains",
    "share_interest", "share_dividends", "share_unemployment", "share_social_security"
]
print(beta_df[["mean", "sd", "hdi_3%", "hdi_97%"]])

# State random effects
state_df = az.summary(trace, var_names=["u_state"], round_to=4)
print(state_df.sort_values("mean", ascending=False).head(10))

# Year effects
year_df = az.summary(trace, var_names=["gamma_year"], round_to=4)
print(year_df)
```

---

## Dataset

**Source:** IRS Statistics of Income (SOI), ZIP Code Data  
**File:** `zip_income_panel_structural_2011_2022.csv`  
**Rows:** 334,071 · **ZIPs:** 30,147 · **States:** 51 · **Years:** 2011–2022

| Column | Description |
|--------|-------------|
| `ZIPCODE` | 5-digit ZIP code |
| `STATE` | 2-letter state abbreviation |
| `YEAR` | Tax year (2011–2022) |
| `total_agi` | Total Adjusted Gross Income ($) |
| `total_wages` | Wages and salaries ($) |
| `total_business` | Business income ($) |
| `total_capital_gains` | Net capital gains ($) |
| `total_interest` | Taxable interest ($) |
| `total_dividends` | Ordinary dividends ($) |
| `total_unemployment` | Unemployment compensation ($) |
| `total_social_security` | Social security benefits ($) |
| `AVG_INCOME` | Average income per return ($) — **model target** |
| `MEDIAN_BAND` | Ordinal median income band (1–6) |

---

## Reproducibility

Results are **fully reproducible** with `RANDOM_SEED = 42` hardcoded in the script.  
Running on the same CSV will produce numerically identical results.

To reproduce without re-running MCMC, share `results/trace.nc` + the CSV.

---

## References

- **Stukalenko, E.A. (2005).** *Estimation Methods of Inequality of the Population Incomes.* KURUS'2005. — Lognormal income distribution; Gini/decile decomposition.
- **Kalivoshko, O. et al. (2020).** *Assessment of Factors Influencing the Volume of Personal Income Tax Revenues.* PIC S&T'2020. — Real vs nominal income; year effects absorbing inflation.
- **Sherri, M. et al. (2021).** *A Comparison of Multiple Markov Chains Algorithms for Bayesian Updating.* ICECET 2021. — Multi-chain MCMC strategy; DE-MC vs Pop-MCMC; target_accept guidance.
- **Zheng, K. et al. (2009).** *Access Model of Web Users Based on Multi-chains Hidden Markov Models.* ICICSE 2009. — Multiple independent chains reduce autocorrelation.

---

## Configuration

Key parameters in `hierarchical_bayesian_panel.py`:

```python
N_SAMPLE_ZIPS = 3000    # ZIPs in subsample (increase for richer posteriors)
MCMC_DRAWS    = 2000    # Posterior draws per chain
MCMC_TUNE     = 1500    # Warm-up steps
MCMC_CHAINS   = 2       # Independent chains
TARGET_ACCEPT = 0.95    # NUTS acceptance rate
RANDOM_SEED   = 42      # Reproducibility seed
```
