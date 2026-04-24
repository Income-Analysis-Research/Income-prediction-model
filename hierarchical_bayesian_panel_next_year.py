"""
===================================================================
Hierarchical Bayesian Panel Regression — Regional Income Model
===================================================================

Model (from paper / image):
    Income_{z,t} = β · X_{z,t}  +  u_state  +  γ_t  +  ε_{z,t}

Where:
  z        = ZIP code
  t        = Year (2011–2022)
  X_{z,t}  = Income composition SHARES of AGI:
               • wages share
               • business income share
               • capital gains share
               • interest share
               • dividends share
               • unemployment compensation share
               • social security share
  u_state  = Hierarchical state-level random intercept  ~ N(0, σ_state²)
  γ_t      = Year fixed effect                          ~ N(0, σ_year²)
  ε_{z,t}  ~ N(0, σ_obs²)

Response:  log(1 + AVG_INCOME)   [log-normal, per Stukalenko 2005]

Theory grounding:
  • Lognormal income distribution  → Stukalenko (2005)
  • Inflation / time controls via γ_t → Kalivoshko et al. (2020)
  • MCMC convergence via multi-chain NUTS → Sherri et al. (2021)
  • Multiple chains reduce autocorrelation → Zheng et al. (2009)

MCMC Strategy (from Sherri 2021):
  • NUTS sampler (PyMC default) — equivalent to population-based MCMC
  • 2 chains, target_accept = 0.95, 2000 draws, 1500 tune
  • g++ C++ backend for fast compiled ops (pytensor cxx flag)

Dataset: zip_income_panel_structural_2011_2022.csv
  334,071 rows · 30,147 ZIPs · 51 states · 12 years
===================================================================
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── g++ backend for PyTensor (4–10x faster C-compiled ops) ──────────────────
_GPP = (
    "C:/Users/Vibhor/AppData/Local/Microsoft/WinGet/Packages/"
    "BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe/"
    "mingw64/bin/g++.exe"
)
if os.path.exists(_GPP.replace("/", "\\")):
    os.environ["PYTENSOR_FLAGS"] = f"cxx={_GPP}"
    _gpp_bin = os.path.dirname(_GPP.replace("/", "\\"))
    os.environ["PATH"] = _gpp_bin + os.pathsep + os.environ.get("PATH", "")

import pymc as pm
import pytensor
import arviz as az

warnings.filterwarnings("ignore")

# ────────────────────────────────────────────────────────────────────────────
# 0.  Configuration
# ────────────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE, "zip_income_panel_structural_2011_2022.csv")
OUTPUT_DIR  = os.path.join(BASE, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_METADATA_PATH = os.path.join(OUTPUT_DIR, "model_metadata.npz")
LATEST_ZIP_FEATURES_PATH = os.path.join(OUTPUT_DIR, "latest_zip_features.csv")
TRAINED_ZIPS_PATH = os.path.join(OUTPUT_DIR, "training_sampled_zips.csv")

# Sampling — tuned per Sherri (2021): multi-chain, high target_accept
MCMC_DRAWS    = 3500
MCMC_TUNE     = 1500
MCMC_CHAINS   = 2          # 2 independent chains (Sherri / Zheng guidance)
MCMC_CORES    = 1          # safe sequential on Windows
TARGET_ACCEPT = 0.95       # higher = fewer divergences
RANDOM_SEED   = 42

# Subsample — use only ZIPs with the full 12-year panel
N_SAMPLE_ZIPS = 3000

# Next-year forecast settings
FORECAST_DRAWS = 500
FORECAST_CHUNK_SIZE = 1200
FORECAST_YEAR_EFFECT_MODE = "sample_sigma"  # zero | last_year | sample_sigma

SHARE_COLS_RAW = [
    "total_wages",
    "total_business",
    "total_capital_gains",
    "total_interest",
    "total_dividends",
    "total_unemployment",
    "total_social_security",
]
SHARE_NAMES  = [c.replace("total_", "") for c in SHARE_COLS_RAW]
FEATURE_NAMES = [f"share_{n}" for n in SHARE_NAMES]

BANNER = "=" * 70

# ────────────────────────────────────────────────────────────────────────────
# 1.  Load & validate
# ────────────────────────────────────────────────────────────────────────────
print(BANNER)
print("  HIERARCHICAL BAYESIAN PANEL REGRESSION — REGIONAL INCOME")
print(f"  Response : log(1 + AVG_INCOME)")
print(f"  Model    : Income_{{z,t}} = beta·X_{{z,t}} + u_state + gamma_t + eps")
print(f"  Backend  : {pytensor.config.cxx or 'Python (no g++)'}")
print(BANNER)

print("\n[1/7] Loading dataset ...")
df = pd.read_csv(DATA_PATH)
print(f"  Raw shape : {df.shape[0]:,} rows x {df.shape[1]} cols")
print(f"  ZIPs      : {df['ZIPCODE'].nunique():,}")
print(f"  States    : {df['STATE'].nunique()}")
print(f"  Years     : {df['YEAR'].min()} - {df['YEAR'].max()}")

# ────────────────────────────────────────────────────────────────────────────
# 2.  Feature engineering — income composition shares
#     (Stukalenko 2005: per-capita money income decomposition)
# ────────────────────────────────────────────────────────────────────────────
print("\n[2/7] Engineering income composition shares (X) ...")

for raw, feat in zip(SHARE_COLS_RAW, FEATURE_NAMES):
    df[feat] = df[raw] / df["total_agi"]

# Clip to [0, 1] — values outside range are data artefacts
for feat in FEATURE_NAMES:
    df[feat] = df[feat].clip(0.0, 1.0)

# Drop rows where AGI <= 0 or income is missing/negative
df = df[df["total_agi"] > 0].copy()
df = df[df["AVG_INCOME"] > 0].copy()
df.dropna(subset=FEATURE_NAMES + ["AVG_INCOME"], inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"  After cleaning : {df.shape[0]:,} rows")

# Log-transform response (Stukalenko: lognormal income distribution)
df["log_income"] = np.log1p(df["AVG_INCOME"])

# ────────────────────────────────────────────────────────────────────────────
# 2b.  NEW ECONOMIC FEATURES
#      These three features extend the composition-share predictors with
#      structural dynamics that differentiate regional income resilience.
# ────────────────────────────────────────────────────────────────────────────
print("\n  [2b/7] Engineering new economic features ...")

# ── 1. Income Composition Stability Index (ICS) ─────────────────────────────
#   ICS_{z,t}  =  Σ_k | share_{k,z,t} − share_{k,z,t−1} |
#   Measures year-over-year volatility in income structure per ZIP.
#   High ICS → rapidly shifting income mix (e.g., boom/bust capital gains).
#   Low  ICS → stable, predictable income base (wage-dominant ZIPs).
#   Economic interpretation: ZIP structural stability (Stukalenko 2005 §3)
_ics_order   = df.sort_values(["ZIPCODE", "YEAR"])
_share_feat_list = [f"share_{n}" for n in SHARE_NAMES]   # original 7 share cols
_ics_diffs = (
    _ics_order.groupby("ZIPCODE")[_share_feat_list]
    .diff()          # NaN for the first year of each ZIP (no prior year)
    .abs()
    .sum(axis=1)
)
df["income_stability_index"] = _ics_diffs
# First observation per ZIP has no prior year → fill with dataset median
_ics_med = df["income_stability_index"].median()
df["income_stability_index"] = df["income_stability_index"].fillna(_ics_med)
df["income_stability_index"] = df["income_stability_index"].clip(0.0, 2.0)
print(f"    income_stability_index   mean={df['income_stability_index'].mean():.4f}  "
      f"std={df['income_stability_index'].std():.4f}  "
      f"[0=perfectly stable, 2=high volatility]")

# ── 2. Income Diversity Index (Shannon Entropy) ──────────────────────────────
#   H_{z,t} = − Σ_k  p_k · log(p_k)    p_k = share_k / Σ shares
#   Measures diversification of income sources at the ZIP level.
#   High H → balanced economy (wages + capital + business + SS + …)
#   Low  H → concentration in one source (dependency risk)
#   Max possible with 7 sources: log(7) ≈ 1.946
_div_vals = df[_share_feat_list].values.clip(1e-10, None)
_div_sums = _div_vals.sum(axis=1, keepdims=True)
_div_sums[_div_sums < 1e-10] = 1.0      # guard against all-zero rows
_p        = _div_vals / _div_sums        # row-normalised to probability simplex
_entropy  = -(_p * np.log(_p + 1e-10)).sum(axis=1)
df["income_diversity_index"] = _entropy
df["income_diversity_index"] = df["income_diversity_index"].clip(0.0, np.log(7) + 0.05)
print(f"    income_diversity_index   mean={df['income_diversity_index'].mean():.4f}  "
      f"std={df['income_diversity_index'].std():.4f}  "
      f"[max=log(7)={np.log(7):.3f}; high=diversified]")

# ── 3. Shock Response Index (COVID-19: 2020 vs 2019) ─────────────────────────
#   shock_{z} = ( AVG_INCOME_{z,2020} − AVG_INCOME_{z,2019} )
#                / ( AVG_INCOME_{z,2019} + 1 )
#   Captures structural resilience/vulnerability to macro shocks.
#   Positive → income rose during COVID (wealth-heavy / capital-gains ZIPs).
#   Negative → income fell (hospitality, service, trade-exposed ZIPs).
#   Applied as a time-invariant ZIP-level property (computed once, same value
#   across all years for a ZIP — encodes structural economic type).
_inc_2019 = (df[df["YEAR"] == 2019]
               .set_index("ZIPCODE")["AVG_INCOME"]
               .rename("inc_2019"))
_inc_2020 = (df[df["YEAR"] == 2020]
               .set_index("ZIPCODE")["AVG_INCOME"]
               .rename("inc_2020"))
_shock_zip = (_inc_2020 - _inc_2019) / (_inc_2019 + 1.0)   # normalised ratio
df["shock_response_index"] = df["ZIPCODE"].map(_shock_zip)
_shock_med = df["shock_response_index"].median()
df["shock_response_index"] = df["shock_response_index"].fillna(_shock_med)
df["shock_response_index"] = df["shock_response_index"].clip(-1.0, 1.0)
print(f"    shock_response_index     mean={df['shock_response_index'].mean():.4f}  "
      f"std={df['shock_response_index'].std():.4f}  "
      f"[−1=hardest hit, +1=largest COVID income gain]")

# Extend FEATURE_NAMES to include the three new features.
# All downstream code (X matrix, PyMC coords, beta_df) picks these up automatically.
_NEW_FEATURE_NAMES = [
    "income_stability_index",
    "income_diversity_index",
    "shock_response_index",
]
FEATURE_NAMES = FEATURE_NAMES + _NEW_FEATURE_NAMES   # 7 → 10 features

# Drop rows where any new feature is still NaN (safety guard)
df.dropna(subset=_NEW_FEATURE_NAMES, inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"    Rows after new-feature NaN drop : {df.shape[0]:,}")

# Quick correlation report
print("\n  Income share/feature -> log(income) correlations:")
for feat in FEATURE_NAMES:
    r = df[feat].corr(df["log_income"])
    bar = "#" * int(abs(r) * 25)
    sign = "+" if r >= 0 else "-"
    print(f"    {feat:35s}  r = {sign}{abs(r):.4f}  {bar}")

# ────────────────────────────────────────────────────────────────────────────
# 3.  Subsample: balanced across states, full-panel ZIPs only
# ────────────────────────────────────────────────────────────────────────────
print(f"\n[3/7] Subsampling {N_SAMPLE_ZIPS:,} full-panel ZIPs (12 years each) ...")

np.random.seed(RANDOM_SEED)

zip_counts = df.groupby("ZIPCODE")["YEAR"].count()
full_zips  = zip_counts[zip_counts == 12].index.values
print(f"  Full-panel ZIPs available : {len(full_zips):,}")

df_full = df[df["ZIPCODE"].isin(full_zips)].copy()
state_zip_counts = df_full.drop_duplicates("ZIPCODE").groupby("STATE")["ZIPCODE"].count()
total_full = len(df_full["ZIPCODE"].unique())

sampled_zips = []
for state, count in state_zip_counts.items():
    n_from_state = max(1, round(N_SAMPLE_ZIPS * count / total_full))
    state_zips = df_full[df_full["STATE"] == state]["ZIPCODE"].unique()
    chosen = np.random.choice(
        state_zips,
        size=min(n_from_state, len(state_zips)),
        replace=False
    )
    sampled_zips.extend(chosen.tolist())

sampled_zips = list(set(sampled_zips))
if len(sampled_zips) > N_SAMPLE_ZIPS:
    sampled_zips = np.random.choice(sampled_zips, N_SAMPLE_ZIPS, replace=False).tolist()

pd.DataFrame({"ZIPCODE": sampled_zips}).to_csv(TRAINED_ZIPS_PATH, index=False)

sub = df[df["ZIPCODE"].isin(sampled_zips)].copy()
sub.reset_index(drop=True, inplace=True)
print(f"  Subsample   : {sub.shape[0]:,} rows")
print(f"  ZIPs        : {sub['ZIPCODE'].nunique():,}")
print(f"  States      : {sub['STATE'].nunique()}")
print("  ZIP scope   : results/training_sampled_zips.csv")

# ────────────────────────────────────────────────────────────────────────────
# 4.  Encode categorical indices
# ────────────────────────────────────────────────────────────────────────────
print("\n[4/7] Encoding state/year indices ...")

state_labels, state_idx = np.unique(sub["STATE"].values, return_inverse=True)
n_states = len(state_labels)
sub["state_idx"] = state_idx

year_labels, year_idx = np.unique(sub["YEAR"].values, return_inverse=True)
n_years = len(year_labels)
sub["year_idx"] = year_idx

print(f"  States : {n_states}  |  Years : {n_years}  ({year_labels[0]}-{year_labels[-1]})")
FORECAST_YEAR = int(year_labels.max()) + 1
print(f"  Forecast target year : {FORECAST_YEAR}")

# ────────────────────────────────────────────────────────────────────────────
# 5.  Standardise design matrix
# ────────────────────────────────────────────────────────────────────────────
print("\n[5/7] Standardising feature matrix X ...")

X_raw  = sub[FEATURE_NAMES].values.astype(np.float64)
X_mean = X_raw.mean(axis=0)
X_std  = X_raw.std(axis=0)
X_std[X_std == 0] = 1.0
X = (X_raw - X_mean) / X_std

y = sub["log_income"].values.astype(np.float64)
n_obs, n_feat = X.shape

print(f"  X : {X.shape}   y : {y.shape}")
print(f"  y range : [{y.min():.3f}, {y.max():.3f}]  mean={y.mean():.3f}")

# ────────────────────────────────────────────────────────────────────────────
# 6.  Hierarchical Bayesian Panel Model (PyMC + NUTS)
#
#   Income_{z,t} = alpha + beta·X_{z,t} + u_state[s] + gamma_year[t] + eps
#
#   Priors (weakly informative, BDA3 / Gelman):
#     alpha          ~ N(mu_y, 2)
#     beta_k         ~ N(0, 5)              k = 1..7 composition shares
#     sigma_state    ~ HalfNormal(2)        hyperprior for state effects
#     u_state_raw[s] ~ N(0, 1)             non-centred reparameterisation
#     u_state        = u_state_raw * sigma_state
#     sigma_year     ~ HalfNormal(2)
#     gamma_raw[t]   ~ N(0, 1)
#     gamma_year     = gamma_raw * sigma_year
#     sigma_obs      ~ HalfNormal(3)
#
#   MCMC: NUTS, 2 chains, target_accept=0.95 (Sherri 2021 recommendation)
# ────────────────────────────────────────────────────────────────────────────
print("\n[6/7] Building and sampling PyMC model ...")
print(f"  Draws={MCMC_DRAWS}  Tune={MCMC_TUNE}  Chains={MCMC_CHAINS}  "
      f"target_accept={TARGET_ACCEPT}")

sidx = sub["state_idx"].values.astype(int)
tidx = sub["year_idx"].values.astype(int)

with pm.Model(coords={
    "features": FEATURE_NAMES,
    "state":    state_labels.tolist(),
    "year":     [str(y_) for y_ in year_labels],
}) as panel_model:

    # Data containers
    X_data     = pm.Data("X_data",    X,    dims=("obs", "features"))
    y_data     = pm.Data("y_data",    y,    dims="obs")
    state_data = pm.Data("state_idx", sidx, dims="obs")
    year_data  = pm.Data("year_idx",  tidx, dims="obs")

    # Global intercept
    alpha = pm.Normal("alpha", mu=float(y.mean()), sigma=2)

    # Beta: income composition share coefficients
    beta = pm.Normal("beta", mu=0, sigma=5, dims="features")

    # u_state: hierarchical state random intercepts (non-centred)
    sigma_state  = pm.HalfNormal("sigma_state", sigma=2)
    u_state_raw  = pm.Normal("u_state_raw", mu=0, sigma=1, dims="state")
    u_state      = pm.Deterministic("u_state", u_state_raw * sigma_state, dims="state")

    # gamma_t: year effects capturing inflation drift (Kalivoshko 2020)
    sigma_year   = pm.HalfNormal("sigma_year", sigma=2)
    gamma_raw    = pm.Normal("gamma_raw", mu=0, sigma=1, dims="year")
    gamma_year_  = pm.Deterministic("gamma_year", gamma_raw * sigma_year, dims="year")

    # Linear predictor: Income_{z,t} = alpha + beta·X + u_state + gamma_t
    mu = (alpha
          + pm.math.dot(X_data, beta)
          + u_state[state_data]
          + gamma_year_[year_data])

    # Likelihood
    sigma_obs  = pm.HalfNormal("sigma_obs", sigma=3)
    _likelihood = pm.Normal("income_obs", mu=mu, sigma=sigma_obs,
                             observed=y_data, dims="obs")

    # NUTS sampling (Sherri 2021: multi-chain, high target_accept)
    trace = pm.sample(
        draws         = MCMC_DRAWS,
        tune          = MCMC_TUNE,
        chains        = MCMC_CHAINS,
        cores         = MCMC_CORES,
        target_accept = TARGET_ACCEPT,
        random_seed   = RANDOM_SEED,
        return_inferencedata = True,
        progressbar   = True,
    )

    # Posterior predictive check
    print("\n  Running posterior predictive check ...")
    ppc = pm.sample_posterior_predictive(
        trace, random_seed=RANDOM_SEED, progressbar=True
    )

# ────────────────────────────────────────────────────────────────────────────
# 7.  Diagnostics & Results
# ────────────────────────────────────────────────────────────────────────────
print(f"\n[7/7] Computing diagnostics ...")
print(BANNER)
print("RESULTS")
print(BANNER)

# R-hat convergence check (Sherri 2021: R-hat < 1.05 = converged)
r_hat_vars = ["alpha", "beta", "sigma_state", "sigma_year", "sigma_obs"]
rhat = az.rhat(trace, var_names=r_hat_vars)
max_rhat = max(float(rhat[v].max()) for v in rhat.data_vars)
status = "CONVERGED" if max_rhat < 1.05 else "WARNING - may not have converged"
print(f"\n  Max R-hat : {max_rhat:.4f}  [{status}]")

# Global parameter summary
summary_all = az.summary(trace, var_names=r_hat_vars, round_to=4)
print("\n-- Global parameters --")
print(summary_all.to_string())
summary_all.to_csv(os.path.join(OUTPUT_DIR, "summary_global.csv"))

# Beta coefficients
beta_df = az.summary(trace, var_names=["beta"], round_to=4)
beta_df.index = FEATURE_NAMES
print("\n-- Beta Coefficients (income composition shares) --")
print(beta_df[["mean", "sd", "hdi_3%", "hdi_97%", "r_hat"]].to_string())
beta_df.to_csv(os.path.join(OUTPUT_DIR, "beta_coefficients.csv"))

# State random effects
state_df = az.summary(trace, var_names=["u_state"], round_to=4)
state_df.index = state_labels
state_df_sorted = state_df.sort_values("mean", ascending=False)
print("\n-- State Random Effects (top 10 highest income states) --")
print(state_df_sorted.head(10)[["mean", "sd", "hdi_3%", "hdi_97%"]].to_string())
state_df.to_csv(os.path.join(OUTPUT_DIR, "state_effects.csv"))

# Year effects
year_df = az.summary(trace, var_names=["gamma_year"], round_to=4)
year_df.index = year_labels
print("\n-- Year Effects (gamma_t) --")
print(year_df[["mean", "sd", "hdi_3%", "hdi_97%"]].to_string())
year_df.to_csv(os.path.join(OUTPUT_DIR, "year_effects.csv"))

# In-sample fit metrics
ppc_vals   = ppc.posterior_predictive["income_obs"].values
y_hat      = ppc_vals.reshape(-1, len(y)).mean(axis=0)
residuals  = y - y_hat
rmse  = np.sqrt(np.mean(residuals**2))
mae   = np.mean(np.abs(residuals))
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y - y.mean())**2)
r2    = 1.0 - ss_res / ss_tot

# Gini on residuals (Stukalenko: residual inequality measure)
def gini(x):
    x = np.sort(np.abs(x))
    n = len(x)
    cum = np.cumsum(x)
    return (2 * np.sum(np.arange(1, n+1) * x) / (n * cum[-1])) - (n + 1) / n

gini_res = gini(residuals)

print("\n-- In-sample Fit Metrics --")
print(f"  RMSE (log-income scale) : {rmse:.4f}")
print(f"  MAE  (log-income scale) : {mae:.4f}")
print(f"  R2                      : {r2:.4f}")
print(f"  Gini of residuals       : {gini_res:.4f}  (Stukalenko inequality measure)")

# Information criteria
print("\n-- Information Criteria --")
try:
    waic = az.waic(trace)
    print(f"  WAIC : {waic.elpd_waic:.2f}  (se={waic.se:.2f})")
    pd.DataFrame({"elpd_waic": [waic.elpd_waic], "se": [waic.se]}).to_csv(
        os.path.join(OUTPUT_DIR, "waic.csv"), index=False)
except Exception as e:
    print(f"  WAIC : could not compute - {e}")

try:
    loo = az.loo(trace)
    print(f"  LOO  : {loo.elpd_loo:.2f}  (se={loo.se:.2f})")
    pd.DataFrame({"elpd_loo": [loo.elpd_loo], "se": [loo.se]}).to_csv(
        os.path.join(OUTPUT_DIR, "loo.csv"), index=False)
except Exception as e:
    print(f"  LOO  : could not compute - {e}")

# Save trace
trace_path = os.path.join(OUTPUT_DIR, "trace.nc")
trace.to_netcdf(trace_path)
print(f"\n  Full MCMC trace saved -> results/trace.nc")

# ────────────────────────────────────────────────────────────────────────────
# 8.  Next-year ZIP predictions (out-of-sample)
# ────────────────────────────────────────────────────────────────────────────
print(f"\n[8/8] Forecasting ZIP-level income for {FORECAST_YEAR} ...")

# Use each ZIP's latest available feature vector as the next-year scenario.
latest_by_zip = (
    df.sort_values(["ZIPCODE", "YEAR"])
      .groupby("ZIPCODE", as_index=False)
      .tail(1)
      .copy()
)

X_fore_raw = latest_by_zip[FEATURE_NAMES].to_numpy(dtype=np.float64)
X_fore = (X_fore_raw - X_mean) / X_std

state_to_idx = {s: i for i, s in enumerate(state_labels.tolist())}
state_fore_idx = latest_by_zip["STATE"].map(state_to_idx).fillna(-1).astype(int).to_numpy()

alpha_all = trace.posterior["alpha"].values.reshape(-1)
beta_all = trace.posterior["beta"].values.reshape(-1, n_feat)
u_state_all = trace.posterior["u_state"].values.reshape(-1, n_states)
sigma_year_all = trace.posterior["sigma_year"].values.reshape(-1)
gamma_year_all = trace.posterior["gamma_year"].values.reshape(-1, n_years)

n_total_draws = alpha_all.shape[0]
n_use_draws = min(FORECAST_DRAWS, n_total_draws)
rng = np.random.default_rng(RANDOM_SEED)
draw_sel = rng.choice(n_total_draws, size=n_use_draws, replace=False)

alpha_draws = alpha_all[draw_sel]
beta_draws = beta_all[draw_sel]
u_state_draws = u_state_all[draw_sel]
sigma_year_draws = sigma_year_all[draw_sel]
gamma_year_draws = gamma_year_all[draw_sel]

if FORECAST_YEAR_EFFECT_MODE == "zero":
    gamma_future = np.zeros(n_use_draws)
elif FORECAST_YEAR_EFFECT_MODE == "last_year":
    gamma_future = gamma_year_draws[:, -1]
elif FORECAST_YEAR_EFFECT_MODE == "sample_sigma":
    gamma_future = rng.normal(0.0, sigma_year_draws)
else:
    raise ValueError(
        "Invalid FORECAST_YEAR_EFFECT_MODE. Use: zero, last_year, or sample_sigma"
    )

n_zip = len(latest_by_zip)
log_mean = np.empty(n_zip, dtype=np.float64)
log_p05 = np.empty(n_zip, dtype=np.float64)
log_p95 = np.empty(n_zip, dtype=np.float64)
income_mean = np.empty(n_zip, dtype=np.float64)
income_p05 = np.empty(n_zip, dtype=np.float64)
income_p50 = np.empty(n_zip, dtype=np.float64)
income_p95 = np.empty(n_zip, dtype=np.float64)

for start in range(0, n_zip, FORECAST_CHUNK_SIZE):
    end = min(start + FORECAST_CHUNK_SIZE, n_zip)
    X_chunk = X_fore[start:end]
    s_chunk = state_fore_idx[start:end]

    mu_draws = (
        alpha_draws[:, None]
        + (beta_draws @ X_chunk.T)
        + gamma_future[:, None]
    )

    known_state = s_chunk >= 0
    if np.any(known_state):
        mu_draws[:, known_state] += u_state_draws[:, s_chunk[known_state]]

    income_draws = np.expm1(mu_draws)

    log_mean[start:end] = mu_draws.mean(axis=0)
    log_p05[start:end] = np.quantile(mu_draws, 0.05, axis=0)
    log_p95[start:end] = np.quantile(mu_draws, 0.95, axis=0)

    income_mean[start:end] = income_draws.mean(axis=0)
    income_p05[start:end] = np.quantile(income_draws, 0.05, axis=0)
    income_p50[start:end] = np.quantile(income_draws, 0.50, axis=0)
    income_p95[start:end] = np.quantile(income_draws, 0.95, axis=0)

forecast_df = pd.DataFrame({
    "ZIPCODE": latest_by_zip["ZIPCODE"].to_numpy(),
    "STATE": latest_by_zip["STATE"].to_numpy(),
    "source_year": latest_by_zip["YEAR"].to_numpy(),
    "forecast_year": FORECAST_YEAR,
    "forecast_year_effect_mode": FORECAST_YEAR_EFFECT_MODE,
    "feature_assumption": "carry_forward_last_observation",
    "pred_log_income_mean": log_mean,
    "pred_log_income_p05": log_p05,
    "pred_log_income_p95": log_p95,
    "pred_income_mean": income_mean,
    "pred_income_p05": income_p05,
    "pred_income_p50": income_p50,
    "pred_income_p95": income_p95,
})

forecast_df.sort_values("ZIPCODE", inplace=True)
forecast_path = os.path.join(OUTPUT_DIR, "zip_next_year_income_predictions.csv")
forecast_df.to_csv(forecast_path, index=False)

# Save lightweight artifacts so predictions can be generated later without retraining.
latest_by_zip[["ZIPCODE", "STATE", "YEAR"] + FEATURE_NAMES].to_csv(
    LATEST_ZIP_FEATURES_PATH, index=False
)
np.savez(
    MODEL_METADATA_PATH,
    feature_names=np.array(FEATURE_NAMES, dtype=object),
    x_mean=X_mean,
    x_std=X_std,
    state_labels=np.array(state_labels, dtype=object),
    year_labels=np.array(year_labels, dtype=int),
)

unknown_state_count = int((state_fore_idx < 0).sum())
print(f"  ZIP forecasts generated : {len(forecast_df):,}")
print(f"  Output file             : results/zip_next_year_income_predictions.csv")
print(f"  Inference metadata      : results/model_metadata.npz")
print(f"  Latest ZIP features     : results/latest_zip_features.csv")
print(f"  Training ZIP scope      : results/training_sampled_zips.csv")
if unknown_state_count > 0:
    print(f"  Warning                 : {unknown_state_count} ZIPs had unseen states and used state effect = 0")

# ────────────────────────────────────────────────────────────────────────────
# 9.  Plots
# ────────────────────────────────────────────────────────────────────────────
print("\n-- Generating plots ...")

# 9a. Trace plot
az.plot_trace(trace, var_names=["alpha", "beta", "sigma_state",
                                 "sigma_year", "sigma_obs"],
              compact=True, figsize=(14, 10))
plt.suptitle("MCMC Trace Plot - chain convergence (Sherri 2021)", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "trace_plot.png"), dpi=150, bbox_inches="tight")
plt.close("all")
print("  -> trace_plot.png")

# 9b. Beta forest plot
fig, ax = plt.subplots(figsize=(9, 5))
means  = beta_df["mean"].values
lo     = beta_df["hdi_3%"].values
hi     = beta_df["hdi_97%"].values
# Clean display labels: strip "share_" prefix; new features already have clean names
labels = [f.replace("share_", "").replace("_index", "\n_idx").replace("_", " ")
          for f in FEATURE_NAMES]
colors = ["#d62728" if m < 0 else "#2ca02c" for m in means]
y_pos  = np.arange(len(labels))

ax.barh(y_pos, means,
        xerr=[means - lo, hi - means],
        color=colors, alpha=0.7, capsize=4, align="center")
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=11)
ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xlabel("Posterior Mean beta  (standardised)", fontsize=11)
ax.set_title(
    "Beta Coefficients: Income Composition Shares + Structural Features\n"
    "green = positive effect on income  |  red = negative  |  94% HDI bars\n"
    "[ICS=stability  DIV=diversity  SHOCK=COVID resilience index]",
    fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "beta_forest.png"), dpi=150, bbox_inches="tight")
plt.close("all")
print("  -> beta_forest.png")

# 9c. State effects bar chart
fig, ax = plt.subplots(figsize=(13, 6))
s_sorted = state_df.sort_values("mean", ascending=True)
clrs = ["#d62728" if m < 0 else "#2ca02c" for m in s_sorted["mean"]]
ax.barh(range(len(s_sorted)), s_sorted["mean"],
        xerr=[s_sorted["mean"] - s_sorted["hdi_3%"],
              s_sorted["hdi_97%"] - s_sorted["mean"]],
        color=clrs, alpha=0.7, capsize=2)
ax.set_yticks(range(len(s_sorted)))
ax.set_yticklabels(s_sorted.index, fontsize=8)
ax.axvline(0, color="black", linewidth=0.6)
ax.set_xlabel("State Random Intercept u_state  (log-income scale)")
ax.set_title("Hierarchical State Random Effects on log(AVG_INCOME)\n"
             "[Stukalenko: regional income differentiation across states]")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "state_effects.png"), dpi=150, bbox_inches="tight")
plt.close("all")
print("  -> state_effects.png")

# 9d. Year effects trend
fig, ax = plt.subplots(figsize=(9, 4))
ym = year_df["mean"].values
yl = year_df["hdi_3%"].values
yh = year_df["hdi_97%"].values
ax.plot(year_labels, ym, "o-", color="navy", linewidth=2, markersize=6)
ax.fill_between(year_labels, yl, yh, alpha=0.2, color="steelblue")
ax.set_xticks(year_labels)
ax.set_xlabel("Year")
ax.set_ylabel("gamma_t  (log-income scale)")
ax.set_title("Year Fixed Effects gamma_t  with 94% HDI\n"
             "[Kalivoshko 2020: year effects absorb inflation and macro trends]")
ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "year_effects.png"), dpi=150, bbox_inches="tight")
plt.close("all")
print("  -> year_effects.png")

# 9e. Posterior predictive check
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.scatter(y, y_hat, alpha=0.04, s=3, color="steelblue", rasterized=True)
mn, mx = y.min(), y.max()
ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect prediction")
ax.set_xlabel("Observed  log(1 + AVG_INCOME)")
ax.set_ylabel("Posterior Mean Predicted")
ax.set_title(f"Observed vs Predicted  |  R2 = {r2:.4f}")
ax.legend(fontsize=9)

ax = axes[1]
ax.hist(residuals, bins=80, color="steelblue", alpha=0.7, edgecolor="white")
ax.axvline(0, color="red", linewidth=1.2, linestyle="--")
ax.set_xlabel("Residual  (log scale)")
ax.set_ylabel("Count")
ax.set_title(f"Residual Distribution  |  Gini = {gini_res:.4f}")

plt.suptitle("Posterior Predictive Check", fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "posterior_predictive.png"),
            dpi=150, bbox_inches="tight")
plt.close("all")
print("  -> posterior_predictive.png")

# 9f. Prior vs Posterior for variance components
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, var, title in zip(
    axes,
    ["sigma_state", "sigma_year", "sigma_obs"],
    ["sigma_state (state heterogeneity)",
     "sigma_year  (year macro trend)",
     "sigma_obs   (residual noise)"]
):
    samples     = trace.posterior[var].values.flatten()
    prior_draws = np.abs(np.random.normal(0, 2, len(samples)))
    ax.hist(samples,     bins=60, alpha=0.65, color="navy",   label="Posterior", density=True)
    ax.hist(prior_draws, bins=60, alpha=0.40, color="orange", label="Prior",     density=True)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Value")
    ax.legend(fontsize=8)

plt.suptitle("Prior vs Posterior: Variance Components", y=1.02, fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "prior_vs_posterior.png"),
            dpi=150, bbox_inches="tight")
plt.close("all")
print("  -> prior_vs_posterior.png")

# ── Final summary ─────────────────────────────────────────────────────────
print(f"""
{BANNER}
DONE  —  results saved to ./results/
{BANNER}

  summary_global.csv         - alpha, sigma_state, sigma_year, sigma_obs
  beta_coefficients.csv      - beta for each feature (10 total)
                               7 income shares + ICS + diversity + shock
  state_effects.csv          - u_state for all states
  year_effects.csv           - gamma_t for 2011-2022
  waic.csv / loo.csv         - model information criteria
  trace.nc                   - full MCMC trace (NetCDF)
    zip_next_year_income_predictions.csv
                                                         - next-year prediction per ZIP (mean + 90% PI)
                                                             using carried-forward features
    model_metadata.npz         - scaler/state/year metadata for inference
    latest_zip_features.csv    - one latest feature row per ZIP for inference
    training_sampled_zips.csv  - ZIPs used in the 3,000 ZIP training subsample

  New features added:
    income_stability_index   - ICS: Σ|share_t - share_t-1|  (ZIP volatility)
    income_diversity_index   - Shannon entropy of income mix (diversification)
    shock_response_index     - (income_2020-income_2019)/(income_2019+1) (COVID)

  trace_plot.png             - chain convergence (Sherri 2021)
  beta_forest.png            - beta coefficients with 94% HDI
  state_effects.png          - state hierarchical intercepts
  year_effects.png           - year fixed effects trend
  posterior_predictive.png   - obs vs predicted + residuals
  prior_vs_posterior.png     - variance component update

  R2       : {r2:.4f}
  RMSE     : {rmse:.4f}  (log scale)
  MAE      : {mae:.4f}  (log scale)
  Max R-hat: {max_rhat:.4f}  [{status}]
""")
