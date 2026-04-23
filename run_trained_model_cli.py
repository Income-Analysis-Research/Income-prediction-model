#!/usr/bin/env python3
"""
CLI inference runner for the trained Bayesian regional income model.

Usage:
    python run_trained_model_cli.py

Optional arguments:
    --zip 10001 --year 2020      Run a single non-interactive query
    --export                      Export last result JSON automatically (single query mode)
    --fallback-nearest-year       Allow nearest-year fallback only with explicit enable

This script is inference-only:
- Loads saved posterior from results/trace.nc
- Does NOT retrain
- Computes predictions from posterior draws
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import arviz as az
import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "zip_income_panel_structural_2011_2022.csv"
RESULTS_DIR = BASE / "results"
TRACE_PATH = RESULTS_DIR / "trace.nc"
AUDIT_LOG_PATH = RESULTS_DIR / "cli_audit_log.jsonl"

RANDOM_SEED = 42
N_SAMPLE_ZIPS = 3000

SHARE_COLS_RAW = [
    "total_wages",
    "total_business",
    "total_capital_gains",
    "total_interest",
    "total_dividends",
    "total_unemployment",
    "total_social_security",
]
SHARE_NAMES = [c.replace("total_", "") for c in SHARE_COLS_RAW]
BASE_FEATURES = [f"share_{n}" for n in SHARE_NAMES]

REQUIRED_DATA_COLS = {
    "ZIPCODE",
    "STATE",
    "YEAR",
    "total_agi",
    "AVG_INCOME",
    *SHARE_COLS_RAW,
}


@dataclass
class LoadedArtifacts:
    idata: az.InferenceData
    df_engineered: pd.DataFrame
    sub_train: pd.DataFrame
    feature_names: List[str]
    x_mean: np.ndarray
    x_std: np.ndarray
    state_lookup: Dict[str, str]
    available_years: List[int]
    n_draws_total: int


def print_presentation_banner(artifacts: LoadedArtifacts, fallback_nearest_year: bool) -> None:
    print("\n" + "=" * 74)
    print("BAYESIAN REGIONAL INCOME MODEL - PRESENTATION CONSOLE")
    print("=" * 74)
    print(f"Inference artifact : {TRACE_PATH}")
    print(f"Dataset            : {DATA_PATH.name}")
    print(f"Posterior draws    : {artifacts.n_draws_total}")
    print(f"Trained features   : {len(artifacts.feature_names)} -> {artifacts.feature_names}")
    print(
        f"Year range         : {artifacts.available_years[0]}-{artifacts.available_years[-1]}"
    )
    print(f"Fallback mode      : {'ENABLED' if fallback_nearest_year else 'DISABLED'}")
    print("No retraining is performed in this script.")


def append_audit_log(event: Dict[str, object]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        **event,
    }
    with AUDIT_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")
    return AUDIT_LOG_PATH


def ensure_required_files() -> None:
    required = [DATA_PATH, TRACE_PATH]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required files:\n- " + "\n- ".join(missing)
        )


def validate_dataset_schema(df: pd.DataFrame) -> None:
    missing_cols = REQUIRED_DATA_COLS.difference(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {sorted(missing_cols)}")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for raw, feat in zip(SHARE_COLS_RAW, BASE_FEATURES):
        df[feat] = df[raw] / df["total_agi"]

    for feat in BASE_FEATURES:
        df[feat] = df[feat].clip(0.0, 1.0)

    df = df[df["total_agi"] > 0].copy()
    df = df[df["AVG_INCOME"] > 0].copy()
    df.dropna(subset=BASE_FEATURES + ["AVG_INCOME"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df["log_income"] = np.log1p(df["AVG_INCOME"])

    # ICS feature
    ics_order = df.sort_values(["ZIPCODE", "YEAR"])
    ics_diffs = (
        ics_order.groupby("ZIPCODE")[BASE_FEATURES]
        .diff()
        .abs()
        .sum(axis=1)
    )
    df["income_stability_index"] = ics_diffs
    ics_med = df["income_stability_index"].median()
    df["income_stability_index"] = df["income_stability_index"].fillna(ics_med)
    df["income_stability_index"] = df["income_stability_index"].clip(0.0, 2.0)

    # Entropy diversity feature
    div_vals = df[BASE_FEATURES].values.clip(1e-10, None)
    div_sums = div_vals.sum(axis=1, keepdims=True)
    div_sums[div_sums < 1e-10] = 1.0
    probs = div_vals / div_sums
    entropy = -(probs * np.log(probs + 1e-10)).sum(axis=1)
    df["income_diversity_index"] = entropy
    df["income_diversity_index"] = df["income_diversity_index"].clip(0.0, np.log(7) + 0.05)

    # Shock response feature
    inc_2019 = (
        df[df["YEAR"] == 2019]
        .set_index("ZIPCODE")["AVG_INCOME"]
        .rename("inc_2019")
    )
    inc_2020 = (
        df[df["YEAR"] == 2020]
        .set_index("ZIPCODE")["AVG_INCOME"]
        .rename("inc_2020")
    )
    shock_zip = (inc_2020 - inc_2019) / (inc_2019 + 1.0)
    df["shock_response_index"] = df["ZIPCODE"].map(shock_zip)
    shock_med = df["shock_response_index"].median()
    df["shock_response_index"] = df["shock_response_index"].fillna(shock_med)
    df["shock_response_index"] = df["shock_response_index"].clip(-1.0, 1.0)

    df.dropna(subset=["income_stability_index", "income_diversity_index", "shock_response_index"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def build_training_subset(df: pd.DataFrame) -> pd.DataFrame:
    np.random.seed(RANDOM_SEED)

    years_per_zip = df.groupby("ZIPCODE")["YEAR"].nunique()
    expected_panel_years = df["YEAR"].nunique()
    full_zips = years_per_zip[years_per_zip == expected_panel_years].index.values

    df_full = df[df["ZIPCODE"].isin(full_zips)].copy()
    state_zip_counts = (
        df_full.drop_duplicates("ZIPCODE")
        .groupby("STATE")["ZIPCODE"]
        .count()
    )
    total_full = len(df_full["ZIPCODE"].unique())

    sampled_zips: List[str] = []
    for state, count in state_zip_counts.items():
        n_from_state = max(1, round(N_SAMPLE_ZIPS * count / total_full))
        state_zips = df_full[df_full["STATE"] == state]["ZIPCODE"].unique()
        chosen = np.random.choice(
            state_zips,
            size=min(n_from_state, len(state_zips)),
            replace=False,
        )
        sampled_zips.extend(chosen.tolist())

    sampled_zips = list(set(sampled_zips))
    if len(sampled_zips) > N_SAMPLE_ZIPS:
        sampled_zips = np.random.choice(sampled_zips, N_SAMPLE_ZIPS, replace=False).tolist()

    sub = df[df["ZIPCODE"].isin(sampled_zips)].copy()
    sub.reset_index(drop=True, inplace=True)
    return sub


def load_artifacts() -> LoadedArtifacts:
    ensure_required_files()

    idata = az.from_netcdf(TRACE_PATH)
    if not hasattr(idata, "posterior"):
        raise ValueError("trace.nc does not contain posterior group")

    posterior = idata.posterior
    required_vars = {"alpha", "beta", "u_state", "gamma_year"}
    missing_vars = required_vars.difference(set(posterior.data_vars))
    if missing_vars:
        raise ValueError(f"trace.nc missing required posterior variables: {sorted(missing_vars)}")

    required_coords = {"features", "state", "year"}
    missing_coords = required_coords.difference(set(posterior.coords))
    if missing_coords:
        raise ValueError(f"trace.nc missing required coords: {sorted(missing_coords)}")

    feature_names = [str(x) for x in posterior.coords["features"].values.tolist()]

    df_raw = pd.read_csv(DATA_PATH)
    validate_dataset_schema(df_raw)
    df = engineer_features(df_raw)

    # Verify feature availability against trained trace feature coordinates.
    absent = [f for f in feature_names if f not in df.columns]
    if absent:
        raise ValueError(
            "Trace expects features not present after feature engineering: "
            f"{absent}"
        )

    sub = build_training_subset(df)

    x_raw = sub[feature_names].values.astype(np.float64)
    x_mean = x_raw.mean(axis=0)
    x_std = x_raw.std(axis=0)
    x_std[x_std == 0] = 1.0

    zip_state = (
        df[["ZIPCODE", "STATE"]]
        .drop_duplicates(subset=["ZIPCODE"])
        .set_index("ZIPCODE")["STATE"]
        .to_dict()
    )

    years_sorted = sorted(int(y) for y in df["YEAR"].dropna().unique().tolist())

    n_draws_total = int(posterior.sizes["chain"] * posterior.sizes["draw"])

    return LoadedArtifacts(
        idata=idata,
        df_engineered=df,
        sub_train=sub,
        feature_names=feature_names,
        x_mean=x_mean,
        x_std=x_std,
        state_lookup=zip_state,
        available_years=years_sorted,
        n_draws_total=n_draws_total,
    )


def get_query_row(df: pd.DataFrame, zip_code: str, year: int) -> pd.Series:
    rows = df[(df["ZIPCODE"].astype(str) == zip_code) & (df["YEAR"] == year)]
    if rows.empty:
        raise LookupError(f"No exact row found for ZIP={zip_code}, YEAR={year}")
    return rows.iloc[0]


def choose_nearest_year(df: pd.DataFrame, zip_code: str, year: int) -> Optional[int]:
    zip_rows = df[df["ZIPCODE"].astype(str) == zip_code]
    if zip_rows.empty:
        return None
    years = sorted(zip_rows["YEAR"].unique().tolist())
    nearest = min(years, key=lambda y: abs(y - year))
    return int(nearest)


def standardize_features(row: pd.Series, feature_names: List[str], x_mean: np.ndarray, x_std: np.ndarray) -> np.ndarray:
    x = row[feature_names].values.astype(np.float64)
    return (x - x_mean) / x_std


def posterior_predict(
    artifacts: LoadedArtifacts,
    zip_code: str,
    year: int,
    fallback_nearest_year: bool,
) -> Dict[str, object]:
    row_year = year
    try:
        row = get_query_row(artifacts.df_engineered, zip_code, row_year)
    except LookupError:
        if not fallback_nearest_year:
            raise
        nearest = choose_nearest_year(artifacts.df_engineered, zip_code, year)
        if nearest is None:
            raise LookupError(f"ZIP={zip_code} is not available in engineered dataset")
        row_year = nearest
        row = get_query_row(artifacts.df_engineered, zip_code, row_year)

    state_code = str(row["STATE"])
    year_label = str(int(row_year))

    posterior = artifacts.idata.posterior
    state_labels = set(str(x) for x in posterior.coords["state"].values.tolist())
    year_labels = set(str(x) for x in posterior.coords["year"].values.tolist())

    if state_code not in state_labels:
        raise ValueError(
            f"State '{state_code}' for ZIP {zip_code} not present in trace state coord"
        )
    if year_label not in year_labels:
        raise ValueError(
            f"Year '{year_label}' not present in trace year coord"
        )

    x_std = standardize_features(row, artifacts.feature_names, artifacts.x_mean, artifacts.x_std)

    alpha_draws = posterior["alpha"].stack(sample=("chain", "draw")).values.astype(np.float64)
    beta_draws = (
        posterior["beta"]
        .stack(sample=("chain", "draw"))
        .transpose("sample", "features")
        .values.astype(np.float64)
    )
    u_state_draws = (
        posterior["u_state"]
        .sel(state=state_code)
        .stack(sample=("chain", "draw"))
        .values.astype(np.float64)
    )
    gamma_year_draws = (
        posterior["gamma_year"]
        .sel(year=year_label)
        .stack(sample=("chain", "draw"))
        .values.astype(np.float64)
    )

    beta_contrib_draws = np.einsum("sf,f->s", beta_draws, x_std)
    mu_draws = alpha_draws + beta_contrib_draws + u_state_draws + gamma_year_draws
    income_draws = np.expm1(mu_draws)

    mu_mean = float(np.mean(mu_draws))
    mu_ci3 = float(np.quantile(mu_draws, 0.03))
    mu_ci97 = float(np.quantile(mu_draws, 0.97))

    income_mean = float(np.mean(income_draws))
    income_median = float(np.median(income_draws))
    income_ci3 = float(np.quantile(income_draws, 0.03))
    income_ci97 = float(np.quantile(income_draws, 0.97))

    feature_vector = {f: float(row[f]) for f in artifacts.feature_names}

    result = {
        "query": {
            "zip": zip_code,
            "year_requested": int(year),
            "year_used": int(row_year),
            "state": state_code,
        },
        "predicted": {
            "log_income_mean": mu_mean,
            "log_income_hdi_94": [mu_ci3, mu_ci97],
            "income_mean": income_mean,
            "income_median": income_median,
            "income_hdi_94": [income_ci3, income_ci97],
        },
        "decomposition_posterior_mean": {
            "alpha": float(np.mean(alpha_draws)),
            "beta_dot_x": float(np.mean(beta_contrib_draws)),
            "state_effect": float(np.mean(u_state_draws)),
            "year_effect": float(np.mean(gamma_year_draws)),
        },
        "feature_vector_used": feature_vector,
        "metadata": {
            "inference_source": str(TRACE_PATH),
            "dataset_source": str(DATA_PATH),
            "posterior_draws_used": int(len(mu_draws)),
            "feature_engineering_mode": "Recomputed to mirror training pipeline",
            "no_retraining_performed": True,
        },
    }
    return result


def format_currency(value: float) -> str:
    return f"${value:,.2f}"


def print_report(result: Dict[str, object]) -> None:
    q = result["query"]
    p = result["predicted"]
    d = result["decomposition_posterior_mean"]
    m = result["metadata"]

    print("\n" + "=" * 74)
    print(f"Prediction Report  |  ZIP={q['zip']}  YEAR={q['year_requested']} (used={q['year_used']})")
    print("=" * 74)

    print("\nPrediction (income scale):")
    print(f"  Mean          : {format_currency(p['income_mean'])}")
    print(f"  Median        : {format_currency(p['income_median'])}")
    print(
        "  94% Cred Int  : "
        f"[{format_currency(p['income_hdi_94'][0])}, {format_currency(p['income_hdi_94'][1])}]"
    )

    print("\nPrediction (log scale):")
    print(f"  Mean          : {p['log_income_mean']:.4f}")
    print(f"  94% Cred Int  : [{p['log_income_hdi_94'][0]:.4f}, {p['log_income_hdi_94'][1]:.4f}]")

    print("\nDecomposition (posterior means):")
    print(f"  alpha         : {d['alpha']:.4f}")
    print(f"  beta dot X    : {d['beta_dot_x']:.4f}")
    print(f"  state effect  : {d['state_effect']:.4f}")
    print(f"  year effect   : {d['year_effect']:.4f}")

    print("\nTransparency:")
    print(f"  Inference source       : {m['inference_source']}")
    print(f"  No retraining performed: {m['no_retraining_performed']}")
    print(f"  Posterior draws used   : {m['posterior_draws_used']}")
    print(f"  Feature mode           : {m['feature_engineering_mode']}")

    print("\nPlain-language notes:")
    print("  - Mean/Median are expected ZIP-level average income estimates for the selected year.")
    print("  - 94% credible interval is Bayesian uncertainty from posterior draws.")
    print("  - Decomposition shows how intercept, features, state, and year effects combine.")


def export_json(result: Dict[str, object]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"cli_prediction_{ts}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return out_path


def normalize_zip(user_input: str) -> str:
    s = user_input.strip()
    if not s.isdigit():
        raise ValueError("ZIP must be numeric")
    if len(s) > 5:
        raise ValueError("ZIP must be <= 5 digits")
    return s.zfill(5)


def parse_year_input(user_input: str, available_years: List[int]) -> int:
    s = user_input.strip()
    if not s.isdigit():
        raise ValueError("Year must be numeric")
    year = int(s)
    if year not in set(available_years):
        raise ValueError(
            f"Year {year} not available. Valid years: {available_years[0]}-{available_years[-1]}"
        )
    return year


def run_single_query(
    artifacts: LoadedArtifacts,
    zip_code: str,
    year: int,
    fallback_nearest_year: bool,
    export: bool,
) -> int:
    result = posterior_predict(
        artifacts=artifacts,
        zip_code=zip_code,
        year=year,
        fallback_nearest_year=fallback_nearest_year,
    )
    append_audit_log(
        {
            "event": "single_query_prediction",
            "query": result["query"],
            "metadata": result["metadata"],
        }
    )
    print_report(result)
    if export:
        out = export_json(result)
        print(f"\nExported JSON -> {out}")
        append_audit_log(
            {
                "event": "single_query_export",
                "query": result["query"],
                "export_path": str(out),
            }
        )
    return 0


def print_quick_help() -> None:
    print("\nMenu:")
    print("  1) Predict ZIP-year")
    print("  2) Demo quick query (ZIP=10001, YEAR=2019)")
    print("  3) Explain method (what happens under the hood)")
    print("  4) Export last result to JSON")
    print("  5) Show feature set and years")
    print("  0) Exit")


def print_method_explainer() -> None:
    print("\nMethod explainer:")
    print("  - Input ZIP and YEAR are matched to an exact row in the IRS panel.")
    print("  - Features are engineered using the same logic as training.")
    print("  - Features are standardized with regenerated training subset statistics.")
    print("  - Posterior draws are loaded from trace.nc (alpha, beta, state, year).")
    print("  - For each draw: mu = alpha + beta.X + state + year.")
    print("  - Income draws are transformed as exp(mu)-1.")
    print("  - Report shows posterior mean/median and 94% credible interval.")
    print("  - No pm.sample() or retraining occurs in this script.")


def execute_prediction_flow(
    artifacts: LoadedArtifacts,
    fallback_nearest_year: bool,
    zip_value: Optional[str] = None,
    year_value: Optional[int] = None,
) -> Optional[Dict[str, object]]:
    if zip_value is None:
        zip_input = input("ZIP code (5-digit) > ").strip()
    else:
        zip_input = zip_value
        print(f"ZIP code (5-digit) > {zip_input}")

    if zip_input.lower() in {"q", "quit", "exit"}:
        return None

    try:
        zip_code = normalize_zip(zip_input)
    except ValueError as e:
        print(f"Input error: {e}")
        return None

    if year_value is None:
        year_input = input("Year > ").strip()
        if year_input.lower() in {"q", "quit", "exit"}:
            return None
    else:
        year_input = str(year_value)
        print(f"Year > {year_input}")

    try:
        year = parse_year_input(year_input, artifacts.available_years)
    except ValueError as e:
        print(f"Input error: {e}")
        return None

    try:
        result = posterior_predict(
            artifacts=artifacts,
            zip_code=zip_code,
            year=year,
            fallback_nearest_year=fallback_nearest_year,
        )
        print_report(result)
        append_audit_log(
            {
                "event": "interactive_prediction",
                "query": result["query"],
                "metadata": result["metadata"],
            }
        )
        return result
    except Exception as e:
        print(f"Prediction error: {e}")
        return None


def run_interactive(artifacts: LoadedArtifacts, fallback_nearest_year: bool) -> int:
    print_presentation_banner(artifacts, fallback_nearest_year)
    print("\nSingle-file presentation mode is active.")
    print("All queries are audit-logged to results/cli_audit_log.jsonl.")
    print_quick_help()

    last_result: Optional[Dict[str, object]] = None

    while True:
        cmd = input("\nSelect option [1/2/3/4/5/0] > ").strip().lower()
        if cmd in {"0", "q", "quit", "exit"}:
            print("Exiting.")
            return 0
        if cmd == "1":
            result = execute_prediction_flow(artifacts, fallback_nearest_year)
            if result is not None:
                last_result = result
            continue
        if cmd == "2":
            result = execute_prediction_flow(
                artifacts,
                fallback_nearest_year,
                zip_value="10001",
                year_value=2019,
            )
            if result is not None:
                last_result = result
            continue
        if cmd == "3":
            print_method_explainer()
            continue
        if cmd == "4":
            if last_result is None:
                print("No previous result available. Run a prediction first.")
            else:
                out = export_json(last_result)
                print(f"Exported JSON -> {out}")
                append_audit_log(
                    {
                        "event": "interactive_export",
                        "query": last_result["query"],
                        "export_path": str(out),
                    }
                )
            continue
        if cmd == "5":
            print(f"Feature set ({len(artifacts.feature_names)}): {artifacts.feature_names}")
            print(f"Available years: {artifacts.available_years}")
            continue
        print("Unknown option. Choose 1, 2, 3, 4, 5, or 0.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run trained Bayesian model inference via terminal.")
    parser.add_argument("--zip", dest="zip_code", type=str, help="ZIP code (e.g., 10001)")
    parser.add_argument("--year", dest="year", type=int, help="Year (must exist in dataset)")
    parser.add_argument(
        "--fallback-nearest-year",
        action="store_true",
        help="Allow nearest-year fallback when exact ZIP-year row is missing",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export result JSON (single-query mode only)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("Loading artifacts...")
    artifacts = load_artifacts()
    print(f"Loaded trace with {artifacts.n_draws_total} posterior draws")
    print(f"Trace feature set ({len(artifacts.feature_names)}): {artifacts.feature_names}")

    if args.zip_code is not None or args.year is not None:
        if args.zip_code is None or args.year is None:
            raise ValueError("Both --zip and --year are required in single-query mode")
        zip_code = normalize_zip(args.zip_code)
        year = parse_year_input(str(args.year), artifacts.available_years)
        return run_single_query(
            artifacts=artifacts,
            zip_code=zip_code,
            year=year,
            fallback_nearest_year=args.fallback_nearest_year,
            export=args.export,
        )

    return run_interactive(artifacts=artifacts, fallback_nearest_year=args.fallback_nearest_year)


if __name__ == "__main__":
    raise SystemExit(main())
