"""
ZIP-level next-year income predictor from saved Bayesian model artifacts.

Usage (CLI):
  python zip_income_predictor.py --zip 10001 --year 2023

Strict mode (only ZIPs used in training subsample):
    python zip_income_predictor.py --zip 10001 --year 2023 --strict-training-zip

Usage (import):
  from zip_income_predictor import ZipIncomePredictor
  predictor = ZipIncomePredictor(results_dir="results", year_effect_mode="sample_sigma")
  pred = predictor.predict_zip("10001", forecast_year=2023, draws=500)
  print(pred)
"""

import argparse
import os
import numpy as np
import pandas as pd
import arviz as az


class ZipIncomePredictor:
    def __init__(
        self,
        results_dir="results",
        year_effect_mode="sample_sigma",
        strict_training_zip=False,
        random_seed=42,
    ):
        self.results_dir = results_dir
        self.year_effect_mode = year_effect_mode
        self.strict_training_zip = strict_training_zip
        self.random_seed = random_seed
        self._load_assets()

        if self.strict_training_zip and not self.has_trained_zip_scope:
            raise FileNotFoundError(
                "Strict training ZIP mode requires results/training_sampled_zips.csv. "
                "Re-run hierarchical_bayesian_panel_next_year.py to generate it."
            )

    @staticmethod
    def _zip_key(value):
        if pd.isna(value):
            return ""

        text = str(value).strip()
        if text.endswith(".0") and text[:-2].isdigit():
            text = text[:-2]

        if text.isdigit():
            return str(int(text))

        return text

    def _zip_variants(self, zipcode):
        raw = str(zipcode).strip()
        if not raw:
            return set()

        variants = {self._zip_key(raw)}
        if raw.isdigit():
            as_int = str(int(raw))
            variants.add(as_int)
            variants.add(raw.zfill(5))
            variants.add(as_int.zfill(5))

        return {v for v in variants if v}

    def _load_assets(self):
        trace_path = os.path.join(self.results_dir, "trace.nc")
        meta_path = os.path.join(self.results_dir, "model_metadata.npz")
        latest_path = os.path.join(self.results_dir, "latest_zip_features.csv")
        trained_zip_path = os.path.join(self.results_dir, "training_sampled_zips.csv")

        missing = [p for p in [trace_path, meta_path, latest_path] if not os.path.exists(p)]
        if missing:
            missing_fmt = "\n  - " + "\n  - ".join(missing)
            raise FileNotFoundError(
                "Missing inference artifacts. Run hierarchical_bayesian_panel_next_year.py first."
                f"\nMissing:{missing_fmt}"
            )

        self.trace = az.from_netcdf(trace_path)

        meta = np.load(meta_path, allow_pickle=True)
        self.feature_names = [str(x) for x in meta["feature_names"].tolist()]
        self.x_mean = np.asarray(meta["x_mean"], dtype=np.float64)
        self.x_std = np.asarray(meta["x_std"], dtype=np.float64)
        self.x_std[self.x_std == 0] = 1.0
        self.state_labels = [str(x) for x in meta["state_labels"].tolist()]
        self.year_labels = np.asarray(meta["year_labels"], dtype=int)
        self.state_to_idx = {s: i for i, s in enumerate(self.state_labels)}

        self.latest_zip_features = pd.read_csv(latest_path)
        self.latest_zip_features["_zip_key"] = self.latest_zip_features["ZIPCODE"].map(self._zip_key)

        self.has_trained_zip_scope = False
        self.trained_zip_keys = set()
        if os.path.exists(trained_zip_path):
            trained_df = pd.read_csv(trained_zip_path)
            zip_col = "ZIPCODE" if "ZIPCODE" in trained_df.columns else trained_df.columns[0]
            self.trained_zip_keys = set(trained_df[zip_col].map(self._zip_key).dropna().tolist())
            self.has_trained_zip_scope = len(self.trained_zip_keys) > 0

    def _get_zip_row(self, zipcode):
        variants = self._zip_variants(zipcode)
        if not variants:
            raise ValueError("ZIP code input is empty.")

        rows = self.latest_zip_features[self.latest_zip_features["_zip_key"].isin(variants)]
        if rows.empty:
            raise ValueError(f"ZIP code '{zipcode}' not found in latest_zip_features.csv")

        return rows.sort_values("YEAR").tail(1).iloc[0]

    def _sample_posterior_draws(self, draws):
        alpha_all = self.trace.posterior["alpha"].values.reshape(-1)
        beta_all = self.trace.posterior["beta"].values.reshape(-1, len(self.feature_names))
        u_state_all = self.trace.posterior["u_state"].values.reshape(-1, len(self.state_labels))
        sigma_year_all = self.trace.posterior["sigma_year"].values.reshape(-1)
        gamma_year_all = self.trace.posterior["gamma_year"].values.reshape(-1, len(self.year_labels))

        n_total = alpha_all.shape[0]
        n_use = min(int(draws), n_total)
        if n_use <= 0:
            raise ValueError("draws must be a positive integer")

        rng = np.random.default_rng(self.random_seed)
        sel = rng.choice(n_total, size=n_use, replace=False)

        return {
            "alpha": alpha_all[sel],
            "beta": beta_all[sel],
            "u_state": u_state_all[sel],
            "sigma_year": sigma_year_all[sel],
            "gamma_year": gamma_year_all[sel],
            "rng": rng,
        }

    def _future_year_effect(self, draws_dict, forecast_year):
        gamma_year = draws_dict["gamma_year"]
        sigma_year = draws_dict["sigma_year"]
        rng = draws_dict["rng"]

        if forecast_year in set(self.year_labels.tolist()):
            idx = int(np.where(self.year_labels == forecast_year)[0][0])
            return gamma_year[:, idx]

        if self.year_effect_mode == "zero":
            return np.zeros(gamma_year.shape[0])

        if self.year_effect_mode == "last_year":
            return gamma_year[:, -1]

        if self.year_effect_mode == "sample_sigma":
            return rng.normal(0.0, sigma_year)

        raise ValueError("Invalid year_effect_mode. Use: zero, last_year, sample_sigma")

    def predict_zip(self, zipcode, forecast_year=2023, draws=500):
        row = self._get_zip_row(zipcode)
        row_zip_key = self._zip_key(row["ZIPCODE"])

        in_training_sample = None
        if self.has_trained_zip_scope:
            in_training_sample = row_zip_key in self.trained_zip_keys

        if self.strict_training_zip and not in_training_sample:
            raise ValueError(
                f"ZIP code '{zipcode}' is outside the training subsample scope. "
                "Disable strict mode to allow out-of-sample ZIP predictions."
            )

        x_raw = row[self.feature_names].to_numpy(dtype=np.float64)
        x = (x_raw - self.x_mean) / self.x_std

        state = str(row["STATE"])
        state_idx = self.state_to_idx.get(state, -1)

        draws_dict = self._sample_posterior_draws(draws)
        gamma_future = self._future_year_effect(draws_dict, int(forecast_year))

        mu_draws = (
            draws_dict["alpha"]
            + np.dot(draws_dict["beta"], x)
            + gamma_future
        )

        if state_idx >= 0:
            mu_draws = mu_draws + draws_dict["u_state"][:, state_idx]

        income_draws = np.expm1(mu_draws)

        result = {
            "zip": row_zip_key,
            "state": state,
            "source_year": int(row["YEAR"]),
            "forecast_year": int(forecast_year),
            "year_effect_mode": self.year_effect_mode,
            "in_training_sample": in_training_sample,
            "pred_log_income_mean": float(np.mean(mu_draws)),
            "pred_log_income_p05": float(np.quantile(mu_draws, 0.05)),
            "pred_log_income_p95": float(np.quantile(mu_draws, 0.95)),
            "pred_income_mean": float(np.mean(income_draws)),
            "pred_income_p05": float(np.quantile(income_draws, 0.05)),
            "pred_income_p50": float(np.quantile(income_draws, 0.50)),
            "pred_income_p95": float(np.quantile(income_draws, 0.95)),
        }

        return result


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Predict ZIP-level income for a target year using saved Bayesian model artifacts."
    )
    parser.add_argument("--results-dir", default="results", help="Directory containing trace.nc and metadata artifacts")
    parser.add_argument("--zip", dest="zip_code", default=None, help="ZIP code to predict")
    parser.add_argument("--year", type=int, default=2023, help="Target forecast year")
    parser.add_argument("--draws", type=int, default=500, help="Posterior draws used for prediction")
    parser.add_argument(
        "--mode",
        dest="year_effect_mode",
        choices=["zero", "last_year", "sample_sigma"],
        default="sample_sigma",
        help="How to handle unseen future-year effect",
    )
    parser.add_argument(
        "--strict-training-zip",
        action="store_true",
        help="Allow predictions only for ZIPs included in training_sampled_zips.csv",
    )
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    zip_code = args.zip_code
    if zip_code is None:
        zip_code = input("Enter ZIP code for prediction: ").strip()

    if not zip_code:
        raise SystemExit("No ZIP code provided.")

    predictor = ZipIncomePredictor(
        results_dir=args.results_dir,
        year_effect_mode=args.year_effect_mode,
        strict_training_zip=args.strict_training_zip,
    )

    pred = predictor.predict_zip(
        zipcode=zip_code,
        forecast_year=args.year,
        draws=args.draws,
    )

    print("\nZIP Prediction")
    print("=" * 60)
    print(f"ZIP                : {pred['zip']}")
    print(f"State              : {pred['state']}")
    print(f"Source Year        : {pred['source_year']}")
    print(f"Forecast Year      : {pred['forecast_year']}")
    print(f"Year Effect Mode   : {pred['year_effect_mode']}")
    if pred["in_training_sample"] is None:
        scope_text = "unknown (training scope file unavailable)"
    elif pred["in_training_sample"]:
        scope_text = "yes"
    else:
        scope_text = "no"
    print(f"In Train Subsample : {scope_text}")
    print(f"Pred Income Mean   : {pred['pred_income_mean']:.2f}")
    print(f"Pred Income P05    : {pred['pred_income_p05']:.2f}")
    print(f"Pred Income P50    : {pred['pred_income_p50']:.2f}")
    print(f"Pred Income P95    : {pred['pred_income_p95']:.2f}")


if __name__ == "__main__":
    main()
