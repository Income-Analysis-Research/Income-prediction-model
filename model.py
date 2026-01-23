"""Train a zipcode-level income model from IRS zip aggregates.

The script loads every CSV in ./raw_data, normalizes the schema, builds
zipcode-year aggregates, trains a regression model, reports metrics, and
writes the fitted pipeline to ./artifacts/model.joblib.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RAW_DATA_DIR = Path(__file__).parent / "raw_data"
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


@dataclass
class LoadedFrame:
	year: int
	frame: pd.DataFrame


def _parse_year_from_name(filename: str) -> Optional[int]:
	match = re.search(r"(20\d{2}|19\d{2})", filename)
	return int(match.group(1)) if match else None


def _load_single_csv(path: Path) -> Optional[LoadedFrame]:
	year = _parse_year_from_name(path.name)
	if year is None:
		print(f"[skip] Could not infer year from {path.name}")
		return None

	try:
		df = pd.read_csv(path, dtype={"zipcode": str, "ZIPCODE": str}, engine="python")
	except Exception as exc:  # pragma: no cover - defensive path
		print(f"[skip] Failed to read {path.name}: {exc}")
		return None

	df.columns = [c.strip().upper() for c in df.columns]

	required = {"ZIPCODE", "STATE", "A00100", "N1"}
	if not required.issubset(df.columns):
		print(f"[skip] {path.name} missing required columns: {required - set(df.columns)}")
		return None

	df = df[list(required)]
	df = df.rename(columns={"ZIPCODE": "zipcode", "STATE": "state", "A00100": "total_agi", "N1": "total_returns"})

	return LoadedFrame(year=year, frame=df)


def load_all_raw(raw_dir: Path) -> List[LoadedFrame]:
	frames: List[LoadedFrame] = []
	for path in sorted(raw_dir.glob("*.csv")):
		loaded = _load_single_csv(path)
		if loaded:
			loaded.frame["year"] = loaded.year
			frames.append(loaded)
	return frames


def build_training_frame(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
	loaded_frames = load_all_raw(raw_dir)
	if not loaded_frames:
		raise RuntimeError(f"No usable CSVs found in {raw_dir}")

	parts: List[pd.DataFrame] = []
	for loaded in loaded_frames:
		df = loaded.frame.copy()
		df["zipcode"] = df["zipcode"].str.zfill(5)
		df = df[df["zipcode"].str.match(r"^\d{5}$")]

		df["total_agi"] = pd.to_numeric(df["total_agi"], errors="coerce")
		df["total_returns"] = pd.to_numeric(df["total_returns"], errors="coerce")
		df = df.dropna(subset=["total_agi", "total_returns"])
		df = df[df["total_returns"] > 0]

		df = df.groupby(["zipcode", "state", "year"], as_index=False).agg(
			total_agi=("total_agi", "sum"),
			total_returns=("total_returns", "sum"),
		)

		df["avg_income"] = df["total_agi"] / df["total_returns"]
		df["zip3"] = df["zipcode"].str.slice(0, 3).astype(int)
		parts.append(df)

	full = pd.concat(parts, ignore_index=True)
	full = full.dropna(subset=["avg_income"])
	return full


def build_pipeline(categorical_features: Iterable[str], numeric_features: Iterable[str]) -> Pipeline:
	categorical_transformer = OneHotEncoder(handle_unknown="ignore")
	numeric_transformer = Pipeline(
		steps=[("scaler", StandardScaler())]
	)

	preprocessor = ColumnTransformer(
		transformers=[
			("categorical", categorical_transformer, list(categorical_features)),
			("numeric", numeric_transformer, list(numeric_features)),
		]
	)

	model = RandomForestRegressor(
		n_estimators=200,
		random_state=42,
		n_jobs=-1,
		min_samples_leaf=2,
	)

	return Pipeline([
		("preprocess", preprocessor),
		("regressor", model),
	])


def train_model(df: pd.DataFrame) -> Tuple[Pipeline, dict]:
	feature_cols = ["state", "year", "zip3", "total_returns"]
	X = df[feature_cols]
	y = df["avg_income"]

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	pipeline = build_pipeline(categorical_features=["state"], numeric_features=["year", "zip3", "total_returns"])
	pipeline.fit(X_train, y_train)

	preds = pipeline.predict(X_test)
	metrics = {
		"mae": float(mean_absolute_error(y_test, preds)),
		"rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
	}
	return pipeline, metrics


def ensure_artifacts_dir(path: Path = ARTIFACTS_DIR) -> Path:
	path.mkdir(parents=True, exist_ok=True)
	return path


def save_pipeline(pipeline: Pipeline, path: Path) -> None:
	joblib.dump(pipeline, path)
	print(f"[saved] {path}")


def example_predictions(pipeline: Pipeline, df: pd.DataFrame, k: int = 5) -> List[Tuple[str, int, float]]:
	sample = df.sample(min(k, len(df)), random_state=7)
	feature_cols = ["state", "year", "zip3", "total_returns"]
	preds = pipeline.predict(sample[feature_cols])
	sample = sample.assign(predicted_income=preds)
	results: List[Tuple[str, int, float]] = []
	for _, row in sample.iterrows():
		results.append((row["zipcode"], int(row["year"]), float(row["predicted_income"])))
	return results


def main() -> None:
	print("[load] building training frame from raw_data/")
	df = build_training_frame(RAW_DATA_DIR)
	print(f"[data] rows={len(df):,} unique_zipcodes={df['zipcode'].nunique():,}")

	pipeline, metrics = train_model(df)
	print(f"[metrics] MAE={metrics['mae']:.2f} | RMSE={metrics['rmse']:.2f}")

	ensure_artifacts_dir()
	model_path = ARTIFACTS_DIR / "income_model.joblib"
	save_pipeline(pipeline, model_path)

	print("[preview] sample predictions (zipcode, year -> predicted avg income)")
	for zipcode, year, pred in example_predictions(pipeline, df):
		print(f"  {zipcode} ({year}): ${pred:,.0f}")


if __name__ == "__main__":
	main()
