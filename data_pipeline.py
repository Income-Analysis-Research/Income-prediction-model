"""Collate all raw IRS CSVs into a single CSV/Parquet without aggregation.

Actions:
1) Load every `irs_YYYY.csv` from ./raw_data
2) Add `year` (from filename) and `source_file`
3) Concatenate rows as-is (no computations or grouping)
4) Save to ./processed/collated_raw.csv and .parquet
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional

import pandas as pd

RAW_DATA_DIR = Path(__file__).parent / "raw_data"
PROCESSED_DIR = Path(__file__).parent / "processed"
CSV_OUT = PROCESSED_DIR / "collated_raw.csv"
PARQUET_OUT = PROCESSED_DIR / "collated_raw.parquet"


def parse_year_from_name(filename: str) -> Optional[int]:
	match = re.search(r"(19|20)\d{2}", filename)
	return int(match.group(0)) if match else None


def load_single_csv(path: Path) -> Optional[pd.DataFrame]:
	year = parse_year_from_name(path.name)
	if year is None:
		print(f"[skip] Could not infer year from {path.name}")
		return None

	try:
		df = pd.read_csv(path, engine="python")
	except Exception as exc:  # defensive read
		print(f"[skip] Failed to read {path.name}: {exc}")
		return None

	df["year"] = year
	df["source_file"] = path.name
	return df


def load_all(raw_dir: Path) -> List[pd.DataFrame]:
	frames: List[pd.DataFrame] = []
	for path in sorted(raw_dir.glob("irs_*.csv")):
		loaded = load_single_csv(path)
		if loaded is not None:
			frames.append(loaded)
	return frames


def collate(frames: List[pd.DataFrame]) -> pd.DataFrame:
	if not frames:
		raise RuntimeError("No CSVs loaded; nothing to collate.")
	# Align on all columns present across files
	combined = pd.concat(frames, ignore_index=True, sort=False)
	return combined


def ensure_output_paths() -> None:
	PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def save_outputs(df: pd.DataFrame, csv_path: Path, parquet_path: Path) -> None:
	ensure_output_paths()

	df.to_csv(csv_path, index=False)
	print(f"[save] {csv_path} ({csv_path.stat().st_size/1_048_576:.2f} MB)")

	try:
		def _coerce_object_cols(frame: pd.DataFrame) -> pd.DataFrame:
			out = frame.copy()
			for col in out.columns:
				if out[col].dtype == object:
					s = out[col]
					# Replace lone dot strings with NA to aid numeric conversion
					s = s.replace({".": pd.NA})
					numeric = pd.to_numeric(s, errors="coerce")
					# If mostly numeric, keep numeric; otherwise store as string
					if numeric.notna().mean() >= 0.8:
						out[col] = numeric
					else:
						out[col] = s.astype(str)
			return out

		df_parquet = _coerce_object_cols(df)
		df_parquet.to_parquet(parquet_path, index=False)
		print(f"[save] {parquet_path} ({parquet_path.stat().st_size/1_048_576:.2f} MB)")
	except ImportError:
		print("[warn] pyarrow/fastparquet not installed; skipping parquet. Install with 'pip install pyarrow' if needed.")


def run_pipeline(raw_dir: Path = RAW_DATA_DIR, csv_path: Path = CSV_OUT, parquet_path: Path = PARQUET_OUT) -> None:
	print(f"[load] scanning {raw_dir}")
	frames = load_all(raw_dir)
	print(f"[load] usable files: {len(frames)}")

	df = collate(frames)
	print(f"[data] rows={len(df):,} cols={len(df.columns)}")

	save_outputs(df, csv_path, parquet_path)
	print("[done] processing complete")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Collate raw IRS CSVs into a single file (no aggregation).")
	parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR, help="Directory containing irs_YYYY.csv files")
	parser.add_argument("--csv-out", type=Path, default=CSV_OUT, help="CSV output path")
	parser.add_argument("--parquet-out", type=Path, default=PARQUET_OUT, help="Parquet output path")
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	run_pipeline(args.raw_dir, args.csv_out, args.parquet_out)
