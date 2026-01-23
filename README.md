# Income-prediction-model

Pipeline to clean IRS zipcode-level aggregates (2005–2022) and prepare them for modeling average income by zipcode.

## Prerequisites
- Python 3.9+
- Install deps:

```bash
pip install pandas scikit-learn joblib numpy
```

## Raw data naming
All files in `raw_data/` follow `irs_YYYY.csv` (e.g., `irs_2018.csv`).

## Process data (no modeling yet)
Run the processing pipeline to build a clean, aggregated dataset:

```bash
python data_pipeline.py
```

Outputs to `processed/`:
- `income_by_zip.parquet`
- `income_by_zip.csv`

Each row: `zipcode`, `state`, `year`, `total_agi`, `total_returns`, `avg_income`, `zip3`.

## Next steps (modeling)
After processing, we can fit a model using `income_by_zip.*` as the source features/target. The existing `model.py` can be adapted to consume the processed file when you are ready to train.