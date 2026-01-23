"""
Convert IRS CSV datasets to Parquet format for faster loading.
Validates that no data is lost in the conversion process.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import time

def validate_conversion(csv_df, parquet_df, filename):
    """Validate that CSV and Parquet dataframes are identical."""
    print(f"\n{'='*70}")
    print(f"VALIDATING: {filename}")
    print(f"{'='*70}")
    
    # Check shape
    assert csv_df.shape == parquet_df.shape, f"Shape mismatch: {csv_df.shape} vs {parquet_df.shape}"
    print(f"✓ Shape preserved: {csv_df.shape}")
    
    # Check columns
    assert list(csv_df.columns) == list(parquet_df.columns), "Column names differ"
    print(f"✓ All {len(csv_df.columns)} columns preserved")
    
    # Check data types (allow minor type changes like int64 vs int32)
    for col in csv_df.columns:
        csv_dtype = csv_df[col].dtype
        pq_dtype = parquet_df[col].dtype
        
        # Allow numeric type variations
        if pd.api.types.is_numeric_dtype(csv_dtype) and pd.api.types.is_numeric_dtype(pq_dtype):
            continue
        # Allow string/object equivalence
        elif (csv_dtype == 'object' or csv_dtype == 'string') and (pq_dtype == 'object' or pq_dtype == 'string'):
            continue
        # Otherwise must match
        elif csv_dtype != pq_dtype:
            print(f"⚠ Type change in {col}: {csv_dtype} → {pq_dtype}")
    
    # Check for missing values preservation
    csv_nulls = csv_df.isnull().sum().sum()
    pq_nulls = parquet_df.isnull().sum().sum()
    assert csv_nulls == pq_nulls, f"Null count mismatch: {csv_nulls} vs {pq_nulls}"
    print(f"✓ Missing values preserved: {csv_nulls:,}")
    
    # Sample data validation (check a few random rows)
    sample_size = min(100, len(csv_df))
    sample_indices = np.random.choice(len(csv_df), sample_size, replace=False)
    
    for idx in sample_indices:
        for col in csv_df.columns:
            csv_val = csv_df.iloc[idx][col]
            pq_val = parquet_df.iloc[idx][col]
            
            # Handle NaN comparison
            if pd.isna(csv_val) and pd.isna(pq_val):
                continue
            # Handle numeric comparison with tolerance
            elif pd.api.types.is_numeric_dtype(csv_df[col].dtype):
                if not np.allclose([csv_val], [pq_val], rtol=1e-9, equal_nan=True):
                    raise ValueError(f"Value mismatch at row {idx}, col {col}: {csv_val} vs {pq_val}")
            # Handle exact string comparison
            else:
                if csv_val != pq_val:
                    raise ValueError(f"Value mismatch at row {idx}, col {col}: {csv_val} vs {pq_val}")
    
    print(f"✓ Sampled {sample_size} rows - all values match")
    print(f"✓ VALIDATION PASSED")


def convert_csv_to_parquet(csv_path, parquet_path, validate=True):
    """Convert a CSV file to Parquet with validation."""
    print(f"\n{'='*70}")
    print(f"CONVERTING: {csv_path.name}")
    print(f"{'='*70}")
    
    # Read CSV
    print("Reading CSV...")
    start_time = time.time()
    csv_df = pd.read_csv(csv_path, low_memory=False)
    csv_read_time = time.time() - start_time
    print(f"CSV read time: {csv_read_time:.2f} seconds")
    print(f"Records: {len(csv_df):,}")
    print(f"Columns: {len(csv_df.columns)}")
    print(f"Memory: {csv_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Write Parquet
    print("\nWriting Parquet...")
    start_time = time.time()
    csv_df.to_parquet(parquet_path, engine='pyarrow', compression='snappy', index=False)
    parquet_write_time = time.time() - start_time
    print(f"Parquet write time: {parquet_write_time:.2f} seconds")
    
    # Get file sizes
    csv_size = csv_path.stat().st_size / 1024**2
    parquet_size = parquet_path.stat().st_size / 1024**2
    compression_ratio = csv_size / parquet_size
    
    print(f"\nFile sizes:")
    print(f"  CSV:     {csv_size:.1f} MB")
    print(f"  Parquet: {parquet_size:.1f} MB")
    print(f"  Compression ratio: {compression_ratio:.2f}x")
    
    # Validate if requested
    if validate:
        print("\nReading Parquet for validation...")
        start_time = time.time()
        parquet_df = pd.read_parquet(parquet_path, engine='pyarrow')
        parquet_read_time = time.time() - start_time
        print(f"Parquet read time: {parquet_read_time:.2f} seconds")
        print(f"Speed improvement: {csv_read_time/parquet_read_time:.2f}x faster")
        
        validate_conversion(csv_df, parquet_df, csv_path.name)
    
    return parquet_path


def main():
    """Convert all IRS CSV files to Parquet."""
    print("="*70)
    print("IRS DATASET CSV → PARQUET CONVERSION")
    print("="*70)
    print("\nThis script converts CSV files to Parquet format for:")
    print("  • 10-100x faster loading times")
    print("  • Better compression (2-5x smaller files)")
    print("  • Preserved data types (no inference needed)")
    print("  • Column-oriented storage (efficient for ML)")
    
    datasets_dir = Path("datasets")
    parquet_dir = datasets_dir / "parquet"
    parquet_dir.mkdir(exist_ok=True)
    
    # Find all IRS ZIP code CSV files
    irs_files = list(datasets_dir.glob("*zpallagi.csv"))
    
    if not irs_files:
        print("\n❌ No IRS CSV files found in datasets/")
        return
    
    print(f"\n✓ Found {len(irs_files)} IRS CSV files")
    
    # Convert each file
    converted = []
    for csv_path in sorted(irs_files):
        parquet_path = parquet_dir / csv_path.name.replace('.csv', '.parquet')
        
        # Skip if already exists and is newer
        if parquet_path.exists() and parquet_path.stat().st_mtime > csv_path.stat().st_mtime:
            print(f"\n⏭ Skipping {csv_path.name} (Parquet already up-to-date)")
            converted.append(parquet_path)
            continue
        
        try:
            result = convert_csv_to_parquet(csv_path, parquet_path, validate=True)
            converted.append(result)
        except Exception as e:
            print(f"\n❌ Error converting {csv_path.name}: {e}")
            raise
    
    # Summary
    print("\n" + "="*70)
    print("CONVERSION SUMMARY")
    print("="*70)
    print(f"\n✓ Successfully converted {len(converted)} files")
    print(f"\nParquet files saved to: {parquet_dir.absolute()}")
    print("\nTo use in training scripts:")
    print('  df = pd.read_parquet("datasets/parquet/22zpallagi.parquet")')
    print("\nOriginal CSV files are preserved - you can delete them later if needed.")


if __name__ == "__main__":
    main()
