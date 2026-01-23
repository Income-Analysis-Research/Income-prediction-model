"""
Download IRS SOI ZIP Code Data for multiple years.

This script downloads Individual Income Tax Statistics from the IRS
for specified years and saves them to datasets/us_multi_year/.

Usage:
    python scripts/download_us_irs_zip_data.py --years 2018 2019 2020 2021 2022
    python scripts/download_us_irs_zip_data.py --all  # Downloads 2018-2022
"""

import argparse
import requests
from pathlib import Path
import sys
import hashlib
from datetime import datetime

# Base URL for IRS SOI ZIP Code Data
IRS_BASE_URL = "https://www.irs.gov/pub/irs-soi"

# File naming pattern: YYzpallagi.csv (e.g., 22zpallagi.csv for 2022)
FILE_PATTERN = "{yy}zpallagi.csv"

# Expected approximate file sizes (in MB) for validation
EXPECTED_SIZES = {
    2018: (200, 220),  # Min, Max in MB
    2019: (200, 220),
    2020: (200, 220),
    2021: (205, 225),
    2022: (205, 225),
}

# Expected columns that should be present (for validation)
REQUIRED_COLUMNS = ['zipcode', 'agi_stub', 'N1', 'A00100', 'STATE']


def download_year(year: int, output_dir: Path, force: bool = False) -> bool:
    """
    Download IRS ZIP code data for a specific year.
    
    Args:
        year: Tax year (2018-2022)
        output_dir: Directory to save the file
        force: If True, re-download even if file exists
    
    Returns:
        True if successful, False otherwise
    """
    # Format year as YY
    yy = str(year)[2:]
    
    # Construct URL and output path
    filename = FILE_PATTERN.format(yy=yy)
    url = f"{IRS_BASE_URL}/{filename}"
    output_path = output_dir / f"{year}.csv"
    
    # Check if file already exists
    if output_path.exists() and not force:
        print(f"[SKIP] {year}.csv already exists (use --force to re-download)")
        return True
    
    print(f"\n{'='*70}")
    print(f"Downloading {year} data...")
    print(f"URL: {url}")
    print(f"{'='*70}")
    
    try:
        # Stream download with progress
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Get total file size
        total_size = int(response.headers.get('content-length', 0))
        total_mb = total_size / (1024 * 1024)
        
        print(f"File size: {total_mb:.1f} MB")
        
        # Download with progress
        downloaded = 0
        chunk_size = 8192
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Print progress every 10 MB
                    if downloaded % (10 * 1024 * 1024) < chunk_size:
                        progress = (downloaded / total_size * 100) if total_size > 0 else 0
                        print(f"  Progress: {progress:.1f}% ({downloaded/(1024*1024):.1f}/{total_mb:.1f} MB)", end='\r')
        
        print(f"\n[OK] Downloaded: {output_path}")
        
        # Validate file size
        actual_size_mb = output_path.stat().st_size / (1024 * 1024)
        min_size, max_size = EXPECTED_SIZES.get(year, (0, 1000))
        
        if min_size <= actual_size_mb <= max_size:
            print(f"[OK] Size validation passed: {actual_size_mb:.1f} MB")
        else:
            print(f"[WARNING] Unexpected file size: {actual_size_mb:.1f} MB (expected {min_size}-{max_size} MB)")
        
        # Quick validation: check first line for expected columns
        with open(output_path, 'r', encoding='utf-8') as f:
            header = f.readline().strip().lower()
            missing_cols = [col for col in REQUIRED_COLUMNS if col.lower() not in header]
            
            if missing_cols:
                print(f"[WARNING] Missing expected columns: {missing_cols}")
            else:
                print(f"[OK] Header validation passed")
        
        return True
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print(f"[ERROR] File not found at IRS website")
            print(f"        The {year} data may not be available yet")
            print(f"        Check: https://www.irs.gov/statistics/soi-tax-stats")
        else:
            print(f"[ERROR] HTTP error: {e}")
        return False
        
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Download failed: {e}")
        return False
        
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download IRS SOI ZIP Code Data for multiple years",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download specific years
  python scripts/download_us_irs_zip_data.py --years 2018 2019 2020 2021 2022
  
  # Download all available years (2018-2022)
  python scripts/download_us_irs_zip_data.py --all
  
  # Force re-download
  python scripts/download_us_irs_zip_data.py --all --force
        """
    )
    
    parser.add_argument(
        '--years',
        type=int,
        nargs='+',
        help='Specific years to download (e.g., 2018 2019 2020 2021 2022)'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Download all available years (2018-2022)'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if files exist'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='datasets/us_multi_year',
        help='Output directory (default: datasets/us_multi_year)'
    )
    
    args = parser.parse_args()
    
    # Determine which years to download
    if args.all:
        years = list(range(2018, 2023))  # 2018-2022
    elif args.years:
        years = sorted(args.years)
    else:
        print("ERROR: Must specify either --years or --all")
        parser.print_help()
        sys.exit(1)
    
    # Validate years
    valid_years = list(range(2018, 2023))
    invalid = [y for y in years if y not in valid_years]
    if invalid:
        print(f"ERROR: Invalid years: {invalid}")
        print(f"       Valid range: 2018-2022")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("IRS SOI ZIP CODE DATA DOWNLOADER")
    print("="*70)
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Years to download: {years}")
    print(f"Force re-download: {args.force}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Download each year
    results = {}
    for year in years:
        success = download_year(year, output_dir, args.force)
        results[year] = success
    
    # Summary
    print("\n" + "="*70)
    print("DOWNLOAD SUMMARY")
    print("="*70)
    
    successful = [y for y, s in results.items() if s]
    failed = [y for y, s in results.items() if not s]
    
    print(f"Successful: {len(successful)}/{len(years)}")
    if successful:
        print(f"  Years: {successful}")
    
    if failed:
        print(f"\nFailed: {len(failed)}/{len(years)}")
        print(f"  Years: {failed}")
        print("\nTroubleshooting:")
        print("  1. Check internet connection")
        print("  2. Verify IRS website is accessible")
        print("  3. Check if data for failed years is published yet")
        print("  4. Visit: https://www.irs.gov/statistics/soi-tax-stats")
    
    # Write download log
    log_path = output_dir / "download_log.txt"
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*70}\n")
        f.write(f"Download completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Years requested: {years}\n")
        f.write(f"Successful: {successful}\n")
        f.write(f"Failed: {failed}\n")
    
    print(f"\nLog saved: {log_path}")
    
    # Exit code
    sys.exit(0 if not failed else 1)


if __name__ == '__main__':
    main()
