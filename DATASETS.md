# Datasets Documentation

## US Multi-Year ZIP Code Income Data

### Official Source: IRS Statistics of Income (SOI)

**Primary Source:**
- **Website:** https://www.irs.gov/statistics/soi-tax-stats-individual-income-tax-statistics-zip-code-data-soi
- **Agency:** Internal Revenue Service, Statistics of Income Division
- **Update Frequency:** Annual (typically 18-24 months lag)

### Available Years

| Year | Dataset Name | Official Link | Status |
|------|--------------|---------------|--------|
| 2022 | Individual Income Tax Statistics - 2022 ZIP Code Data | https://www.irs.gov/pub/irs-soi/22zpallagi.csv | ✅ Available |
| 2021 | Individual Income Tax Statistics - 2021 ZIP Code Data | https://www.irs.gov/pub/irs-soi/21zpallagi.csv | ✅ Available |
| 2020 | Individual Income Tax Statistics - 2020 ZIP Code Data | https://www.irs.gov/pub/irs-soi/20zpallagi.csv | ✅ Available |
| 2019 | Individual Income Tax Statistics - 2019 ZIP Code Data | https://www.irs.gov/pub/irs-soi/19zpallagi.csv | ✅ Available |
| 2018 | Individual Income Tax Statistics - 2018 ZIP Code Data | https://www.irs.gov/pub/irs-soi/18zpallagi.csv | ✅ Available |

### File Format

- **Format:** CSV (comma-separated values)
- **Encoding:** UTF-8
- **Size:** ~200-250 MB per year (uncompressed)
- **Records:** ~165,000 rows per year (ZIP-level aggregates)

### Dataset Variants

The IRS publishes multiple variants of ZIP code data:

1. **`YYzpallagi.csv`** ✅ **CHOSEN**
   - All AGI ranges combined
   - Most comprehensive
   - Used for this project

2. `YYzpallnoagi.csv`
   - No AGI (Adjusted Gross Income) field
   - Smaller file size
   - Not suitable for income prediction

3. State-specific files
   - Separate files per state
   - More granular but requires aggregation
   - Not used (we use national file)

### Why `zpallagi.csv` Was Chosen

✅ **Comprehensive:** Includes all income brackets and tax statistics
✅ **Consistent:** Same schema across years (2018-2022)
✅ **ZIP-level:** Natural geographic unit for analysis
✅ **Rich features:** 150+ columns including wages, business income, capital gains, deductions
✅ **Target variable:** Contains AGI (Adjusted Gross Income) for prediction

### Column Schema (Key Fields)

| Column | Description | Type | Use |
|--------|-------------|------|-----|
| `zipcode` | 5-digit ZIP code | string | Primary key |
| `agi_stub` | AGI range (1-6) | int | Aggregation level |
| `N1` | Number of returns | int | Sample size |
| `A00100` | Adjusted Gross Income | float | Target (income) |
| `N00200` | Number with salaries/wages | int | Feature |
| `A00200` | Total salaries and wages | float | Feature |
| `N00300` | Number with taxable interest | int | Feature |
| `A00300` | Taxable interest amount | float | Feature |
| `N00650` | Number with capital gains | int | Feature |
| `A00650` | Capital gain/loss amount | float | Feature |
| ... | 150+ additional columns | ... | Features |

Full schema documentation: https://www.irs.gov/pub/irs-soi/22zpdoc.doc

---

## Generated Datasets

We transform the raw IRS data into several processed datasets for different modeling purposes:

### 1. US Panel Dataset

**File:** `datasets/us_panel_dataset.csv`  
**Size:** 113.9 MB  
**Records:** 110,430 ZIP-year observations  
**Coverage:** 27,769 unique ZIPs across 50 states (2018-2022)  

**Purpose:** Unified panel data structure with engineered features.

**Generation:**
```bash
python src/data/build_us_panel_dataset.py
```

**Schema:**
- `zip`: 5-digit ZIP code
- `year`: Tax year (2018-2022)
- `state`: 2-letter state code
- `n_returns`: Number of tax returns
- `total_wages`, `n_wages`: Wage income aggregates
- `total_interest`, `n_interest`: Interest income
- `total_cap_gains`, `n_cap_gains`: Capital gains
- `total_business`, `n_business`: Business income
- `avg_agi`: **Target variable** - Average adjusted gross income
- `pct_with_wages`: Percentage with wage income
- `avg_wage_per_earner`: Average wage per earner
- `pct_with_business`: Percentage with business income
- `pct_with_cap_gains`: Percentage with capital gains
- `log_returns`: Log-transformed return count

**Filtering:**
- Removed ZIPs with <10 returns (privacy/reliability)
- Removed records where ZIP=0 (data errors)
- Kept all 50 states + DC

**Feature Engineering:**
- Computed percentages (e.g., % with wages)
- Calculated per-capita metrics (e.g., avg wage per earner)
- Log-transformed count variables

---

### 2. Multi-Year Forecasting Datasets

These datasets contain temporal pairs for training horizon-specific models.

#### 2a. 1-Year Forecasting Dataset

**File:** `datasets/us_forecasting_1yr.csv`  
**Size:** 16.5 MB  
**Records:** 82,596 transition pairs  
**Years:** 2019→2020, 2020→2021, 2021→2022  

**Structure:** Each row contains:
- 15 features from year `t` (suffix `_t`)
- Target income from year `t+1`: `target_income_tplus1`

**Train/Test Split:**
- Train: 55,073 pairs (2019→2020, 2020→2021)
- Test: 27,523 pairs (2021→2022)

#### 2b. 2-Year Forecasting Dataset

**File:** `datasets/us_forecasting_2yr.csv`  
**Size:** 11.0 MB  
**Records:** 55,027 transition pairs  
**Years:** 2019→2021, 2020→2022  

**Train/Test Split:**
- Train: 27,509 pairs (2019→2021)
- Test: 27,518 pairs (2020→2022)

#### 2c. 3-Year Forecasting Dataset

**File:** `datasets/us_forecasting_3yr.csv`  
**Size:** 5.5 MB  
**Records:** 27,487 transition pairs  
**Years:** 2019→2022  

**Train/Test Split:**
- Train: 27,487 pairs (all data - no test set due to limited samples)
- Test: 0 pairs

**Generation:**
```bash
python src/data/build_us_multiyear_forecasting_dataset.py
```

**Feature Schema (15 features):**
All features have `_t` suffix indicating base year:
- `n_returns_t`: Number of returns
- `total_wages_t`: Total wage income
- `n_wages_t`: Count with wages
- `total_interest_t`: Total interest income
- `n_interest_t`: Count with interest
- `total_cap_gains_t`: Total capital gains
- `n_cap_gains_t`: Count with capital gains
- `total_business_t`: Total business income
- `n_business_t`: Count with business income
- `avg_agi_t`: **Most important** - Average AGI at base year
- `pct_with_wages_t`: Percentage with wages
- `avg_wage_per_earner_t`: Average wage per earner
- `pct_with_business_t`: Percentage with business
- `pct_with_cap_gains_t`: Percentage with capital gains
- `log_returns_t`: Log of return count

**Target Variable:**
- `target_income_tplusn`: Average AGI at year `t+n` (in thousands)

---

### 3. Parquet Optimized Files

**Directory:** `datasets/parquet/`  
**Format:** Apache Parquet (columnar storage)  
**Size:** ~60% smaller than CSV  
**Load Speed:** 10-100x faster than CSV  

**Files:**
- `09zpallagi.parquet` through `22zpallagi.parquet`
- Raw IRS data in optimized format

**Conversion:**
```bash
python scripts/convert_to_parquet.py
```

**Usage:**
```python
import pandas as pd
df = pd.read_parquet('datasets/parquet/22zpallagi.parquet')
```

---

### Data Quality Notes

**Strengths:**
- Official government data (highly reliable)
- Consistent methodology across years
- Large sample size (entire US tax-filing population)
- Rich feature set (100+ tax statistics)

**Limitations:**
- **Privacy:** ZIP codes with <100 returns suppressed
- **Lag:** Data released 18-24 months after tax year
- **Coverage:** Only includes tax filers (misses non-filers)
- **Schema changes:** Minor column additions/removals between years
- **COVID impact:** 2020-2021 data shows anomalies (stimulus, unemployment)

### Download Instructions

#### Automated Download (Recommended)

```bash
python scripts/download_us_irs_zip_data.py --years 2018 2019 2020 2021 2022
```

This will:
1. Download CSV files from IRS.gov
2. Validate file integrity (size, headers)
3. Save to `datasets/us_multi_year/YYYY.csv`
4. Generate download log

#### Manual Download (If Automation Fails)

1. Visit: https://www.irs.gov/statistics/soi-tax-stats-individual-income-tax-statistics-zip-code-data-soi
2. For each year (2018-2022):
   - Find "Individual Income Tax Statistics - YYYY ZIP Code Data"
   - Click "ZIP Code Data (csv)" link
   - Direct link format: `https://www.irs.gov/pub/irs-soi/YYzpallagi.csv`
3. Save files as:
   ```
   datasets/us_multi_year/2018.csv
   datasets/us_multi_year/2019.csv
   datasets/us_multi_year/2020.csv
   datasets/us_multi_year/2021.csv
   datasets/us_multi_year/2022.csv
   ```

#### Verification

After download, verify:
```bash
python scripts/verify_dataset_integrity.py
```

Expected file sizes:
- 2018: ~206 MB
- 2019: ~208 MB
- 2020: ~210 MB
- 2021: ~212 MB
- 2022: ~215 MB

---

## India District Income Data

### Current Status: PROXY-BASED (Not True Multi-Year Forecast)

**Source:** Synthetic/proxy dataset based on 2011 census + economic indicators

**Limitations:**
- ❌ No official district-level income time series available
- ❌ Census conducted only once per decade (2011, 2021 pending)
- ❌ Economic surveys at state level only
- ⚠️ Current model uses proxy features (literacy, employment, urbanization)

**Future Data Requirements for True Forecasting:**

To enable year-to-year forecasting for India, we would need:

1. **Annual District Domestic Product (DDP)**
   - Source: State economic surveys (partial coverage)
   - Availability: Limited, inconsistent across states
   - Status: Not publicly available in structured format

2. **NSSO Employment Surveys**
   - Source: National Sample Survey Office
   - Frequency: Every 5 years
   - Latest: 2017-18, 2022-23 (preliminary)
   - Limitation: District-level not published

3. **Alternative Proxy Data:**
   - Night-time lights (satellite) - continuous
   - Mobile phone activity - commercial
   - Banking transactions - RBI aggregates
   - Status: Research-grade only, not production-ready

**Recommended Approach:**
- Continue with district classification (income bracket prediction)
- Document as "proxy-based estimation" not "time-series forecast"
- Clearly state limitations in research paper
- Future work: integrate emerging data sources when available

---

## Dataset Storage Structure

```
datasets/
├── us_multi_year/              # Raw IRS CSV files
│   ├── 2018.csv
│   ├── 2019.csv
│   ├── 2020.csv
│   ├── 2021.csv
│   └── 2022.csv
├── parquet/                    # Optimized format (converted)
│   ├── 2018.parquet
│   ├── 2019.parquet
│   ├── 2020.parquet
│   ├── 2021.parquet
│   └── 2022.parquet
├── us_panel_dataset.csv        # Stacked multi-year panel
├── us_forecasting_dataset.csv  # Year t → t+1 pairs
└── india_proxy_dataset.csv     # India district proxies
```

---

## Data Citation

**US IRS Data:**
```
Internal Revenue Service. (2024). Individual Income Tax Statistics - 
ZIP Code Data (2018-2022). Statistics of Income Division. 
Retrieved from https://www.irs.gov/statistics/soi-tax-stats
```

**Research Paper Citation:**
```bibtex
@techreport{IRS2024SOI,
  author = {{Internal Revenue Service}},
  title = {Individual Income Tax Statistics - ZIP Code Data},
  institution = {Statistics of Income Division},
  year = {2024},
  url = {https://www.irs.gov/statistics/soi-tax-stats},
  note = {Years 2018-2022}
}
```

---

## Last Updated

**Date:** January 23, 2026  
**Verified By:** Research Team  
**Next Verification:** When IRS releases 2023 data (expected Q3 2025)
