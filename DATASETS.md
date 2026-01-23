# Datasets Documentation

**Complete data source documentation and download instructions**

---

## Table of Contents
- [Data Sources & Citations](#data-sources--citations)
- [Download Instructions](#download-instructions)
- [Generated Datasets](#generated-datasets)
- [Data Quality Notes](#data-quality-notes)

---

## Data Sources & Citations

### 1. US Income Data (Primary Dataset)

**Official Source:** IRS Statistics of Income (SOI) - Individual Income Tax Statistics

**Publisher:** Internal Revenue Service, United States Department of the Treasury  
**License:** Public domain (U.S. Government works, 17 U.S.C. § 105)  
**Website:** https://www.irs.gov/statistics/soi-tax-stats-individual-income-tax-statistics-zip-code-data-soi

**Dataset Details:**
- **Format:** CSV (comma-separated values)
- **Size:** ~200-250 MB per year (uncompressed)
- **Records:** ~165,000 ZIP-level aggregates per year
- **Coverage:** All 50 US states + DC
- **Update Frequency:** Annual (18-24 months lag)

**Available Years:**

| Year | Dataset | Direct Download | Status |
|------|---------|----------------|--------|
| 2022 | 22zpallagi.csv | https://www.irs.gov/pub/irs-soi/22zpallagi.csv | ✅ Used |
| 2021 | 21zpallagi.csv | https://www.irs.gov/pub/irs-soi/21zpallagi.csv | ✅ Used |
| 2020 | 20zpallagi.csv | https://www.irs.gov/pub/irs-soi/20zpallagi.csv | ✅ Used |
| 2019 | 19zpallagi.csv | https://www.irs.gov/pub/irs-soi/19zpallagi.csv | ✅ Used |
| 2018 | 18zpallagi.csv | https://www.irs.gov/pub/irs-soi/18zpallagi.csv | ✅ Used |

**What We Used:**
- **Target Variable:** Adjusted Gross Income (AGI) - average income per ZIP code
- **Features:** 150+ columns including wages, business income, capital gains, interest, deductions
- **Geographic Granularity:** 5-digit ZIP code level
- **Sample Size:** 27,769 unique ZIP codes, 110,430 ZIP-year observations (2018-2022)

**Why This Dataset:**
- ✅ Official government data (highly reliable)
- ✅ Comprehensive tax statistics
- ✅ Consistent methodology across years
- ✅ Fine geographic granularity
- ✅ Large sample size (entire tax-filing population)

**Citation:**
> Internal Revenue Service. (2024). *SOI Tax Stats - Individual Income Tax Statistics - ZIP Code Data (Tax Years 2018-2022)*. United States Department of the Treasury. Retrieved from https://www.irs.gov/statistics/soi-tax-stats-individual-income-tax-statistics-zip-code-data-soi

**Schema Documentation:** https://www.irs.gov/pub/irs-soi/22zpdoc.doc

---

### 2. India District Census Data (Secondary Dataset)

**Official Source:** Census of India 2011

**Publisher:** Office of the Registrar General & Census Commissioner, India  
**Ministry:** Ministry of Home Affairs, Government of India  
**License:** Government of India Open Data License  
**Original Website:** https://censusindia.gov.in/

**Kaggle Mirror:** https://www.kaggle.com/datasets/sudalairajkumar/india-census-2011 (CC0 Public Domain)

**Dataset Details:**
- **Format:** CSV
- **Size:** ~2 MB
- **Records:** 640 districts
- **Coverage:** 19 major states (95% population)

**What We Used:**
- **Proxy Target:** Socioeconomic status indicators (no direct income data available)
- **Features:** Literacy rate, worker participation, urban ratio, household size, infrastructure access, asset ownership
- **Processing:** Created composite scores from census variables
- **Final Dataset:** 631 valid districts × 10 features

**Citation:**
> Office of the Registrar General & Census Commissioner, India. (2011). *Census of India 2011: District-level Data*. Ministry of Home Affairs, Government of India. Retrieved from https://censusindia.gov.in/

---

## Download Instructions

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
