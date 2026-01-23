# Data Sources & Attribution

This document provides complete citations for all datasets used in this research project, including their publishers, access methods, licenses, and how they were utilized.

---

## 📊 **Datasets Used in Final Model**

### 1. **US Income Data**

**Dataset Name:** Statistics of Income (SOI) Tax Stats - Individual Income Tax Statistics - ZIP Code Data (Tax Year 2022)

**Publisher/Owner:** Internal Revenue Service (IRS), United States Department of the Treasury

**Year:** 2022 (Tax Year), Published 2024

**Access Link:** https://www.irs.gov/statistics/soi-tax-stats-individual-income-tax-statistics-zip-code-data-soi

**License/Usage:** Public domain - U.S. Government works are not subject to copyright protection under 17 U.S.C. § 105

**Description:** Provides aggregate income statistics at the ZIP code level, including:
- Adjusted Gross Income (AGI) amounts by income brackets
- Number of returns filed
- Wages and salaries
- Interest and dividend income
- State-level geographic aggregation

**What We Used It For:**
- Primary target variable: Average Adjusted Gross Income (AGI) per ZIP code
- Features: Number of returns, wage percentages, income distribution across brackets
- Geographic scope: All 50 states + DC

**File Location in Repository:**
- Raw data: `data_raw/us/22zpallagi.csv` (206 MB, original IRS file)
- Processed: `datasets/us_irs_zipcode_data.csv` (27,589 ZIP codes × 8 features)
- Training-ready: `data_processed/us/processed_data.csv`

**Processing Steps:**
1. Aggregated 6 income brackets into single average AGI per ZIP
2. Calculated wage ratios and income diversity metrics
3. Removed outliers and missing data
4. Standardized ZIP code format (5-digit with leading zeros)

**Citation Format:**
> Internal Revenue Service. (2024). *SOI Tax Stats - Individual Income Tax Statistics - ZIP Code Data (Tax Year 2022)*. Retrieved from https://www.irs.gov/statistics/soi-tax-stats-individual-income-tax-statistics-zip-code-data-soi

---

### 2. **India District-Level Census Data**

**Dataset Name:** India Districts Census 2011

**Publisher/Owner:** Office of the Registrar General & Census Commissioner, India (Ministry of Home Affairs, Government of India)

**Year:** 2011 (Census conducted)

**Access Link (via Kaggle):** https://www.kaggle.com/datasets/sudalairajkumar/india-census-2011

**Original Source:** https://censusindia.gov.in/

**License/Usage:** 
- Original data: Government of India Open Data License
- Kaggle dataset: CC0 Public Domain

**Description:** District-level demographic and socioeconomic indicators from India's 2011 Census:
- Population demographics (total, male/female breakdown, age groups)
- Literacy rates (overall and gender-specific)
- Worker participation (main workers, marginal workers)
- Household characteristics (size, type)
- Urban/rural split
- Infrastructure access (electricity, water, sanitation)
- Asset ownership (bicycle, car, television, telephone, computer)

**What We Used It For:**
- Proxy features for district-level socioeconomic status (since direct income data unavailable)
- Features: Literacy rate, worker participation rate, urban ratio, avg household size, infrastructure access scores
- Geographic scope: 640 districts (cleaned to 631 valid districts)
- Coverage: 19 out of 20 major Indian states (95% population coverage)

**File Location in Repository:**
- Raw data: `data_raw/india/india-districts-census-2011.csv` (640 districts)
- Processed: `datasets/india_district_census_data.csv` (631 districts × 10 features)
- Training-ready: `data_processed/india/processed_data.csv`

**Processing Steps:**
1. Calculated derived features (literacy_rate, worker_participation, urban_ratio)
2. Created infrastructure score from amenities (water, electricity, latrine access)
3. Created asset score from ownership rates (TV, telephone, computer, car, bicycle)
4. Standardized state names (ORISSA → ODISHA, NCT OF DELHI → DELHI)
5. Removed districts with missing critical data

**Citation Format:**
> Office of the Registrar General & Census Commissioner, India. (2011). *Census of India 2011: District-level Data*. Ministry of Home Affairs, Government of India. Retrieved from https://censusindia.gov.in/

---

## 📁 **Planned Data Sources (Not Used in Final Model)**

The following datasets were identified for potential inclusion but were not integrated into the final model due to extraction complexity, data quality issues, or time constraints.

### 3. **National Family Health Survey (NFHS-5) - India**

**Dataset Name:** National Family Health Survey - 5 (2019-21)

**Publisher:** International Institute for Population Sciences (IIPS), Mumbai (Ministry of Health and Family Welfare, Government of India)

**Year:** 2019-2021

**Access Link:** http://rchiips.org/nfhs/

**File Location:** `archive(10)/NFHS-5_India_Report.pdf` (PDF format, ~700 pages)

**Status:** ❌ **Not used** - Data extraction pending

**Reason Not Used:**
- PDF format requires manual extraction or OCR processing
- Significant preprocessing effort needed to align with district boundaries
- Time constraints for initial model development

**Potential Value:**
- More recent data (8 years newer than Census 2011)
- Additional health and nutrition indicators
- Asset ownership updates

**Future Integration Plan:** High priority - Extract district-level indicators for model v2.0

---

### 4. **District GDP Data - India**

**Dataset Name:** District Domestic Product (DDP) Estimates

**Publisher:** Ministry of Statistics and Programme Implementation (MoSPI), Government of India

**Year:** Various years (2014-2019 based on filenames)

**File Location:** 
- `archive(10)/gdp_district_estimates_2014.csv`
- `archive(10)/gdp_district_estimates_2016.csv`
- Additional GDP files in archive(10)/

**Status:** ⚠️ **Available but not used** - Prioritized proxy-based approach

**Reason Not Used:**
- Initial model focused on feature-based prediction rather than direct GDP target
- Data availability/completeness for all 631 districts needs verification
- Chose Census 2011 data for consistency and completeness

**Potential Value:**
- Direct economic output measure (actual GDP instead of proxy index)
- Ground truth for validating proxy-based predictions
- Could serve as regression target instead of synthetic index

**Future Integration Plan:** **HIGH PRIORITY** - Use as target variable in next iteration to replace proxy-based index with actual economic data

---

### 5. **Nighttime Light Intensity Data**

**Dataset Name:** VIIRS Nighttime Light Data (Visible Infrared Imaging Radiometer Suite)

**Publisher:** NASA Earth Observations / NOAA National Centers for Environmental Information

**Year:** 2020-2023 (monthly/annual composites)

**Access Link:** https://eogdata.mines.edu/products/vnl/

**Status:** ❌ **Not used** - Requires geospatial processing

**Reason Not Used:**
- Requires georeferencing district boundaries to satellite raster data
- Complex preprocessing (cloud masking, atmospheric correction)
- Python geospatial libraries (rasterio, geopandas) needed
- Significant computational overhead

**Potential Value:**
- Proxy for economic activity and development
- Captures informal economy not reflected in official statistics
- Correlation with GDP demonstrated in literature

**Future Integration Plan:** Medium priority - Add as supplementary feature for urban districts

---

## 🔍 **Additional Data Sources (Explored but Not Viable)**

### 6. **PIN Code Level Data - India**

**Status:** ❌ **Not publicly available**

**Reason:** India Postal Service does not publish income, demographic, or economic statistics at PIN code level. Census data aggregated at district level only.

**Alternative:** Used district-level data as finest granularity available

---

### 7. **Tax Statistics - India**

**Status:** ⚠️ **Aggregate only**

**Reason:** Income Tax Department publishes only state-level aggregates, not district or PIN-level statistics due to privacy regulations

**Source Explored:** https://www.incometax.gov.in/iec/foportal/

---

## ⚖️ **Data Usage Compliance**

### Government Data Policy
All datasets used are:
- ✅ Publicly available government statistics
- ✅ Used for non-commercial academic research
- ✅ Properly attributed to original publishers
- ✅ Not redistributed in violation of terms

### Privacy Protection
- US IRS data: Pre-aggregated by IRS, no individual taxpayer information
- India Census: District-level aggregates, no household/individual records
- No personally identifiable information (PII) in any dataset

### Reproducibility
- All raw data files preserved in `data_raw/` (not tracked in git due to size)
- Processing scripts documented in `prepare_india_data.py` and `prepare_us_data.py`
- Exact data sources and access dates recorded in this document

---

## 📚 **References for Methodology**

### Academic Papers Cited
1. Henderson, J. V., Storeygard, A., & Weil, D. N. (2012). Measuring economic growth from outer space. *American Economic Review*, 102(2), 994-1028.
   - Justification for using nightlight data as economic proxy

2. Jean, N., Burke, M., Xie, M., Davis, W. M., Lobell, D. B., & Ermon, S. (2016). Combining satellite imagery and machine learning to predict poverty. *Science*, 353(6301), 790-794.
   - Multi-modal approaches to poverty estimation in developing countries

---

## 📧 **Data Access Notes**

### For Reproducibility

**US IRS Data:**
- Download: Visit IRS SOI website → "ZIP Code Data" → Select Tax Year 2022
- File: `22zpallagi.csv` (206 MB)
- No registration required

**India Census Data:**
- Option 1: Official Census of India website (requires navigation through multiple pages)
- Option 2: Kaggle dataset (single CSV download, requires free Kaggle account)
- File: `india-districts-census-2011.csv`

**Preprocessed Datasets:**
If you cannot access raw data sources, the preprocessed versions are available in `datasets/` folder after running:
```bash
python prepare_us_data.py
python prepare_india_data.py
```

---

## 📅 **Last Updated**

**Document Version:** 1.0  
**Date:** January 20, 2026  
**Maintained By:** Research Team

**Change Log:**
- 2026-01-20: Initial documentation of all sources
- Future updates will track dataset version changes and new source additions

---

## 📄 **How to Cite This Project**

If you use this project's methodology or code, please cite:

```bibtex
@misc{income_prediction_us_india_2026,
  title={Income Prediction Models: A Comparative Study of US and India},
  author={[Authors - Add Names]},
  year={2026},
  note={University Research Project},
  url={[GitHub URL]}
}
```

---

**End of Document**
