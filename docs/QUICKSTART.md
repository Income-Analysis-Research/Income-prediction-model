# 🚀 QUICK START GUIDE

**Get up and running in 10 minutes**

---

## ✅ Prerequisites

- Python 3.8+ installed
- 8GB RAM minimum
- 2GB disk space
- Internet connection (for dataset download)

---

## 📥 Step 1: Install Dependencies (2 minutes)

```bash
# Navigate to project directory
cd c:\Users\versu\Github\newresearchproj

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On Mac/Linux

# Install required packages
pip install -r requirements.txt
```

Expected output: All packages installed successfully

---

## 📊 Step 2: Download Datasets (Manual - see DATASETS.md)

### US Dataset:

1. Visit: https://www.irs.gov/statistics/soi-tax-stats-individual-income-tax-statistics-zip-code-data-soi
2. Download latest year CSV
3. Save as: `datasets/us_irs_zipcode_data.csv`

### India Dataset:

Since India data must be compiled from multiple sources, see [DATASETS.md](DATASETS.md) for detailed instructions.

Place compiled data as: `datasets/india_district_census_data.csv`

Alternatively, create a sample dataset for testing (see bottom of this guide).

---

## 🎯 Step 3: Train Models

### Train US Model (~5 minutes):

```bash
python train_us.py
```

**Expected output:**
```
============================================================
🎯 US INCOME PREDICTION MODEL TRAINING PIPELINE
============================================================
📁 LOADING DATA
✅ Loaded XX,XXX records
⚙️  PREPROCESSING & FEATURE ENGINEERING
✅ Valid records after filtering: XX,XXX
🚀 TRAINING MODELS
📊 Training Linear Regression...
📊 Training Random Forest...
📊 Training XGBoost...
🏆 BEST MODEL: [Model Name]
💾 SAVING MODEL ARTIFACTS
✅ ALL ARTIFACTS SAVED SUCCESSFULLY
```

**Artifacts created:**
- `models/us/best_model.pkl`
- `models/us/scaler.pkl`
- `models/us/feature_names.txt`
- `models/us/metadata.json`
- `models/us/results_summary.txt`

### Train India Model (~3 minutes):

```bash
python train_india.py
```

**Artifacts created:**
- `models/india/regression_model.pkl`
- `models/india/classification_model.pkl`
- `models/india/scaler.pkl`
- And other supporting files

---

## 🌐 Step 4: Start Backend API

```bash
cd backend
python app.py
```

**Expected output:**
```
============================================================
🚀 STARTING INCOME PREDICTION API
============================================================
✅ US model loaded successfully
✅ India model loaded successfully
============================================================

INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Test the API:**

Open browser and visit:
- Health check: http://localhost:8000/health
- API docs: http://localhost:8000/docs

**Or use curl:**
```bash
# Test US prediction
curl -X POST "http://localhost:8000/predict/us" \
  -H "Content-Type: application/json" \
  -d "{\"zipCode\": \"10001\", \"method\": \"ml\"}"

# Test India prediction
curl -X POST "http://localhost:8000/predict/india" \
  -H "Content-Type: application/json" \
  -d "{\"state\": \"Maharashtra\", \"district\": \"Mumbai\", \"method\": \"ml\"}"
```

---

## 💻 Step 5: Launch Frontend

**Option A: Direct file open**
1. Navigate to `frontend/` folder
2. Double-click `index.html`

**Option B: HTTP Server (recommended)**
```bash
cd frontend
python -m http.server 3000
```
Then visit: http://localhost:3000

**Frontend Usage:**
1. Select region (US or India)
2. Enter ZIP code (US) or State/District (India)
3. Click "Get Prediction"
4. View results

---

## ✅ Step 6: Verify Everything Works

Run the verification script:
```bash
python verify_project.py
```

Should show all green checkmarks ✅

---

## 🎓 For Research Paper

### Document Your Results:

1. **Check training results:**
   - US: `models/us/results_summary.txt`
   - India: `models/india/results_summary.txt`

2. **Update README.md Results section:**
   - Fill in the "TBD" placeholders with actual metrics
   - Add your training date
   - Document dataset details

3. **Cite data sources:**
   - Document exact IRS data year used
   - Document India data sources and compilation date
   - Update DATA_SOURCES.md files

4. **Run reproducibility test:**
   - Delete models folder
   - Re-run training scripts
   - Verify results are similar (small variation is normal)

---

## 🐛 Troubleshooting

### Issue: "Dataset not found"
**Solution:** Download datasets per DATASETS.md instructions

### Issue: "XGBoost/LightGBM not available"
**Solution:** This is OK - Random Forest will be used as best model

### Issue: "Model not loaded" in API
**Solution:** Ensure models are trained first with train_us.py and train_india.py

### Issue: "CORS error" in frontend
**Solution:** 
- Make sure backend is running on port 8000
- Use HTTP server for frontend (not file://)

### Issue: "Out of memory"
**Solution:** Reduce dataset size or increase RAM

---

## 📝 Sample Dataset Creation (For Testing Only)

If you need to test the pipeline without real datasets:

### Create Sample US Data:

```python
import pandas as pd
import numpy as np

# Generate sample data
np.random.seed(42)
n = 1000

data = {
    'zipcode': [f"{i:05d}" for i in range(10001, 10001 + n)],
    'agi_amount': np.random.uniform(50000000, 500000000, n),
    'num_returns': np.random.randint(500, 5000, n),
    'state': np.random.choice(['NY', 'CA', 'TX', 'FL'], n)
}

df = pd.DataFrame(data)
df.to_csv('datasets/us_irs_zipcode_data.csv', index=False)
print("Sample US dataset created!")
```

### Create Sample India Data:

```python
import pandas as pd
import numpy as np

np.random.seed(42)
states = ['Maharashtra', 'Karnataka', 'Tamil Nadu', 'Delhi']
districts_per_state = 5
n = len(states) * districts_per_state

data = {
    'state': [state for state in states for _ in range(districts_per_state)],
    'district': [f"District_{i}" for i in range(n)],
    'literacy_rate': np.random.uniform(60, 95, n),
    'worker_participation': np.random.uniform(30, 60, n),
    'urban_ratio': np.random.uniform(20, 80, n),
    'avg_household_size': np.random.uniform(3.5, 5.5, n),
    'asset_score': np.random.uniform(40, 90, n),
    'electricity_access': np.random.uniform(70, 100, n),
    'water_access': np.random.uniform(60, 100, n),
    'sanitation_access': np.random.uniform(50, 95, n),
    'nightlight_mean': np.random.uniform(1, 50, n)
}

df = pd.DataFrame(data)
df.to_csv('datasets/india_district_census_data.csv', index=False)
print("Sample India dataset created!")
```

**⚠️ WARNING:** These are synthetic datasets for testing only.  
**DO NOT USE FOR RESEARCH PAPER.** Use real official datasets.

---

## 🎯 Next Steps After Setup

1. **Exploratory Analysis:**
   - Use Jupyter notebooks in `notebooks/` folder
   - Visualize data distributions
   - Analyze feature importance

2. **Hyperparameter Tuning:**
   - Modify training scripts
   - Experiment with different model parameters
   - Document improvements

3. **Deployment (Optional):**
   - Deploy backend to cloud (Azure/AWS/GCP)
   - Host frontend on GitHub Pages
   - Set up CI/CD pipeline

4. **Paper Writing:**
   - Use README.md as template
   - Add visualizations
   - Document all experiments
   - Include limitations section

---

## 📞 Getting Help

**Check these in order:**

1. **README.md** - Comprehensive documentation
2. **DATASETS.md** - Dataset download/compilation
3. **verify_project.py** - Automated verification
4. **Training script output** - Error messages
5. **GitHub Issues** - Report problems

---

**🎉 Congratulations! Your research project is ready.**

**Remember:**
- ✅ Use real datasets from official sources
- ✅ Train models from scratch
- ✅ Document all limitations
- ✅ No hardcoded predictions
- ✅ Reproducible results

**Good luck with your research paper! 📊📝**
