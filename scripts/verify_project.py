"""
Project Verification Script
============================

This script checks the integrity of the project structure and verifies
that no hardcoded predictions exist in the codebase.

Run this before submission to ensure research authenticity.
"""

import os
import re
from pathlib import Path
import json


class ProjectVerifier:
    """Verifies project authenticity and completeness"""
    
    def __init__(self, project_root="."):
        self.root = Path(project_root)
        self.issues = []
        self.warnings = []
        self.success = []
        
    def verify_structure(self):
        """Check that required directories and files exist"""
        print("\n" + "="*60)
        print("VERIFYING PROJECT STRUCTURE")
        print("="*60)
        
        required_dirs = [
            "data_raw/us",
            "data_raw/india",
            "data_processed/us",
            "data_processed/india",
            "models/us",
            "models/india",
            "backend",
            "frontend"
        ]
        
        required_files = [
            "train_us.py",
            "train_india.py",
            "backend/app.py",
            "frontend/index.html",
            "requirements.txt",
            "README.md",
            "DATASETS.md",
            ".gitignore"
        ]
        
        for dir_path in required_dirs:
            full_path = self.root / dir_path
            if full_path.exists():
                self.success.append(f"✅ Directory exists: {dir_path}")
            else:
                self.issues.append(f"❌ Missing directory: {dir_path}")
        
        for file_path in required_files:
            full_path = self.root / file_path
            if full_path.exists():
                self.success.append(f"✅ File exists: {file_path}")
            else:
                self.issues.append(f"❌ Missing file: {file_path}")
    
    def check_for_hardcoded_predictions(self):
        """Scan code for suspicious hardcoded prediction patterns"""
        print("\n" + "="*60)
        print("CHECKING FOR HARDCODED PREDICTIONS")
        print("="*60)
        
        # Patterns that indicate hardcoded predictions
        suspicious_patterns = [
            (r'zipcode.*:\s*\{.*income.*:.*\d+', "ZIP to income dictionary"),
            (r'\{.*["\']1000[1-9].*["\'].*:.*\d{4,}', "Hardcoded ZIP predictions"),
            (r'if.*zipcode.*==.*return.*\d{4,}', "Conditional ZIP returns"),
            (r'district.*income.*=.*\{', "District to income mapping"),
            (r'PREDICTIONS\s*=\s*\{', "Prediction dictionary"),
        ]
        
        files_to_check = [
            "train_us.py",
            "train_india.py",
            "backend/app.py"
        ]
        
        for file_path in files_to_check:
            full_path = self.root / file_path
            if not full_path.exists():
                continue
                
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            found_issues = False
            for pattern, description in suspicious_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    self.issues.append(
                        f"⚠️  Suspicious pattern in {file_path}: {description}"
                    )
                    found_issues = True
            
            if not found_issues:
                self.success.append(
                    f"✅ No hardcoded predictions in {file_path}"
                )
    
    def check_model_artifacts(self):
        """Check if model artifacts exist (after training)"""
        print("\n" + "="*60)
        print("CHECKING MODEL ARTIFACTS")
        print("="*60)
        
        us_artifacts = [
            "models/us/best_model.pkl",
            "models/us/scaler.pkl",
            "models/us/feature_names.txt",
            "models/us/metadata.json"
        ]
        
        india_artifacts = [
            "models/india/regression_model.pkl",
            "models/india/classification_model.pkl",
            "models/india/scaler.pkl",
            "models/india/feature_names.txt",
            "models/india/metadata.json"
        ]
        
        us_trained = all((self.root / p).exists() for p in us_artifacts)
        india_trained = all((self.root / p).exists() for p in india_artifacts)
        
        if us_trained:
            self.success.append("✅ US model artifacts present")
            
            # Check metadata
            metadata_path = self.root / "models/us/metadata.json"
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.success.append(
                        f"   Model: {metadata.get('model_name', 'Unknown')}"
                    )
                    self.success.append(
                        f"   Test R²: {metadata.get('best_test_r2', 'N/A'):.4f}"
                    )
            except Exception as e:
                self.warnings.append(f"⚠️  Could not read US metadata: {e}")
        else:
            self.warnings.append(
                "⚠️  US model not trained yet. Run: python train_us.py"
            )
        
        if india_trained:
            self.success.append("✅ India model artifacts present")
            
            # Check metadata
            metadata_path = self.root / "models/india/metadata.json"
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    test_r2 = metadata.get('results', {}).get('regression', {}).get('test_r2', 'N/A')
                    self.success.append(f"   Test R²: {test_r2}")
            except Exception as e:
                self.warnings.append(f"⚠️  Could not read India metadata: {e}")
        else:
            self.warnings.append(
                "⚠️  India model not trained yet. Run: python train_india.py"
            )
    
    def check_datasets(self):
        """Check if datasets are in place"""
        print("\n" + "="*60)
        print("CHECKING DATASETS")
        print("="*60)
        
        us_data = self.root / "datasets/us_irs_zipcode_data.csv"
        india_data = self.root / "datasets/india_district_census_data.csv"
        
        if us_data.exists():
            self.success.append("✅ US dataset present")
            # Check size
            size_mb = us_data.stat().st_size / (1024 * 1024)
            self.success.append(f"   Size: {size_mb:.2f} MB")
        else:
            self.warnings.append(
                "⚠️  US dataset missing. Download IRS data and place at: datasets/us_irs_zipcode_data.csv"
            )
        
        if india_data.exists():
            self.success.append("✅ India dataset present")
            size_mb = india_data.stat().st_size / (1024 * 1024)
            self.success.append(f"   Size: {size_mb:.2f} MB")
        else:
            self.warnings.append(
                "⚠️  India dataset missing. Compile census data and place at: datasets/india_district_census_data.csv"
            )
    
    def verify_api_code(self):
        """Verify API uses model.predict() not hardcoded logic"""
        print("\n" + "="*60)
        print("VERIFYING API INFERENCE CODE")
        print("="*60)
        
        api_file = self.root / "backend/app.py"
        if not api_file.exists():
            self.issues.append("❌ API file not found")
            return
        
        with open(api_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for model loading
        if "joblib.load" in content:
            self.success.append("✅ API loads models from disk")
        else:
            self.issues.append("❌ API doesn't load models properly")
        
        # Check for model.predict calls
        if ".predict(" in content:
            self.success.append("✅ API uses model.predict() for inference")
        else:
            self.issues.append("❌ API doesn't use model.predict()")
        
        # Check for NO hardcoded dictionaries in prediction endpoints
        predict_us_match = re.search(
            r'async def predict_us.*?(?=async def|$)',
            content,
            re.DOTALL
        )
        
        if predict_us_match:
            predict_us_code = predict_us_match.group(0)
            if "{" in predict_us_code and "income" in predict_us_code.lower():
                # Check if it's just for response construction
                if "result['prediction']" in predict_us_code:
                    self.success.append(
                        "✅ US prediction endpoint uses model results"
                    )
                else:
                    self.warnings.append(
                        "⚠️  US prediction endpoint has dictionary - review manually"
                    )
    
    def run(self):
        """Run all verification checks"""
        print("\n" + "="*80)
        print("INCOME PREDICTION PROJECT VERIFICATION")
        print("Research Authenticity Check")
        print("="*80)
        
        self.verify_structure()
        self.check_for_hardcoded_predictions()
        self.check_datasets()
        self.check_model_artifacts()
        self.verify_api_code()
        
        # Print results
        print("\n" + "="*80)
        print("VERIFICATION RESULTS")
        print("="*80)
        
        if self.success:
            print("\n✅ SUCCESSES:")
            for item in self.success:
                print(f"  {item}")
        
        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for item in self.warnings:
                print(f"  {item}")
        
        if self.issues:
            print("\n❌ ISSUES:")
            for item in self.issues:
                print(f"  {item}")
        
        # Final verdict
        print("\n" + "="*80)
        if self.issues:
            print("❌ VERIFICATION FAILED - Fix issues before submission")
            print("="*80)
            return False
        elif self.warnings:
            print("⚠️  VERIFICATION PASSED WITH WARNINGS")
            print("Address warnings before final submission")
            print("="*80)
            return True
        else:
            print("✅ VERIFICATION PASSED - Project ready for research submission")
            print("="*80)
            return True


if __name__ == "__main__":
    verifier = ProjectVerifier()
    success = verifier.run()
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    
    if not success:
        print("\n1. Fix the issues listed above")
        print("2. Re-run this verification script")
        print("3. Review code manually for any hardcoded logic")
    else:
        print("\n1. Train models: python train_us.py && python train_india.py")
        print("2. Test API: cd backend && python app.py")
        print("3. Test frontend: Open frontend/index.html")
        print("4. Document results in README.md")
        print("5. Commit to git (models will be excluded via .gitignore)")
        print("6. Submit for research review")
    
    print("\n⚠️  RESEARCH INTEGRITY REMINDER:")
    print("   - All predictions must come from trained models")
    print("   - No hardcoded ZIP-to-income dictionaries")
    print("   - Document all limitations honestly")
    print("   - Results must be reproducible")
    print("\n")
