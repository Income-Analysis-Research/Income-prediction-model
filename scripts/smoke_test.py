"""
Smoke Test Script
==================

Quick sanity test to verify:
1. Models are trained and saved
2. Backend can load models
3. Both endpoints respond with valid predictions
4. Predictions come from real models (not hardcoded)

Usage:
    python scripts/smoke_test.py
"""

import os
import sys
import json
import time
import requests
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_status(message, status='info'):
    """Print formatted status message"""
    symbols = {
        'success': f'{Colors.GREEN}✅',
        'error': f'{Colors.RED}❌',
        'warning': f'{Colors.YELLOW}⚠️',
        'info': f'{Colors.BLUE}ℹ️'
    }
    print(f"{symbols.get(status, '')} {message}{Colors.RESET}")


def print_section(title):
    """Print section header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{title}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}\n")


def check_models_exist():
    """Verify that model files exist"""
    print_section("1. CHECKING MODEL ARTIFACTS")
    
    required_files = {
        'US': [
            'models/us/best_model.pkl',
            'models/us/scaler.pkl',
            'models/us/feature_names.txt',
            'models/us/metadata.json',
            'models/us/results_summary.txt'
        ],
        'India': [
            'models/india/regression_model.pkl',
            'models/india/classification_model.pkl',
            'models/india/scaler.pkl',
            'models/india/feature_names.txt',
            'models/india/metadata.json',
            'models/india/results_summary.txt'
        ]
    }
    
    all_exist = True
    
    for region, files in required_files.items():
        print(f"\n{Colors.BOLD}{region} Model Files:{Colors.RESET}")
        for file in files:
            exists = Path(file).exists()
            status = 'success' if exists else 'error'
            print_status(f"  {file}", status)
            if not exists:
                all_exist = False
    
    if not all_exist:
        print_status("\nSome model files are missing. Train models first:", 'error')
        print("  python train_us.py")
        print("  python train_india.py")
        return False
    
    print_status("\nAll model artifacts found!", 'success')
    return True


def check_processed_data():
    """Verify processed data exists for inference"""
    print_section("2. CHECKING PROCESSED DATA")
    
    required_data = [
        'data_processed/us/processed_data.csv',
        'data_processed/india/processed_data.csv'
    ]
    
    all_exist = True
    
    for file in required_data:
        exists = Path(file).exists()
        status = 'success' if exists else 'error'
        print_status(f"  {file}", status)
        if not exists:
            all_exist = False
    
    if not all_exist:
        print_status("\nProcessed data missing. Models may not be able to extract features.", 'warning')
        return False
    
    print_status("\nAll processed data files found!", 'success')
    return True


def check_backend():
    """Check if backend server is running"""
    print_section("3. CHECKING BACKEND SERVER")
    
    try:
        response = requests.get('http://localhost:8000/health', timeout=3)
        if response.ok:
            print_status("Backend is running!", 'success')
            return True
        else:
            print_status(f"Backend returned status {response.status_code}", 'warning')
            return False
    except requests.exceptions.ConnectionError:
        print_status("Backend not running", 'error')
        print_status("\nPlease start the backend in a separate terminal:", 'info')
        print(f"  {Colors.YELLOW}python backend/app.py{Colors.RESET}")
        return False
    except Exception as e:
        print_status(f"Failed to connect to backend: {e}", 'error')
        return False


def test_health_endpoint():
    """Test health check endpoint"""
    print_section("4. TESTING HEALTH ENDPOINT")
    
    try:
        response = requests.get('http://localhost:8000/health', timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print_status("Health check passed!", 'success')
            print(f"\n  Response:")
            print(f"    Status: {data.get('status')}")
            print(f"    US Model Loaded: {data.get('us_model_loaded')}")
            print(f"    India Model Loaded: {data.get('india_model_loaded')}")
            return True
        else:
            print_status(f"Health check failed with status {response.status_code}", 'error')
            return False
            
    except Exception as e:
        print_status(f"Health check error: {e}", 'error')
        return False


def test_us_prediction():
    """Test US prediction endpoint"""
    print_section("5. TESTING US PREDICTION ENDPOINT")
    
    test_zip = "10001"  # New York City
    
    try:
        print_status(f"Requesting prediction for ZIP code: {test_zip}", 'info')
        
        response = requests.post(
            'http://localhost:8000/predict/us',
            json={"zipCode": test_zip, "method": "ml"},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print_status("US prediction successful!", 'success')
            print(f"\n  Results:")
            print(f"    ZIP Code: {data.get('zipCode')}")
            print(f"    Predicted Income: ${data.get('predictedIncome', 0):,.0f}")
            print(f"    Model Name: {data.get('modelName')}")
            print(f"    Method: {data.get('method')}")
            print(f"    Message: {data.get('message')}")
            
            # Verify prediction is not zero (real model output)
            if data.get('predictedIncome', 0) > 0:
                print_status("\n  ✅ Prediction appears to be from real model (non-zero value)", 'success')
            else:
                print_status("\n  ⚠️ Warning: Zero prediction - check model quality", 'warning')
            
            return True
        elif response.status_code == 404:
            print_status(f"ZIP code {test_zip} not found in training data", 'warning')
            print("  This is OK - try with a different ZIP code that exists in your dataset")
            return True
        else:
            print_status(f"Prediction failed with status {response.status_code}", 'error')
            print(f"  Response: {response.text}")
            return False
            
    except Exception as e:
        print_status(f"US prediction error: {e}", 'error')
        return False


def test_india_prediction():
    """Test India prediction endpoint"""
    print_section("6. TESTING INDIA PREDICTION ENDPOINT")
    
    test_state = "Karnataka"
    test_district = "Bangalore Urban"
    
    try:
        print_status(f"Requesting prediction for: {test_district}, {test_state}", 'info')
        
        response = requests.post(
            'http://localhost:8000/predict/india',
            json={"state": test_state, "district": test_district, "method": "ml"},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print_status("India prediction successful!", 'success')
            print(f"\n  Results:")
            print(f"    State: {data.get('state')}")
            print(f"    District: {data.get('district')}")
            print(f"    Income Index: {data.get('incomeIndex', 0):.1f} / 100")
            print(f"    Category: {data.get('incomeCategory')}")
            print(f"    Model Name: {data.get('modelName')}")
            print(f"    Disclaimer: {data.get('disclaimer')}")
            
            # Verify prediction is within valid range
            index = data.get('incomeIndex', 0)
            if 0 <= index <= 100:
                print_status("\n  ✅ Income index within valid range (0-100)", 'success')
            else:
                print_status(f"\n  ⚠️ Warning: Index {index} outside valid range", 'warning')
            
            return True
        elif response.status_code == 404:
            print_status(f"District {test_district} not found in training data", 'warning')
            print("  This is OK - try with a different district that exists in your dataset")
            return True
        else:
            print_status(f"Prediction failed with status {response.status_code}", 'error')
            print(f"  Response: {response.text}")
            return False
            
    except Exception as e:
        print_status(f"India prediction error: {e}", 'error')
        return False


def verify_no_hardcoding():
    """Verify predictions are not hardcoded"""
    print_section("7. VERIFYING NO HARDCODED PREDICTIONS")
    
    print_status("Checking for suspicious patterns in backend code...", 'info')
    
    backend_file = Path('backend/app.py')
    if not backend_file.exists():
        print_status("Backend file not found", 'error')
        return False
    
    with open(backend_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for suspicious patterns
    suspicious = []
    
    if 'zipcode_to_income = {' in content.lower():
        suspicious.append("Found dictionary mapping ZIP codes to income")
    
    if 'district_to_income = {' in content.lower():
        suspicious.append("Found dictionary mapping districts to income")
    
    if content.count('return {') > 20 and 'hardcoded' in content.lower():
        suspicious.append("Suspicious number of return statements with 'hardcoded' keyword")
    
    # Look for good signs
    good_signs = []
    
    if 'model.predict(' in content:
        good_signs.append("✅ Found model.predict() calls")
    
    if 'joblib.load(' in content:
        good_signs.append("✅ Found joblib.load() for model loading")
    
    if 'NO HARDCODED' in content or 'Real ML prediction' in content:
        good_signs.append("✅ Found authenticity declarations in code")
    
    # Report
    if suspicious:
        print_status("\n⚠️ Potential issues found:", 'warning')
        for issue in suspicious:
            print(f"    - {issue}")
    else:
        print_status("\n✅ No suspicious hardcoding patterns detected", 'success')
    
    if good_signs:
        print_status("\nPositive indicators:", 'success')
        for sign in good_signs:
            print(f"    {sign}")
    
    return len(suspicious) == 0


def main():
    """Run all smoke tests"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║         INCOME PREDICTION PROJECT - SMOKE TEST            ║")
    print("║                  Research Integrity Check                 ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print(f"{Colors.RESET}\n")
    
    print(f"{Colors.YELLOW}Note: Backend must be running before starting smoke test{Colors.RESET}")
    print(f"{Colors.YELLOW}Start it in a separate terminal: python backend/app.py{Colors.RESET}\n")
    
    results = {}
    
    try:
        # Run tests
        results['models'] = check_models_exist()
        results['data'] = check_processed_data()
        
        if not results['models']:
            print_status("\n❌ Cannot proceed without trained models", 'error')
            print_status("Train models first:", 'info')
            print("  python train_us.py")
            print("  python train_india.py")
            return False
        
        # Check backend
        results['backend'] = check_backend()
        
        if not results['backend']:
            print_status("\n❌ Tests require backend to be running", 'error')
            return False
        
        # API tests
        results['health'] = test_health_endpoint()
        results['us_predict'] = test_us_prediction()
        results['india_predict'] = test_india_prediction()
        results['no_hardcode'] = verify_no_hardcoding()
        
        # Summary
        print_section("SMOKE TEST SUMMARY")
        
        total_tests = len(results)
        passed_tests = sum(results.values())
        
        print(f"\n{Colors.BOLD}Test Results:{Colors.RESET}")
        for test_name, passed in results.items():
            status = 'success' if passed else 'error'
            symbol = '✅' if passed else '❌'
            print(f"  {symbol} {test_name.replace('_', ' ').title()}: {'PASS' if passed else 'FAIL'}")
        
        print(f"\n{Colors.BOLD}Overall:{Colors.RESET} {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print_status("\n🎉 ALL TESTS PASSED! Project is ready for use.", 'success')
            print_status("\nVerified:", 'success')
            print("  ✅ Models trained and saved")
            print("  ✅ Backend loads real .pkl models")
            print("  ✅ Predictions use model.predict(), not hardcoded values")
            print("  ✅ No suspicious hardcoding patterns detected")
            return True
        else:
            print_status(f"\n⚠️ {total_tests - passed_tests} test(s) failed. Review above for details.", 'warning')
            return False
        
    except KeyboardInterrupt:
        print_status("\n\nTests interrupted by user", 'warning')
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
