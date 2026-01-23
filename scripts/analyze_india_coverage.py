"""
Analyze India Dataset Coverage
"""
import pandas as pd

df = pd.read_csv('datasets/india_district_census_data.csv')

print("=" * 80)
print("CURRENT INDIA DATASET COVERAGE ANALYSIS")
print("=" * 80)

print(f"\n📊 Total districts: {len(df)}")
print(f"📍 Total states/UTs: {df['state'].nunique()}")

print("\n" + "=" * 80)
print("DISTRICTS PER STATE (All States)")
print("=" * 80)

state_counts = df['state'].value_counts().sort_values(ascending=False)
for state, count in state_counts.items():
    print(f"{state:35s}: {count:3d} districts")

print("\n" + "=" * 80)
print("MAJOR STATES COVERAGE CHECK")
print("=" * 80)

# Top 20 most populous states (2011 Census)
major_states = {
    'UTTAR PRADESH': 'Most populous (200M+)',
    'MAHARASHTRA': '2nd most populous (112M+)',
    'BIHAR': '3rd most populous (104M+)',
    'WEST BENGAL': '4th most populous (91M+)',
    'MADHYA PRADESH': '5th most populous (72M+)',
    'TAMIL NADU': '6th most populous (72M+)',
    'RAJASTHAN': '7th most populous (68M+)',
    'KARNATAKA': '8th most populous (61M+)',
    'GUJARAT': '9th most populous (60M+)',
    'ANDHRA PRADESH': '10th most populous (49M+)',
    'ODISHA': 'Eastern state (42M+)',
    'TELANGANA': 'New state (35M+)',
    'KERALA': 'Southern state (33M+)',
    'JHARKHAND': 'Eastern state (33M+)',
    'ASSAM': 'Northeast (31M+)',
    'PUNJAB': 'Northern state (28M+)',
    'CHHATTISGARH': 'Central state (25M+)',
    'HARYANA': 'Northern state (25M+)',
    'DELHI': 'Capital (17M+)',
    'JAMMU AND KASHMIR': 'Northern UT (13M+)'
}

print("\nState                               Districts  Status   Notes")
print("-" * 80)
for state, description in major_states.items():
    count = len(df[df['state'] == state])
    status = "✅ Good" if count >= 10 else "⚠️ Low" if count > 0 else "❌ Missing"
    print(f"{state:35s} {count:3d}       {status:10s} {description}")

print("\n" + "=" * 80)
print("COVERAGE SUMMARY")
print("=" * 80)

total_major_districts = sum(len(df[df['state'] == state]) for state in major_states.keys())
print(f"\nDistricts from 20 major states: {total_major_districts} / {len(df)} ({total_major_districts/len(df)*100:.1f}%)")

missing_states = [state for state in major_states.keys() if len(df[df['state'] == state]) == 0]
low_coverage = [state for state in major_states.keys() if 0 < len(df[df['state'] == state]) < 10]

if missing_states:
    print(f"\n❌ Missing major states: {len(missing_states)}")
    for state in missing_states:
        print(f"   - {state}")
        
if low_coverage:
    print(f"\n⚠️ Low coverage states (<10 districts): {len(low_coverage)}")
    for state in low_coverage:
        count = len(df[df['state'] == state])
        print(f"   - {state}: {count} districts")

print("\n" + "=" * 80)
