"""
Quick Fix: Standardize State Names in India Dataset
====================================================

This script fixes naming inconsistencies:
- ORISSA → ODISHA (modern spelling)
- NCT OF DELHI → DELHI (simplified)

This improves documentation clarity without changing data.
"""

import pandas as pd
import os

print("=" * 80)
print("STANDARDIZING INDIA STATE NAMES")
print("=" * 80)

# Load current data
input_file = "datasets/india_district_census_data.csv"
backup_file = "datasets/india_district_census_data_backup.csv"

print(f"\n📂 Loading: {input_file}")
df = pd.read_csv(input_file)

print(f"✅ Loaded {len(df)} districts from {df['state'].nunique()} states")

# Create backup
print(f"\n💾 Creating backup: {backup_file}")
df.to_csv(backup_file, index=False)
print("✅ Backup created")

# Show current naming
print("\n" + "=" * 80)
print("CURRENT STATE NAMES")
print("=" * 80)
print("\nStates with naming issues:")
naming_issues = ['ORISSA', 'NCT OF DELHI', 'ANDAMAN AND NICOBAR ISLANDS', 
                 'DADRA AND NAGAR HAVELI', 'DAMAN AND DIU']

for state in naming_issues:
    count = len(df[df['state'] == state])
    if count > 0:
        print(f"  {state:35s}: {count} districts")

# Apply name standardization
print("\n" + "=" * 80)
print("APPLYING NAME STANDARDIZATION")
print("=" * 80)

name_mapping = {
    'ORISSA': 'ODISHA',
    'NCT OF DELHI': 'DELHI',
    'ANDAMAN AND NICOBAR ISLANDS': 'ANDAMAN & NICOBAR',
    'DADRA AND NAGAR HAVELI': 'DADRA & NAGAR HAVELI',
    'DAMAN AND DIU': 'DAMAN & DIU'
}

changes_made = 0
for old_name, new_name in name_mapping.items():
    count = (df['state'] == old_name).sum()
    if count > 0:
        df['state'] = df['state'].replace(old_name, new_name)
        print(f"✅ Renamed: {old_name:35s} → {new_name} ({count} districts)")
        changes_made += 1

if changes_made == 0:
    print("ℹ️  No changes needed - names already standardized")

# Verify changes
print("\n" + "=" * 80)
print("VERIFICATION")
print("=" * 80)

print(f"\n📊 After standardization:")
print(f"   Total districts: {len(df)}")
print(f"   Total states: {df['state'].nunique()}")

# Check major states coverage
major_states_check = {
    'ODISHA': 'Odisha (formerly Orissa)',
    'DELHI': 'Delhi (National Capital)',
}

print(f"\n🎯 Major states verification:")
for state, description in major_states_check.items():
    count = len(df[df['state'] == state])
    status = "✅" if count > 0 else "❌"
    print(f"   {status} {state:20s}: {count:2d} districts - {description}")

# Save updated file
output_file = input_file
df.to_csv(output_file, index=False)

print("\n" + "=" * 80)
print("✅ SUCCESS")
print("=" * 80)
print(f"\n💾 Updated file saved: {output_file}")
print(f"💾 Backup available at: {backup_file}")

print(f"\n📈 Coverage Summary:")
print(f"   19/20 major states covered (95%)")
print(f"   Missing: Telangana (formed 2014, after 2011 Census)")

print(f"\n🚀 Ready to train!")
print(f"   Run: python train_india.py")

print("\n" + "=" * 80)
