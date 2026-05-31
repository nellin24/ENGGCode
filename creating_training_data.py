import os
import pandas as pd
import numpy as np

# ==========================================
# 1. SETUP CONFIGURATION PATHS
# ==========================================

# Real crash datasets
MACRO_REAL_PATH =  "2016-2024-deduplicated.csv"
GOLD_REAL_PATH = "2021-2023-filtered.csv"

# Generated pseudo-absence datasets
TIER1_ABS_PATH = "tier1_pseudo_absences.csv"
TIER2_ABS_PATH = "tier2_pseudo_absences.csv"

try:
    print("Loading data layers...")
    df_macro_real = pd.read_csv(MACRO_REAL_PATH)
    df_gold_real = pd.read_csv(GOLD_REAL_PATH)
    df_tier1_abs = pd.read_csv(TIER1_ABS_PATH)
    df_tier2_abs = pd.read_csv(TIER2_ABS_PATH)

    # Assign binary machine learning targets
    df_macro_real['target'] = 1
    df_gold_real['target'] = 1
    df_tier1_abs['target'] = 0
    df_tier2_abs['target'] = 0

    # ==========================================
    # 2. GENERATE TIER 1 MASTER FILE (2016-2024)
    # ==========================================
    print("\nProcessing Tier 1 Master File...")
    
    # Standardize time representation (Convert "05:00" to "5:00" if needed to match absence generation)
    df_macro_real['time_of_crash'] = df_macro_real['time_of_crash'].astype(str).str.lstrip('0')
    df_tier1_abs['time_of_crash'] = df_tier1_abs['time_of_crash'].astype(str).str.lstrip('0')
    # If the stripping leaves it empty (i.e., Midnight was "00:00"), fix it to "0:00"
    df_macro_real['time_of_crash'] = df_macro_real['time_of_crash'].replace('', '0:00')
    df_tier1_abs['time_of_crash'] = df_tier1_abs['time_of_crash'].replace('', '0:00')

    # Concatenate real data and absences
    # Unmatched columns (street_type, weather, speed_limit) will automatically initialize as NaN for absences
    df_master_tier1 = pd.concat([df_macro_real, df_tier1_abs], ignore_index=True)

    # Force uniform column sorting to mirror your clean schema exactly
    tier1_columns = [
        'Day of week of crash', 'year', 'month', 'time_of_crash', 'street_type', 
        'latitude', 'longitude', 'weather', 'speed_limit', 'target'
    ]
    df_master_tier1 = df_master_tier1[tier1_columns]
    
    # Save Tier 1 to disk
    tier1_output = "master_tier1_training.csv"
    df_master_tier1.to_csv(tier1_output, index=False)
    print(f"-> Saved: {tier1_output} ({len(df_master_tier1)} total rows)")

    # ==========================================
    # 3. GENERATE TIER 2 MASTER FILE (2021-2023)
    # ==========================================
    print("\nProcessing Tier 2 Master File...")
    
    # Concatenate real data and absences for Tier 2
    df_master_tier2 = pd.concat([df_gold_real, df_tier2_abs], ignore_index=True)

    # Force uniform column sorting to match your 2021-23 gold dataset
    # Leaving out attributes that don't belong here so your daily environmental markers can be added cleanly
    tier2_columns = [
        'latitude', 'longitude', 'year', 'date_of_crash', 'time_of_crash', 
        'street_type', 'speed_limit', 
        'weather', 'target'
    ]
    df_master_tier2 = df_master_tier2[tier2_columns]

    # Save Tier 2 to disk
    tier2_output = "master_tier2_training.csv"
    df_master_tier2.to_csv(tier2_output, index=False)
    print(f"-> Saved: {tier2_output} ({len(df_master_tier2)} total rows)")

    print("\n==========================================")
    print("SUCCESS: Combined Training files generated successfully!")
    print("==========================================")

except FileNotFoundError as e:
    print(f"\nCRITICAL ERROR: {e}")
except Exception as e:
    print(f"\nAn unexpected compilation error occurred: {e}")