import os
import pandas as pd

# ==========================================
# SETUP CONFIGURATION PATHS
# ==========================================

TIER1_PATH = "master_tier1_training.csv"
TIER2_PATH = "master_tier2_training.csv"

print("Loading master training data arrays...")
df_tier1 = pd.read_csv(TIER1_PATH)
df_tier2 = pd.read_csv(TIER2_PATH)


# EXTRACT UNIQUE COORDINATES

print("Extracting coordinates...")

# 1. Pull latitude and longitude columns from both files
coords_t1 = df_tier1[['latitude', 'longitude', 'year']]
coords_t2 = df_tier2[['latitude', 'longitude', 'year']]

# 2. Stack them on top of each other
combined_coords = pd.concat([coords_t1, coords_t2], ignore_index=True)

# 3. Drop exact duplicates so you aren't wasting Earth Engine compute cycles
unique_coords = combined_coords.drop_duplicates().reset_index(drop=True)

# 4. Generate a clean, unique ID column starting from 0 (or 1)
unique_coords['point_index'] = unique_coords.index

# Rearrange columns for a tidy layout
unique_coords = unique_coords[['point_index', 'latitude', 'longitude', 'year']]

# ==========================================
# EXPORT TO GEE READY CSV
# ==========================================
output_path = "gee_coors.csv"
unique_coords.to_csv(output_path, index=False)

print("\n==========================================")
print(f"SUCCESS: Extracted {len(unique_coords)} unique coordinate points.")
print(f"Saved: {output_path}")
print("==========================================")