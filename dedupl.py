import pandas as pd

# 1. Load your datasets (replace with your actual file paths)
df_gold = pd.read_csv("2021-2023-filtered.csv")   # The 650 points
df_macro = pd.read_csv("2016-2024.csv")  # The 2,500 points

# ==========================================
# STEP 2: Normalize the 2021-23 (Gold) Data
# ==========================================
# Convert time_of_crash to string, pad with zeros (e.g., '500' -> '0500')
df_gold['time_str'] = df_gold['time_of_crash'].astype(str).str.zfill(4)
# Extract the first two characters as the hour integer
df_gold['hour_match'] = df_gold['time_str'].str[:2].astype(int)

# Normalize coordinates
df_gold['lat_match'] = df_gold['latitude'].round(4)
df_gold['lon_match'] = df_gold['longitude'].round(4)


# ==========================================
# STEP 3: Normalize the 2016-24 (Macro) Data
# ==========================================
# Split '05:00' by the colon and take the first part (the hour)
df_macro['hour_match'] = df_macro['time_of_crash'].astype(str).str.split(':').str[0].astype(int)

# Normalize coordinates
df_macro['lat_match'] = df_macro['latitude'].round(4)
df_macro['lon_match'] = df_macro['longitude'].round(4)


# ==========================================
# STEP 4: Identify and Filter the Overlap
# ==========================================
# Define our standardized matching columns
match_keys = ['lat_match', 'lon_match', 'year', 'hour_match']

# Find the overlapping rows to see how many match
overlap = df_macro.merge(df_gold[match_keys], on=match_keys, how='inner')
print(f"Identified {len(overlap)} overlapping records between the datasets.")

# Perform an outer/left merge with an indicator to drop the matches from Macro
df_macro_cleaned = df_macro.merge(
    df_gold[match_keys].drop_duplicates(), 
    on=match_keys, 
    how='left', 
    indicator=True
)

# Keep only the rows that exist solely in the Macro dataset
df_macro_cleaned = df_macro_cleaned[df_macro_cleaned['_merge'] == 'left_only']

# Drop the temporary helper columns used for matching
df_macro_cleaned = df_macro_cleaned.drop(columns=['lat_match', 'lon_match', 'hour_match', '_merge', 'LGA', 'Key TU type' , 'surface_condition', 'Alignment'])

df_macro_cleaned.to_csv("2016-2024-deduplicated.csv", index=False)


print(f"Deduplication complete.")
print(f"Remaining rows in Tier 1 Macro dataset: {len(df_macro_cleaned)}")