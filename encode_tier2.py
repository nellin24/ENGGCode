# ============================================================
# TIER 2 DATA PREPARATION
# ============================================================
# Handles:
#   1. Time encoding (hour, day-of-week, month → sin/cos)
#   2. NDVI merge
#   3. KDE hotspot merge
# Output: tier2_prepared.csv — ready to feed into t2_model.py

import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from sklearn.preprocessing import MinMaxScaler

# ============================================================
# 1. LOAD RAW TIER 2 DATA
# ============================================================

df = pd.read_csv("tier_2_final_upd.csv")  
print(f"Loaded {len(df)} rows")


# ============================================================
# 2. MERGE NDVI (do this first, before any columns are dropped)
# ============================================================

ndvi = pd.read_csv("master_tier2_with_ndvi.csv")

# Both files are the same 3870 rows from the same source dataset.
# Direct assignment bypasses the lat/lon precision mismatch entirely.
df["NDVI_Mean"] = ndvi["NDVI_Mean"].values

print(f"✓ NDVI assigned — range: {df['NDVI_Mean'].min():.3f} – {df['NDVI_Mean'].max():.3f}")
print(f"   Missing: {df['NDVI_Mean'].isna().sum()}")

# ============================================================
# 2. PARSE DATE → MONTH + DAY-OF-WEEK
# ============================================================

df["date_parsed"] = pd.to_datetime(df["date_of_crash"], dayfirst=True, errors="coerce")
df["month"]       = df["date_parsed"].dt.month
df["dow"]         = df["date_parsed"].dt.dayofweek    # 0=Mon … 6=Sun

# Report any rows where date failed to parse
bad_dates = df["date_parsed"].isna().sum()
if bad_dates > 0:
    print(f"⚠️  {bad_dates} rows had unparseable dates — month/dow will be NaN for these rows")

df = df.drop(columns=["date_of_crash", "date_parsed"])

# ============================================================
# 3. PARSE TIME → HOUR
# ============================================================

# time_of_crash is stored as HHMM integers (e.g. 2100, 830, 100)
df["time_of_crash"] = pd.to_numeric(df["time_of_crash"], errors="coerce")
df["hour"]          = (df["time_of_crash"] // 100).clip(0, 23)

bad_times = df["hour"].isna().sum()
if bad_times > 0:
    print(f"⚠️  {bad_times} rows had unparseable times — hour will be NaN for these rows")

df = df.drop(columns=["time_of_crash"])

# ============================================================
# 4. CYCLICAL ENCODINGS
# ============================================================

df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
df["dow_sin"]   = np.sin(2 * np.pi * df["dow"]   / 7)
df["dow_cos"]   = np.cos(2 * np.pi * df["dow"]   / 7)
df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]  / 24)
df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]  / 24)

df = df.drop(columns=["month", "dow", "hour"])

print("✓ Cyclical time encodings added")

# ============================================================
# 6. MERGE KDE HOTSPOT
# ============================================================
# ============================================================
# LOAD FILES
# ============================================================

kde_df = pd.read_csv("carcass_density_lookup.csv")

# ============================================================
# PREPARE COORDINATES
# ============================================================

# Convert to radians for BallTree haversine distance

tier2_coords = np.radians(
    df[["latitude", "longitude"]].values
)

kde_coords = np.radians(
    kde_df[["latitude", "longitude"]].values
)

# ============================================================
# BUILD SPATIAL TREE
# ============================================================

tree = BallTree(
    kde_coords,
    metric='haversine'
)

# ============================================================
# FIND NEAREST KDE POINT
# ============================================================

distances, indices = tree.query(
    tier2_coords,
    k=1
)

# ============================================================
# ATTACH KDE VALUES
# ============================================================

# Extract the scores as a raw NumPy array BEFORE assigning to avoid index alignment issues
nearest_scores = kde_df["carcass_density_score"].iloc[indices.flatten()].to_numpy()

df["KDE_hotspot"] = nearest_scores

# ============================================================
# OPTIONAL: DISTANCE CHECK
# ============================================================

# Convert radians to meters

earth_radius = 6371000

df["distance_to_kde_m"] = (
    distances.flatten() * earth_radius
)

scaler = MinMaxScaler()

df["KDE_hotspot"] = scaler.fit_transform(
    df[["KDE_hotspot"]]
)

print("KDE successfully joined.")

# ---- END OF YOUR KDE CODE ----

if "KDE_hotspot" in df.columns:
    missing_kde = df["KDE_hotspot"].isna().sum()
    if missing_kde > 0:
        median_kde = df["KDE_hotspot"].median()
        df["KDE_hotspot"] = df["KDE_hotspot"].fillna(median_kde)
        print(f"⚠️  Imputed {missing_kde} missing KDE values with median ({median_kde:.4f})")
    print(f"✓ KDE merged — range: {df['KDE_hotspot'].min():.3f} – {df['KDE_hotspot'].max():.3f}")
else:
    print("⚠️  KDE_hotspot column not found — add your merge code at step 6")

# Drop unwanted columns

df = df.drop(columns=["street_type", "speed_limit", "weather"])

# ============================================================
# 7. FINAL CHECK
# ============================================================

print(f"\nFinal shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nMissing values per column:")
print(df.isnull().sum()[df.isnull().sum() > 0] if df.isnull().any().any() else "  None ✓")

# ============================================================
# 8. SAVE
# ============================================================

df.to_csv("tier2_prepared.csv", index=False)
print("\n✓ Saved: tier2_prepared.csv")