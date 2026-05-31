import pandas as pd

# ============================================================
# LOAD DATA
# ============================================================

ndvi_data = pd.read_csv("master_tier1_with_ndvi.csv")

drought_data = pd.read_csv(
    "master_tier1_training_with_monthly_drought_index (1).csv"
)

temperature_data = pd.read_csv(
    "master_tier1_training_average temperature(new).csv"
)

# ============================================================
# ROUND COORDINATES
# ============================================================

# 4 decimal places ≈ 11 metres
# Good for environmental joins

for df in [ndvi_data, drought_data, temperature_data]:

    df["lat_round"] = df["latitude"].round(4)
    df["lon_round"] = df["longitude"].round(4)

# ============================================================
# MERGE DROUGHT
# ============================================================

merged_data = pd.merge(

    ndvi_data,

    drought_data[
        [
            "lat_round",
            "lon_round",
            "year",
            "month",
            "monthly_drought_index (0-100)"
        ]
    ],

    on=[
        "lat_round",
        "lon_round",
        "year",
        "month"
    ],

    how="left"
)

# ============================================================
# MERGE TEMPERATURE
# ============================================================

merged_data = pd.merge(

    merged_data,

    temperature_data[
        [
            "lat_round",
            "lon_round",
            "year",
            "month",
            "Monthly_Temperature_Average"
        ]
    ],

    on=[
        "lat_round",
        "lon_round",
        "year",
        "month"
    ],

    how="left"
)

# ============================================================
# CLEANUP
# ============================================================

merged_data = merged_data.drop(
    columns=["lat_round", "lon_round"]
)

# ============================================================
# SAVE
# ============================================================

merged_data.to_csv(
    "tier1_avc_data.csv",
    index=False
)

print("Data merged successfully.")