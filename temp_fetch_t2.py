import pandas as pd
import numpy as np
import requests
import time

# ============================================================
# LOAD YOUR DATA
# ============================================================

df = pd.read_csv("tier_2_final_upd.csv")
print(f"Loaded {len(df)} rows")

# ============================================================
# ROUND COORDS (optional but good for dedup)
# ============================================================

def round_coord(x):
    return np.round(x, 2)

df["lat_r"] = round_coord(df["latitude"])
df["lon_r"] = round_coord(df["longitude"])

locations = df[["lat_r", "lon_r"]].drop_duplicates()
print(f"Unique locations: {len(locations)}")

# ============================================================
# FETCH FUNCTION (OPEN-METEO)
# ============================================================

def fetch_weather(lat, lon):
    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": "2021-01-01",
        "end_date": "2023-12-31",
        "daily": "temperature_2m_max,temperature_2m_min",
        "timezone": "auto"
    }

    try:
        r = requests.get(url, params=params, timeout=30)
        data = r.json()

        if "daily" not in data:
            return None

        df_w = pd.DataFrame(data["daily"])

        df_w["date"] = pd.to_datetime(df_w["time"])
        df_w["year"] = df_w["date"].dt.year
        df_w["month"] = df_w["date"].dt.month

        df_w["mean_temp"] = (
            df_w["temperature_2m_max"] + df_w["temperature_2m_min"]
        ) / 2

        monthly = (
            df_w.groupby(["year", "month"])["mean_temp"]
            .mean()
            .reset_index()
            .rename(columns={"mean_temp": "Monthly_Temperature_Average"})
        )

        monthly["lat_r"] = lat
        monthly["lon_r"] = lon

        return monthly

    except Exception:
        return None

# ============================================================
# FETCH ALL (parallel optional later)
# ============================================================

climate_records = []
failed = []

for i, (_, row) in enumerate(locations.iterrows()):
    lat, lon = row["lat_r"], row["lon_r"]

    result = fetch_weather(lat, lon)

    if result is not None:
        climate_records.append(result)
    else:
        failed.append((lat, lon))

    if (i + 1) % 50 == 0:
        print(f"Processed {i+1}/{len(locations)}")

    time.sleep(0.2)  # very light rate limiting

print(f"\n✓ Success: {len(climate_records)}")
print(f"⚠️ Failed: {len(failed)}")

# ============================================================
# COMBINE
# ============================================================

climate_df = pd.concat(climate_records, ignore_index=True)

# ============================================================
# MERGE BACK (same as your pipeline)
# ============================================================

df = df.merge(
    climate_df,
    left_on=["lat_r", "lon_r", "year", "month"],
    right_on=["lat_r", "lon_r", "year", "month"],
    how="left"
)

# ============================================================
# SAVE
# ============================================================

df.to_csv("tier2_with_temperature.csv", index=False)
print("\n✓ DONE")