import pandas as pd
import numpy as np

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("tier1_avc_data.csv")

# ============================================================
# REMOVE UNUSED COLUMNS
# ============================================================

df = df.drop(columns=[
    "street_type",
    "weather",
    "speed_limit"
])

# ============================================================
# MAP MONTH
# ============================================================

month_map = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12
}

df["month"] = df["month"].map(month_map)

# ============================================================
# MAP DAY OF WEEK
# ============================================================

day_of_week_map = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}

df["Day of week of crash"] = df[
    "Day of week of crash"
].map(day_of_week_map)

# ============================================================
# HANDLE DROUGHT INDEX NaNs
# ============================================================

# Fill missing drought values using the mean of that specific month
# (e.g., a missing January value gets January's average drought index)
df["monthly_drought_index (0-100)"] = df.groupby("month")["monthly_drought_index (0-100)"].transform(
    lambda x: x.fillna(x.mean())
)

# Safety fallback: If an entire month happens to be missing in your data, 
# fill with the overall global average.
global_drought_mean = df["monthly_drought_index (0-100)"].mean()
df["monthly_drought_index (0-100)"] = df["monthly_drought_index (0-100)"].fillna(global_drought_mean)

# ============================================================
# CONVERT TIME TO HOUR & FIX NaNs
# ============================================================

# 1. Cast column to string and strip empty spaces
time_strings = df["time_of_crash"].astype(str).str.strip()

# 2. Fix the clean ":00" pattern by transforming it to midnight "00:00"
time_strings = time_strings.replace({":00": "00:00"})

# 3. Handle edge cases where it might be just "0" or "0.0"
time_strings = time_strings.replace({"0": "00:00", "0.0": "00:00"})

# 4. Parse dates (format="mixed" automatically reads both "1:01" and "13:05" cleanly)
parsed_times = pd.to_datetime(time_strings, format="mixed", errors="coerce")

# 5. Extract hours
df["time_of_crash"] = parsed_times.dt.hour

# 6. Safety check: If there are ANY leftover structural issues, fill with midnight (0)
df["time_of_crash"] = df["time_of_crash"].fillna(0)

# ============================================================
# CREATE CYCLICAL FEATURES
# ============================================================

# MONTH
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

# DAY OF WEEK
df["dow_sin"] = np.sin(
    2 * np.pi * df["Day of week of crash"] / 7
)

df["dow_cos"] = np.cos(
    2 * np.pi * df["Day of week of crash"] / 7
)

# HOUR
df["hour_sin"] = np.sin(
    2 * np.pi * df["time_of_crash"] / 24
)

df["hour_cos"] = np.cos(
    2 * np.pi * df["time_of_crash"] / 24
)

# ============================================================
# SAVE
# ============================================================

df.to_csv("tier1_encoded.csv", index=False)

print("Encoded dataset saved.")