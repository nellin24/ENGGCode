import pandas as pd
import csv
import numpy as np
from datetime import datetime, timedelta

# TODO
# Decide what columns to keep
# Remove duplicates from dataset (if any?)
# Combine the datasets to form one dataset with shared column titles
# NaN values
# Remove data w/o coor


def convert_time_interval(interval):
    if not isinstance(interval, str):
        return None

    start_str = interval.split("-")[0].strip()

    if start_str.lower() == "midnight":
        start_str = "00:00"

    # Try multiple formats
    for fmt in ("%H:%M", "%H:%M:%S", "%I:%M", "%I:%M %p"):
        try:
            start = datetime.strptime(start_str, fmt)
            return (start + timedelta(hours=1)).strftime("%H:%M")
        except ValueError:
            continue

    # If nothing works, return None instead of crashing
    return None

# FILTER THE 2016-2020 DATASET

input_file = "2016-2020_crash.csv"
output_file = "2016-2020-filtered.csv"

column_name = "RUM - code"
value_to_keep = "67"

with open(input_file, newline='') as infile, open(output_file, 'w', newline='') as outfile:
    reader = csv.DictReader(infile)
    writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)

    writer.writeheader()

    for row in reader:
        if row[column_name] == value_to_keep:
            writer.writerow(row)


# FILTER THE 2020-2024 DATASET


input_file = "2020-2024_crash.csv"
output_file = "2020-2024-filtered.csv"

column_name = "RUM - code"
value_to_keep = "67"

with open(input_file, newline='') as infile, open(output_file, 'w', newline='') as outfile:
    reader = csv.DictReader(infile)
    writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)

    writer.writeheader()

    for row in reader:
        if row[column_name] == value_to_keep:
            writer.writerow(row)

# Combine and remove duplicates from the two datasets
df1 = pd.read_csv("2016-2020-filtered.csv")
df2 = pd.read_csv("2020-2024-filtered.csv") 

df1.rename(columns={"Time of crash - Two-hour intervals": "Two-hour intervals"}, inplace=True)

columns_to_keep = ["Day of week of crash", "Year of crash", "Month of crash", "Two-hour intervals", "Street type", 
                   "Latitude", "Longitude", "LGA", "Alignment",
                   "Surface condition", "Weather", "Speed limit", "Key TU type"]

df1_eq = df1[columns_to_keep]
df2_eq = df2[columns_to_keep]

combined = pd.concat([df1_eq, df2_eq]).drop_duplicates()
combined.to_csv("combined.csv", index=False)

final_df = pd.read_csv("combined.csv")

# Convert time intervals to single time value (1 hour from start)

final_df["Two-hour intervals"] = final_df["Two-hour intervals"].apply(convert_time_interval)
final_df.rename(columns={"Two-hour intervals": "time_of_crash", "Month of crash": "month", "Year of crash": "year", "Street type": "street_type", "Latitude": "latitude", "Longitude": "longitude", "Surface condition": "surface_condition", "Weather": "weather", "Speed limit": "speed_limit"}, inplace=True)

final_df.to_csv("2016-2024.csv", index=False)


