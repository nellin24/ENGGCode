# Code for adding NaN values to the NDVI data where there are clouds or other obstructions in the satellite imagery.
import os
import numpy as np
import pandas as pd
# ==========================================
# PATH CONFIGURATION
print("Loading NDVI data...")
ndvi_2016 = pd.read_csv("NDVI_2016.csv")
ndvi_2017 = pd.read_csv("NDVI_2017.csv")
ndvi_2018 = pd.read_csv("NDVI_2018.csv")
ndvi_2019 = pd.read_csv("NDVI_2019.csv")
ndvi_2020 = pd.read_csv("NDVI_2020.csv")
ndvi_2021 = pd.read_csv("NDVI_2021.csv")
ndvi_2022 = pd.read_csv("NDVI_2022.csv")
ndvi_2023 = pd.read_csv("NDVI_2023.csv")
ndvi_2024 = pd.read_csv("NDVI_2024.csv")
# ==========================================

# Combine all the NDVI data into a single DataFrame
ndvi_data = pd.concat([ndvi_2016, ndvi_2017, ndvi_2018, ndvi_2019, ndvi_2020, ndvi_2021, ndvi_2022, ndvi_2023, ndvi_2024], ignore_index=True)

# Strip all columns except for Latitude,Longitude,Matched_Year,NDVI_Mean
ndvi_data = ndvi_data[["Latitude", "Longitude", "Matched_Year", "NDVI_Mean"]]

# Output the cleaned NDVI data to a new CSV file
ndvi_data.to_csv("Cleaned_NDVI.csv", index=False)
print("NDVI data cleaned and saved to Cleaned_NDVI.csv")
