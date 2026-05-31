import pandas as pd
from sklearn.neighbors import BallTree
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# ============================================================
# LOAD FILES
# ============================================================

tier1_df = pd.read_csv("tier1_encoded.csv")

kde_df = pd.read_csv("carcass_density_lookup.csv")

# ============================================================
# PREPARE COORDINATES
# ============================================================

# Convert to radians for BallTree haversine distance

tier1_coords = np.radians(
    tier1_df[["latitude", "longitude"]].values
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
    tier1_coords,
    k=1
)

# ============================================================
# ATTACH KDE VALUES
# ============================================================

# Extract the scores as a raw NumPy array BEFORE assigning to avoid index alignment issues
nearest_scores = kde_df["carcass_density_score"].iloc[indices.flatten()].to_numpy()

tier1_df["KDE_hotspot"] = nearest_scores

# ============================================================
# OPTIONAL: DISTANCE CHECK
# ============================================================

# Convert radians to meters

earth_radius = 6371000

tier1_df["distance_to_kde_m"] = (
    distances.flatten() * earth_radius
)


scaler = MinMaxScaler()

tier1_df["KDE_hotspot"] = scaler.fit_transform(
    tier1_df[["KDE_hotspot"]]
)

# Drop the columns: Day of week of crash,month,time_of_crash

tier1_df = tier1_df.drop(columns=["Day of week of crash", "month", "time_of_crash"])

# ============================================================
# SAVE
# ============================================================

tier1_df.to_csv(
    "tier1_with_kde.csv",
    index=False
)

print("KDE successfully joined.")

print(tier1_df.isna().sum())

print(tier1_df.describe())

print(tier1_df.head())


