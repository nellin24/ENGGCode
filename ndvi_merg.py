import pandas as pd
from scipy.spatial import KDTree

# 1. Load datasets
ndvi_data = pd.read_csv("Cleaned_NDVI.csv")
tier1_data = pd.read_csv("master_tier1_training.csv")
tier2_data = pd.read_csv("master_tier2_training.csv")


def merge_by_proximity(tier_df, ndvi_df, tolerance=0.0001):
    """Merges NDVI data into a tier dataframe by finding the closest coordinate

    match within a specific year.
    """
    final_dfs = []

    # Process year by year to guarantee exact year matches
    for year in tier_df["year"].unique():
        tier_year = tier_df[tier_df["year"] == year].copy()
        ndvi_year = ndvi_df[ndvi_df["Matched_Year"] == year].copy()

        if ndvi_year.empty or tier_year.empty:
            final_dfs.append(tier_year)
            continue

        # Build a spatial tree of the NDVI coordinates for this year
        spatial_tree = KDTree(ndvi_year[["Latitude", "Longitude"]].values)

        # Query the tree using the tier coordinates
        distances, indices = spatial_tree.query(
            tier_year[["latitude", "longitude"]].values
        )

        # Map the matching NDVI values back using the indices
        tier_year["NDVI_Mean"] = ndvi_year["NDVI_Mean"].values[indices]
        tier_year["spatial_drift"] = distances

        # Optional: Set NDVI to NaN if the closest point is further away than our tolerance
        tier_year.loc[tier_year["spatial_drift"] > tolerance, "NDVI_Mean"] = (
            None
        )

        final_dfs.append(tier_year)

    # Combine everything back together and drop the temporary drift column
    return pd.concat(final_dfs, ignore_index=True).drop(
        columns=["spatial_drift"]
    )


# 2. Run the robust proximity merge
print("Merging Tier 1...")
tier1_merged = merge_by_proximity(tier1_data, ndvi_data)

print("Merging Tier 2...")
tier2_merged = merge_by_proximity(tier2_data, ndvi_data)

# 3. Save the merged datasets
tier1_merged.to_csv("master_tier1_with_ndvi.csv", index=False)
tier2_merged.to_csv("master_tier2_with_ndvi.csv", index=False)

print("Success! Perfect proximity-based merge complete.")