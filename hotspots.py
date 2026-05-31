import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from shapely.geometry import Point

# ==========================================
# PATH CONFIGURATION
# ==========================================

print("Loading carcass collection data...")
df = pd.read_csv("animal-remov.csv")

# Drop any rows that are missing coordinates in our target columns
df = df.dropna(subset=['start_lati', 'start_long'])

# 1. Convert to a GeoDataFrame
geometry = [Point(xy) for xy in zip(df['start_long'], df['start_lati'])]
gdf_carcass = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

# 2. Project to metric coordinates (EPSG:3112) so our grid resolution is in meters, not degrees
print("Projecting coordinates for spatial analysis...")
gdf_carcass = gdf_carcass.to_crs(epsg=3112)
coords = np.vstack([gdf_carcass.geometry.x, gdf_carcass.geometry.y]).T

# 3. Define a grid system across the bounding area of NSW data
x_min, y_min = coords[:, 0].min() - 10000, coords[:, 1].min() - 10000
x_max, y_max = coords[:, 0].max() + 10000, coords[:, 1].max() + 10000

# Set resolution: create reference nodes every 2000 meters (2km square blocks)
grid_res = 2000 
x_grid = np.arange(x_min, x_max, grid_res)
y_grid = np.arange(y_min, y_max, grid_res)
X, Y = np.meshgrid(x_grid, y_grid)
grid_coords = np.vstack([X.ravel(), Y.ravel()]).T

# 4. Run Kernel Density Estimation (KDE)
print("Calculating Kernel Density Estimation (this may take a minute)...")
# bandwidth=5000 means the search radius around each point is 5km
kde = KernelDensity(bandwidth=5000, metric='euclidean', kernel='gaussian')
kde.fit(coords)

# Score_samples returns log-density; exp converts it back to standard density values
log_dens = kde.score_samples(grid_coords)
dens = np.exp(log_dens) * 1e6 # Multiplied by 1M to show density per sq km

# Re-shape density calculations back to our visual grid layout
Z = dens.reshape(X.shape)

# ==========================================
# STEP 5: SAVE MAP & CREATE LOOKUP DATASET
# ==========================================
print("Saving outputs...")

# A. Save the Visual Map Image
plt.figure(figsize=(10, 8))
contour = plt.contourf(X, Y, Z, levels=30, cmap='YlOrRd')
plt.colorbar(contour, label='Carcass Density (Incidents / $km^2$)')
plt.title('Static Carcass Hotspot Distribution Map (NSW)')
plt.xlabel('Easting (Meters)')
plt.ylabel('Northing (Meters)')
plt.savefig("carcass_hotspots.png", dpi=300)
plt.close()

# B. Export the data array to a clean lookup CSV
# Convert grid coordinates back to normal Lat/Long degrees for modeling lookups
df_grid = pd.DataFrame({'x_metric': grid_coords[:, 0], 'y_metric': grid_coords[:, 1], 'carcass_density_score': dens})
gdf_grid = gpd.GeoDataFrame(df_grid, geometry=gpd.points_from_xy(df_grid.x_metric, df_grid.y_metric), crs="EPSG:3112")
gdf_grid = gdf_grid.to_crs(epsg=4326)

df_lookup = pd.DataFrame({
    'latitude': gdf_grid.geometry.y,
    'longitude': gdf_grid.geometry.x,
    'carcass_density_score': gdf_grid['carcass_density_score']
})
df_lookup.to_csv("carcass_density_lookup.csv", index=False)

print("\n==========================================")
print("SUCCESS: Hotspot generation complete!")
print("1. Visual map saved as 'carcass_hotspots.png'")
print("2. Lookup dataset saved as 'carcass_density_lookup.csv'")
print("==========================================")