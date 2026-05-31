import geopandas as gpd
import pandas as pd
import numpy as np
import random
import os

def load_nsw_roads(zip_path):
    """
    Loads the NSW road network directly from a zip file and projects it 
    to a metric system (NSW Lambert epsg=3112) for accurate length-weighted sampling.
    """
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Could not find the zip file: {zip_path} in your workspace directory.")
        
    print("Reading NSW road network directly from zip archive...")
    # 'zip://' prefix allows geopandas to read directly inside the compressed archive
    roads = gpd.read_file(f"zip://{zip_path}")
    
    print("Projecting roads to NSW Lambert (EPSG:3112) for geometric calculations...")
    roads = roads.to_crs(epsg=3112)
    return roads

def generate_spatial_points(roads_gdf, num_points):
    """
    Samples random points strictly along the road network.
    Longer roads have a proportionally higher probability of being selected.
    """
    print(f"Sampling {num_points} spatial points along the road lines...")
    lengths = roads_gdf.geometry.length
    weights = lengths / lengths.sum()
    
    # Select road line segments based on their length distribution weight
    sampled_indices = np.random.choice(roads_gdf.index, size=num_points, p=weights)
    
    points = []
    for idx in sampled_indices:
        road_line = roads_gdf.loc[idx, 'geometry']
        # Extract a random coordinate position along the line length
        random_dist = random.uniform(0, road_line.length)
        point = road_line.interpolate(random_dist)
        points.append(point)
        
    return points

# ==========================================
# GENERATION FOR TIER 1: MACRO (2016-2024)
# ==========================================
def create_tier1_absences(roads_gdf, num_crashes, ratio=5):
    """Generates absences matching the schema of the 2016-2024 Macro dataset."""
    num_absences = num_crashes * ratio
    points = generate_spatial_points(roads_gdf, num_absences)
    
    print("Assigning Macro temporal profiles (2016-2024)...")
    years = np.random.randint(2016, 2025, size=num_absences)
    hours = np.random.randint(0, 24, size=num_absences)
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    random_months = np.random.choice(months, size=num_absences)
    
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    random_days = np.random.choice(days_of_week, size=num_absences)
    
    # Convert hour integers to your native string format (e.g., 5 -> "05:00")
    time_strings = [f"{str(h).zfill(2)}:00" for h in hours]
    
    df = pd.DataFrame({
        'Day of week of crash': random_days,
        'year': years,
        'month': random_months,
        'time_of_crash': time_strings,
        'target': 0  # 0 indicates control/absence point
    })
    
    # Bundle into a GeoDataFrame to handle conversion back to standard GPS coordinates
    gdf = gpd.GeoDataFrame(df, geometry=points, crs=roads_gdf.crs)
    gdf = gdf.to_crs(epsg=4326) # Project back to standard Lat/Long degrees
    
    gdf['latitude'] = gdf.geometry.y
    gdf['longitude'] = gdf.geometry.x
    
    return pd.DataFrame(gdf.drop(columns='geometry'))

# ==========================================
# GENERATION FOR TIER 2: DEEP-DIVE (2021-2023)
# ==========================================
def create_tier2_absences(roads_gdf, num_crashes, ratio=5):
    """Generates absences matching the schema of the 2021-2023 Gold dataset with exact dates."""
    num_absences = num_crashes * ratio
    points = generate_spatial_points(roads_gdf, num_absences)
    
    print("Assigning precise timestamps and date intervals (2021-2023)...")
    # Generate random unique epochs across the 3-year timeline
    start_ts = pd.Timestamp('2021-01-01').value
    end_ts = pd.Timestamp('2023-12-31').value
    random_timestamps = pd.to_datetime(
    np.random.randint(start_ts, end_ts, size=num_absences, dtype=np.int64)
)
    
    df = pd.DataFrame({
        'year': random_timestamps.year,
        'month': random_timestamps.month,
        'date_of_crash': random_timestamps.strftime('%d/%m/%Y'), # Native DD/MM/YYYY layout
        'time_of_crash': random_timestamps.strftime('%H%M').astype(int), # Native integer layout (e.g. 2100)
        'target': 0  # 0 indicates control/absence point
    })
    
    gdf = gpd.GeoDataFrame(df, geometry=points, crs=roads_gdf.crs)
    gdf = gdf.to_crs(epsg=4326) # Project back to standard Lat/Long degrees
    
    gdf['latitude'] = gdf.geometry.y
    gdf['longitude'] = gdf.geometry.x
    
    return pd.DataFrame(gdf.drop(columns='geometry'))

# ==========================================
# CENTRAL EXECUTION FLOW
# ==========================================
if __name__ == "__main__":
    # CONFIGURATION: Update file names to match your workspace exactly
    ZIP_PATH = "nsw_roads.zip"                    # Name of your uploaded shapefile zip
    MACRO_CLEANED_PATH = "2016-2024-deduplicated.csv"   # Your deduplicated 2016-24 crash data
    GOLD_REAL_PATH = "2021-2023-filtered.csv"      # Your 2021-23 exact crash data
    
    try:
        # 1. Read real files to extract lengths for calculation
        print("Reading clean real data files...")
        df_macro_real = pd.read_csv('2016-2024-deduplicated.csv')
        df_gold_real = pd.read_csv('2021-2023-filtered.csv')
        
        num_macro_crashes = len(df_macro_real)
        num_gold_crashes = len(df_gold_real)
        
        # 2. Parse Spatial Infrastructure Layer
        roads_layer = load_nsw_roads("roads.zip")
        
        # 3. Generate Tier 1 Absences (1:5 ratio)
        print(f"\n--- Processing Tier 1 Matrix (Target: {num_macro_crashes * 5} rows) ---")
        tier1_absences = create_tier1_absences(roads_layer, num_macro_crashes, ratio=5)
        
        # 4. Generate Tier 2 Absences (1:5 ratio)
        print(f"\n--- Processing Tier 2 Matrix (Target: {num_gold_crashes * 5} rows) ---")
        tier2_absences = create_tier2_absences(roads_layer, num_gold_crashes, ratio=5)
        
        # 5. Export clean datasets to your project folder
        print("\nSaving generated pipelines to disk...")
        tier1_absences.to_csv("tier1_pseudo_absences.csv", index=False)
        tier2_absences.to_csv("tier2_pseudo_absences.csv", index=False)
        
        print("\n==========================================")
        print("SUCCESS: Generation complete!")
        print(f"Generated Tier 1 Absences: {tier1_absences.shape[0]} rows -> Saved as 'tier1_pseudo_absences.csv'")
        print(f"Generated Tier 2 Absences: {tier2_absences.shape[0]} rows -> Saved as 'tier2_pseudo_absences.csv'")
        print("==========================================")
        
    except FileNotFoundError as e:
        print(f"\nCRITICAL ERROR: {e}")
    except Exception as e:
        print(f"\nAn unexpected runtime execution error occurred: {e}") 