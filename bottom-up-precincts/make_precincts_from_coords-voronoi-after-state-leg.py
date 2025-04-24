import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
import numpy as np
from scipy.spatial import Voronoi
import hdbscan

#---------------------------------------------------------------------
# 1. Load and Prepare Data
#---------------------------------------------------------------------
# Read precinct points
df = pd.read_csv(r'/Users/aspencage/Documents/Data/input/post_g2024/bottom_up_precincts_i/va_precinct_lat_long_241212_10M.csv')
df = df.dropna(subset=["reglongitude", "reglatitude"])

# Create a GeoDataFrame
gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df.reglongitude, df.reglatitude),
    crs="EPSG:4326"
)

# Project to a suitable projection (e.g., EPSG:3857)
gdf = gdf.to_crs(epsg=3857)

# Load legislative districts and water
house_gdf = gpd.read_file(r"/Users/aspencage/Documents/Data/input/post_g2024/comparative_presidential_performance/TIGER2023_SLDL/tl_2023_51_sldl/tl_2023_51_sldl.shp").to_crs(gdf.crs)
senate_gdf = gpd.read_file(r"/Users/aspencage/Documents/Data/input/post_g2024/comparative_presidential_performance/TIGER2023_SLDU/tl_2023_51_sldu/tl_2023_51_sldu.shp").to_crs(gdf.crs)
water_gdf = gpd.read_file(r"/Users/aspencage/Documents/Data/input/post_g2024/bottom_up_precincts_i/USA_Detailed_Water_Bodies").to_crs(gdf.crs)

virginia = gpd.read_file(r"/Users/aspencage/Documents/Data/input/post_g2024/bottom_up_precincts_i/tl_2023_us_state/tl_2023_us_state.shp").to_crs(gdf.crs)
virginia = virginia.loc[virginia['STATEFP'] == "51"]
virginia_polygon = unary_union(virginia.geometry)

# Combine all water polygons into a single geometry for difference
#water_union = unary_union(water_gdf.geometry)
#water_union = water_union.intersection(virginia_polygon)
water_union = None


#---------------------------------------------------------------------
# 2. Assign Each Point to House and Senate Districts
#---------------------------------------------------------------------
# Spatially join points with house and senate districts to find out in which districts each point lies
house_gdf.rename(columns={"GEOIDFQ": "house_id"}, inplace=True)
senate_gdf.rename(columns={"GEOIDFQ": "senate_id"}, inplace=True)

points_house = gpd.sjoin(gdf, house_gdf, how="left", predicate="within")
points_senate = gpd.sjoin(gdf, senate_gdf, how="left", predicate="within")

# Merge house and senate attributes back onto a single frame
# We'll assume house_gdf has a column 'house_id' and senate_gdf has 'senate_id'
# After sjoin: points_house['house_id'], points_senate['senate_id'] exist
points_merged = points_house.drop(columns=house_gdf.columns.difference(["house_id"]), errors='ignore')
points_merged = points_merged.join(points_senate[['senate_id']], how='left')

# If there's a suffix or name conflict, adjust accordingly

#---------------------------------------------------------------------
# 3. For Each Precinct, Determine the Most Common (House_id, Senate_id) Combination
#---------------------------------------------------------------------
precinct_districts = []
for precinct_name, grp in points_merged.groupby("precinctname"):
    # Count occurrences of each (house_id, senate_id) pair
    combo_counts = grp.groupby(["house_id", "senate_id"]).size().reset_index(name='count')
    # Pick the combo with the maximum count
    if not combo_counts.empty:
        max_idx = combo_counts['count'].idxmax()
        chosen_house = combo_counts.iloc[max_idx]['house_id']
        chosen_senate = combo_counts.iloc[max_idx]['senate_id']
    else:
        # If no assignment possible (e.g., no districts found?), handle fallback
        chosen_house = None
        chosen_senate = None
    precinct_districts.append((precinct_name, chosen_house, chosen_senate))

precinct_districts_df = pd.DataFrame(precinct_districts, columns=["precinctname", "house_id", "senate_id"])

#---------------------------------------------------------------------
# 4. Determine Representative Point for Each Precinct
#    (Optional: If you wish to cluster points first to reduce outlier influence)
#---------------------------------------------------------------------
# We'll do a quick clustering as before, though it's optional
precinct_centroids = []
for precinct_name, group in gdf.groupby("precinctname"):
    coords = np.array([(pt.x, pt.y) for pt in group.geometry])
    if len(coords) < 3:
        centroid_x, centroid_y = coords[:,0].mean(), coords[:,1].mean()
        precinct_centroids.append((precinct_name, centroid_x, centroid_y))
        continue

    clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
    labels = clusterer.fit_predict(coords)
    unique_labels, counts = np.unique(labels, return_counts=True)
    mask = unique_labels != -1
    filtered_labels = unique_labels[mask]
    filtered_counts = counts[mask]

    if len(filtered_labels) == 0:
        # No cluster found, just use mean of all points
        centroid_x, centroid_y = coords[:,0].mean(), coords[:,1].mean()
    else:
        largest_cluster = filtered_labels[np.argmax(filtered_counts)]
        cluster_points = coords[labels == largest_cluster]
        centroid_x, centroid_y = cluster_points[:,0].mean(), cluster_points[:,1].mean()

    precinct_centroids.append((precinct_name, centroid_x, centroid_y))

centroids_df = pd.DataFrame(precinct_centroids, columns=["precinctname", "x", "y"])

# Merge centroids with district assignments
centroids_df = centroids_df.merge(precinct_districts_df, on="precinctname", how="left")

# Filter out precincts that have no assigned districts
centroids_df = centroids_df.dropna(subset=["house_id", "senate_id"])

#---------------------------------------------------------------------
# 5. Group Precincts by (House_id, Senate_id) and Compute Voronoi for Each Group
#---------------------------------------------------------------------
final_precinct_polygons = []

unique_combos = centroids_df[["house_id", "senate_id"]].drop_duplicates()

for _, combo in unique_combos.iterrows():
    h_id = combo["house_id"]
    s_id = combo["senate_id"]

    # Extract precincts for this combo
    subset = centroids_df[(centroids_df.house_id == h_id) & (centroids_df.senate_id == s_id)]
    if len(subset) == 1:
        # Only one precinct in this combo, just use a small buffer around centroid
        row = subset.iloc[0]
        poly = Point(row["x"], row["y"]).buffer(5000)
        # Clip by house & senate intersection
        house_poly = house_gdf.loc[house_gdf.house_id == h_id].geometry.unary_union
        senate_poly = senate_gdf.loc[senate_gdf.senate_id == s_id].geometry.unary_union
        district_poly = house_poly.intersection(senate_poly)
        poly = poly.intersection(district_poly)
        if water_union is not None:
            poly = poly.difference(water_union)
        final_precinct_polygons.append((row["precinctname"], poly))
        continue

    # Compute Voronoi for points in this combo
    coords = subset[["x","y"]].to_numpy()
    vor = Voronoi(coords)

    # Get district intersection polygon
    house_poly = house_gdf.loc[house_gdf.house_id == h_id].geometry.unary_union
    senate_poly = senate_gdf.loc[senate_gdf.senate_id == s_id].geometry.unary_union
    district_poly = house_poly.intersection(senate_poly)

    for i, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        precinct_name = subset.iloc[i]["precinctname"]

        if not region or -1 in region:
            # Infinite region
            finite_vertices = [vor.vertices[v] for v in region if v != -1 and v < len(vor.vertices)]
            if len(finite_vertices) < 3:
                poly = Point(coords[i]).buffer(5000)
            else:
                poly = Polygon(finite_vertices)
        else:
            polygon_coords = [vor.vertices[v] for v in region]
            poly = Polygon(polygon_coords)

        # Clip by district polygon
        poly = poly.intersection(district_poly)

        # Remove water areas
        if water_union is not None:
            poly = poly.difference(water_union)

        if not poly.is_empty:
            final_precinct_polygons.append((precinct_name, poly))

# scipy.spatial._qhull.QhullError: QH6214 qhull input error: not enough points(2) to construct initial simplex (need 4)

#---------------------------------------------------------------------
# 6. Construct final GeoDataFrame and Save
#---------------------------------------------------------------------
final_gdf = gpd.GeoDataFrame(
    final_precinct_polygons,
    columns=["precinctname", "geometry"],
    crs=gdf.crs
)

import os 
import time 
os.chdir(r'/Users/aspencage/Documents/Data/output/bottom_up_precincts_o')

final_gdf.to_file(f"final_precinct_boundaries_voronoi_by_district_{time.strftime('%y%m%d-%H%M')}.shp")

print("Final precinct boundaries created using the second approach, assigned by majority points to districts first, then Voronoi within each combined district.")
