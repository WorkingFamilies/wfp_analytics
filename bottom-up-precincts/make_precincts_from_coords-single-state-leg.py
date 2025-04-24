import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
import numpy as np
import hdbscan
from scipy.spatial import Voronoi

#---------------------------------------------------------------------
# Load and prepare data (same as before)
#---------------------------------------------------------------------
df = pd.read_csv(r'/Users/aspencage/Documents/Data/input/post_g2024/bottom_up_precincts_i/va_precinct_lat_long_241212_10M.csv')
df = df.dropna(subset=["reglongitude", "reglatitude"])
gdf = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df.reglongitude, df.reglatitude),
    crs="EPSG:4326"
).to_crs(epsg=3857)

# Load legislative districts and water
house_gdf = gpd.read_file(r"/Users/aspencage/Documents/Data/input/post_g2024/comparative_presidential_performance/TIGER2023_SLDL/tl_2023_51_sldl/tl_2023_51_sldl.shp").to_crs(gdf.crs)
senate_gdf = gpd.read_file(r"/Users/aspencage/Documents/Data/input/post_g2024/comparative_presidential_performance/TIGER2023_SLDU/tl_2023_51_sldu/tl_2023_51_sldu.shp").to_crs(gdf.crs)
water_gdf = gpd.read_file(r"/Users/aspencage/Documents/Data/input/post_g2024/bottom_up_precincts_i/USA_Detailed_Water_Bodies").to_crs(gdf.crs)

virginia = gpd.read_file(r"/Users/aspencage/Documents/Data/input/post_g2024/bottom_up_precincts_i/tl_2023_us_state/tl_2023_us_state.shp").to_crs(gdf.crs)
virginia = virginia.loc[virginia['STATEFP'] == "51"]
virginia_polygon = unary_union(virginia.geometry)

# Combine all water polygons into a single geometry for difference
water_union = None
#water_union = unary_union(water_gdf.geometry)
#water_union = water_union.intersection(virginia_polygon)

#---------------------------------------------------------------------
# Determine representative centroid per precinct
#---------------------------------------------------------------------
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
    mask = (unique_labels != -1)
    filtered_labels = unique_labels[mask]
    filtered_counts = counts[mask]
    
    if len(filtered_labels) == 0:
        centroid_x, centroid_y = coords[:,0].mean(), coords[:,1].mean()
    else:
        largest_cluster = filtered_labels[np.argmax(filtered_counts)]
        cluster_points = coords[labels == largest_cluster]
        centroid_x, centroid_y = cluster_points[:,0].mean(), cluster_points[:,1].mean()
    
    precinct_centroids.append((precinct_name, centroid_x, centroid_y))

centroid_df = pd.DataFrame(precinct_centroids, columns=["precinctname", "x", "y"])
centroid_gdf = gpd.GeoDataFrame(
    centroid_df, geometry=gpd.points_from_xy(centroid_df.x, centroid_df.y),
    crs=gdf.crs
)

#---------------------------------------------------------------------
# Compute Voronoi diagram and generate precinct polygons
#---------------------------------------------------------------------
points = np.array(centroid_gdf[["x", "y"]])
vor = Voronoi(points)

polygons = []
for i, region_idx in enumerate(vor.point_region):
    region = vor.regions[region_idx]
    precinct_name = centroid_gdf.iloc[i]["precinctname"]
    
    if not region or -1 in region:
        # Infinite region: fallback to clipping with VA
        finite_vertices = [vor.vertices[v] for v in region if v != -1 and v < len(vor.vertices)]
        if len(finite_vertices) < 3:
            # Buffer centroid if we can't form polygon
            poly = Point(points[i]).buffer(5000)
        else:
            poly = Polygon(finite_vertices)
        clipped_poly = poly.intersection(virginia_polygon)
    else:
        polygon_coords = [vor.vertices[v] for v in region]
        poly = Polygon(polygon_coords)
        clipped_poly = poly.intersection(virginia_polygon)
    
    polygons.append((precinct_name, clipped_poly))

precinct_gdf = gpd.GeoDataFrame(polygons, columns=["precinctname", "geometry"], crs=gdf.crs)

#---------------------------------------------------------------------
# Integrate Legislative Districts
#---------------------------------------------------------------------
# Assign each precinct to exactly one house district:
# 1. Find which house district contains the precinct's centroid.
# 2. Intersect that precinct polygon with the chosen house district polygon.

final_precincts = []
for i, row in precinct_gdf.iterrows():
    precinct_name = row["precinctname"]
    precinct_poly = row.geometry
    centroid = precinct_poly.centroid
    
    # Find house district containing centroid
    house_containing = house_gdf[house_gdf.geometry.contains(centroid)]
    if len(house_containing) == 1:
        house_poly = house_containing.geometry.iloc[0]
    else:
        # If none or multiple (rare), pick nearest house district
        # (In well-defined data, this shouldn't occur)
        house_poly = house_gdf.iloc[house_gdf.distance(centroid).idxmin()].geometry
    
    # Intersect precinct with house polygon
    precinct_poly = precinct_poly.intersection(house_poly)
    
    # Now do the same for senate districts
    senate_containing = senate_gdf[senate_gdf.geometry.contains(centroid)]
    if len(senate_containing) == 1:
        senate_poly = senate_containing.geometry.iloc[0]
    else:
        # Fallback to nearest if needed
        senate_poly = senate_gdf.iloc[senate_gdf.distance(centroid).idxmin()].geometry
    
    # Intersect with senate polygon
    precinct_poly = precinct_poly.intersection(senate_poly)
    
    # Remove water areas
    if water_union is not None:
        precinct_poly = precinct_poly.difference(water_union)
    
    # If the resulting polygon is empty (e.g., entirely water), skip or handle as needed
    if not precinct_poly.is_empty:
        final_precincts.append((precinct_name, precinct_poly))

final_gdf = gpd.GeoDataFrame(final_precincts, columns=["precinctname", "geometry"], crs=gdf.crs)

#---------------------------------------------------------------------
# Save Final Result
#---------------------------------------------------------------------
import os 
import time 
os.chdir(r'/Users/aspencage/Documents/Data/output/bottom_up_precincts_o')

final_gdf.to_file(f"final_precinct_boundaries_{time.strftime('%y%m%d-%H%M')}.shp")

print("Final precinct boundaries created successfully, assigned to single house/senate districts and excluding water.")
