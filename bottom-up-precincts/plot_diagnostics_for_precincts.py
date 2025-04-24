import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import geopandas as gpd
import contextily as ctx  # For basemap tiles (optional)
from shapely.geometry import Point
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import geopandas as gpd
import contextily as ctx
from shapely.geometry import Point

def plot_precinct_diagnostics(precinct_name, precinct_gdf, points_gdf, precinct_col="precinctna", bandwidth=None, basemap=True):
    """
    Plot a diagnostic map showing:
    - The selected precinct boundary highlighted.
    - Points within the precinct and a KDE-based density heatmap of these points.
    - Other precinct boundaries for context.
    
    Parameters:
    -----------
    precinct_name : str
        The name of the precinct to highlight.
    precinct_gdf : GeoDataFrame
        GeoDataFrame of precinct polygons with a 'precinctna' column.
    points_gdf : GeoDataFrame
        GeoDataFrame of points representing registrations or other data.
    bandwidth : float or None
        Bandwidth parameter for seaborn.kdeplot. If None, seaborn default is used.
    basemap : bool
        If True, adds a basemap (requires contextily and EPSG:3857 projection).
    """

    # Filter for the selected precinct
    target_precinct = precinct_gdf[precinct_gdf[precinct_col] == precinct_name]
    
    if target_precinct.empty:
        raise ValueError(f"No precinct found with name {precinct_name}")
    
    # Get the bounding box of the target precinct
    minx, miny, maxx, maxy = target_precinct.total_bounds
    
    # Extract points that lie within this precinct
    points_in_precinct = points_gdf[points_gdf.within(target_precinct.unary_union)]
    
    # Convert points to arrays for KDE
    x_coords = points_in_precinct.geometry.x
    y_coords = points_in_precinct.geometry.y
    
    fig, ax = plt.subplots(figsize=(10,10))
    
    # Plot all precincts for context
    precinct_gdf.plot(ax=ax, facecolor='none', edgecolor='grey', linewidth=0.5)
    
    # Highlight the target precinct
    target_precinct.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=2)
    
    # Plot the points
    ax.scatter(x_coords, y_coords, s=10, color='black', alpha=0.7, zorder=3)
    
    # Check if we can plot KDE
    if len(points_in_precinct) > 2 and x_coords.std() > 0 and y_coords.std() > 0:
        # Plot KDE without fixed levels to avoid contour issues
        sns.kdeplot(
            x=x_coords, y=y_coords, 
            fill=True, cmap='Reds', 
            alpha=0.5, ax=ax,
            bw_method=bandwidth  # If None, seaborn uses a default
        )
    
    # Zoom to precinct with some margin
    ax.set_xlim(minx - (maxx - minx)*0.1, maxx + (maxx - minx)*0.1)
    ax.set_ylim(miny - (maxy - miny)*0.1, maxy + (maxy - miny)*0.1)
    
    # Add a basemap if desired and CRS is EPSG:3857
    if basemap and precinct_gdf.crs == "EPSG:3857":
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    
    ax.set_title(f"Precinct Diagnostics: {precinct_name}")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    plt.show()

precinct_gdf = gpd.read_file(r"/Users/aspencage/Documents/Data/output/bottom_up_precincts_o/precinct_boundaries_voronoi_241218-1032.shp")

# Input CSV with columns: precinctname, reglatitude, reglongitude
points_df = pd.read_csv(r'/Users/aspencage/Documents/Data/input/post_g2024/bottom_up_precincts_i/va_precinct_lat_long_241212_10M.csv')
points_df = points_df.dropna(subset=["reglongitude", "reglatitude"])

# Create a GeoDataFrame
points_gdf = gpd.GeoDataFrame(
    points_df,
    geometry=gpd.points_from_xy(points_df.reglongitude, points_df.reglatitude),
    crs="EPSG:4326"
)

# Project to a suitable projection (e.g., EPSG:3857)
precinct_gdf = precinct_gdf.to_crs(epsg=3857)
precinct_col = "precinctna"

points_gdf = points_gdf.to_crs(epsg=3857)

# Example usage: plot_precinct_diagnostics("001 - CENTRAL", precinct_gdf, points_gdf, precinct_col=precinct_col)

# If you want an interactive toggle in a Jupyter notebook:
from ipywidgets import interact, fixed
precinct_list = precinct_gdf[precinct_col].unique().tolist()
interact(plot_precinct_diagnostics, precinct_name=precinct_list, 
          precinct_gdf=fixed(precinct_gdf), points_gdf=fixed(points_gdf), bandwidth=(0.0001, 0.01, 0.0001), basemap=True)
