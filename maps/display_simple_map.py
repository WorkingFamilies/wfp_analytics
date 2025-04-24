"""
display_simple_map.py

Usage:
    python3 display_simple_map.py <geo_path> [column_name] [add_basemap] [show_nulls]

Examples:
    1) Simple map (no column):
        python3 display_simple_map.py /path/to/file.shp
    2) Choropleth by 'population':
        python3 display_simple_map.py /path/to/file.shp population
    3) Disable basemap:
        python3 display_simple_map.py /path/to/file.shp population false
    4) Highlight NULLs in 'population':
        python3 display_simple_map.py /path/to/file.shp population true true
"""
# We can allow 2-5 arguments:
#   sys.argv[1] -> geo_path (required)
#   sys.argv[2] -> column_name (optional)
#   sys.argv[3] -> add_basemap (optional, default True)
#   sys.argv[4] -> show_nulls (optional, default False)

import sys
import pandas as pd 
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from mapping_utilities import load_as_gdf

def display_simple_map(geo_path, column_name=None, add_basemap=True, show_nulls=False):
    """
    Display a map or choropleth from a shapefile, with an option to highlight NULL values.

    Parameters
    ----------
    geo_path : str
        Path to the shapefile to be displayed.
    column_name : str, optional
        The column name used to create a choropleth. If None or invalid, 
        a simple uniform-color map is shown.
    add_basemap : bool, default True
        Whether to add an OpenStreetMap basemap.
    show_nulls : bool, default False
        If True, features whose values in 'column_name' are NULL are 
        highlighted in gray with a hatch pattern.

    Notes
    -----
    - If 'column_name' is provided and exists, the map is colored by that column 
      (choropleth). Otherwise, a uniform color plot is shown.
    - If 'show_nulls=True', any features with NULL in 'column_name' are rendered
      in a distinct style (gray/hatching) behind the main layer. 
    - The shapefile is reprojected to EPSG:3857 for compatibility with contextily.
    """

    # 1) Load the shapefile into a GeoDataFrame
    gdf = gpd.read_file(geo_path)

    # because a GeoDataFrame can be a subclass of DataFrame, we need to check
    df_not_gdf_mask = isinstance(gdf, pd.DataFrame) and not isinstance(gdf, gpd.GeoDataFrame)
    if df_not_gdf_mask:
        gdf = load_as_gdf(
            geo_path,
            geometry_col='geometry',
            guess_format='auto'
        )

    # 2) Ensure the GeoDataFrame has a valid CRS
    if gdf.crs is None:
        raise ValueError("The shapefile does not have a valid CRS.")

    # 3) Reproject to Web Mercator for compatibility with contextily
    gdf = gdf.to_crs(epsg=3857)

    # 4) Split the data into null vs. not-null if show_nulls is True & column_name is valid
    if show_nulls and column_name and column_name in gdf.columns:
        null_mask = gdf[column_name].isnull()
        gdf_nulls = gdf[null_mask]
        gdf_non_nulls = gdf[~null_mask]
    else:
        gdf_nulls = None
        gdf_non_nulls = gdf

    # 5) Create the plot
    fig, ax = plt.subplots(figsize=(12, 10))

    # -- (a) If requested, plot NULL values layer in gray/hatching
    if gdf_nulls is not None and not gdf_nulls.empty:
        gdf_nulls.plot(
            ax=ax,
            color="gray",
            edgecolor="black",
            hatch="///",  # hatch pattern for visibility
            alpha=0.5
        )

    # -- (b) Plot non-null data
    if column_name and column_name in gdf_non_nulls.columns:
        # Choropleth
        gdf_non_nulls.plot(
            ax=ax,
            column=column_name,
            legend=True,
            cmap="OrRd",
            edgecolor="black"
        )
        ax.set_title(f"Choropleth Map by '{column_name}'", fontsize=16)
    else:
        # Fallback: uniform color
        gdf_non_nulls.plot(
            ax=ax,
            color="blue",
            alpha=0.5,
            edgecolor="black"
        )
        if column_name and column_name not in gdf.columns:
            print(f"[WARNING] Column '{column_name}' not found. Showing simple map.")
        ax.set_title("Shapefile Map (Simple)", fontsize=16)

    # 6) Optionally add OpenStreetMap basemap
    if add_basemap:
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

    # 7) Label axes, adjust layout
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()

    # 8) Show the plot
    plt.show()


if __name__ == "__main__":

    if len(sys.argv) < 2 or len(sys.argv) > 5:
        print("Usage: python3 display_simple_map.py <geo_path> [column_name] [add_basemap] [show_nulls]")
        sys.exit(1)

    geo_path = sys.argv[1]

    column_name = sys.argv[2] if len(sys.argv) > 2 else None

    if len(sys.argv) >= 4:
        add_basemap_arg = sys.argv[3].lower()
        add_basemap = (add_basemap_arg == 'true')
    else:
        add_basemap = True

    if len(sys.argv) == 5:
        show_nulls_arg = sys.argv[4].lower()
        show_nulls = (show_nulls_arg == 'true')
    else:
        show_nulls = False

    display_simple_map(geo_path, column_name, add_basemap, show_nulls)
