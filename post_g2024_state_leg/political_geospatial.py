
from mapping_utilities import fix_invalid_geometries

# NOTE - inspired by nyt_2024_process_and_overlay.process_votes_area_weighted but more flexible 
import geopandas as gpd
import pandas as pd

def create_state_stats(districts, cols=["coverage_percentage", "split_coverage_percentage"]):
    """
    Summarizes statistics by state, returning aggregated stats.
    """
    print("Calculating summary stats by State...")
    agg_dict = {col: ["mean", "median", "min", "max"] for col in cols}
    
    state_stats = districts.groupby("State").agg(agg_dict).reset_index()
    
    # Flatten MultiIndex columns
    state_stats.columns = ["_".join(col).strip("_") for col in state_stats.columns.to_flat_index()]
    
    return state_stats


def process_area_weighted_metric(
    gdf_source, 
    gdf_target, 
    source_cols,
    source_id_col=None,
    target_id_col="District",
    suffix="_weighted",
    agg_dict=None,
    print_warnings=False,
    return_intersection=False
):
    """
    Performs an area-weighted overlay of `gdf_source` onto `gdf_target`,
    aggregating numeric columns in `source_cols` by the fraction of intersected
    area. For string columns, collects a list of unique values.
    The result is merged back onto `gdf_target`.

    Parameters
    ----------
    gdf_source : GeoDataFrame
        The layer containing the metric(s) you want to aggregate (e.g., counties).
    gdf_target : GeoDataFrame
        The layer you want to aggregate onto (e.g., legislative districts).
    source_cols : list of str
        The columns in `gdf_source` to be aggregated. 
        - Numeric columns are area-weighted and summed.
        - String columns are collected as a unique list.
    source_id_col : str, optional
        If your source has an ID column (e.g., 'GEOID'), you can specify it
        for clarity or debugging. Not strictly required.
    target_id_col : str, default="District"
        The column in the target used to group (i.e., your district identifier).
    suffix : str, default="_weighted"
        The suffix appended to each aggregated column.
    agg_dict : dict or None, default=None
        If provided, this dictionary will be used to aggregate the weighted
        columns. If None, defaults to:
          - sum for numeric columns
          - unique list for string columns
    print_warnings : bool, default=False
        If True, prints some overlay warnings/info.
    return_intersection : bool, default=False
        If True, returns the intersection GeoDataFrame (useful for debugging).

    Returns
    -------
    GeoDataFrame
        A GeoDataFrame with the original geometry of `gdf_target`, 
        plus new columns for each area-weighted or string-collected 
        `source_cols` aggregated by `target_id_col`.
    """
    # 1) Ensure both layers have the same CRS
    print(f"Target CRS: {gdf_target.crs}")
    if gdf_source.crs != gdf_target.crs:
        gdf_source = gdf_source.to_crs(gdf_target.crs)
        print(f"Source CRS changed to match target: {gdf_target.crs}")

    # 2) Compute the area of each feature in the source if not present
    gdf_source = gdf_source.copy()
    if "_source_area" not in gdf_source.columns:
        gdf_source["_source_area"] = gdf_source.geometry.area

    # 3) Fix invalid geometries if needed (optional step)
    gdf_source = fix_invalid_geometries(gdf_source)
    gdf_target = fix_invalid_geometries(gdf_target)

    # 4) Perform the overlay
    if print_warnings is False:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            intersection = gpd.overlay(gdf_source, gdf_target, how="intersection")
    else:
        intersection = gpd.overlay(gdf_source, gdf_target, how="intersection")

    # 5) Calculate the intersection area and fraction
    intersection["_intersection_area"] = intersection.geometry.area
    intersection["_area_fraction"] = (
        intersection["_intersection_area"] / intersection["_source_area"]
    )

    # 6) For each column:
    #    - If it's numeric, create a weighted version
    #    - If it's string, just copy it so we can gather them later
    #    We'll also build a default aggregator if none provided.
    default_agg_dict = {}
    
    for col in source_cols:
        if pd.api.types.is_numeric_dtype(intersection[col]):
            # Weighted column
            out_col = f"{col}{suffix}"
            intersection[out_col] = intersection[col] * intersection["_area_fraction"]
            # If user didn't provide agg_dict, default to summing numeric columns
            default_agg_dict[out_col] = "sum"
        else:
            # String column: keep it and gather unique values
            out_col = f"{col}{suffix}"
            # Copy the original string to a new column so we can groupby it
            intersection[out_col] = intersection[col]
            # If user didn't provide agg_dict, default to a list of unique strings
            default_agg_dict[out_col] = lambda x: sorted(set(x))

    # If user gave a custom agg_dict, that overrides our defaults
    if agg_dict is None:
        agg_dict = default_agg_dict
    else:
        # Merge user agg_dict with defaults if needed
        for k, v in default_agg_dict.items():
            agg_dict.setdefault(k, v)

    # Optionally return the raw intersection for debugging
    if return_intersection:
        return intersection

    # 7) Group by the target’s district/ID with the aggregator
    grouped = intersection.groupby(target_id_col).agg(agg_dict).reset_index()

    # 8) Merge the aggregated columns back to the original target geometry
    target_unique = gdf_target.copy()
    final = target_unique.merge(grouped, on=target_id_col, how="left")

    # 9) Convert back to GeoDataFrame
    final_gdf = gpd.GeoDataFrame(final, geometry="geometry", crs=gdf_target.crs)

    return final_gdf
