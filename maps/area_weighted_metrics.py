import warnings
import geopandas as gpd
import pandas as pd
import numpy as np


###############################################################################
# VOTE/METRIC AREA-WEIGHTED AGGREGATION FUNCTIONS
###############################################################################

# Build a flexible aggregator that sums numeric columns, takes most common for extra precinct cols
def top_mode(s):
    # Return the top value (ties are broken arbitrarily by value_counts)
    return s.value_counts(dropna=False).index[0]

# while non geospatial, this is used within the geospatial functions
def process_votes_nonspatial(
    df_precincts,
    district_col="District",
    year="2024",
    extra_precinct_cols=None
):
    """
    For precincts that do NOT have geometry (or otherwise can't be area-weighted),
    compute total votes by district. Returns a DataFrame aggregated by `district_col`
    with columns named consistently with the area-weighted version.
    """
    # Make a copy to avoid mutating caller data
    df = df_precincts.copy()

    # Ensure we have "third party" and "two_way" columns
    df["votes_third_party"] = (
        df["votes_total"] - df["votes_rep"] - df["votes_dem"]
    )
    df["votes_two_way"] = df["votes_rep"] + df["votes_dem"]

    # Name the columns in a way that matches the area-weighted approach
    df[f"votes_rep_by_area_{year}"] = df["votes_rep"]
    df[f"votes_dem_by_area_{year}"] = df["votes_dem"]
    df[f"votes_third_party_by_area_{year}"] = df["votes_third_party"]
    df[f"votes_total_by_area_{year}"] = df["votes_total"]
    df[f"votes_two_way_by_area_{year}"] = df["votes_two_way"]

    # Basic aggregator dictionary 
    agg_dict = {
        f"votes_rep_by_area_{year}": "sum",
        f"votes_dem_by_area_{year}": "sum",
        f"votes_third_party_by_area_{year}": "sum",
        f"votes_total_by_area_{year}": "sum",
        f"votes_two_way_by_area_{year}": "sum",
    }

    # If user wants extra precinct columns aggregated, pick the mode
    if extra_precinct_cols:
        for c in extra_precinct_cols:
            if c in df.columns:
                agg_dict[c] = top_mode

    grouped_nongeo = df.groupby(district_col).agg(agg_dict).reset_index()

    return grouped_nongeo


def process_votes_area_weighted(
    gdf_precinct, 
    gdf_districts, 
    year="2024",
    district_col="District",
    state_col="State",
    extra_district_cols=None,
    extra_precinct_cols=None
):
    """
    Given precinct-level vote columns (rep, dem, total) and a district GDF,
    performs area-weighted overlay to compute aggregated vote totals + shares
    by district. Also handles precincts without geometry by directly summing
    votes by district. Returns a new GeoDataFrame containing these aggregations
    merged with `gdf_districts`.

    If a given district has rows in both the geospatial data and the 
    non-geospatial data, the non-geospatial row completely overrides 
    the geospatial row (i.e., it replaces it).

    Parameters
    ----------
    gdf_precinct : GeoDataFrame
        Precinct-level data, must contain columns: "votes_rep", "votes_dem", 
        "votes_total", and a geometry column (except for precincts with None 
        geometry).  
    gdf_districts : GeoDataFrame
        District polygons, must have at least a 'geometry' column and 
        a column matching `district_col`.  
    year : str, default "2024"
        Used for naming the resulting vote columns, e.g., 'votes_rep_by_area_2024'.  
    district_col : str, default "District"
        Column name in gdf_districts that identifies each district.  
    state_col : str, default "State"
        Column name in gdf_districts that identifies the state (kept in the 
        final output).  
    extra_district_cols : list of str, optional
        Additional columns in `gdf_districts` to carry into the final merged GDF.  
    extra_precinct_cols : list of str, optional
        Additional columns in `gdf_precinct` to carry forward. These columns 
        will be aggregated by district using the most common (mode) value.  
        
    Returns
    -------
    GeoDataFrame
        One row per district, containing aggregated vote totals, shares, and any
        requested columns.  Districts that only have geometry (i.e. no precincts 
        in them) will appear with null/NaN vote data.
    """
    # 1) Ensure matching CRS
    if gdf_precinct.crs != gdf_districts.crs:
        gdf_precinct = gdf_precinct.to_crs(gdf_districts.crs)

    # 2) Split precincts into those with geometry vs. no geometry
    precincts_geo = gdf_precinct[~gdf_precinct.geometry.isna()].copy()
    precincts_nogeo = gdf_precinct[gdf_precinct.geometry.isna()].copy()

    # =========================
    # Part A: Handle geometry
    # =========================
    #
    # Drop the district_col in geometry precincts if present – we only want 
    # that district_col from the district layer.
    precincts_geo.drop(columns=[district_col], errors="ignore", inplace=True)

    # Compute extra columns in geometric precincts
    precincts_geo["votes_third_party"] = (
        precincts_geo["votes_total"] 
        - precincts_geo["votes_rep"] 
        - precincts_geo["votes_dem"]
    )
    precincts_geo["votes_two_way"] = (
        precincts_geo["votes_dem"] + precincts_geo["votes_rep"]
    )

    # If total precinct area isn't precomputed, do it now
    if "precinct_area" not in precincts_geo.columns:
        precincts_geo["precinct_area"] = precincts_geo.geometry.area

    # Perform spatial intersection
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # e.g. topological warnings
        subset_geo = gpd.overlay(precincts_geo, gdf_districts, how="intersection")

    # Compute partial-area fraction
    subset_geo["split_area"] = subset_geo.geometry.area
    subset_geo["area_fraction"] = subset_geo["split_area"] / subset_geo["precinct_area"]

    # Multiply vote columns by that fraction
    subset_geo[f"votes_rep_by_area_{year}"] = (
        subset_geo["votes_rep"] * subset_geo["area_fraction"]
    )
    subset_geo[f"votes_dem_by_area_{year}"] = (
        subset_geo["votes_dem"] * subset_geo["area_fraction"]
    )
    subset_geo[f"votes_third_party_by_area_{year}"] = (
        subset_geo["votes_third_party"] * subset_geo["area_fraction"]
    )
    subset_geo[f"votes_total_by_area_{year}"] = (
        subset_geo["votes_total"] * subset_geo["area_fraction"]
    )
    subset_geo[f"votes_two_way_by_area_{year}"] = (
        subset_geo["votes_two_way"] * subset_geo["area_fraction"]
    )

    # Aggregator dictionary for area-weighted votes
    agg_dict_geo = {
        f"votes_rep_by_area_{year}": "sum",
        f"votes_dem_by_area_{year}": "sum",
        f"votes_third_party_by_area_{year}": "sum",
        f"votes_total_by_area_{year}": "sum",
        f"votes_two_way_by_area_{year}": "sum",
    }

    # If user wants extra precinct columns aggregated, pick the mode
    if extra_precinct_cols:
        for c in extra_precinct_cols:
            if c in subset_geo.columns:
                agg_dict_geo[c] = top_mode

    grouped_geo = subset_geo.groupby(district_col).agg(agg_dict_geo).reset_index()

    # =========================
    # Part B: Handle no-geometry
    # =========================
    #
    # Summation by district (or however `process_votes_nonspatial` does it).
    # Note that columns produced match the area-based naming.
    if not precincts_nogeo.empty:
        grouped_nogeo = process_votes_nonspatial(
            df_precincts=precincts_nogeo,
            district_col=district_col,
            year=year,
            extra_precinct_cols=extra_precinct_cols
        )
    else:
        grouped_nogeo = pd.DataFrame(columns=[district_col])

    # =========================
    # Part C: Combine with override logic
    # =========================
    #
    # For districts that appear in grouped_nogeo, that row overrides 
    # any row in grouped_geo.
    if not grouped_nogeo.empty:
        # Identify override districts
        override_districts = grouped_nogeo[district_col].unique()
        # Remove them from the area-weighted results
        grouped_geo = grouped_geo[~grouped_geo[district_col].isin(override_districts)]
        # Then combine
        combined = pd.concat([grouped_geo, grouped_nogeo], ignore_index=True)
    else:
        combined = grouped_geo

    # We now have exactly one row per district: either from geometry or from no-geometry

    # =========================
    # Part D: Compute shares
    # =========================
    combined[f"pres_dem_share_total_{year}"] = (
        combined[f"votes_dem_by_area_{year}"] / combined[f"votes_total_by_area_{year}"]
    )
    combined[f"pres_dem_share_two_way_{year}"] = (
        combined[f"votes_dem_by_area_{year}"] / combined[f"votes_two_way_by_area_{year}"]
    )
    combined[f"third_party_vote_share_{year}"] = (
        combined[f"votes_third_party_by_area_{year}"] 
        / combined[f"votes_total_by_area_{year}"]
    )

    # =========================
    # Part E: Merge with District Geometry
    # =========================
    keep_cols = [state_col, district_col, "geometry"]
    if extra_district_cols:
        keep_cols += extra_district_cols

    # Drop duplicates in district shapes
    districts_subset = gdf_districts[keep_cols].drop_duplicates(subset=[district_col])
    final = districts_subset.merge(combined, on=district_col, how="left")

    # Return as GeoDataFrame
    final = gpd.GeoDataFrame(final, geometry="geometry", crs=gdf_districts.crs)

    return final


def fix_invalid_geometries(gdf):
    """Example helper to fix invalid geometries via buffer(0)."""
    if not isinstance(gdf, gpd.GeoDataFrame):
        return gdf
    invalid_mask = ~gdf.geometry.is_valid
    if invalid_mask.any():
        gdf.loc[invalid_mask, "geometry"] = gdf.loc[invalid_mask, "geometry"].buffer(0)
    return gdf

# NOTE - inspired by process_votes_area_weighted but more flexible 
# NOTE this is used in the county aggregation 
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

    Now also handles rows in `gdf_source` that have no geometry (NaN):
    - These will be aggregated directly (non-spatially) by the target_id_col.

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
    if gdf_source.crs != gdf_target.crs:
        gdf_source = gdf_source.to_crs(gdf_target.crs)
        if print_warnings:
            print(f"Source CRS changed to match target: {gdf_target.crs}")

    gdf_source = gdf_source.copy()
    gdf_target = gdf_target.copy()

    # Separate source rows that have geometry from those that do not
    source_with_geo = gdf_source[~gdf_source.geometry.isna()].copy()
    source_no_geo = gdf_source[gdf_source.geometry.isna()].copy()

    # ========== Handle geometry-based rows via overlay ==========

    # 2) Compute the area of each feature in the source if not present
    if "_source_area" not in source_with_geo.columns:
        source_with_geo["_source_area"] = source_with_geo.geometry.area

    # 3) Fix invalid geometries (optional step)
    source_with_geo = fix_invalid_geometries(source_with_geo)
    gdf_target = fix_invalid_geometries(gdf_target)

    # 4) Perform the overlay
    if not print_warnings:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            intersection = gpd.overlay(source_with_geo, gdf_target, how="intersection")
    else:
        intersection = gpd.overlay(source_with_geo, gdf_target, how="intersection")

    # 5) Calculate the intersection area and fraction
    intersection["_intersection_area"] = intersection.geometry.area
    intersection["_area_fraction"] = (
        intersection["_intersection_area"] / intersection["_source_area"]
    )

    # 6) Build a default aggregator if none was provided
    default_agg_dict = {}
    # We'll create new columns that reflect the area-weighted or aggregated values
    for col in source_cols:
        # Weighted numeric or string aggregator
        if pd.api.types.is_numeric_dtype(intersection[col]):
            out_col = f"{col}{suffix}"
            intersection[out_col] = intersection[col] * intersection["_area_fraction"]
            default_agg_dict[out_col] = "sum"
        else:
            out_col = f"{col}{suffix}"
            intersection[out_col] = intersection[col]
            # Collect unique sorted values (same as original code's default for strings)
            default_agg_dict[out_col] = lambda x: sorted(set(x))

    # If user gave a custom agg_dict, merge it with the defaults
    if agg_dict is None:
        agg_dict = default_agg_dict
    else:
        for k, v in default_agg_dict.items():
            agg_dict.setdefault(k, v)

    # ========== Optionally return the raw intersection for debugging ==========
    if return_intersection:
        return intersection

    # 7) Group the intersection by the target's district/ID
    grouped_geo = intersection.groupby(target_id_col).agg(agg_dict).reset_index()

    # ========== Handle rows with no geometry (direct aggregator) ==========

    if not source_no_geo.empty:
        # We'll do a simpler grouping here, since there's no geometry to weight
        # Use the same aggregator logic, except there's no area fraction
        # We do need to create matching out_cols in order to unify them.
        for col in source_cols:
            if pd.api.types.is_numeric_dtype(source_no_geo[col]):
                out_col = f"{col}{suffix}"
                source_no_geo[out_col] = source_no_geo[col]
            else:
                out_col = f"{col}{suffix}"
                source_no_geo[out_col] = source_no_geo[col]

        grouped_nogeo = source_no_geo.groupby(target_id_col).agg(agg_dict).reset_index()

        # Combine the geometry-based and no-geometry-based results
        combined = pd.concat([grouped_geo, grouped_nogeo], ignore_index=True)

        # Because a district may appear in both, do a final groupby
        # so that numeric columns sum up, string columns unify.
        grouped_all = combined.groupby(target_id_col).agg(agg_dict).reset_index()
    else:
        # If no non-geometry rows exist, just use grouped_geo
        grouped_all = grouped_geo

    # ========== 8) Merge aggregated columns back onto the target geometry ==========

    final = gdf_target.drop_duplicates(subset=[target_id_col]).merge(
        grouped_all, on=target_id_col, how="left"
    )

    # 9) Convert back to GeoDataFrame
    final_gdf = gpd.GeoDataFrame(final, geometry="geometry", crs=gdf_target.crs)

    return final_gdf

