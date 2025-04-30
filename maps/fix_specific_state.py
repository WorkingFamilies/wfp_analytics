import os
import time
import warnings
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.errors import TopologicalError
from shapely.geometry import Polygon
from typing import Union, List

from maps.political_geospatial import *
from maps.prepare_precinct_data import * 
from maps.include_historical_data import * 
from maps.mapping_utilities import load_as_gdf

def create_diagnostic_table(
    old_df, 
    new_df, 
    join_col="District", 
    compare_cols=(
        "coverage_percentage", 
        "split_coverage_percentage",
        "votes_dem_by_area_2024",
        "votes_rep_by_area_2024",
        "votes_third_party_by_area_2024",
        "votes_total_by_area_2024",
        "pres_dem_share_total_2024",
        "pres_dem_share_two_way_2024",
        "third_party_vote_share_2024"
        )
):
    """
    Creates a diagnostic table comparing old vs. new coverage metrics 
    for each district. Returns a pandas DataFrame with columns for:
      - District
      - old_* for each compare_col
      - new_* for each compare_col
      - diff_* for each compare_col
    """
    # 1) Merge on District (left: old, right: new)
    merged = pd.merge(
        old_df[[join_col] + list(compare_cols)], 
        new_df[[join_col] + list(compare_cols)], 
        on=join_col, 
        how="outer",
        suffixes=("_old", "_new")
    )

    # 2) For each compare_col, compute difference
    for col in compare_cols:
        merged[f"diff_{col}"] = merged[f"{col}_new"] - merged[f"{col}_old"]

    return merged


def fix_single_state(
    full_gdf,
    state_to_fix,
    improved_precinct_data,
    improved_districts_data,
    do_merge_2020=True
):
    """
    Fixes coverage and vote aggregation for a single state using 
    improved precinct/district data.
    
    Returns:
      (full_gdf_fixed, diag_df)
      where full_gdf_fixed is the full dataset with updated rows for 
      the target state, and diag_df is the diagnostic comparison table 
      for that state.
    """
    # 1) Split out the target state
    if "State" not in full_gdf.columns:
        # Attempt to parse from "District" (format "XX-123")
        if "District" in full_gdf.columns:
            full_gdf["State"] = full_gdf["District"].str.split("-", expand=True)[0]
        else:
            raise ValueError("No 'State' or 'District' column found. Cannot isolate state.")

    mask_fix = (full_gdf["State"] == state_to_fix)
    gdf_to_fix = full_gdf.loc[mask_fix].copy()
    gdf_others = full_gdf.loc[~mask_fix].copy()

    print(f"\n===== Fixing State: {state_to_fix} =====")
    print(f"Extracted {len(gdf_to_fix)} rows for {state_to_fix}.")
    if gdf_to_fix.empty:
        print(f"No rows found for {state_to_fix}, skipping.")
        return full_gdf, None

    # 2) Prepare improved precinct data
    precincts_fixed = None
    if isinstance(improved_precinct_data, str):
        print(f"Loading improved precinct data from {improved_precinct_data} ...")
        precincts_fixed = load_and_prepare_precincts(improved_precinct_data, crs="EPSG:2163")
    elif isinstance(improved_precinct_data, gpd.GeoDataFrame):
        print("Using in-memory GeoDataFrame for improved precincts.")
        if improved_precinct_data.crs is None:
            improved_precinct_data.set_crs("EPSG:2163", inplace=True)
        precincts_fixed = improved_precinct_data.to_crs("EPSG:2163").copy()
    else:
        raise TypeError("`improved_precinct_data` must be a filepath or a GeoDataFrame.")

    # 3) Prepare improved district data
    districts_fixed = None
    if isinstance(improved_districts_data, str):
        print(f"Loading improved district data from {improved_districts_data} ...")
        districts_fixed = load_and_prepare_state_leg_districts(improved_districts_data, crs="EPSG:2163")
        if "State" in districts_fixed.columns:
            districts_fixed = districts_fixed.query("State == @state_to_fix").copy()
    elif isinstance(improved_districts_data, gpd.GeoDataFrame):
        print("Using in-memory GeoDataFrame for improved districts.")
        if improved_districts_data.crs is None:
            improved_districts_data.set_crs("EPSG:2163", inplace=True)
        districts_fixed = improved_districts_data.to_crs("EPSG:2163").copy()
        if "State" in districts_fixed.columns:
            districts_fixed = districts_fixed.query("State == @state_to_fix").copy()
    else:
        raise TypeError("`improved_districts_data` must be a filepath or a GeoDataFrame.")

    # 4) Re-run coverage & vote aggregation
    print("Recalculating coverage...")
    coverage_fixed = calculate_coverage(precincts_fixed, districts_fixed)
    coverage_fixed = calculate_split_precinct_coverage(precincts_fixed, coverage_fixed)

    print("Recalculating area-weighted votes for 2024...")
    results_2024_fixed = process_votes_area_weighted(
        gdf_precinct=precincts_fixed,
        gdf_districts=coverage_fixed,
        year="2024",
        district_col="District",
        extra_district_cols=["coverage_percentage", "split_coverage_percentage"]
    )

    # 5) Merge 2020 data if requested
    if do_merge_2020:
        print("Merging 2020 data...")
        gdf_2020 = prepare_2016_v_2020_data()
        results_2024_fixed = merge_in_2020_data(results_2024_fixed, gdf_2020)

    # 6) Create diagnostic table comparing old vs. new coverage
    diag_df = create_diagnostic_table(
        old_df=gdf_to_fix,
        new_df=results_2024_fixed,
        join_col="District"
    )

    # 7) Reintegrate updated rows
    updated_cols = set(results_2024_fixed.columns)
    other_cols = set(gdf_others.columns)
    for col in updated_cols.difference(other_cols):
        gdf_others[col] = np.nan
    for col in other_cols.difference(updated_cols):
        results_2024_fixed[col] = np.nan

    full_gdf_fixed = pd.concat([gdf_others, results_2024_fixed], ignore_index=True)
    print(f"Done fixing {state_to_fix}. Updated rows: {len(results_2024_fixed)}.\n")

    return full_gdf_fixed, diag_df


def export_data(full_gdf_fixed, output_dir, output_prefix, gpkg_layer):
    """
    Exports the final corrected GeoDataFrame to GPKG + CSV.
    Returns the file paths used.
    """
    os.makedirs(output_dir, exist_ok=True)
    time_str = time.strftime("%y%m%d-%H%M%S")
    outbase = f"{output_prefix}_{time_str}"

    output_gpkg = os.path.join(output_dir, outbase + ".gpkg")
    print(f"\nWriting updated dataset to {output_gpkg} ...")
    full_gdf_fixed.to_file(output_gpkg, layer=gpkg_layer, driver="GPKG")

    output_csv = os.path.join(output_dir, outbase + ".csv")
    full_gdf_fixed.drop(columns="geometry").to_csv(output_csv, index=False)
    print(f"Also wrote CSV to {output_csv}.\n")

    return output_gpkg, output_csv


###############################################################################
# STATE-LEVEL FUNCTION (Post hoc)
###############################################################################

def fix_bad_states_data(
    gpkg_path,
    gpkg_layer,
    states_to_fix,
    improved_precincts_list,
    improved_districts_list,
    output_dir=".",
    output_prefix="pres_in_state_leg_fixed",
    do_merge_2020=True,
    export_fixed_file=True,
    export_diagnostic=False,
    export_state_stats=False
):
    """
    Fixes coverage/vote data for multiple states in an existing GPKG, 
    given improved precinct/district data for each state. Exports a 
    final corrected dataset, a diagnostic table for each state, and 
    a create_state_stats table across all states.

    Parameters
    ----------
    gpkg_path : str
        Path to the existing GPKG with coverage + vote data.
    gpkg_layer : str
        Name of the layer in the GPKG (e.g. "sldl_coverage").
    states_to_fix : list of str
        A list of state abbreviations to fix (e.g. ["TX","FL"]).
    improved_precincts_list : list
        A list (same length as states_to_fix) of either filepaths or GDFs 
        for the precinct data of each state.
    improved_districts_list : list
        A list (same length as states_to_fix) of either filepaths or GDFs 
        for the district data of each state.
    output_dir : str
        Directory to which the final GPKG/CSV will be saved.
    output_prefix : str
        Prefix for the output filename. A timestamp is appended automatically.
    do_merge_2020 : bool
        Whether to re-merge 2020 data for each state.
    export_diagnostic : bool
        If True, export a CSV comparing old vs. new coverage for each state.

    Returns
    -------
    None
    """

    if len(states_to_fix) != len(improved_precincts_list) or len(states_to_fix) != len(improved_districts_list):
        raise ValueError("states_to_fix, improved_precincts_list, and improved_districts_list must all be the same length.")

    # 1) Read the full "flawed" data
    print(f"\nLoading existing coverage data from {gpkg_path} (layer: {gpkg_layer}) ...")
    full_gdf = gpd.read_file(gpkg_path, layer=gpkg_layer)
    print(f"Total records in GPKG: {len(full_gdf)}")

    # 2) Fix each state in turn
    diagnostic_tables = {}  # store diag DF for each state
    for state, prec_data, dist_data in zip(states_to_fix, improved_precincts_list, improved_districts_list):
        full_gdf, diag_df = fix_single_state(
            full_gdf=full_gdf,
            state_to_fix=state,
            improved_precinct_data=prec_data,
            improved_districts_data=dist_data,
            do_merge_2020=do_merge_2020
        )
        diagnostic_tables[state] = diag_df

    # 3) After all states are fixed, export the final dataset
    if export_fixed_file:
        output_gpkg, output_csv = export_data(full_gdf, output_dir, output_prefix, gpkg_layer)

    # 4) (Optional) Export each state's diagnostic table
    if export_diagnostic:
        for state, diag_df in diagnostic_tables.items():
            if diag_df is None:
                continue
            diag_file = os.path.join(output_dir, f"diagnostic_{state}_{time.strftime('%y%m%d-%H%M%S')}.csv")
            diag_df.to_csv(diag_file, index=False)
            print(f"Exported diagnostic table for {state} to {diag_file}")

    # 5) Create and export state-level stats across the entire updated dataset
    #    (Assuming create_state_stats is from your new_unified_script or similar)
    if export_state_stats:
        try:
            final_stats = create_state_stats(full_gdf)
            stats_file = os.path.join(output_dir, f"create_state_stats_{time.strftime('%y%m%d-%H%M%S')}.csv")
            final_stats.to_csv(stats_file, index=False)
            print(f"Exported state-level stats to {stats_file}")
        except Exception as e:
            print("Warning: Could not create state stats. Error:", e)

    print("\nAll requested states fixed. Final dataset and diagnostics exported.")

    return full_gdf


###############################################################################
# PRECINCT VARIATION: WHERE POSSIBLE TO FIX BEFORE THE STATE LEG ANALYSIS 
###############################################################################

def fix_precinct_data_for_states(
    original_precinct_data: Union[str, gpd.GeoDataFrame, pd.DataFrame],
    states_to_fix: List[str],
    improved_precincts_list: List[Union[str, gpd.GeoDataFrame, pd.DataFrame]],
    state_col: str = "State",
    fix_invalid_geometries: bool = True,
    target_crs: str = None,
    output_dir: str = ".",
    output_prefix: str = "precinct_data_fixed",
    export_fixed_file: bool = False
) -> gpd.GeoDataFrame:
    """
    Replaces precinct-level data for specific states in a multi-state precinct dataset,
    with optional geometry fixes and CRS alignment. Columns are NOT renamed or standardized.

    Parameters
    ----------
    original_precinct_data : str, GeoDataFrame, or DataFrame
        - If a string, path to the original precinct data (read via gpd.read_file).
        - If a GeoDataFrame, used as-is.
        - If a DataFrame, used as-is; geometry-based operations will be skipped 
          unless it has a 'geometry' column.

    states_to_fix : list of str
        List of state abbreviations for which we want to replace precinct data.

    improved_precincts_list : list
        Same length as states_to_fix. Each is either:
          - A file path (str)
          - A GeoDataFrame
          - A DataFrame
        containing improved precinct data for that state.

    state_col : str, default="State"
        Column in both datasets that identifies which state each row belongs to.

    fix_invalid_geometries : bool, default=True
        Whether to attempt to fix invalid geometries via buffer(0) if geometry is present.

    target_crs : str, optional
        If provided, reproject both the original data and the improved data to this CRS
        (only applies if geometry is present).

    output_dir : str, default="."
        Directory to which the final combined data is exported if export_fixed_file=True.

    output_prefix : str, default="precinct_data_fixed"
        Filename prefix for the output dataset if export_fixed_file=True.

    export_fixed_file : bool, default=False
        If True, writes the resulting corrected precinct data to a GeoPackage
        in output_dir named "{output_prefix}_{timestamp}.gpkg".

    Returns
    -------
    gpd.GeoDataFrame
        The updated precinct dataset with replaced rows for the specified states,
        optionally fixed and reprojected geometries (if present).
        If geometry is entirely absent, returns a GeoDataFrame with no valid geometry.
    """

    # -------------------- 1) Load or adapt the original precinct data --------------------

    if isinstance(original_precinct_data, str):
        # Load from file
        print(f"Loading original precinct data from: {original_precinct_data}")
        combined_gdf = gpd.read_file(original_precinct_data)
    elif isinstance(original_precinct_data, gpd.GeoDataFrame):
        combined_gdf = original_precinct_data.copy()
    elif isinstance(original_precinct_data, pd.DataFrame):
        # If it has a geometry column, convert to a GeoDataFrame
        if "geometry" in original_precinct_data.columns:
            combined_gdf = gpd.GeoDataFrame(
                original_precinct_data.copy(),
                geometry="geometry",
                crs=None  # We'll check/fix this below
            )
        else:
            # Otherwise, just store as a DataFrame for now
            combined_gdf = original_precinct_data.copy()
    else:
        raise TypeError(
            "original_precinct_data must be one of: str (filepath), GeoDataFrame, or DataFrame."
        )

    # Ensure states_to_fix and improved_precincts_list match
    if len(states_to_fix) != len(improved_precincts_list):
        raise ValueError("states_to_fix and improved_precincts_list must be the same length.")

    # Make sure the state_col exists
    if state_col not in combined_gdf.columns:
        raise ValueError(f"Column '{state_col}' not found in original precinct data.")

    combined_gdf = combined_gdf[~combined_gdf[state_col].isin(states_to_fix)]

    # -------------------- 2) Attempt geometry-based fixes if geometry is present --------------------
    # We'll define a helper function to check if combined_gdf is truly geospatial
    def has_valid_geometry(gdf_like):
        return (
            isinstance(gdf_like, gpd.GeoDataFrame) 
            and "geometry" in gdf_like.columns 
            and not gdf_like["geometry"].isna().all()
        )

    # If it's a GeoDataFrame with actual geometry:
    if has_valid_geometry(combined_gdf):
        # If the original data lacks a CRS, assume EPSG:2163
        if combined_gdf.crs is None:
            print("Original data has no CRS. Assuming EPSG:2163.")
            combined_gdf.set_crs("EPSG:2163", inplace=True)

        # Fix invalid geometries if requested
        if fix_invalid_geometries:
            invalid_count = (~combined_gdf.is_valid).sum()
            if invalid_count > 0:
                print(f"Fixing {invalid_count} invalid geometries in the original dataset via buffer(0).")
                combined_gdf["geometry"] = combined_gdf["geometry"].buffer(0)

        # Optional reprojection
        if target_crs:
            print(f"Reprojecting original data to {target_crs}...")
            combined_gdf = combined_gdf.to_crs(target_crs)

    # Record count
    print(f"Initial precinct record count: {len(combined_gdf)}")

    # -------------------- 3) Replace data for each state in states_to_fix --------------------
    for st, improved_src in zip(states_to_fix, improved_precincts_list):
        print(f"\nReplacing precinct data for state: {st}")

        # A) Remove old precinct rows for this state
        before_remove = len(combined_gdf)
        # We can do this for either DataFrame or GeoDataFrame
        mask = (combined_gdf[state_col] == st)
        combined_gdf = combined_gdf.loc[~mask].copy()
        removed_count = before_remove - len(combined_gdf)
        print(f" - Removed {removed_count} old rows for {st}.")

        # B) Load or accept improved data
        if isinstance(improved_src, str):
            print(f" - Reading improved precinct data from: {improved_src}")
            improved_gdf = gpd.read_file(improved_src)
        elif isinstance(improved_src, gpd.GeoDataFrame):
            improved_gdf = improved_src.copy()
        elif isinstance(improved_src, pd.DataFrame):
            # Convert to GeoDataFrame if there's a geometry column
            if "geometry" in improved_src.columns:
                improved_gdf = gpd.GeoDataFrame(
                    improved_src.copy(),
                    geometry="geometry",
                    crs=None  # We'll set or fix below
                )
            else:
                improved_gdf = improved_src.copy()
        else:
            raise TypeError(
                "Each item in improved_precincts_list must be a file path (str), "
                "a GeoDataFrame, or a DataFrame."
            )

        # Confirm it has the state_col
        if state_col not in improved_gdf.columns:
            raise ValueError(
                f"Improved data for {st} is missing the '{state_col}' column. "
                "You must add/rename it before calling this function."
            )

        # If geometry is valid, fix geometry issues or reproject
        if has_valid_geometry(improved_gdf):
            if improved_gdf.crs is None:
                print(f"  -> Improved data for {st} has no CRS. Assuming EPSG:2163.")
                improved_gdf.set_crs("EPSG:2163", inplace=True)

            # Fix invalid geometries
            if fix_invalid_geometries:
                inv_improved = (~improved_gdf.is_valid).sum()
                if inv_improved > 0:
                    print(f"  -> Fixing {inv_improved} invalid geometries for {st} via buffer(0).")
                    improved_gdf["geometry"] = improved_gdf["geometry"].buffer(0)

            # Reproject if requested
            if target_crs:
                improved_gdf = improved_gdf.to_crs(target_crs)

        # C) Append the improved data
        before_append = len(combined_gdf)
        combined_gdf = pd.concat([combined_gdf, improved_gdf], ignore_index=True)
        after_append = len(combined_gdf)
        appended_count = after_append - before_append
        print(f" - Appended {appended_count} improved rows for {st}.")

    # -------------------- 4) (Optional) Export the final combined data --------------------
    if export_fixed_file:
        os.makedirs(output_dir, exist_ok=True)
        time_str = time.strftime("%Y%m%d_%H%M%S")
        out_gpkg = os.path.join(output_dir, f"{output_prefix}_{time_str}.gpkg")
        print(f"\nExporting final combined precinct data to: {out_gpkg}")

        # If final data is not a GeoDataFrame (e.g. no geometry), convert to one
        # with an empty geometry column so we can export to GPKG
        if not isinstance(combined_gdf, gpd.GeoDataFrame):
            combined_gdf = gpd.GeoDataFrame(combined_gdf, geometry=[])

        # If there's still no geometry col, add one
        if "geometry" not in combined_gdf.columns:
            combined_gdf["geometry"] = None

        # If no CRS is set, pick a default
        if combined_gdf.crs is None:
            combined_gdf.set_crs("EPSG:2163", inplace=True)

        combined_gdf.to_file(out_gpkg, driver="GPKG")
        print("Export complete.")

    print(f"\nFinal precinct record count: {len(combined_gdf)}")

    # -------------------- 5) Return as a GeoDataFrame --------------------
    # If combined_gdf is already a GeoDataFrame, we return as is.
    # Otherwise, convert to GeoDataFrame, possibly with no valid geometry.
    if not isinstance(combined_gdf, gpd.GeoDataFrame):
        gdf_out = gpd.GeoDataFrame(combined_gdf, geometry=[])
        gdf_out.crs = None
        return gdf_out
    else:
        return combined_gdf

################################################################################
# PARTIAL STATE OVERWRITE (e.g., particular counties)
################################################################################


def partially_overwrite_precinct_data(
    original_precinct_data: Union[str, gpd.GeoDataFrame, pd.DataFrame],
    improved_precinct_data: Union[str, gpd.GeoDataFrame, pd.DataFrame],
    columns_to_scale: List[str],
    state_col: str = "State",
    states_to_fix: List[str] = None,
    fix_invalid_geometries: bool = True,
    target_crs: str = None,
    output_dir: str = ".",
    output_prefix: str = "precinct_data_partially_overwritten",
    export_fixed_file: bool = False,
) -> gpd.GeoDataFrame:
    """
    Partially overwrites precinct polygons in 'original_precinct_data' with 
    polygons in 'improved_precinct_data', scaling numeric columns by area 
    proportion in the leftover portion of the original precinct.

    Parameters
    ----------
    original_precinct_data : str or GeoDataFrame or DataFrame
        Original precinct data or path to it. Must at least contain geometry 
        if you want area-based scaling.  
    improved_precinct_data : str or GeoDataFrame or DataFrame
        Overwriting precinct data or path to it. Must have geometry.  
    columns_to_scale : list of str
        Name(s) of the numeric columns to be scaled by area fraction 
        for any leftover portion of the original precinct. E.g. 
        ['votes_dem', 'votes_rep', 'votes_total'].  
    state_col : str, default="State"
        Column that identifies the state (used to filter out only the states 
        you want to partially overwrite).  
    states_to_fix : list of str, optional
        If provided, only polygons with State in this list are subject to partial 
        overwrite. Everything else is kept as-is.  
    fix_invalid_geometries : bool, default=True
        Attempt to fix invalid geometries via buffering if needed.  
    target_crs : str, optional
        If provided, reproject both the original and improved data to this CRS.  
    output_dir : str, default="."
        Directory to which the final data is exported if `export_fixed_file=True`.  
    output_prefix : str, default="precinct_data_partially_overwritten"
        Filename prefix for the output dataset if `export_fixed_file=True`.  
    export_fixed_file : bool, default=False
        If True, writes the resulting data to a GeoPackage 
        in `output_dir` named "{output_prefix}_{timestamp}.gpkg".

    Returns
    -------
    gpd.GeoDataFrame
        The final precinct dataset, where overlapping portions are replaced by 
        improved polygons and leftover portions of the originals are scaled 
        according to area fraction.
    """

    # -------------------- 1) Helper to load/convert input data to GeoDataFrame --------------------
    def to_geodataframe(obj) -> gpd.GeoDataFrame:
        """Helper to convert 'obj' to a GeoDataFrame if possible."""
        if isinstance(obj, str):
            print(f"Loading data from: {obj}")
            return gpd.read_file(obj)
        elif isinstance(obj, gpd.GeoDataFrame):
            return obj.copy()
        elif isinstance(obj, pd.DataFrame):
            if "geometry" in obj.columns:
                # Convert DataFrame -> GeoDataFrame directly
                return gpd.GeoDataFrame(obj.copy(), geometry="geometry", crs=None)
            else:
                # Create an empty geometry column so merges/overlays won't break
                gdf_tmp = gpd.GeoDataFrame(obj.copy(), geometry=None)
                print("Warning: DataFrame has no geometry column – partial overwrite won't be geometric.")
                return gdf_tmp
        else:
            raise TypeError("Data must be a file path, GeoDataFrame, or DataFrame.")

    original_gdf = to_geodataframe(original_precinct_data)
    improved_gdf = to_geodataframe(improved_precinct_data)

    # -------------------- 2) Filter to the states we want to fix (optional) --------------------
    if states_to_fix:
        mask_keep = ~original_gdf[state_col].isin(states_to_fix)
        # Keep these polygons as-is
        original_gdf_outside = original_gdf[mask_keep].copy()
        # Only overwrite these
        original_gdf_inside = original_gdf[~mask_keep].copy()
    else:
        # Overwrite all polygons
        original_gdf_outside = None
        original_gdf_inside = original_gdf.copy()

    # -------------------- 3) Ensure geometry validity & CRS alignment --------------------
    def has_valid_geometry(gdf_like):
        return (
            isinstance(gdf_like, gpd.GeoDataFrame) and
            "geometry" in gdf_like.columns and
            not gdf_like["geometry"].isna().all()
        )

    def fix_and_reproject(gdf):
        """Fix invalid geoms and reproject if needed."""
        if has_valid_geometry(gdf):
            # If no CRS, assume EPSG:2163 (or your choice)
            if gdf.crs is None:
                print("Input data has no CRS. Assuming EPSG:2163.")
                gdf.set_crs("EPSG:2163", inplace=True)
            # Fix invalid geometries if requested
            if fix_invalid_geometries:
                invalid_count = (~gdf.is_valid).sum()
                if invalid_count > 0:
                    print(f"Fixing {invalid_count} invalid geometries with buffer(0).")
                    gdf["geometry"] = gdf["geometry"].buffer(0)
            # Reproject if requested
            if target_crs:
                gdf = gdf.to_crs(target_crs)
        return gdf

    original_gdf_inside = fix_and_reproject(original_gdf_inside)
    improved_gdf = fix_and_reproject(improved_gdf)
    # Also fix 'outside' if it exists, so all final pieces can be consistent
    if original_gdf_outside is not None:
        original_gdf_outside = fix_and_reproject(original_gdf_outside)

    # If the improved data also has a state_col, filter it to the same states if needed
    if states_to_fix and state_col in improved_gdf.columns:
        improved_gdf = improved_gdf[improved_gdf[state_col].isin(states_to_fix)].copy()

    # -------------------- 4) Partial Overwrite Logic (intersection & leftover) --------------------
    if not has_valid_geometry(original_gdf_inside) or not has_valid_geometry(improved_gdf):
        # If either side is not truly geospatial, just do simple overwrite
        print("No valid geometry on one dataset; performing a simple overwrite for those states.")
        frames_to_concat = []
        if original_gdf_outside is not None and not original_gdf_outside.empty:
            frames_to_concat.append(original_gdf_outside)
        frames_to_concat.append(improved_gdf)
        partially_overwritten = pd.concat(frames_to_concat, ignore_index=True)
    else:
        # A) Create a unique ID to track each original precinct row
        original_gdf_inside["orig_id"] = np.arange(len(original_gdf_inside))

        # B) Overlay for intersection
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            overlap_gdf = gpd.overlay(original_gdf_inside, improved_gdf, how="intersection")

        # Compute intersection area
        overlap_gdf["__overlap_area__"] = overlap_gdf.geometry.area

        # C) Sum overlap area by orig_id
        overlap_area_sum = (
            overlap_gdf.groupby("orig_id")["__overlap_area__"]
            .sum()
            .reset_index()
        )

        # D) Merge back to compute leftover area for each original row
        original_gdf_inside["__orig_area__"] = original_gdf_inside.geometry.area

        # Fill area sums with 0 to avoid in-place fillna warning
        overlap_area_sum["__overlap_area__"] = overlap_area_sum["__overlap_area__"].fillna(0)

        original_with_overlap = pd.merge(
            original_gdf_inside,
            overlap_area_sum,
            on="orig_id",
            how="left"
        )

        # If no overlap, the merged column might be NaN
        original_with_overlap["__overlap_area__"] = original_with_overlap["__overlap_area__"].fillna(0)

        original_with_overlap["__leftover_area__"] = (
            original_with_overlap["__orig_area__"] - original_with_overlap["__overlap_area__"]
        )

        # E) Keep only rows that still have leftover area
        leftover_mask = original_with_overlap["__leftover_area__"] > 1e-12
        leftover_gdf = original_with_overlap[leftover_mask].copy()

        # F) Union all improved polygons (shapely 2.x tip: use union_all() to avoid the deprecation warning)
        improved_union = improved_gdf.unary_union

        leftover_gdf["geometry"] = leftover_gdf["geometry"].difference(improved_union)

        # Remove any empty geometries (fully overwritten)
        leftover_gdf = leftover_gdf[~leftover_gdf.geometry.is_empty].copy()

        # G) Recompute actual leftover area after difference
        leftover_gdf["__actual_leftover_area__"] = leftover_gdf.geometry.area

        # H) Scale numeric columns
        leftover_gdf["__area_fraction__"] = (
            leftover_gdf["__actual_leftover_area__"] / leftover_gdf["__orig_area__"]
        )
        for col in columns_to_scale:
            if col in leftover_gdf.columns:
                leftover_gdf[col] = leftover_gdf[col] * leftover_gdf["__area_fraction__"]
            else:
                print(f"Warning: Column '{col}' not found in original data – skipping scale.")

        # I) Clean up leftover columns
        leftover_gdf.drop(
            columns=[
                "orig_id",
                "__orig_area__",
                "__overlap_area__",
                "__leftover_area__",
                "__actual_leftover_area__",
                "__area_fraction__",
            ],
            inplace=True,
            errors="ignore"
        )

        # -------------------- 4a) Unify CRS before concatenation (prevent ValueError) --------------------
        # We unify leftover_gdf, improved_gdf, and original_gdf_outside to the same CRS 
        # so pd.concat won't fail if they differ slightly.
        common_crs = improved_gdf.crs  # or leftover_gdf.crs, your choice

        if has_valid_geometry(leftover_gdf) and leftover_gdf.crs != common_crs:
            leftover_gdf = leftover_gdf.to_crs(common_crs)
        if has_valid_geometry(improved_gdf) and improved_gdf.crs != common_crs:
            improved_gdf = improved_gdf.to_crs(common_crs)
        if (
            original_gdf_outside is not None 
            and has_valid_geometry(original_gdf_outside) 
            and original_gdf_outside.crs != common_crs
        ):
            original_gdf_outside = original_gdf_outside.to_crs(common_crs)

        # J) Combine leftover portion + improved polygons + outside states
        frames_to_concat = []
        if original_gdf_outside is not None and not original_gdf_outside.empty:
            frames_to_concat.append(original_gdf_outside)
        frames_to_concat.append(leftover_gdf)
        frames_to_concat.append(improved_gdf)

        partially_overwritten = pd.concat(frames_to_concat, ignore_index=True)

    # -------------------- 5) (Optional) Export --------------------
    if export_fixed_file:
        # Ensure we have a GeoDataFrame
        if not isinstance(partially_overwritten, gpd.GeoDataFrame):
            partially_overwritten = gpd.GeoDataFrame(partially_overwritten, geometry="geometry")

        # If still no geometry col, create one
        if "geometry" not in partially_overwritten.columns:
            partially_overwritten["geometry"] = None

        # If no CRS is set, assign default
        if partially_overwritten.crs is None:
            partially_overwritten.set_crs("EPSG:2163", inplace=True)

        os.makedirs(output_dir, exist_ok=True)
        time_str = time.strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(output_dir, f"{output_prefix}_{time_str}.gpkg")
        print(f"Exporting partial overwrite results to: {out_path}")
        partially_overwritten.to_file(out_path, driver="GPKG")

    # -------------------- 6) Return as a GeoDataFrame --------------------
    if not isinstance(partially_overwritten, gpd.GeoDataFrame):
        gdf_out = gpd.GeoDataFrame(partially_overwritten, geometry=[])
        gdf_out.crs = None
        return gdf_out
    else:
        return partially_overwritten

if __name__ == "__main__":

    '''
    wi_precinct_fp = r'/Users/aspencage/Documents/Data/input/post_g2024/2024_precinct_level_data/wi/2024_Election_Data_with_2025_Wards_-5879223691586298781.geojson'

    wi_gdf_precincts = standardize_precinct_data(
        input_data=wi_precinct_fp,
        precinct_col="GEOID",          # rename GEOID -> 'precinct'
        dem_col="PREDEM24",      # rename -> 'votes_dem'
        rep_col="PREREP24",      # rename -> 'votes_rep'
        total_col='PRETOT24',                # no total col in data, so compute from dem+rep
        rename_map=None,               # no extra renaming
        fix_invalid_geometries=True,
        target_crs="EPSG:2163",
        retain_addl_cols=False
    )

    gpkg_path = r"/Users/aspencage/Documents/Data/output/post_2024/2020_2024_pres_compare/pres_in_state_leg__20_v_24_comparison_250226-111450.gpkg"

    fix_bad_states_data(
        gpkg_path=gpkg_path,
        gpkg_layer="sldl_coverage",
        states_to_fix=["WI"],
        improved_precincts_list=[wi_gdf_precincts],
        improved_districts_list=['/Users/aspencage/Documents/Data/input/post_g2024/comparative_presidential_performance/TIGER2023_SLDL/tl_2023_55_sldl/'],
        output_dir="/Users/aspencage/Documents/Data/output/post_2024/2020_2024_pres_compare/",
        output_prefix="pres_in_state_leg__20_v_24_comparison__fixed",
        do_merge_2020=True,
        export_diagnostic=True
    )
    '''

    precincts_fp = (
        r'/Users/aspencage/Documents/Data/output/post_2024/2020_2024_pres_compare/'
        r'precincts__2024_pres__fixed__20250314_211335.gpkg'
    )
    combined_precincts = gpd.read_file(precincts_fp)

    mi_selected_fp = (
        r'/Users/aspencage/Documents/Data/output/post_2024/2020_2024_pres_compare/'
        r'MI_SELECT_COUNTIES_ALL_cleaned_reallocated_votes_20250408_152315.csv'
        )
    
    mi_selected_gdf = load_as_gdf(
                mi_selected_fp,
                geometry_col='geometry',
                guess_format='auto'
            )
    cols_to_keep = [
        'State', 
        'precinct' ,
        'geometry',
        'votes_dem', 
        'votes_rep', 
        'votes_total'
        ]
    mi_selected_gdf = mi_selected_gdf[cols_to_keep].copy()
    
    gdf = partially_overwrite_precinct_data(
        original_precinct_data=combined_precincts,
        improved_precinct_data=mi_selected_gdf,
        columns_to_scale=[
            'votes_dem',
            'votes_rep',
            'votes_total'
            ],
        state_col = "State",
        states_to_fix = ["MI"],
        fix_invalid_geometries = True,
        target_crs = "EPSG:2163",
        output_dir = ".",
        output_prefix = "precincts__2024_pres__fixed_",
        export_fixed_file = True
    )
