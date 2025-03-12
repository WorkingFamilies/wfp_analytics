import os
import time
import warnings
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.errors import TopologicalError
from typing import Union, List

from nyt_2024_process_and_overlay import (
    load_and_prepare_precincts,
    load_and_prepare_state_leg_districts,
    calculate_coverage,
    calculate_split_precinct_coverage,
    process_votes_area_weighted
)

from state_leg__2020_2024 import (
    prepare_2016_v_2020_data,
    merge_in_2020_data
)

from political_geospatial import create_state_stats

from prepare_precinct_data import standardize_precinct_data

time_str = time.strftime("%y%m%d-%H%M%S")

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
# MAIN FUNCTION
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

import os
import time
import pandas as pd
import geopandas as gpd
from typing import Union, List

def fix_precinct_data_for_states(
    original_precinct_data: Union[str, gpd.GeoDataFrame],
    states_to_fix: List[str],
    improved_precincts_list: List[Union[str, gpd.GeoDataFrame]],
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
    original_precinct_data : str or GeoDataFrame
        If a string, path to the original precinct data (read via gpd.read_file).
        If a GeoDataFrame, used as-is.

    states_to_fix : list of str
        List of state abbreviations for which we want to replace precinct data.

    improved_precincts_list : list
        Same length as states_to_fix, each a file path or GeoDataFrame
        containing improved precinct data for that state.

    state_col : str, default="State"
        Column in both datasets that identifies which state each row belongs to.

    fix_invalid_geometries : bool, default=True
        Whether to attempt to fix invalid geometries via buffer(0).

    target_crs : str, optional
        If provided, reproject both the original data and the improved data to this CRS.

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
        optionally fixed and reprojected geometries.
    """

    # -------------------- 1) Load the original precinct data --------------------
    if isinstance(original_precinct_data, str):
        print(f"Loading original precinct data from: {original_precinct_data}")
        combined_gdf = gpd.read_file(original_precinct_data)
    else:
        combined_gdf = original_precinct_data.copy()

    if len(states_to_fix) != len(improved_precincts_list):
        raise ValueError("states_to_fix and improved_precincts_list must be the same length.")

    if state_col not in combined_gdf.columns:
        raise ValueError(f"Column '{state_col}' not found in original precinct data.")

    # If the original data lacks a CRS, assume EPSG:2163
    if combined_gdf.crs is None:
        print("Original data has no CRS. Assuming EPSG:2163.")
        combined_gdf.set_crs("EPSG:2163", inplace=True)

    # -------------------- 2) Optional fix invalid geometries --------------------
    if fix_invalid_geometries:
        invalid_count = (~combined_gdf.is_valid).sum()
        if invalid_count > 0:
            print(f"Fixing {invalid_count} invalid geometries in the original dataset via buffer(0).")
            combined_gdf["geometry"] = combined_gdf["geometry"].buffer(0)

    # -------------------- 3) Optional reprojection --------------------
    if target_crs:
        print(f"Reprojecting original data to {target_crs}...")
        combined_gdf = combined_gdf.to_crs(target_crs)

    # Record count
    print(f"Initial precinct record count: {len(combined_gdf)}")

    # -------------------- 4) Replace data for each state in states_to_fix --------------------
    for st, improved_src in zip(states_to_fix, improved_precincts_list):
        print(f"\nReplacing precinct data for state: {st}")

        # A) Remove old precinct rows for this state
        before_remove = len(combined_gdf)
        mask = (combined_gdf[state_col] == st)
        combined_gdf = combined_gdf.loc[~mask].copy()
        removed_count = before_remove - len(combined_gdf)
        print(f" - Removed {removed_count} old rows for {st}.")

        # B) Load or accept improved data as-is
        if isinstance(improved_src, str):
            print(f" - Reading improved precinct data from: {improved_src}")
            improved_gdf = gpd.read_file(improved_src)
        elif isinstance(improved_src, gpd.GeoDataFrame):
            improved_gdf = improved_src.copy()
        else:
            raise TypeError("Each item in improved_precincts_list must be a file path (str) or a GeoDataFrame.")

        # Check that state_col exists in improved data
        if state_col not in improved_gdf.columns:
            raise ValueError(
                f"Improved data for {st} is missing the '{state_col}' column. "
                "You must add/rename it before calling this function."
            )

        # If improved data lacks a CRS, assume EPSG:2163
        if improved_gdf.crs is None:
            print(f"  -> Improved data for {st} has no CRS. Assuming EPSG:2163.")
            improved_gdf.set_crs("EPSG:2163", inplace=True)

        # Optionally fix invalid geometries
        if fix_invalid_geometries:
            inv_improved = (~improved_gdf.is_valid).sum()
            if inv_improved > 0:
                print(f"  -> Fixing {inv_improved} invalid geometries for {st} via buffer(0).")
                improved_gdf["geometry"] = improved_gdf["geometry"].buffer(0)

        # Optionally reproject improved data to target_crs
        if target_crs:
            improved_gdf = improved_gdf.to_crs(target_crs)

        # C) Append the improved data
        before_append = len(combined_gdf)
        combined_gdf = pd.concat([combined_gdf, improved_gdf], ignore_index=True)
        after_append = len(combined_gdf)
        appended_count = after_append - before_append
        print(f" - Appended {appended_count} improved rows for {st}.")

    # -------------------- 5) (Optional) Export the final combined data --------------------
    if export_fixed_file:
        os.makedirs(output_dir, exist_ok=True)
        time_str = time.strftime("%Y%m%d_%H%M%S")
        out_gpkg = os.path.join(output_dir, f"{output_prefix}_{time_str}.gpkg")
        print(f"\nExporting final combined precinct data to: {out_gpkg}")
        combined_gdf.to_file(out_gpkg, driver="GPKG")
        print("Export complete.")

    # -------------------- 6) Return the updated precinct dataset --------------------
    print(f"\nFinal precinct record count: {len(combined_gdf)}")
    return gpd.GeoDataFrame(combined_gdf, geometry="geometry", crs=combined_gdf.crs)


if __name__ == "__main__":

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
        output_prefix="pres_in_state_leg__20_v_24_comparison",
        do_merge_2020=True,
        export_diagnostic=True
    )

