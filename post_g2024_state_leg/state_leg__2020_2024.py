import warnings
import geopandas as gpd
import pandas as pd
import numpy as np
import shapely
from shapely.geometry import shape
import matplotlib.pyplot as plt
from mapping_utilities import get_all_mapfiles, concatenate_geodata
from political_geospatial import create_state_stats
import os
import time
import fiona
from nyt_2024_process_and_overlay import *


def prepare_2016_v_2020_data(
    gpkg_directory__16_v_20=r"/Users/aspencage/Documents/Data/output/g2024/2020_2016_pres_state_leg",
    incl_geo=False
):
    """
    Loads 2016 vs. 2020 data from multiple .gpkg files and filters to relevant columns.
    """
    # Path to the folder containing multiple .gpkg files
    gpkg_files = get_all_mapfiles(
        gpkg_directory__16_v_20,
        extension=".gpkg",
        filename_regex=".*_SLDL.*"
    )
    gdf_2020 = concatenate_geodata(gpkg_files, print_=0)

    keep_cols = [
        "State",
        "District", 
        "pres_dem_share_total_2020",
        "pres_dem_share_two_way_2020",
        "third_party_vote_share_2020",
        # possibly more columns if needed
    ] 
    if incl_geo:
        keep_cols.append("geometry")

    # Ensure District IDs are unique across states
    gdf_2020["District Number"] = gdf_2020["District"]
    gdf_2020["District"] = gdf_2020["State"] + "-" + gdf_2020["District"].astype(str)

    gdf_2020 = gdf_2020[keep_cols].copy()

    return gdf_2020


def merge_in_2020_data(results_2024, gdf_2020):
    """
    Merges 2020 data into 2024 results on District, calculates differences,
    and sets up a consistent column ordering.
    """
    grouped = results_2024.merge(gdf_2020, on='District', how='left')

    # Calculate differences
    grouped['pres_dem_share_total_diff'] = grouped['pres_dem_share_total_2024'] - grouped['pres_dem_share_total_2020']
    grouped['pres_dem_share_two_way_diff'] = grouped['pres_dem_share_two_way_2024'] - grouped['pres_dem_share_two_way_2020']
    grouped['third_party_vote_share_diff'] = grouped['third_party_vote_share_2024'] - grouped['third_party_vote_share_2020']

    # Example "likely_error_detected" threshold
    grouped['likely_error_detected'] = np.where(
        grouped['pres_dem_share_two_way_diff'].abs() > 0.20,
        "Greater than 20 pp swing",
        None
    )

    column_order = [
        'State',
        'District',
        'coverage_percentage',
        'split_coverage_percentage', 
        'likely_error_detected',
        'accuracy_score',
        'votes_rep_by_area_2024',
        'votes_dem_by_area_2024', 
        'votes_third_party_by_area_2024',
        'votes_total_by_area_2024', 
        'votes_two_way_by_area_2024',
        'pres_dem_share_total_2024', 
        'pres_dem_share_two_way_2024',
        'third_party_vote_share_2024', 
        'pres_dem_share_total_2020', 
        'pres_dem_share_two_way_2020',
        'third_party_vote_share_2020',
        'pres_dem_share_total_diff',
        'pres_dem_share_two_way_diff',
        'third_party_vote_share_diff',
        'geometry'
    ]
    # Reorder columns (if they exist)
    try:
        grouped = grouped[column_order]
    except KeyError:
        print("Some columns from column_order are not present in the dataframe.")
    
    return grouped 


if __name__ == "__main__":

    fp_precinct_geojson_2024 = (
        r"/Users/aspencage/Documents/Data/output/post_2024/"
        r"2020_2024_pres_compare/nyt_pres_2024_simplified.geojson"
    ) # NOTE - this is the NYT TopoJSON file, adjusting with a command line utility to GeoJSON
    fp_precincts__by_state_2024 = (
        r"/Users/aspencage/Documents/Data/input/post_g2024/nyt_2024_prez_data/geojsons-by-state"
    ) # NOTE this is the NYT GeoJSONs for each state, downloaded one-by-one

    sldl_directory = (
        r"/Users/aspencage/Documents/Data/input/post_g2024/"
        r"comparative_presidential_performance/TIGER2023_SLDL"
    )

    # -------------------------------------------------------------------------
    # A) LOAD AND PREPARE DATA
    # -------------------------------------------------------------------------
    nyt_geojson_fps = get_all_mapfiles(fp_precincts__by_state_2024, extension=".geojson",print_=False)
    precincts_2024 = load_and_prepare_precincts(
        nyt_geojson_fps, 
        crs="EPSG:2163"
    )
    districts = load_and_prepare_state_leg_districts(
        sldl_directory, 
        crs="EPSG:2163")

    # -------------------------------------------------------------------------
    # B) CALCULATE COVERAGE AND SPLIT PRECINCTS
    # -------------------------------------------------------------------------
    coverage_gdf = calculate_coverage(precincts_2024, districts)
    coverage_split_gdf = calculate_split_precinct_coverage(precincts_2024, coverage_gdf)
    coverage_stats_by_state = create_state_stats(coverage_split_gdf).sort_values(
        "accuracy_score", ascending=True
    )
    print("\nCoverage Stats by State (Sorted by accuracy_score):")
    print(coverage_stats_by_state.head(10))

    # -------------------------------------------------------------------------
    # C) PROCESS AREA-WEIGHTED VOTE TOTALS (e.g., 2024)
    # -------------------------------------------------------------------------
    print("\nCalculating area-weighted vote totals for 2024...")
    final_2024_gdf = process_votes_area_weighted(
        gdf_precinct=precincts_2024,
        gdf_districts=coverage_split_gdf, 
        year="2024",
        district_col="District",
        extra_district_cols=[
            "coverage_percentage",
            "split_coverage_percentage"
        ]
    )

    # ------------------------------------------------------------------
    # 4. Merge in 2020 data for comparison
    # ------------------------------------------------------------------
    print('Merging 2020 and 2024 Presidential results by 2024 State Legislative District...')
    gdf_2020 = prepare_2016_v_2020_data()  # reads 2020 data from GPKGs
    grouped = merge_in_2020_data(final_2024_gdf, gdf_2020)

    # ------------------------------------------------------------------
    # 5. Export results
    # ------------------------------------------------------------------
    out_dir = (
        r'/Users/aspencage/Documents/Data/output/post_2024/'
        r'2020_2024_pres_compare'
    )
    os.makedirs(out_dir, exist_ok=True)
    os.chdir(out_dir)

    time_str = time.strftime("%y%m%d-%H%M%S")
    outfile = f"pres_in_district__20_v_24_comparison_{time_str}"
    
    grouped.to_file(outfile + ".gpkg", layer="sldl_coverage", driver="GPKG")
    df_no_geom = grouped.drop(columns="geometry")
    df_no_geom.to_csv(outfile + ".csv", index=False)

    # ------------------------------------------------------------------
    # 6. Create & Export State Stats
    # ------------------------------------------------------------------
    print("Creating summary stats by state...")
    state_stats = create_state_stats(df_no_geom)
    state_stats.to_csv("pres_in_district__20_v_24_comparison__state_stats_"+time_str+".csv", index=False)
    print("Done!")
