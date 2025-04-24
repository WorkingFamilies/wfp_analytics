import warnings
import geopandas as gpd
import pandas as pd
import numpy as np
import shapely
from shapely.geometry import shape
import matplotlib.pyplot as plt
from mapping_utilities import get_all_mapfiles, concatenate_geodata
from political_geospatial import create_state_stats, load_and_prepare_precincts, load_and_prepare_state_leg_districts, calculate_coverage, calculate_split_precinct_coverage, process_votes_area_weighted, merge_in_2020_data
from include_historical_data import prepare_2016_v_2020_data, merge_in_2020_data
import os
import time
import fiona

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
    coverage_stats_by_state = create_state_stats(coverage_split_gdf)

    print("\nCoverage Stats by State:")
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
