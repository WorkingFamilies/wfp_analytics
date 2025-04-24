import warnings
import geopandas as gpd
import pandas as pd
import numpy as np
import shapely
from shapely.geometry import shape
import matplotlib.pyplot as plt
from mapping_utilities import get_all_mapfiles, concatenate_geodata
from political_geospatial import *
import os
import time
import fiona
from nyt_2024_process_and_overlay import *

# This script is to house functions that are used to include historical data in the analysis.

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
    grouped = results_2024.merge(gdf_2020, on='District', how='left',suffixes=('', '_2020'))

    # Calculate differences
    grouped['pres_dem_share_total_diff'] = grouped['pres_dem_share_total_2024'] - grouped['pres_dem_share_total_2020']
    grouped['pres_dem_share_two_way_diff'] = grouped['pres_dem_share_two_way_2024'] - grouped['pres_dem_share_two_way_2020']
    grouped['third_party_vote_share_diff'] = grouped['third_party_vote_share_2024'] - grouped['third_party_vote_share_2020']

    # Lots of code to column order
    column_order = [
        'State',
        'District',
        'coverage_percentage',
        'split_coverage_percentage',
        'likely_error_detected',
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

    # Get all existing columns in the DataFrame
    all_cols = list(grouped.columns)

    # Determine which columns from column_order actually exist in the DataFrame
    existing_in_df = [col for col in column_order if col in all_cols]

    # Determine which columns from column_order are missing in the DataFrame
    missing_in_df = [col for col in column_order if col not in all_cols]

    # Warn if some of the columns from column_order are missing
    if missing_in_df:
        print("Warning: The following columns from column_order are not present in the DataFrame:")
        print("  ", missing_in_df)

    # Identify any additional columns that are not in column_order
    additional_cols = [c for c in all_cols if c not in column_order]

    # Final order = columns that exist in column_order + any additional columns
    final_col_order = existing_in_df + additional_cols

    # Reorder the DataFrame
    grouped = grouped[final_col_order]

    return grouped 

