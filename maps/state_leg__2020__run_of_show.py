# Filename: state_leg__2020__run_of_show.py

# ------------------------------------------------------------------
#       This script re-runs 2020 data with newly hard-coded
#       State Legislative District overrides (e.g., MI, WI, etc.).
#       It largely mirrors the approach from `state_leg__2024__run_of_show.py`
#       but focuses on 2020 precinct data only.
#
# NOTE: This should be rerun when new district maps are used.
# ------------------------------------------------------------------

import os
import time
import pandas as pd
import numpy as np
import re
import warnings 

# -- Import your custom modules used in the 2024 script
from political_geospatial import (
    load_and_prepare_precincts,
    load_and_prepare_districts_w_overrides,
    process_votes_area_weighted,
    calculate_coverage,
    calculate_split_precinct_coverage
)
from dataset_utilities import reorder_columns, prettify_dataset
from mapping_utilities import map_code_to_value, get_all_mapfiles
from mapping_dicts import *  # if needed

# Disable SettingWithCopyWarning
pd.options.mode.chained_assignment = None

# Suppress all warnings from pyogrio
warnings.filterwarnings("ignore", module="pyogrio")

# ------------------------------------------------------------------
# USER SETTINGS / STATE SELECTION
# ------------------------------------------------------------------

# Select which chamber to run
chamber = 'sldu'

print(f"\nRunning state legislative analysis for {chamber}...")

# Adjust these as needed
base_output_dir = r"/Users/aspencage/Documents/Data/output/post_2024/2020_rerun_new_districts"
os.makedirs(base_output_dir, exist_ok=True)
os.chdir(base_output_dir)
print(f"Output directory: {base_output_dir}")

# Choose which states to include. For example:
max_tier = 1

state_tiers = {
    1: ["NJ","VA","PA","NC","AZ","MI","WI","GA"],
    2: ["TX","CO","OH"] 
}

if max_tier > 0:
    selected_states = [
        state
        for tier, states in state_tiers.items()
        if tier <= max_tier
        for state in states
    ]
else:
    # selected_states = ["MI","WI"] # for testing
    selected_states = ['PA','MI','VA']

print(f"States selected for 2020 re-run analysis: {selected_states}")

# For states where geometry calculations are not performed
no_geom_states = ["PA"]  # Example: geometry usage might be disabled for these states

# ------------------------------------------------------------------
# FILE PATHS
# ------------------------------------------------------------------

# VEST precincts 
base_fp_2020 = (
    r"/Users/aspencage/Documents/Data/input/post_g2024/"
    r"comparative_presidential_performance/"
    r"2020_precinct_level_election_returns_dataverse_files"
)

# Path to your main Tiger 2024 legislative districts
if chamber == "sldl":
    state_leg_map_directory = (
        r"/Users/aspencage/Documents/Data/input/post_g2024/"
        r"comparative_presidential_performance/TIGER2024_SLDL"
    )
elif chamber == "sldu":
    state_leg_map_directory = (
        r"/Users/aspencage/Documents/Data/input/post_g2024/"
        r"comparative_presidential_performance/TIGER2024_SLDU"
    )

# Any “override” district shapefiles for states where the 
# legislature has changed the lines from the default TIGER/LINE 2024
mi_state_house_fp = (
    r"/Users/aspencage/Documents/Data/"
    r"input/post_g2024/comparative_presidential_performance/"
    r"G2024/Redistricted/State House/"
    r"mi_sldl_2024_Motown_Sound_FC_E1_Shape_Files/"
    r"91b9440a853443918ad4c8dfdf52e495.shp"
)

wi_state_house_fp = (
    r"/Users/aspencage/Documents/Data/input/post_g2024/"
    r"comparative_presidential_performance/G2024/Redistricted/"
    r"State House/wi_sldl_adopted_2023/"
    r"AssemblyDistricts_2023WIAct94/AssemblyDistricts_2023WIAct94.shp"
)

nc_state_house_fp = (
    r"/Users/aspencage/Documents/Data/input/post_g2024/"
    r"comparative_presidential_performance/G2024/Redistricted/"
    r"State House/nc_sldl_adopted_2023/SL 2023-149 House - Shapefile/"
    r"SL 2023-149.shp"
)

ga_state_house_fp = (
    r"/Users/aspencage/Documents/Data/input/post_g2024/"
    r"comparative_presidential_performance/G2024/Redistricted/"
    r"State House/ga_sldl_adopted_2023/House-2023 shape.shp"
)

mi_state_senate_fp = (
    r'/Users/aspencage/Documents/Data/input/post_g2024/comparative_presidential_performance/'
    r'G2024/Redistricted/State Senate/mi_sldu_adopted_2026/Shapefile/2d55a8a8eb43439bacb5be8d7596f4ad.shp'
    )

# ------------------------------------------------------------------
# OVERRIDE CONFIGS: 
#   States re-mapped lines
# ------------------------------------------------------------------
district_col = "District"

override_configs = {
    "sldl": {
        "MI": {
                "path" : mi_state_house_fp,
                "col_map" : {'DISTRICTNO':district_col}, 
                "geo_source": "Michigan SOS: 2024 Motown Sound FC E1 Shape File",
                "process_override" : True,
                "keep_all_columns" : False
            },
        "WI": {
                "path" : wi_state_house_fp,
                "col_map" : {'ASSEMBLY':district_col}, 
                "geo_source": "Wisconsin: 2023-94 State House",
                "process_override" : True,
                "keep_all_columns" : False
            },
        "NC": {
                "path" : nc_state_house_fp,
                "col_map" : {'DISTRICT':district_col}, 
                "geo_source": "North Carolina: 2023-149 State House",
                "process_override" : True,
                "keep_all_columns" : False
            },
        "GA": {
                "path" : ga_state_house_fp,
                "col_map" : {'DISTRICT':district_col},
                "geo_source": "Georgia: 2023 State House, Bill 1EX, via RDH",
                "process_override" : True,
                "keep_all_columns" : False
            }
        },
    "sldu": {
        "MI": {
                "path" : mi_state_senate_fp,
                "col_map" : {'DISTRICTNO':district_col}, 
                "geo_source": "Michigan SOS: 2024 405 Crane A1 Shapefile for 2026+",
                "process_override" : True,
                "keep_all_columns" : False
            }
        }
    }


# ------------------------------------------------------------------
# 1. LOAD & PREPARE PRECINCTS (2020)
# ------------------------------------------------------------------

print("\nLoading 2020 precinct-level data...")
mapfiles_2020 = get_all_mapfiles(
    base_fp_2020,
    extension=".shp"
    )

precincts_2020 = load_and_prepare_precincts(
    fp_precinct=mapfiles_2020,
    crs="EPSG:2163",           # recommended for area calculations in US
    drop_na_columns=["G20PRERTRU", "G20PREDBID"],  # adjust columns to match your 2020 data
    print_=True
)
g20_pres_cols = [col for col in precincts_2020.columns if re.match("G20PRE*", col)]
precincts_2020["votes_total"] = precincts_2020[g20_pres_cols].sum(axis=1)

cols_to_keep = [
    'COUNTY', 
    'DISTRICT', 
    'NAME', 
    'G20PRERTRU', 
    'G20PREDBID',
    'votes_total',
    'geometry'
  ]
precincts_2020 = precincts_2020[cols_to_keep]
precincts_2020.rename(columns={
    "G20PREDBID": "votes_dem",
    "G20PRERTRU": "votes_rep"
    }, inplace=True)
precincts_2020["precinct_source"] = "UFL Vest 2020 Precinct-level data"

# Filter to selected states, if applicable
state_col = "State"
#if selected_states is not None:
#    precincts_2020 = precincts_2020.loc[precincts_2020[state_col].isin(selected_states)]

# ------------------------------------------------------------------
# 2. LOAD & PREPARE DISTRICTS (with overrides)
# ------------------------------------------------------------------

print("\nLoading and preparing district shapefiles with 2020 overrides...")

districts_2020 = load_and_prepare_districts_w_overrides(
    state_leg_map_directory, 
    crs="EPSG:2163",
    override_configs=override_configs[chamber],
    district_col=district_col,
    main_source_name="Census TIGER/LINE 2024",
    chamber=chamber
)

if selected_states is not None:
    districts_2020 = districts_2020.loc[districts_2020[state_col].isin(selected_states)]

# ------------------------------------------------------------------
# 3. PROCESS 2020 AREA-WEIGHTED VOTE TOTALS
# ------------------------------------------------------------------

print("\nCalculating 2020 area-weighted vote totals in updated districts...")

districts_w_2020_vote = process_votes_area_weighted(
    gdf_precinct=precincts_2020,
    gdf_districts=districts_2020,
    year="2020",
    district_col=district_col,
    extra_precinct_cols=["precinct_source"],  # or any extra columns from precincts
    extra_district_cols=["geo_source","dist_area"]
)

# 4. CALCULATE COVERAGE METRICS -> skipped, not needed for 2020 re-run

# ------------------------------------------------------------------
# 5. FINAL CLEANUP / EXPORT
# ------------------------------------------------------------------

# If you want to reorder columns or remove extraneous ones:
first_cols = [
    "State",
    "District",
    "votes_dem_by_area_2020",
    "votes_rep_by_area_2020",
    "votes_total_by_area_2020",
    "pres_dem_share_two_way_2020"
    # etc. – whatever columns you prefer on the front
]
final_2020_district_gdf = reorder_columns(districts_w_2020_vote, first_cols)

# If you want to prettify (round decimals, etc.):
gdf_pretty_2020 = prettify_dataset(
    final_2020_district_gdf,
    round_decimals=3,
    int_columns=["votes_dem_by_area_2020","votes_rep_by_area_2020","votes_total_by_area_2020"],
    percent_columns=None,
    sort_by="District"
)

# Output filenames
time_str = time.strftime("%Y%m%d_%H%M%S")
outfile_base = f"pres_in_district__2020_rerun__{time_str}__{chamber}"

print(f"\nExporting final 2020 re-run results to {outfile_base}.gpkg and .csv...")
gdf_pretty_2020.to_file(outfile_base + ".gpkg", layer=f"{chamber}_coverage_2020", driver="GPKG")

# Export a CSV without geometry
df_no_geom_2020 = gdf_pretty_2020.drop(columns="geometry")
df_no_geom_2020.to_csv(outfile_base + ".csv", index=False)

print("\nDone! Your 2020 data has been re-run using the updated district overrides.")
