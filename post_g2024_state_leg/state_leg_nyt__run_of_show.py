# ------------------------------------------------------------------
# NOTE: Instead of fixing bad state data afterwards, it can be fixed 
# before the main script runs. This way additional imports (e.g., 
# for county-level data) will also be fixed.
# ------------------------------------------------------------------

# see state_leg__fix_precinct_testing.py

# TODO - PA, VA precinct data 
# State House NC boundaries 
# State Senate MI boundaries 

# ------------------------------------------------------------------
# I. RUN THE MAIN STATE LEG SCRIPT
# ------------------------------------------------------------------


import os

out_dir = (
    r'/Users/aspencage/Documents/Data/output/post_2024/'
    r'2020_2024_pres_compare'
)
os.makedirs(out_dir, exist_ok=True)
os.chdir(out_dir)

print("Running main state leg script...")

from state_leg__2020_2024 import * 

# -------------------------------
# File paths (adjust as needed)
# -------------------------------
fp_precincts_fixed = (
    r'precincts__2024_pres__fixed__20250311_150129.gpkg'
) # NOTE - currently the output of state_leg__fix_precinct_testing

sldl_directory = (
    r"/Users/aspencage/Documents/Data/input/post_g2024/"
    r"comparative_presidential_performance/TIGER2023_SLDL"
)

# -------------------------------------------------------------------------
# A) LOAD AND PREPARE DATA
# -------------------------------------------------------------------------
precincts_2024 = load_and_prepare_precincts(
    fp_precincts_fixed, 
    crs="EPSG:2163"
)
districts = load_and_prepare_state_leg_districts(sldl_directory, crs="EPSG:2163")

# -------------------------------------------------------------------------
# B) CALCULATE COVERAGE AND SPLIT PRECINCTS
# -------------------------------------------------------------------------
coverage_gdf = calculate_coverage(precincts_2024, districts)
coverage_split_gdf = calculate_split_precinct_coverage(precincts_2024, coverage_gdf)
coverage_stats_by_state = create_state_stats(coverage_split_gdf)

# -------------------------------------------------------------------------
# C) PROCESS AREA-WEIGHTED VOTE TOTALS (e.g., 2024)
# -------------------------------------------------------------------------
# Assume precincts_2024 has 'votes_dem', 'votes_rep', 'votes_total', etc.
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
) # NOTE - this currently has WI

# ------------------------------------------------------------------
# 4. Merge in 2020 data for comparison
# ------------------------------------------------------------------
print('Merging 2020 and 2024 Presidential results by 2024 State Legislative District...')
gdf_2020 = prepare_2016_v_2020_data()  # reads 2020 data from GPKGs
grouped = merge_in_2020_data(final_2024_gdf, gdf_2020)

# ------------------------------------------------------------------
# 5. Export results
# ------------------------------------------------------------------


time_str = time.strftime("%y%m%d-%H%M%S")
outfile = f"pres_in_district__20_v_24_comparison_{time_str}"

grouped.to_file(outfile + ".gpkg", layer="sldl_coverage", driver="GPKG")
df_no_geom = grouped.drop(columns="geometry")
df_no_geom.to_csv(outfile + ".csv", index=False)

# Formerly bad data fixed in states afterwards, but fixing at the precinct level fixes county results as well. 
# These functions are still available in state_leg__fix_specific_state. 

# ------------------------------------------------------------------
# FROM THE NY COUNTY DIAGNOSTICS SCRIPT
# NOTE - instead of loading from NYT precinct, where data is fixed/modified from NYT precincts, we should use that gpkg instead
# Otherwise, the county analysis will retain the incorrect data from the NYT precincts
# However, might be easier said than done, as fix_bad_states_data goes state by state. 
# ------------------------------------------------------------------

print("Running county-level diagnostics...")

from nyt_2024_diagnostics_county_level import *

# -------------------------------
# File paths 
# -------------------------------

county_shapefile = (
    r"/Users/aspencage/Documents/Data/input/post_g2024/"
    r"comparative_presidential_performance/tl_2019_us_county"
)
time_string = time.strftime("%Y%m%d_%H%M%S")
outfile_base = (
    f"/Users/aspencage/Documents/Data/output/post_2024/2020_2024_pres_compare/nyt_pres_2024_by_county_{time_string}"
)

# -------------------------------------------------------------------------
# A) LOAD AND PREPARE DATA
# -------------------------------------------------------------------------
districts = load_and_prepare_counties(county_shapefile, crs="EPSG:2163")

# precincts_2024 already loaded

# -------------------------------------------------------------------------
# B) CALCULATE COVERAGE AND SPLIT PRECINCTS
# -------------------------------------------------------------------------
coverage_gdf = calculate_coverage(precincts_2024, districts, district_col='County')
coverage_split_gdf = calculate_split_precinct_coverage(precincts_2024, coverage_gdf, district_col='County')
coverage_stats_by_state = create_state_stats(coverage_split_gdf)

# -------------------------------------------------------------------------
# C) PROCESS AREA-WEIGHTED VOTE TOTALS (e.g., 2024)
# -------------------------------------------------------------------------
# Assume precincts_2024 has 'votes_dem', 'votes_rep', 'votes_total', etc.
print("\nCalculating area-weighted vote totals for 2024 in counties...")
coverage_split_gdf.rename(columns={"GEOID":"fips"},inplace=True)
county_diagnostics_gdf = process_votes_area_weighted(
    gdf_precinct=precincts_2024,
    gdf_districts=coverage_split_gdf, 
    year="2024",
    district_col="County",
    extra_district_cols=[
      "coverage_percentage",
      "split_coverage_percentage",
      "fips"
    ]
)

# -------------------------------------------------------------------------
# D) SAVE RESULTS
# -------------------------------------------------------------------------
print(f"\nSaving final county coverage & vote data to {outfile_base} as .gpkg and .csv...")
county_diagnostics_gdf.to_file(outfile_base+".gpkg", layer="sldl_coverage", driver="GPKG")
county_diagnostics_df_no_geom = county_diagnostics_gdf.drop(columns="geometry")
county_diagnostics_df_no_geom.to_csv(outfile_base + ".csv", index=False)
print("Saved successfully.\n")


# ------------------------------------------------------------------
# FROM THE NYT COUNTY V COUNTY AUXILLIARY COMPARISON SCRIPT
# compare_county_nyt_and_alternative.py 
# ------------------------------------------------------------------

print("Comparing NYT county data to alternative county data...")

from prepare_precinct_data import * 
from mapping_utilities import load_generic_file
import geopandas as gpd
import time 
import matplotlib.pyplot as plt

county_aux_data_fp = r"/Users/aspencage/Documents/Data/input/post_g2024/comparative_presidential_performance/G2024/County/2024_Presidential_by_County_Merged__most_recent_and_fips.csv"

county_comparison = prepare_precinct_data_multirow(
    data_input=county_aux_data_fp,
    precinct_col="fips",
    office_col="office",
    party_col="party",
    vote_col="votes",
    offices=["President "],
    dem_party="D",
    rep_party="R",
    keep_geometry=False,
    has_header=True,
    encoding="utf-8",
    col_names=None,
    dtypes={"votes":int},
    standardize_precinct=False,
    state=None,
    unique_when_combined=["fips","office","candidate"]
)

county_comparison.rename(columns={"D_President ": "D_President", "R_President ": "R_President"}, inplace=True)
county_comparison['D_President'] = county_comparison.D_President.astype(int)
county_comparison['R_President'] = county_comparison.R_President.astype(int)
county_comparison["Twoway_Total_President"] = county_comparison["D_President"] + county_comparison["R_President"]
county_comparison["D_President_Twoway_Share"] = county_comparison["D_President"] / county_comparison["Twoway_Total_President"]

nyt_counties = county_diagnostics_gdf

county_comparison['fips'] = county_comparison['fips'].astype(int)
nyt_counties['fips'] = nyt_counties['fips'].astype(int)

county_comp_merged = pd.merge(
  nyt_counties, 
  county_comparison, 
  on="fips", 
  how="outer")
county_comp_merged['pres_dem_share_two_way_2024__diff'] = county_comp_merged['pres_dem_share_two_way_2024'] - county_comp_merged['D_President_Twoway_Share']
county_comp_merged['pres_dem_share_two_way_2024__diff_abs'] = county_comp_merged['pres_dem_share_two_way_2024__diff'].abs()
county_comp_merged['votes_two_way_by_area_2024__diff'] = county_comp_merged['votes_two_way_by_area_2024'] - county_comp_merged['Twoway_Total_President']
county_comp_merged['votes_two_way_by_area_2024__diff_abs'] = county_comp_merged['votes_two_way_by_area_2024__diff'].abs()
county_comp_merged['votes_dem_by_area_2024__diff'] = county_comp_merged['votes_dem_by_area_2024'] - county_comp_merged['D_President']
county_comp_merged['votes_dem_by_area_2024__diff_abs'] = county_comp_merged['votes_dem_by_area_2024__diff'].abs()

time_str = time.strftime("%Y%m%d_%H%M%S")
outfile_base = f'/Users/aspencage/Documents/Data/output/post_2024/2020_2024_pres_compare/nyt_pres_2024_by_county_comparison_' + time_str

county_comp_merged.to_file(outfile_base + ".gpkg", driver="GPKG")
county_comp_df_no_geom = county_comp_merged.drop(columns="geometry",errors="ignore")
county_comp_df_no_geom.to_csv(outfile_base + ".csv", index=False)

print(f"Comparison saved to {outfile_base}.csv and {outfile_base}.gpkg")


# ------------------------------------------------------------------
# FROM THE SCRIPT TO APPEND COUNTY RESULTS
# ------------------------------------------------------------------

from append_county_accuracy_metric import *
from dataset_utilities import *
import re 

print("Appending county-level accuracy metric to State Legislative data...")
tier_1_states = ["NJ","VA","PA","NC","AZ","MI","WI"]

main_gdf = grouped.copy()
# main_gdf = main_gdf[main_gdf["State"].isin(tier_1_states)]

counties = county_comp_merged

counties["State"] = counties["County"].str.extract(r', ([A-Z]{2})$')
# counties = counties[counties["State"].isin(tier_1_states)]

# Run the area-weighted aggregation
counties.rename(columns=
                {
                    "votes_dem_by_area_2024": "votes_dem_nyt_county",
                    'votes_rep_by_area_2024': 'votes_rep_nyt_county',
                    'D_President': 'votes_dem_county_aux',
                    'R_President': 'votes_rep_county_aux',
                    'votes_two_way_by_area_2024':'votes_two_way_nyt_county',
                    'Twoway_Total_President':'votes_two_way_county_aux'
                  }, inplace=True)
print("Running area-weighted aggregation from one dataset into the other...")
cols_to_add = [
    'votes_dem_nyt_county',
    'votes_rep_nyt_county',
    'votes_dem_county_aux',
    'votes_rep_county_aux',
    'votes_two_way_nyt_county',
    'votes_two_way_county_aux',
    'votes_two_way_by_area_2024__diff',
    'County'
    ]
suffix = "__from_county_comp"  

gdf_with_county_comp = process_area_weighted_metric(
    gdf_source=counties,
    gdf_target=main_gdf,
    source_cols=cols_to_add,
    target_id_col="District",
    suffix=suffix,
    agg_dict=None,
    print_warnings=True,
    return_intersection=False
)

gdf_with_county_comp["ratio__county_aux_twoway_to_precinct_twoway"] = gdf_with_county_comp["votes_two_way_county_aux" + suffix] / gdf_with_county_comp["votes_two_way_by_area_2024"]
gdf_with_county_comp["ratio__nyt_county_two_way_to_precinct_two_way"] = gdf_with_county_comp["votes_two_way_nyt_county" + suffix] / gdf_with_county_comp["votes_two_way_by_area_2024"]
gdf_with_county_comp["ratio__nyt_county_two_way_to_county_aux_two_way"] = gdf_with_county_comp["votes_two_way_nyt_county" + suffix] / gdf_with_county_comp["votes_two_way_county_aux" + suffix]
gdf_with_county_comp["pct_error__btw_county_data_ported_to_districts"] = gdf_with_county_comp["votes_two_way_by_area_2024__diff" + suffix] / gdf_with_county_comp["votes_two_way_nyt_county" + suffix]
gdf_with_county_comp["pct_error__btw_county_data_ported_to_districts_abs"] = gdf_with_county_comp["pct_error__btw_county_data_ported_to_districts"].abs()

# ------------------------------------------------------------------
# FINAL ADJUSTMENTS BEFORE EXPORT 
# ------------------------------------------------------------------

state_stats = create_state_stats(gdf_with_county_comp, cols=["pct_error__btw_county_data_ported_to_districts_abs","coverage_percentage","split_coverage_percentage"])

first_cols = [
    'State',
    'District',
    'pct_error__btw_county_data_ported_to_districts_abs',
    'coverage_percentage',
    'split_coverage_percentage',
    'likely_error_detected',
    'pres_dem_share_two_way_2024'
    ]

gdf_with_county_comp = reorder_columns(gdf_with_county_comp, first_cols)

gdf_with_county_comp['likely_error_detected'] = np.where(
  gdf_with_county_comp["pres_dem_share_two_way_diff"].abs() > 0.20,
  "Greater than 20 pp swing; ",
  ""
)

gdf_with_county_comp['likely_error_detected'] = np.where(
    gdf_with_county_comp["pct_error__btw_county_data_ported_to_districts_abs"] > 0.05,
    gdf_with_county_comp["likely_error_detected"] + "Greater than 5% discrepancy from ported over county comparison; ",
    gdf_with_county_comp["likely_error_detected"]
)

gdf_with_county_comp['likely_error_detected'] = np.where(
    gdf_with_county_comp["coverage_percentage"] < 0.80,
    gdf_with_county_comp["likely_error_detected"] + "Less than 80% of district is covered in precinct data; ",
    gdf_with_county_comp["likely_error_detected"]
)

gdf_with_county_comp['likely_error_detected'] = np.where(
    gdf_with_county_comp["split_coverage_percentage"] > 0.50,
    gdf_with_county_comp["likely_error_detected"] + "Greater than 50% of district is split in precinct data; ",
    gdf_with_county_comp["likely_error_detected"]
)

gdf_pretty = prettify_dataset(
    gdf_with_county_comp, 
    round_decimals=4, 
    int_columns=[col for col in gdf_with_county_comp.columns if re.search('^votes', col)], 
    percent_columns=None, 
    sort_by='District'
    )

gdf = gdf_pretty.copy() 

# Save to disk 
time_str = time.strftime("%Y%m%d_%H%M%S")
os.chdir(r'/Users/aspencage/Documents/Data/output/post_2024/2020_2024_pres_compare')
outfile_base = f"pres_in_district__20_v_24_comparison__{time_str}__w_county_metric"

gdf_with_county_comp.to_file(outfile_base + ".gpkg", driver="GPKG")
df_no_geom = gdf_pretty.drop(columns="geometry")
df_no_geom.to_csv(outfile_base + ".csv", index=False)

state_stats.to_csv(outfile_base + "_state_stats.csv", index=False)