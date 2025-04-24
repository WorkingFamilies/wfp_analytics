# ------------------------------------------------------------------
# NOTE: Vote returns data from official sources is overriden at 
# the precinct level at the beginning of the script.
# Data is output from state_leg__fix_precinct.py
# ------------------------------------------------------------------

# TODO - VA precinct data (may not be possible) 

# Imports 
import os
import re 
import pandas as pd 
import warnings 

from political_geospatial import *
from nyt_2024_diagnostics_county_level import *
from append_county_accuracy_metric import *
from dataset_utilities import *
from mapping_utilities import map_code_to_value
from mapping_dicts import * 
from state_leg_pres__likely_errors import state_leg_pres__likely_errors

# disable SettingWithCopyWarning
pd.options.mode.chained_assignment = None

# Suppress all warnings from pyogrio
warnings.filterwarnings("ignore", module="pyogrio")

# ------------------------------------------------------------------
# SET UP ENVIRONMENT
# ------------------------------------------------------------------
# Select which chamber to run
chamber = 'sldu'

print(f"\nRunning state legislative analysis for {chamber}...")

fp_out = (
    r'/Users/aspencage/Documents/Data/output/post_2024/'
    f'2020_2024_pres_compare/{chamber}'
)
os.makedirs(fp_out, exist_ok=True)
os.chdir(fp_out)
print(f"Output directory: {fp_out}")

# Select which states to include 
max_tier = 1

state_tiers = {
    1: ["NJ","VA","PA","NC","AZ","MI","WI","GA"],
    2: ["TX","CO","OH"] # WI + GA is technically tier 2 but it was used for validation
        } # others to come 
if max_tier > 0:
    selected_states = [state for tier, states in state_tiers.items() if tier <= max_tier for state in states]
else:
    # selected_states = None
    # selected_states = ["MI","WI"] # for testing
    selected_states = ['PA','MI','VA']

print(f"States selected for analysis: {selected_states}")

# ------------------------------------------------------------------
# RUN THE MAIN STATE LEG SCRIPT
# ------------------------------------------------------------------

# -------------------------------
# File paths (adjust as needed)
# -------------------------------

print("Running main state leg script...")

# Output of state_leg__fix_precinct_testing
if chamber == "sldl":
    fp_precincts_fixed_2024 = (
        r'/Users/aspencage/Documents/Data/output/post_2024/'
        r'2020_2024_pres_compare/sldl/'
        r'precincts__2024_pres__fixed_sldl___20250408_173552.gpkg'
    ) 
elif chamber == "sldu": 
    fp_precincts_fixed_2024 = (
        r'/Users/aspencage/Documents/Data/output/post_2024/'
        r'2020_2024_pres_compare/sldu/'
        r'precincts__2024_pres__fixed_sldu___20250409_123004.gpkg'
    )

# 2020 precinct level data
if chamber == "sldl":
    fp_precincts_fixed_2020 = (
        r'/Users/aspencage/Documents/Data/output/post_2024/'
        r'2020_rerun_new_districts/'
        r'pres_in_district__2020_rerun__sldl_20250408_173352_tier_1.gpkg'
    )
elif chamber == "sldu":
    fp_precincts_fixed_2020 = (
        r'/Users/aspencage/Documents/Data/output/post_2024/'
        r'2020_rerun_new_districts/'
        r'pres_in_district__2020_rerun__20250409_122754__sldu_tier_1.gpkg'
    )

no_geom_states = ['PA'] # states where geometry is not used to calculate vote totals, so certain metrics are not relevant

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

# filepaths to districts where we want to override the initial data 
mi_state_house_fp = (
    r'/Users/aspencage/Documents/Data/'
    r'input/post_g2024/comparative_presidential_performance/'
    r'G2024/Redistricted/State House/'
    r'mi_sldl_2024_Motown_Sound_FC_E1_Shape_Files/91b9440a853443918ad4c8dfdf52e495.shp'
    )

wi_state_house_fp = (
    r'/Users/aspencage/Documents/Data/input/post_g2024/'
    r'comparative_presidential_performance/G2024/Redistricted/'
    r'State House/wi_sldl_adopted_2023/'
    r'AssemblyDistricts_2023WIAct94/AssemblyDistricts_2023WIAct94.shp'
    )

nc_state_house_fp = (
    r'/Users/aspencage/Documents/Data/input/post_g2024/comparative_presidential_performance/'
    r'G2024/Redistricted/State House/nc_sldl_adopted_2023/'
    r'SL 2023-149 House - Shapefile/SL 2023-149.shp'
    ) # not 100% sure if current 

ga_state_house_fp = (
    r'/Users/aspencage/Documents/Data/input/post_g2024/comparative_presidential_performance/'
    r'G2024/Redistricted/State House/ga_sldl_adopted_2023/House-2023 shape.shp'
    )

mi_state_senate_fp = (
    r'/Users/aspencage/Documents/Data/input/post_g2024/comparative_presidential_performance/'
    r'G2024/Redistricted/State Senate/mi_sldu_adopted_2026/Shapefile/2d55a8a8eb43439bacb5be8d7596f4ad.shp'
    )

# -------------------------------------------------------------------------
# LOAD AND PREPARE DATA
# -------------------------------------------------------------------------

state_col = 'State'

# precinct level data 
precincts_2024 = load_and_prepare_precincts(
    fp_precincts_fixed_2024, 
    crs="EPSG:2163",
    drop_na_columns=["votes_dem","votes_rep"], # previously included 'votes_total' and 'geometry'
    print_=True
)

if selected_states is not None:
    precincts_2024 = precincts_2024.loc[precincts_2024[state_col].isin(selected_states)]

district_col = 'District'
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

districts = load_and_prepare_districts_w_overrides(
    state_leg_map_directory, 
    crs="EPSG:2163",
    override_configs=override_configs[chamber],
    district_col=district_col,
    main_source_name=f'Census TIGER/LINE 2024 Current State Legislative Districts_{chamber.upper()}',
    chamber=chamber,
)

if selected_states is not None:
    districts = districts.loc[districts[state_col].isin(selected_states)]

# -------------------------------------------------------------------------
# PROCESS AREA-WEIGHTED VOTE TOTALS (e.g., 2024)
# -------------------------------------------------------------------------
# Assume precincts_2024 has 'votes_dem', 'votes_rep', 'votes_total', etc.
print("\nCalculating area-weighted vote totals for 2024...")
districts_w_2024_vote = process_votes_area_weighted(
    gdf_precinct=precincts_2024,
    gdf_districts=districts, 
    year="2024",
    district_col="District",
    extra_precinct_cols=["precinct_source"],
    extra_district_cols=["geo_source","dist_area"]
) 
# FIXME - this currently has many 0s in WI data 

# -------------------------------------------------------------------------
# CALCULATE COVERAGE AND SPLIT PRECINCTS
# -------------------------------------------------------------------------
# drop district from precincts, which is no longer needed to avoid error from intersection with identically-named districts in both gpds
district_coverage = calculate_coverage(precincts_2024, districts_w_2024_vote, states_to_excl=no_geom_states)
final_2024_district_gdf = calculate_split_precinct_coverage(precincts_2024, district_coverage, states_to_excl=no_geom_states)

# ------------------------------------------------------------------
# Merge in 2020 data for comparison
# ------------------------------------------------------------------
print('Merging 2020 and 2024 Presidential results by 2024 State Legislative District...')
gdf_2020 = gpd.read_file(fp_precincts_fixed_2020)
grouped = merge_in_2020_data(final_2024_district_gdf, gdf_2020)
grouped.drop(columns=['geometry_2020'], errors='ignore', inplace=True)

# ------------------------------------------------------------------
# Export results
# ------------------------------------------------------------------

time_str = time.strftime("%y%m%d-%H%M%S")
outfile = f"pres_in_district__20_v_24_comparison_{time_str}"

grouped.to_file(outfile + ".gpkg", layer=f"{chamber}_coverage", driver="GPKG")
df_no_geom = grouped.drop(columns="geometry")
df_no_geom.to_csv(outfile + ".csv", index=False)

# Formerly bad data fixed in states afterwards, but fixing at the precinct level fixes county results as well. 
# These functions are still available in state_leg__fix_specific_state. 



# ------------------------------------------------------------------
# FROM THE NY COUNTY DIAGNOSTICS SCRIPT
# ------------------------------------------------------------------

print("Running county-level diagnostics...")


# -------------------------------
# File paths 
# -------------------------------

county_shapefile = (
    r"/Users/aspencage/Documents/Data/input/post_g2024/"
    r"comparative_presidential_performance/tl_2019_us_county"
)
time_string = time.strftime("%Y%m%d_%H%M%S")
outfile_base = (
    f"nyt_pres_2024_by_county__{chamber}__{time_string}"
)

# -------------------------------------------------------------------------
# A) LOAD AND PREPARE DATA
# -------------------------------------------------------------------------
counties = load_and_prepare_counties(county_shapefile, crs="EPSG:2163")

if selected_states is not None:
    counties = counties[counties[state_col].isin(selected_states)]

# precincts_2024 already loaded

# -------------------------------------------------------------------------
# CALCULATE COVERAGE AND SPLIT PRECINCTS
# -------------------------------------------------------------------------
county_coverage = calculate_coverage(
    precincts_2024, 
    counties, 
    district_col='County', 
    states_to_excl=no_geom_states
    )
county_coverage_split = calculate_split_precinct_coverage(
    precincts_2024, 
    county_coverage, 
    district_col='County', 
    states_to_excl=no_geom_states
    )

# -------------------------------------------------------------------------
# PROCESS AREA-WEIGHTED VOTE TOTALS (e.g., 2024)
# -------------------------------------------------------------------------
# Assume precincts_2024 has 'votes_dem', 'votes_rep', 'votes_total', etc.
print("\nCalculating area-weighted vote totals for 2024 in counties...")
county_coverage_split.rename(columns={"GEOID":"fips"},inplace=True)
counties_w_precinct_vote_data = process_votes_area_weighted(
    gdf_precinct=precincts_2024,
    gdf_districts=county_coverage_split, 
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
counties_w_precinct_vote_data.to_file(outfile_base+".gpkg", layer=f"{chamber}_coverage", driver="GPKG")
counties_with_precinct_data_no_geom = counties_w_precinct_vote_data.drop(columns="geometry")
counties_with_precinct_data_no_geom.to_csv(outfile_base + ".csv", index=False)
print("Saved successfully.\n")


# ------------------------------------------------------------------
# FROM THE NYT COUNTY V COUNTY AUXILLIARY COMPARISON SCRIPT
# compare_county_nyt_and_alternative.py 
# ------------------------------------------------------------------

print("\nComparing NYT county data to alternative county data...")

county_aux_data_fp = (
    r'/Users/aspencage/Documents/Data/input/post_g2024/'
    r'comparative_presidential_performance/G2024/County/'
    r'2024_Presidential_by_County_Merged_2024__250317.csv'
    )

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

county_comparison.rename(columns={
    "D_President ": "votes_dem_county_aux", 
    "R_President ": "votes_rep_county_aux"
    }, inplace=True)

# torturous way to convert fips to accurate string then get state
county_comparison['fips'] = pd.to_numeric(county_comparison['fips'], errors='coerce').astype(int) 
county_comparison['fips'] = county_comparison['fips'].astype(str).str.zfill(5)
county_comparisons = state_from_county_fips(county_comparison, "fips")
if selected_states is not None:
    county_comparison = county_comparison.loc[county_comparison["State"].isin(selected_states)]

county_comparison['votes_dem_county_aux'] = county_comparison['votes_dem_county_aux'].astype(int)
county_comparison['votes_rep_county_aux'] = county_comparison['votes_rep_county_aux'].astype(int)
county_comparison['votes_two_way_county_aux'] = county_comparison["votes_dem_county_aux"] + county_comparison["votes_rep_county_aux"]
county_comparison["pres_dem_share_two_way_county_aux"] = county_comparison["votes_dem_county_aux"] / county_comparison['votes_two_way_county_aux']

counties_w_precinct_vote_data['fips'] = counties_w_precinct_vote_data['fips'].astype(str).str.zfill(5)
county_comparison['fips'] = county_comparison['fips'].astype(str).str.zfill(5)

county_comp_merged = pd.merge(
  counties_w_precinct_vote_data, 
  county_comparison, 
  on="fips", 
  how="outer",
  suffixes=(None,"_county_aux"))
county_comp_merged['pres_dem_share_two_way_2024__diff'] = county_comp_merged['pres_dem_share_two_way_2024'] - county_comp_merged['pres_dem_share_two_way_county_aux']
county_comp_merged['pres_dem_share_two_way_2024__diff_abs'] = county_comp_merged['pres_dem_share_two_way_2024__diff'].abs()
county_comp_merged['votes_two_way_by_area_2024__diff'] = county_comp_merged['votes_two_way_by_area_2024'] - county_comp_merged['votes_two_way_county_aux']
county_comp_merged['votes_two_way_by_area_2024__diff_abs'] = county_comp_merged['votes_two_way_by_area_2024__diff'].abs()
county_comp_merged['votes_dem_by_area_2024__diff'] = county_comp_merged['votes_dem_by_area_2024'] - county_comp_merged['votes_dem_county_aux']
county_comp_merged['votes_dem_by_area_2024__diff_abs'] = county_comp_merged['votes_dem_by_area_2024__diff'].abs()
county_comp_merged.dropna(inplace=True)

time_str = time.strftime("%Y%m%d_%H%M%S")
outfile_base = 'nyt_pres_2024_by_county_comparison__' + chamber + "__" + time_str

county_comp_merged.to_file(outfile_base + ".gpkg", driver="GPKG")
county_comp_df_no_geom = county_comp_merged.drop(columns="geometry",errors="ignore")
county_comp_df_no_geom.to_csv(outfile_base + ".csv", index=False)

print(f"Comparison saved to {outfile_base}.csv and {outfile_base}.gpkg")


# ------------------------------------------------------------------
# FROM THE SCRIPT TO APPEND COUNTY RESULTS
# ------------------------------------------------------------------

print("Appending county-level accuracy metric to State Legislative data...")

main_gdf = grouped.copy()
if selected_states is not None:
    main_gdf = main_gdf[main_gdf[state_col].isin(selected_states)]

#counties[state_col] = counties[state_col].str.extract(r', ([A-Z]{2})$')
if selected_states is not None:
    county_comp_merged = county_comp_merged[county_comp_merged[state_col].isin(selected_states)]

# Run the area-weighted aggregation
county_comp_merged.rename(columns=
                {
                    "votes_dem_by_area_2024": "votes_dem_nyt_county",
                    'votes_rep_by_area_2024': 'votes_rep_nyt_county',
                    'votes_dem_county_aux': 'votes_dem_county_aux',
                    'votes_rep_county_aux': 'votes_rep_county_aux',
                    'votes_two_way_by_area_2024':'votes_two_way_nyt_county',
                    'Twoway_Total_President':'votes_two_way_county_aux'
                  }, inplace=True)
print("Running area-weighted aggregation from one dataset into the other...")
cols_to_add = [
    'votes_dem_nyt_county',
    #'votes_rep_nyt_county',
    'votes_dem_county_aux',
    #'votes_rep_county_aux',
    'votes_two_way_nyt_county',
    'votes_two_way_county_aux',
    'votes_two_way_by_area_2024__diff',
    'County'
    ]
suffix = "__from_county_comp"  

gdf_with_county_comp = process_area_weighted_metric(
    gdf_source=county_comp_merged,
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
# MERGE IN RESULTS FROM STATE LEGISLATIVE RACES
# ------------------------------------------------------------------
from process_ballotpedia_data import process_ballotpedia_data 

fp_bp_2024_state_leg_votes = (
    r'/Users/aspencage/Documents/Data/input/ballotpedia_paid/'
    r'2024_state_legislative_candidates_for_Working_Families.csv'
)

states_to_exclude = ['AK']

if chamber == "sldl":
    states_to_exclude += ['AZ'] # AZ has different rules around two people per house district

ballotpedia_df = process_ballotpedia_data(
    fp_bp_2024_state_leg_votes,
    states_to_exclude=states_to_exclude, 
    district_pattern=f'/{chamber}:'
    )

ballotpedia_df.rename(columns={"standardized_district":"District"}, inplace=True)
ballotpedia_keep_cols = [
    "District",
    "top_two_parties",
    "dem_votes",
    "opp_votes",
    "dem_two_way_vote_share",
    "incumbent_status",
    "incumbent_party",
    "top_two_candidates"
    ]
ballotpedia_df = ballotpedia_df[ballotpedia_keep_cols]
ballotpedia_df.columns = [
    col if col == "District" else f"{col}__state_leg_2024"
    for col in ballotpedia_df.columns
]

gdf_w_state_leg_votes = pd.merge(
    gdf_with_county_comp,
    ballotpedia_df,
    on="District",
    how="left",
    suffixes=(None,"__state_leg_2024")
)

# create a column for the difference in two-way vote share between the 2024 presidential vote and state legilative vote as long as the state legislative vote is available
gdf_w_state_leg_votes["2024_two_way_dem_pres_minus_state_leg"] = np.where(
        ~pd.isnull(gdf_w_state_leg_votes["dem_two_way_vote_share__state_leg_2024"])
        & (gdf_w_state_leg_votes["dem_two_way_vote_share__state_leg_2024"] < 0.99)
        & (gdf_w_state_leg_votes["dem_two_way_vote_share__state_leg_2024"] > 0.01),
        gdf_w_state_leg_votes["pres_dem_share_two_way_2024"] - gdf_w_state_leg_votes["dem_two_way_vote_share__state_leg_2024"],
        np.nan
    )

# Diagnostic 
compare_column_difference(gdf_w_state_leg_votes, "dist_area", "dist_area_2020", tolerance=0.01, group_by='State')

# ------------------------------------------------------------------
# FINAL ADJUSTMENTS BEFORE EXPORT 
# ------------------------------------------------------------------

gdf_final = state_leg_pres__likely_errors(gdf_w_state_leg_votes)

state_stats = create_state_stats(
    gdf_final, 
    cols=[
        "pct_error__btw_county_data_ported_to_districts_abs",
        "coverage_percentage",
        "split_coverage_percentage",
        "2024_two_way_dem_pres_minus_state_leg",
        "num_vote_warnings"
        ],
    #states_to_blank_out=no_geom_states
)

first_cols = [
    'State',
    'District',
    'pct_error__btw_county_data_ported_to_districts_abs',
    'coverage_percentage',
    'split_coverage_percentage',
    'num_vote_warnings',
    'likely_error_detected',
    'pres_dem_share_two_way_2024',
    'pres_dem_share_two_way_2020',
    'pres_dem_share_two_way_diff',
    'dem_two_way_vote_share__state_leg_2024',
    '2024_two_way_dem_pres_minus_state_leg'
    ]

gdf_final = reorder_columns(gdf_final, first_cols)

cols_to_drop = [
    'votes_total_by_area_2024',
    'State_2020',
    'intersection_area',
    'votes_third_party_by_area_2024',
    'votes_total_by_area_2020',
    'pres_dem_share_total_2024',
    'pres_dem_share_total_2020',
    'pres_dem_share_total_diff',
    'third_party_vote_share_2024',
    'third_party_vote_share_2020',
    'third_party_vote_share_diff',
    'votes_dem_nyt_county__from_county_comp',
    'votes_dem_county_aux__from_county_comp',
    'votes_two_way_nyt_county__from_county_comp',
    'votes_two_way_county_aux__from_county_comp',
    'votes_two_way_by_area_2024__diff__from_county_comp',
    'coverage_fraction',
    'split_area_sum',
    'split_coverage_fraction',
    'dist_area',
    'dist_area_2020'
]
gdf_final.drop(columns=cols_to_drop, errors='ignore', inplace=True)

# Drop values that are not relevant for states where geometry is not used to calculate vote totals
gdf_final.loc[
    gdf_final['State'].isin(no_geom_states), 
    [
        'pct_error__btw_county_data_ported_to_districts_abs',
        'coverage_percentage',
        'split_coverage_percentage',
        'ratio__county_aux_twoway_to_precinct_twoway',
        'ratio__nyt_county_two_way_to_precinct_two_way',
        'ratio__nyt_county_two_way_to_county_aux_two_way',	
        'pct_error__btw_county_data_ported_to_districts'
        ]
    ] = np.nan

try:
    state_stats.loc[state_stats['State'].isin(no_geom_states), 1:] = np.nan
except Exception as e:
    print(f"Error when trying to set state stats to NaN for {no_geom_states}:\n\t{e}")
    print("\tSkipping this for now...")

gdf_pretty = prettify_dataset(
    gdf_final, 
    round_decimals=4, 
    int_columns=[col for col in gdf_final.columns if re.search('^votes', col)], 
    percent_columns=None, 
    sort_by='District'
    )

gdf = gdf_pretty.copy() 

# Save to disk 
time_str = time.strftime("%Y%m%d_%H%M%S")
outfile_base = f"pres_in_district__20_v_24_comparison__{chamber}__{time_str}__w_county_and_leg_races"

gdf_final.to_file(outfile_base + ".gpkg", driver="GPKG")
df_no_geom = gdf_pretty.drop(columns="geometry")
df_no_geom.to_csv(outfile_base + ".csv", index=False)
print(f"Saved files to {outfile_base}.csv and {outfile_base}.gpkg")

state_stats.to_csv(outfile_base + "_state_stats.csv", index=False)
print(f"Saved state stats to {outfile_base}__state_stats.csv")