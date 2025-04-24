import geopandas as gpd
import pandas as pd 
import os 
import time

from prepare_precinct_data import attach_votes_to_precincts  
from political_geospatial import (
    load_and_prepare_precincts,
    calculate_coverage,
    calculate_split_precinct_coverage,
    process_votes_area_weighted,
)
from mapping_utilities import dissolve_groups

os.chdir(r'/Users/aspencage/Documents/Data/projects/ohio_precinct_level_vote')

# ---------- user inputs ----------
fp_votes_csv       = r'input/Ahn 2024 Results.xlsx - Raw Data.csv'
# This is county wards only, not precincts
# fp_past_precincts  = r'input/Election_Wards_Cuyahoga_region/Election_Wards_Cuyahoga_region.shp'
# These precincts cover only Cleveland
fp_past_precincts = (
  r'input/Cleveland Precincts Nov 2024_region/'
  r'Cleveland Precincts Nov 2024_region.shp'
)
fp_future_precincts = (
    r'input/2025_Cleveland_Wards Cuyahoga BOE Official_region/'
    r'2025_Cleveland_Wards Cuyahoga BOE Official_region.shp'
    )
vote_cols          = ["Ahn", "O'Malley", "Total Votes"]
precinct_key_csv   = "PRECINCT"       # column in CSV
precinct_key_geo  = "Name"       # column in shapefile


# ---------- 1. load data ----------
votes = pd.read_csv(fp_votes_csv)
# remove first 4 digits from PRECINCT - starts with 4 numbers, mapping unclear 
votes[precinct_key_csv] = votes[precinct_key_csv].astype(str).str[4:]
# retain only rows that have "CLEVELAND" in the precinct name
votes = votes[votes[precinct_key_csv].str.contains("CLEVELAND", na=False)]

precincts_past_in = load_and_prepare_precincts(
    fp_past_precincts,
    crs="EPSG:2163",
    drop_na_columns=None,
    print_=True,
)
# combine by Name for clean merging with votes
precincts_past = dissolve_groups(
    precincts_past_in,
    group_col=precinct_key_geo
)

precincts_future = gpd.read_file(fp_future_precincts).to_crs(precincts_past.crs)

# ---------- 2. attach votes to shapes ----------
prec_votes = attach_votes_to_precincts(
    fp_votes_csv       = votes,
    gdf_precincts      = precincts_past,
    precinct_key_csv   = precinct_key_csv,
    precinct_key_geo   = precinct_key_geo,
    vote_cols          = vote_cols,
    fuzzy              = False,
    fuzzy_score_cutoff = 2,
    global_fuzzy       = True,
    global_min_score   = 2,
)

# ---------- 3. coverage & splits (old helpers) ----------

# To allow remaining functions to run without modification
prec_votes["State"] = "OH"
prec_votes.rename(columns={
  "Ahn":"votes_dem",
  "O'Malley":"votes_rep",
  "Total Votes":"votes_total"
}, inplace=True)
# NOTE - these designations are not correct since it was in the Dem primary

precincts_future["State"] = "OH"
precincts_future['Name'] = precincts_future['Ward'].astype(str) # to align for later functions

'''
cov = calculate_coverage(
    prec_votes,
    precincts_future,
    district_col=precinct_key_geo,     # whatever identifies future precincts
)
precincts_future_split = calculate_split_precinct_coverage(
    prec_votes,
    cov,
    district_col=precinct_key_geo,
)
'''

# ---------- 4. area-weight votes into 2024 precincts ----------

future_votes = process_votes_area_weighted(
    gdf_precinct=prec_votes,
    gdf_districts=precincts_future, #precincts_future_split
    year="2024",
    district_col=precinct_key_geo,
)

future_votes.rename(columns={
  'votes_rep_by_area_2024':'votes_omalley_weighted', 
  'votes_dem_by_area_2024':'votes_ahn_weighted',
  'votes_total_by_area_2024':'votes_total_weighted',
  'pres_dem_share_total_2024':'anh_share_total_weighted',
}, inplace=True)
future_votes.drop(columns=[
  'votes_third_party_by_area_2024', 'votes_two_way_by_area_2024',
  'pres_dem_share_two_way_2024', 'third_party_vote_share_2024'
  ], inplace=True)

time_string = time.strftime("%Y%m%d")
future_votes.to_file(f"output/anh_omalley_vote_by_2025_ward__{time_string}.gpkg", driver="GPKG")
