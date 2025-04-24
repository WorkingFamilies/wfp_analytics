from fix_specific_state import fix_precinct_data_for_states
from prepare_precinct_data import (
    standardize_precinct_data, 
    prepare_precinct_data_multirow, 
    prepare_precinct_data_single_row
    )
from prepare_district_data import normalize_district_col
from nyt_2024_process_and_overlay import load_and_prepare_precincts
import os 
import pandas as pd
from mapping_utilities import map_code_to_value
from mapping_dicts import pa_counties

if __name__ == "__main__":
    
    chamber = 'sldu' 

    # NYT baseline data 
    out_dir = f'/Users/aspencage/Documents/Data/output/post_2024/2020_2024_pres_compare/{chamber}'
    os.chdir(out_dir)

    nyt_data_fp = (
        r'/Users/aspencage/Documents/Data/output/'
        r'post_2024/2020_2024_pres_compare/'
        r'nyt_pres_2024_simplified.geojson'
      )
    nyt_precincts = load_and_prepare_precincts(
        nyt_data_fp, 
        crs="EPSG:2163",
        drop_na_columns=["votes_dem","votes_rep","votes_total",'geometry'],
        print_=True
    )
    nyt_precincts.rename(columns={'state':'State'}, inplace=True)
    nyt_precincts['precinct_source'] = 'NYT'

    # WI precinct data
    wi_precinct_fp = (
        r'/Users/aspencage/Documents/Data/input/post_g2024/'
        r'2024_precinct_level_data/wi/'
        r'2024_Election_Data_with_2025_Wards_-5879223691586298781.geojson'
    )

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
    wi_gdf_precincts['State'] = 'WI'
    wi_gdf_precincts['official_bounary'] = True
    wi_gdf_precincts['precinct_source'] = 'WI SOS Precinct-level returns'

    # PA precinct data 

    # PA precinct level data from SOS 
    pa_2024_precincts_fp = (
        r'/Users/aspencage/Documents/Data/input/post_g2024/'
        r'2024_precinct_level_data/pa/erstat_2024_g_268768_20250110.txt'
    )

    pa_col_names = [
        "ElectionYear", "ElectionType", "CountyCode", "PrecinctCode", "CandidateOfficeRank",
        "CandidateDistrict", "CandidatePartyRank", "CandidateBallotPosition", "CandidateOfficeCode",
        "CandidatePartyCode", "CandidateNumber", "CandidateLastName", "CandidateFirstName",
        "CandidateMiddleName", "CandidateSuffix", "VoteTotal", "YesVoteTotal", "NoVoteTotal",
        "USCongressionalDistrict", "StateSenatorialDistrict", "StateHouseDistrict", "MunicipalityTypeCode",
        "MunicipalityName", "MunicipalityBreakdownCode1", "MunicipalityBreakdownName1",
        "MunicipalityBreakdownCode2", "MunicipalityBreakdownName2", "BiCountyCode", "MCDCode",
        "FIPSCode", "VTDCode", "BallotQuestion", "RecordType", "PreviousPrecinctCode",
        "PreviousUSCongressionalDist", "PreviousStateSenatorialDist", "PreviousStateHouseDist"
    ]

    dtypes = {
        'CountyCode': str,
        'PrecinctCode': str,
        "StateSenatorialDistrict": str,
        "StateHouseDistrict": str,
    }

    if chamber == 'sldl':
      district_col = 'StateHouseDistrict'
    elif chamber == 'sldu':
      district_col = 'StateSenatorialDistrict'

    pa_2024_precincts = prepare_precinct_data_multirow(
    pa_2024_precincts_fp,
    precinct_col='PrecinctCode',
    office_col='CandidateOfficeCode',
    party_col='CandidatePartyCode',
    vote_col='VoteTotal',
    offices=['USP'],
    dem_party='DEM',
    rep_party='REP',
    keep_geometry=False,
    has_header=False,
    col_names=pa_col_names,
    dtypes=dtypes,
    state='PA',
    mode_cols=[district_col,'CountyCode']
    )
    # working with future functions 
    pa_2024_precincts.rename(
        columns={
        district_col:'District',
        'DEM_USP':'votes_dem',
        'REP_USP':'votes_rep',
        }, inplace=True)
    pa_2024_precincts['State'] = 'PA'
    pa_2024_precincts = normalize_district_col(pa_2024_precincts, 'District', 'State', chamber)
    # metadata 
    pa_2024_precincts['precinct_source'] = 'PA SOS Precinct-level returns'
    pa_2024_precincts["CountyCode"] = pa_2024_precincts['CountyCode'].astype(str).str.zfill(2)
    pa_2024_precincts["County"] = pa_2024_precincts["CountyCode"].apply(
        map_code_to_value,
        args=(pa_counties,)
        )
    pa_2024_precincts["County"] = pa_2024_precincts["County"] + " County" + ", PA"

    # 4) Run the function, telling it which columns correspond to precinct, Dem votes, etc.
    combined_precincts = fix_precinct_data_for_states(
        original_precinct_data=nyt_precincts,
        states_to_fix=['WI','PA'],
        improved_precincts_list=[wi_gdf_precincts,pa_2024_precincts],
        state_col="State",           # Column that identifies each row's state
        target_crs="EPSG:2163",
        export_fixed_file=True,      # If True, writes out a corrected GPKG
        output_dir=".",
        output_prefix=f"precincts__2024_pres__fixed_{chamber}__",
    )

    # then compare older fixed dataset (WI)
    ## TODO add VA? 
    ## TODO add diagnostic function 

    # export to use this for run of show 