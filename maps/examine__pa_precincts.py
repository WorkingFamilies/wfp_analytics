from dataset_utilities import verbose_merge
from prepare_precinct_data import prepare_precinct_data_multirow
from prepare_district_data import normalize_district_col 
from political_geospatial import (
    fix_precinct_data_for_states, 
    load_and_prepare_precincts,
    load_and_prepare_districts_w_overrides,
    process_votes_area_weighted
    )
from validation_geo_internal import calculate_coverage, calculate_split_precinct_coverage
import pandas as pd

'''
NOTE - this is largely to test/confirm the new analysis for non-geospatial rollups 
of precinct to district votes 

NOTE getting some spillover from NJ to PA
'''

# NYT data 
fp_nyt_precincts = (
    r'/Users/aspencage/Documents/Data/output/post_2024/'
    r'2020_2024_pres_compare/nyt_pres_2024_simplified.geojson'
)

# PA precinct level data from SOS 
pa_2024_precincts_fp = (
    r'/Users/aspencage/Documents/Data/input/post_g2024/'
    r'2024_precinct_level_data/pa/erstat_2024_g_268768_20250110.txt'
)

states = ['PA','NJ']

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

pa_2024_precincts_unproc = pd.read_csv(pa_2024_precincts_fp, names=pa_col_names, dtype=dtypes)

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
  mode_cols=['StateHouseDistrict']
)
# working with future functions 
pa_2024_precincts.rename(
    columns={
      'StateHouseDistrict':'District',
      'DEM_USP':'votes_dem',
      'REP_USP':'votes_rep',
      }, inplace=True)
pa_2024_precincts['State'] = 'PA'
pa_2024_precincts = normalize_district_col(pa_2024_precincts, 'District', 'State')
# metadata 
pa_2024_precincts['precinct_source'] = 'PA SOS Precinct-level returns'

precincts_2024 = load_and_prepare_precincts(
    fp_nyt_precincts, 
    crs="EPSG:2163",
    drop_na_columns=["votes_dem","votes_rep","votes_total",'geometry'],
    print_=True
)
precincts_2024.rename(columns={'state':'State'}, inplace=True)
precincts_2024 = precincts_2024.loc[precincts_2024["State"].isin(states)]
precincts_2024['precinct_source'] = 'NYT'

combined_precincts = fix_precinct_data_for_states(
        original_precinct_data=precincts_2024,
        states_to_fix=['PA'],
        improved_precincts_list=[pa_2024_precincts],
        state_col="State",           # Column that identifies each row's state
        target_crs="EPSG:2163",
        export_fixed_file=False
    )

# districts 

sldl_directory = (
    r"/Users/aspencage/Documents/Data/input/post_g2024/"
    r"comparative_presidential_performance/TIGER2023_SLDL"
)

# No overrides needed
districts = load_and_prepare_districts_w_overrides(
    sldl_directory, 
    crs="EPSG:2163",
    override_configs=None,
    district_col="District"
)
districts = districts.loc[districts["State"].isin(states)]

districts_w_vote_estimates = process_votes_area_weighted(
    gdf_precinct=combined_precincts,
    gdf_districts=districts, 
    year="2024",
    district_col="District",
    extra_precinct_cols=["precinct_source"],
    extra_district_cols=["geo_source","dist_area"]
) 

# drop district from precincts, which is no longer needed to avoid error from intersection with identically-named districts in both gpds
combined_precincts.drop(columns=['District'], inplace=True)
coverage_gdf = calculate_coverage(combined_precincts, districts_w_vote_estimates, states_to_excl=['PA'])
coverage_split_gdf = calculate_split_precinct_coverage(combined_precincts, coverage_gdf,states_to_excl=['PA'])
