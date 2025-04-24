from compare_pres_v_state_leg_performance import *

# ----------------------------------------------------------
# 1) Setup: file path, column names, data types, etc.
# ----------------------------------------------------------
pres_house_data_file_24 = r"/Users/aspencage/Documents/Data/input/post_g2024/2024_precinct_level_data/pa/erstat_2024_g_268768_20250110.txt"
senate_data_file_22 = r'/Users/aspencage/Documents/Data/input/post_g2024/2024_precinct_level_data/pa/ElectionReturns_2022_General_PrecinctReturns.txt'

# For PA 2024, based on your dictionary:
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

# Define valid parties and offices
valid_parties = ['DEM', 'REP']
senate_col = 'STS'
house_col = 'STH'

# ----------------------------------------------------------
# 2) Load the data
# ----------------------------------------------------------
pa_df_24 = load_data(
    filepath=pres_house_data_file_24, col_names=pa_col_names,
    has_header=False, encoding='utf-8', dtypes=dtypes
)

pa_df_22 = load_data(
    filepath=senate_data_file_22, col_names=pa_col_names,
    has_header=False, encoding='utf-8', dtypes=dtypes
)

# ----------------------------------------------------------
# 3) Create precinct identifier
# ----------------------------------------------------------
pa_df_24 = create_precinct_identifier(
    df=pa_df_24, county_col='CountyCode', precinct_col='PrecinctCode', new_col='PrecinctName'
)

pa_df_22 = create_precinct_identifier(
    df=pa_df_22, county_col='CountyCode', precinct_col='PrecinctCode', new_col='PrecinctName'
)

# ----------------------------------------------------------
# 4) Filter data by office and party
# ----------------------------------------------------------
# State Senate
df_senate = filter_data_by_office_and_party(
    df=pa_df_22, office_col='CandidateOfficeCode', party_col='CandidatePartyCode',
    valid_office_codes=[senate_col], 
    valid_party_codes=valid_parties,
    exclude_if_precinct_contains="provisional", 
    precinct_col='PrecinctName'
)

# State House
df_house = filter_data_by_office_and_party(
    df=pa_df_24, office_col='CandidateOfficeCode', party_col='CandidatePartyCode',
    valid_office_codes=[house_col], valid_party_codes=valid_parties,
    exclude_if_precinct_contains="provisional", precinct_col='PrecinctName'
)

# President
df_pres = filter_data_by_office_and_party(
    df=pa_df_24, office_col='CandidateOfficeCode', party_col='CandidatePartyCode',
    valid_office_codes=['USP'], valid_party_codes=valid_parties,
    exclude_if_precinct_contains="provisional", precinct_col='PrecinctName'
)

# ----------------------------------------------------------
# 5) Aggregate and pivot
# ----------------------------------------------------------
wide_senate = aggregate_and_pivot(
    df_senate, group_cols=['PrecinctName', 'CandidatePartyCode'],
    agg_col='VoteTotal', pivot_index='PrecinctName',
    pivot_columns='CandidatePartyCode', pivot_values='VoteTotal'
)
wide_house = aggregate_and_pivot(
    df_house, group_cols=['PrecinctName', 'CandidatePartyCode'],
    agg_col='VoteTotal', pivot_index='PrecinctName',
    pivot_columns='CandidatePartyCode', pivot_values='VoteTotal'
)
wide_pres = aggregate_and_pivot(
    df_pres, group_cols=['PrecinctName', 'CandidatePartyCode'],
    agg_col='VoteTotal', pivot_index='PrecinctName',
    pivot_columns='CandidatePartyCode', pivot_values='VoteTotal'
)

# Rename columns to avoid conflicts
wide_senate = wide_senate.rename(columns={'DEM': 'DEM_SEN', 'REP': 'REP_SEN'})
wide_house = wide_house.rename(columns={'DEM': 'DEM_HOUSE', 'REP': 'REP_HOUSE'})
wide_pres = wide_pres.rename(columns={'DEM': 'DEM_PRES', 'REP': 'REP_PRES'})

# ----------------------------------------------------------
# 6) Merge all datasets
# ----------------------------------------------------------
merged_sen_pres = merge_two_datasets(
    wide_left=wide_senate, wide_right=wide_pres,
    on_col='PrecinctName', suffixes=('', ''), how='outer'
)

merged_all = merge_two_datasets(
    wide_left=merged_sen_pres, wide_right=wide_house,
    on_col='PrecinctName', suffixes=('', ''), how='outer'
)

# ----------------------------------------------------------
# 7) Add two-way vote share columns
# ----------------------------------------------------------
merged_all = add_two_way_vote_share_columns(
    df=merged_all,
    dem_left_col='DEM_SEN', rep_left_col='REP_SEN',
    dem_right_col='DEM_PRES', rep_right_col='REP_PRES',
    new_dem_share_left='DemShare_SEN', new_dem_share_right='DemShare_PRES'
)

merged_all = add_two_way_vote_share_columns(
    df=merged_all,
    dem_left_col='DEM_HOUSE', rep_left_col='REP_HOUSE',
    dem_right_col='DEM_PRES', rep_right_col='REP_PRES',
    new_dem_share_left='DemShare_HOUSE', new_dem_share_right='DemShare_PRES'
)

# ----------------------------------------------------------
# 8) Calculate vote share by district for both State House and State Senate
# ----------------------------------------------------------
by_senate = calculate_two_way_voteshare_by_district(
    df=merged_all, original_df=pa_df_24, district_col='StateSenatorialDistrict',
    precinct_col='PrecinctName', dem_left_col='DEM_SEN', rep_left_col='REP_SEN',
    dem_right_col='DEM_PRES', rep_right_col='REP_PRES',
    share_left_col='DemShare_SEN', share_right_col='DemShare_PRES'
)

by_house = calculate_two_way_voteshare_by_district(
    df=merged_all, original_df=pa_df_24, district_col='StateHouseDistrict',
    precinct_col='PrecinctName', dem_left_col='DEM_HOUSE', rep_left_col='REP_HOUSE',
    dem_right_col='DEM_PRES', rep_right_col='REP_PRES',
    share_left_col='DemShare_HOUSE', share_right_col='DemShare_PRES'
)

# ----------------------------------------------------------
# 9) Final table with both State House and State Senate
# ----------------------------------------------------------
# Add a column to designate the legislative body and rename columns for stacking
by_senate["LegislativeBody"] = "State Senate"
by_senate = by_senate.rename(columns={
    "DEM_SEN": "DEM_LEG",
    "REP_SEN": "REP_LEG",
    "DemShare_SEN": "DemShare_LEG",
    "StateSenatorialDistrict": "LegislativeDistrict"
})

by_house["LegislativeBody"] = "State House"
by_house = by_house.rename(columns={
    "DEM_HOUSE": "DEM_LEG",
    "REP_HOUSE": "REP_LEG",
    "DemShare_HOUSE": "DemShare_LEG",
    "StateHouseDistrict": "LegislativeDistrict"
})

# Ensure both datasets have the same columns
common_columns = ["LegislativeDistrict", "LegislativeBody", "DEM_LEG", "REP_LEG", "DemShare_LEG", "DEM_PRES", "REP_PRES", "DemShare_PRES"]
by_senate = by_senate[common_columns]
by_house = by_house[common_columns]

# Stack the datasets (concatenate)
by_both_leg = pd.concat([by_senate, by_house], ignore_index=True)
by_both_leg.LegislativeDistrict = by_both_leg.LegislativeDistrict.astype(int)
by_both_leg.sort_values(by='LegislativeDistrict',inplace=True)
by_both_leg["PRES_minus_LEG"] = by_both_leg["DemShare_PRES"] - by_both_leg["DemShare_LEG"]