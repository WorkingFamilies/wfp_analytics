from compare_pres_v_state_leg_performance import *

# ----------------------------------------------------------
# 1) Setup: file path, column names, data types, etc.
# ----------------------------------------------------------
pres_data_file = r"/Users/aspencage/Documents/Data/input/post_g2024/2024_precinct_level_data/pa/erstat_2024_g_268768_20250110.txt"
leg_data_file = r'/Users/aspencage/Documents/Data/input/post_g2024/2024_precinct_level_data/pa/ElectionReturns_2022_General_PrecinctReturns.txt'

# For PA 2024, based on your dictionary:
pa_col_names = [
    "ElectionYear",
    "ElectionType",
    "CountyCode",
    "PrecinctCode",
    "CandidateOfficeRank",
    "CandidateDistrict",
    "CandidatePartyRank",
    "CandidateBallotPosition",
    "CandidateOfficeCode",
    "CandidatePartyCode",
    "CandidateNumber",
    "CandidateLastName",
    "CandidateFirstName",
    "CandidateMiddleName",
    "CandidateSuffix",
    "VoteTotal",
    "YesVoteTotal",
    "NoVoteTotal",
    "USCongressionalDistrict",
    "StateSenatorialDistrict",
    "StateHouseDistrict",
    "MunicipalityTypeCode",
    "MunicipalityName",
    "MunicipalityBreakdownCode1",
    "MunicipalityBreakdownName1",
    "MunicipalityBreakdownCode2",
    "MunicipalityBreakdownName2",
    "BiCountyCode",
    "MCDCode",
    "FIPSCode",
    "VTDCode",
    "BallotQuestion",
    "RecordType",
    "PreviousPrecinctCode",
    "PreviousUSCongressionalDist",
    "PreviousStateSenatorialDist",
    "PreviousStateHouseDist"
]

# Example: keep 'CountyCode' and 'PrecinctCode' as strings
dtypes = {
    'CountyCode': str,
    'PrecinctCode': str,
    "StateSenatorialDistrict":str,
    "StateHouseDistrict":str,
}

# Define office level for Y variable (X assumed to be President)
state = 'PA'
leg_col = 'STS' 
leg_string = 'State Senate' 
district_col = 'StateSenatorialDistrict'  

# ----------------------------------------------------------
# 2) Load the data (no header, pass the column names)
# ----------------------------------------------------------
pa_df_24_pres = load_data(
    filepath=pres_data_file,
    col_names=pa_col_names,
    has_header=False,
    encoding='utf-8',
    dtypes=dtypes
)

pa_df_24_leg = load_data(
    filepath=leg_data_file,
    col_names=pa_col_names,
    has_header=False,
    encoding='utf-8',
    dtypes=dtypes
)

# ----------------------------------------------------------
# 3) Create precinct identifier
# ----------------------------------------------------------
pa_df_24_pres = create_precinct_identifier(
    df=pa_df_24_pres,
    county_col='CountyCode',
    precinct_col='PrecinctCode',
    new_col='PrecinctName'
)

pa_df_24_leg = create_precinct_identifier(
    df=pa_df_24_leg,
    county_col='CountyCode',
    precinct_col='PrecinctCode',
    new_col='PrecinctName'
)

# ----------------------------------------------------------
# 4) Filter data: example, compare State Senate (STS) vs. President (USP)
#    for DEM and REP only
# ----------------------------------------------------------
valid_parties = ['DEM', 'REP']  # PA codes
# State Senate
df_leg = filter_data_by_office_and_party(
    df=pa_df_24_leg,
    office_col='CandidateOfficeCode',
    party_col='CandidatePartyCode',
    valid_office_codes=[leg_col],  # "STS" for State Senate
    valid_party_codes=valid_parties,
    exclude_if_precinct_contains="provisional",  # or None if not needed
    precinct_col='PrecinctName'
)
# President
df_pres = filter_data_by_office_and_party(
    df=pa_df_24_pres,
    office_col='CandidateOfficeCode',
    party_col='CandidatePartyCode',
    valid_office_codes=['USP'],  # "USP" for President
    valid_party_codes=valid_parties,
    exclude_if_precinct_contains="provisional",
    precinct_col='PrecinctName'
)

# ----------------------------------------------------------
# 5) Aggregate and pivot
# ----------------------------------------------------------
wide_leg = aggregate_and_pivot(
    df_leg,
    group_cols=['PrecinctName', 'CandidatePartyCode'],
    agg_col='VoteTotal',
    pivot_index='PrecinctName',
    pivot_columns='CandidatePartyCode',
    pivot_values='VoteTotal'
)
wide_pres = aggregate_and_pivot(
    df_pres,
    group_cols=['PrecinctName', 'CandidatePartyCode'],
    agg_col='VoteTotal',
    pivot_index='PrecinctName',
    pivot_columns='CandidatePartyCode',
    pivot_values='VoteTotal'
)

# Rename columns so they don't clash when merging
wide_leg = wide_leg.rename(columns={'DEM': 'DEM_LEG', 'REP': 'REP_LEG'})
wide_pres = wide_pres.rename(columns={'DEM': 'DEM_PRES', 'REP': 'REP_PRES'})

# ----------------------------------------------------------
# 6) Merge the two wide datasets
# ----------------------------------------------------------
merged = merge_two_datasets(
    wide_left=wide_leg,
    wide_right=wide_pres,
    on_col='PrecinctName',
    suffixes=('', ''),
    how='outer'
)

# ----------------------------------------------------------
# 7) Add two-way vote share columns
# ----------------------------------------------------------
merged = add_two_way_vote_share_columns(
    df=merged,
    dem_left_col='DEM_LEG',
    rep_left_col='REP_LEG',
    dem_right_col='DEM_PRES',
    rep_right_col='REP_PRES',
    new_dem_share_left='DemShare_LEG',
    new_dem_share_right='DemShare_PRES'
)

# ----------------------------------------------------------
# 8) Calculate vote share by district (example: StateSenatorialDistrict)
# ----------------------------------------------------------
by_leg = calculate_two_way_voteshare_by_district(
    df=merged,
    original_df=pa_df_24_pres,
    district_col=district_col,
    precinct_col='PrecinctName',
    dem_left_col='DEM_LEG',
    rep_left_col='REP_LEG',
    dem_right_col='DEM_PRES',
    rep_right_col='REP_PRES',
    share_left_col='DemShare_LEG',
    share_right_col='DemShare_PRES'
)

# ----------------------------------------------------------
# 9) Remove uncontested districts
# ----------------------------------------------------------
by_leg = remove_uncontested_districts(
    df=by_leg,
    share_col='DemShare_LEG',  # threshold check on legislative share, for instance
    threshold=0.05
)

# ----------------------------------------------------------
# 10) Plot a comparison of the two-way Dem shares
# ----------------------------------------------------------
plot_two_way_voteshare_comparison(
    df=by_leg,
    xcol='DemShare_PRES',
    ycol='DemShare_LEG',
    label_col=district_col,
    title=f"{state} 2026 Opportunities: {leg_string} vs. Presidential Dem Vote Share",
    xlabel='Presidential Dem Share (2024)',
    ylabel=f'{leg_string} Dem Share (2022)',
    vert_line_label=f'50% Voteshare ({leg_string})',
    horiz_line_label=f'50% Voteshare (Presidential)'
)

# ----------------------------------------------------------
# Done! Additional analysis, saving, or further plots can follow.
# ----------------------------------------------------------
# Example: print out the final district-level summary
print(by_leg.head(10))

# TODO add y = - x "challenge" line to show whether it's an easier or a harder pickup? 