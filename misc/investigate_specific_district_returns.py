from compare_pres_v_state_leg_performance import *

# ----------------------------------------------------------
# 1) Setup: file path, column names, data types, etc.
# ----------------------------------------------------------
data_file = r"/Users/aspencage/Documents/Data/input/post_g2024/2024_precinct_level_data/pa/erstat_2024_g_268768_20250110.txt"

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
leg_col = 'STS'  # State Senate # 'STH' # Representative in the General Assembly 
leg_string = 'State Senate' # 'State House' 
district_col = 'StateSenatorialDistrict' # 'StateHouseDistrict' 

# ----------------------------------------------------------
# 2) Load the data (no header, pass the column names)
# ----------------------------------------------------------
pa_df = load_data(
    filepath=data_file,
    col_names=pa_col_names,
    has_header=False,
    encoding='utf-8',
    dtypes=dtypes
)

# ----------------------------------------------------------
# 3) Create precinct identifier
# ----------------------------------------------------------
pa_df = create_precinct_identifier(
    df=pa_df,
    county_col='CountyCode',
    precinct_col='PrecinctCode',
    new_col='PrecinctName'
)

# ----------------------------------------------------------
# 4) Investigate Specific District
# ----------------------------------------------------------
df = get_district_election_results(pa_df, district_col='StateHouseDistrict', district_value='35') 

