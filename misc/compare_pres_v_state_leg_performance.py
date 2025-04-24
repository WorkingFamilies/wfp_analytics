import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def load_data(filepath, col_names=None, has_header=False, encoding='utf-8', dtypes=None):
    """
    Loads precinct-level data from a CSV (or comma-delimited) file.
    
    Parameters:
    -----------
    filepath : str
        Path to the input file.
    col_names : list or None
        If the file has no header row and columns need to be specified,
        provide them as a list of column names.
    has_header : bool
        If True, the file includes a header row to be read by pandas.
    encoding : str
        Encoding type for the file.
    dtypes : dict or None
        Optionally specify column data types.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the loaded data.
    """
    if has_header:
        df = pd.read_csv(filepath, encoding=encoding, dtype=dtypes)
    else:
        df = pd.read_csv(filepath, names=col_names, header=None, encoding=encoding, dtype=dtypes)
    return df


def create_precinct_identifier(df, county_col='CountyCode', precinct_col='PrecinctCode',
                               new_col='PrecinctName'):
    """
    Creates a new column by combining county and precinct codes (or similar fields)
    to produce a single, unique precinct identifier.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input data containing county and precinct columns.
    county_col : str
        Name of the column corresponding to the county code.
    precinct_col : str
        Name of the column corresponding to the precinct code.
    new_col : str
        Name of the new column to be created.
    
    Returns:
    --------
    pd.DataFrame
        Modified DataFrame with the new precinct identifier.
    """
    df[new_col] = df[county_col].str.zfill(2) + "-" + df[precinct_col].str.zfill(7)
    return df


def filter_data_by_office_and_party(df,
                                    office_col='OfficeCode',
                                    party_col='PartyCode',
                                    valid_office_codes=None,
                                    valid_party_codes=None,
                                    exclude_if_precinct_contains=None,
                                    precinct_col='PrecinctName'):
    """
    Filters the DataFrame to only include rows whose office code and party code
    match user-specified lists. Optionally excludes precincts containing a substring
    (e.g. 'provisional').
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input data to filter.
    office_col : str
        Column name for the office code (e.g., 'OfficeCode').
    party_col : str
        Column name for the party code (e.g., 'PartyCode').
    valid_office_codes : list of str
        Office codes to keep (e.g., ['STS', 'STH', 'USP']).
    valid_party_codes : list of str
        Party codes to keep (e.g., ['DEM', 'REP']).
    exclude_if_precinct_contains : str or None
        If not None, excludes any precinct where `precinct_col` contains this string.
    precinct_col : str
        Column name for precinct identifier.
    
    Returns:
    --------
    pd.DataFrame
        Filtered DataFrame.
    """
    if valid_office_codes is None:
        valid_office_codes = []
    if valid_party_codes is None:
        valid_party_codes = []

    mask_office = df[office_col].isin(valid_office_codes)
    mask_party = df[party_col].isin(valid_party_codes)
    
    filtered = df[mask_office & mask_party]

    if exclude_if_precinct_contains is not None:
        filtered = filtered[~filtered[precinct_col]
                                .str.lower()
                                .str.contains(exclude_if_precinct_contains, na=False)]
    return filtered


def aggregate_and_pivot(df,
                        group_cols=('PrecinctName','PartyCode'),
                        agg_col='VoteTotal',
                        pivot_index='PrecinctName',
                        pivot_columns='PartyCode',
                        pivot_values='VoteTotal'):
    """
    Aggregates vote totals by the group columns, then pivots to produce wide format
    (e.g., columns for 'DEM', 'REP').
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame to aggregate and pivot.
    group_cols : tuple or list
        Columns to group by (e.g., ['PrecinctName','PartyCode']).
    agg_col : str
        Column to sum (e.g., 'VoteTotal').
    pivot_index : str
        Column name to use as the pivot index.
    pivot_columns : str
        Column name to pivot into multiple columns.
    pivot_values : str
        Values to place in pivot cells.
    
    Returns:
    --------
    pd.DataFrame
        Wide-format DataFrame with one row per precinct, multiple columns for each party.
    """
    aggregated = df.groupby(list(group_cols), as_index=False)[agg_col].sum()
    wide = aggregated.pivot(index=pivot_index, columns=pivot_columns, values=pivot_values).reset_index()
    wide = wide.fillna(0)
    return wide


def merge_two_datasets(wide_left, wide_right,
                       on_col='PrecinctName',
                       suffixes=('_X', '_Y'),
                       how='outer'):
    """
    Merges two wide-format DataFrames (e.g., legislative vs. presidential),
    returning a single DataFrame with suffixes for overlapping columns.
    """
    merged = pd.merge(wide_left, wide_right, on=on_col, how=how, suffixes=suffixes)
    return merged


def add_two_way_vote_share_columns(df,
                                   dem_left_col='DEM_X', rep_left_col='REP_X',
                                   dem_right_col='DEM_Y', rep_right_col='REP_Y',
                                   new_dem_share_left='DemShare_X',
                                   new_dem_share_right='DemShare_Y'):
    """
    Adds two-way Democratic vote share columns for the left (X) and right (Y) sets
    of columns (e.g., legislative vs. presidential).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Merged DataFrame with columns for DEM_X, REP_X, DEM_Y, REP_Y, etc.
    dem_left_col, rep_left_col, dem_right_col, rep_right_col : str
        Column names for the respective DEM/REP vote totals.
    new_dem_share_left, new_dem_share_right : str
        Names for the new vote share columns to be created.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with added columns for two-way Democratic vote share.
    """
    df[new_dem_share_left] = df[dem_left_col] / (df[dem_left_col] + df[rep_left_col])
    df[new_dem_share_right] = df[dem_right_col] / (df[dem_right_col] + df[rep_right_col])
    df[[new_dem_share_left, new_dem_share_right]] = df[[new_dem_share_left, new_dem_share_right]].fillna(0)
    return df


def calculate_two_way_voteshare_by_district(df,
                                            original_df,
                                            district_col='StateSenatorialDistrict',
                                            precinct_col='PrecinctName',
                                            dem_left_col='DEM_X',
                                            rep_left_col='REP_X',
                                            dem_right_col='DEM_Y',
                                            rep_right_col='REP_Y',
                                            share_left_col='DemShare_X',
                                            share_right_col='DemShare_Y'):
    """
    Aggregates vote totals at the district level (e.g., StateSenatorialDistrict),
    then computes two-way Democratic vote shares. Also calculates the change between
    the right share and the left share (e.g., PRES - LEG).
    """
    # 1) Get precinct->district mapping
    district_mapping = original_df[[precinct_col, district_col]].drop_duplicates()
    
    # 2) Merge
    merged_with_district = pd.merge(df, district_mapping, on=precinct_col, how='left')
    
    # 3) Aggregate
    aggregated = merged_with_district.groupby(district_col, as_index=False).agg({
        dem_left_col: 'sum',
        rep_left_col: 'sum',
        dem_right_col: 'sum',
        rep_right_col: 'sum'
    })
    
    # 4) Calculate two-way vote share
    aggregated[share_left_col] = (aggregated[dem_left_col] /
                                  (aggregated[dem_left_col] + aggregated[rep_left_col]))
    aggregated[share_right_col] = (aggregated[dem_right_col] /
                                   (aggregated[dem_right_col] + aggregated[rep_right_col]))
    
    aggregated[[share_left_col, share_right_col]] = aggregated[[share_left_col, share_right_col]].fillna(0)
    
    # 5) Calculate difference
    aggregated['Change_in_Share'] = aggregated[share_right_col] - aggregated[share_left_col]
    
    return aggregated


def remove_uncontested_districts(df, share_col='DemShare_X', threshold=0.05):
    """
    Removes rows where the specified Democratic vote share is < threshold (e.g., 5%)
    or > (1 - threshold) (e.g., 95%), indicating effectively uncontested districts.
    """
    return df[(df[share_col] >= threshold) & (df[share_col] <= 1 - threshold)]


def plot_two_way_voteshare_comparison(df,
                                      xcol='DemShare_Y',
                                      ycol='DemShare_X',
                                      label_col='DistrictName',
                                      title="Comparison of Two-Way Dem Vote Share",
                                      xlabel='Right Office Dem Share',
                                      ylabel='Left Office Dem Share',
                                      vert_line_label='50% (Y)',
                                      horiz_line_label='50% (X)',
                                      ):
    """
    Plots a simple scatter comparing two two-way Democratic vote share columns,
    adding two diagonal lines:
        1) y = x
        2) y = x + (avg_x - avg_y)  -- i.e., a diagonal offset line
           that passes through the intersection of y=0.5 and the "penalty" line.
    """

    # 1) Preliminary checks
    if xcol not in df.columns or ycol not in df.columns:
        raise ValueError("DataFrame must contain columns for xcol and ycol.")
    
    mean_x = df[xcol].mean()
    mean_y = df[ycol].mean()

    # 2) Create figure and scatter
    plt.figure(figsize=(8, 6))
    plt.scatter(df[xcol], df[ycol], alpha=0.6)
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)

    # 3) Calculate the difference
    avg_x_v_y_diff = mean_x - mean_y
    if avg_x_v_y_diff > 0:
        x_v_y_type = "Penalty"
    else:
        x_v_y_type = "Bonus"

    # 4) Plot reference lines at 50%, means, and penalty/bonus offset
    plt.axhline(y=0.5, color='red', linestyle='-', alpha=0.5, label=vert_line_label)
    plt.axvline(x=0.5, color='blue', linestyle='-', alpha=0.5, label=horiz_line_label)
    plt.axvline(
        x= 0.5 + avg_x_v_y_diff,
        color='purple',
        linestyle='--',
        alpha=0.9,
        label=f'{x_v_y_type} to Y Variable ({abs(avg_x_v_y_diff):.2f} difference)'
    )
    plt.axvline(x=mean_x, color='blue', linestyle='--', alpha=0.4, linewidth=1, label=f'Avg {xcol} ({mean_x:.2f})')
    plt.axhline(y=mean_y, color='red', linestyle='--', alpha=0.4, linewidth=1, label=f'Avg {ycol} ({mean_y:.2f})')

    # 5) Plot two diagonal lines
    #    a) exact tie line => y = x
    #    b) offset line    => y = x - avg_x_v_y_diff
    # Determine a plotting range based on the min & max of actual data
    min_val = min(df[xcol].min(), df[ycol].min())
    max_val = max(df[xcol].max(), df[ycol].max())
    
    x_vals = np.linspace(min_val, max_val, 200)
    line_exact = x_vals  # y = x
    line_offset = x_vals - avg_x_v_y_diff  # y = x - delta

    # Exact line represent under/over performance of one office compared to the other 
    plt.plot(x_vals, line_exact, color='orange', linestyle='--', alpha=0.6, label='Identical voteshare in both elections')
    # Offset line this, adjusting for systematic (average) difference in Democratic voteshare between elections/positions
    plt.plot(x_vals, line_offset, color='gold', linestyle='--', alpha=0.8,
             label=f'{x_v_y_type} Line (y = x + {avg_x_v_y_diff:.2f})')

    # 6) Label each point with DistrictName (or whichever label_col)
    for _, row in df.iterrows():
        plt.text(row[xcol], row[ycol], str(row[label_col]), fontsize=8, alpha=0.7)

    plt.legend()
    plt.show()

# ----------------------------------------------------------
# Additional Functions 
# ----------------------------------------------------------

def get_district_election_results(df, district_col='StateHouseDistrict', district_value='35', office_codes=['USP', 'STH']):
    """
    Filters and summarizes election results for a given district, providing total votes
    and percentage of votes within each office category.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame containing election results.
    district_col : str
        Column name corresponding to the district identifier (e.g., 'StateHouseDistrict').
    district_value : str
        The specific district to filter for (e.g., '35').
    office_codes : list of str
        List of office codes to include (e.g., ['USP'] for President, ['STH'] for State House).
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing vote totals and percentages for each candidate within the specified district.
    """
    # Filter by district and office codes
    district_df = df[(df[district_col] == district_value) & (df['CandidateOfficeCode'].isin(office_codes))]
    
    # Group by candidate details and sum their vote totals
    candidate_totals = district_df.groupby(
        ['ElectionYear', 'CandidateOfficeCode', 'CandidateLastName', 
         'CandidateFirstName', 'CandidatePartyCode']
    )['VoteTotal'].sum().reset_index()
    
    # Calculate total votes per office
    office_totals = candidate_totals.groupby('CandidateOfficeCode')['VoteTotal'].transform('sum')
    
    # Calculate the percentage of total votes for each candidate within their office
    candidate_totals['VotePercentage'] = (candidate_totals['VoteTotal'] / office_totals) * 100
    
    return candidate_totals


if __name__ == "__main__":
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
    # 4) Filter data: example, compare State Senate (STS) vs. President (USP)
    #    for DEM and REP only
    # ----------------------------------------------------------
    valid_parties = ['DEM', 'REP']  # PA codes
    # State Senate
    df_leg = filter_data_by_office_and_party(
        df=pa_df,
        office_col='CandidateOfficeCode',
        party_col='CandidatePartyCode',
        valid_office_codes=[leg_col],  # "STS" for State Senate
        valid_party_codes=valid_parties,
        exclude_if_precinct_contains="provisional",  # or None if not needed
        precinct_col='PrecinctName'
    )
    # President
    df_pres = filter_data_by_office_and_party(
        df=pa_df,
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
        original_df=pa_df,
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
        title=f"{state} 2024: {leg_string} vs. Presidential Dem Vote Share",
        xlabel='Presidential Dem Share (2024)',
        ylabel=f'{leg_string} Dem Share (2024)',
        vert_line_label=f'50% Voteshare ({leg_string})',
        horiz_line_label=f'50% Voteshare (Presidential)'
    )

    # ----------------------------------------------------------
    # Done! Additional analysis, saving, or further plots can follow.
    # ----------------------------------------------------------
    # Example: print out the final district-level summary
    print(by_leg.head(10))

# TODO add y = - x "challenge" line to show whether it's an easier or a harder pickup 