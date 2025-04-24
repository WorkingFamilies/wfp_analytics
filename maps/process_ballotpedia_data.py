import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

def process_ballotpedia_data(
    filepath: str,
    stage_filter=('General', 'General runoff'),
    states_to_exclude=['AK'],
    district_pattern='/sldl:',
    party_dem_strings=('Democratic', 'Democrat', 'Dem'),
    party_rep_strings=('Republican', 'GOP'),
    drop_duplicates=True
):
    """
    Reads the Ballotpedia data from CSV and processes it according to the specified requirements:
    
    1) Exclude rows from the given state_to_exclude (Alaska).
    2) Filter to the specified stage_filter (Default: 'General', 'General runoff').
    3) Keep only rows for lower legislative districts (matching district_pattern).
    4) For each race, select only the top 2 vote-getters.
    5) Construct a standardized district name of the form "ST HD-XYZ".
       - ST = state abbreviation
       - HD = prefix for lower chamber
       - XYZ = zero-padded (3 digits) district number if it's an integer, otherwise keep the substring as-is.
    6) Create a column with the party of each of the top two candidates as a list.
    7) Create a column with the names of the top two candidates, in the same order as their parties.
    8) If exactly one top-two candidate is a Dem, create columns for:
       - dem_votes
       - opp_votes
       - dem_two_way_vote_share = dem_votes / (dem_votes + opp_votes)
    9) Create a column "incumbent_status" indicating the incumbent's outcome if is_incumbent == True.
    10) Create a column "incumbent_party" indicating the incumbent’s party.
    
    Returns a processed DataFrame.
    """

    # 1) Read the CSV
    df = pd.read_csv(filepath)

    # 2) Remove rows for excluded state (e.g., Alaska)
    df = df[~df['state'].isin(states_to_exclude)]

    # 3) Filter to specified stage_filter
    df = df[df['stage'].isin(stage_filter)]
    
    # 4) Keep only rows for lower legislative districts (matching district_pattern)
    df = df[df['district_ocdid'].str.contains(district_pattern, na=False)]
    
    # Determine if we’re dealing with lower house (sldl) or upper house (sldu)
    if 'sldl' in district_pattern:
        district_prefix = 'HD'
    elif 'sldu' in district_pattern:
        district_prefix = 'SD'
    else:
        district_prefix = ''  # fallback
    
    # Function to parse district info and standardize
    def extract_standardized_district(row):
        """
        Takes a district_ocdid string, extracts the numeric (or non-numeric) portion,
        and constructs a label like 'ST HD-XYZ'.
        """
        state_abbr = row['state']
        oc = row['district_ocdid']
        
        try:
            district_str = oc.split(district_pattern)[1]
        except IndexError:
            return f"{state_abbr} {district_prefix}-{oc}"  # fallback
        
        district_str = district_str.strip()
        
        # If this is an integer, zero-pad to 3 digits
        if district_str.isdigit():
            district_str = district_str.zfill(3)
        
        return f"{state_abbr} {district_prefix}-{district_str}"
    
    df['standardized_district'] = df.apply(extract_standardized_district, axis=1)
    
    # 5) In each race, select only top 2 vote-getters
    group_cols = ['state', 'standardized_district', 'stage', 'election_year']
    
    # Sort by votes descending so the top rows come first
    df = df.sort_values(by=group_cols + ['votes_for'], 
                        ascending=[True]*len(group_cols) + [False])
    
    # Rank within each group, then filter to top 2
    df['rank_within_race'] = df.groupby(group_cols)['votes_for'].rank(method='first', ascending=False)
    df = df[df['rank_within_race'] <= 2].copy()
    
    # 6) Create a column with the top-two parties
    party_map = (
        df.groupby(group_cols)['party_affiliation']
          .apply(lambda x: list(x))
          .reset_index(name='top_two_parties')
    )
    df = pd.merge(df, party_map, on=group_cols, how='left')
    
    # 7) Create a column with the top-two candidate names in the same order
    name_map = (
        df.groupby(group_cols)['name']
          .apply(lambda x: list(x))
          .reset_index(name='top_two_candidates')
    )
    df = pd.merge(df, name_map, on=group_cols, how='left')
    
    # 8) If exactly one candidate is a Dem, compute dem_votes, opp_votes, etc.
    def compute_party_stats(group):
        parties = group['party_affiliation'].tolist()
        dem_mask = [
            any(str(p).lower().startswith(d.lower()) for d in party_dem_strings)
            for p in parties
        ]
        rep_mask = [
            any(str(p).lower().startswith(r.lower()) for r in party_rep_strings)
            for p in parties
        ]
        
        group['dem_votes'] = np.nan
        group['opp_votes'] = np.nan
        group['dem_two_way_vote_share'] = np.nan
        
        if sum(dem_mask) == 1:
            # Identify the Dem row vs. the other row(s)
            dem_idx = group.index[dem_mask]
            opp_idx = group.index[~pd.Series(dem_mask, index=group.index)]

            dem_votes_val = group.loc[dem_idx, 'votes_for'].sum()
            opp_votes_val = group.loc[opp_idx, 'votes_for'].sum()
            total = dem_votes_val + opp_votes_val
            
            share_val = dem_votes_val / total if total > 0 else np.nan

            group['dem_votes'] = dem_votes_val
            group['opp_votes'] = opp_votes_val
            group['dem_two_way_vote_share'] = share_val
        
        group['at_least_one_dem'] = np.where(sum(dem_mask) >= 1, True, False)
        group['at_least_one_rep'] = np.where(sum(rep_mask) >= 1, True, False)
        
        return group
    
    df = df.groupby(group_cols, group_keys=False).apply(compute_party_stats)
    
    # 9) Create "incumbent_status" and "incumbent_party" columns
    def compute_incumbent_fields(group):
        mask_inc = (group['is_incumbent'] == True)
        group['incumbent_status'] = np.nan
        group['incumbent_party'] = np.nan

        if mask_inc.any():
            # If there's an incumbent, grab that row's status and party
            incumbent_outcome = group.loc[mask_inc, 'candidate_status'].values[0]
            incumbent_party = group.loc[mask_inc, 'party_affiliation'].values[0]
            
            group['incumbent_status'] = incumbent_outcome
            group['incumbent_party'] = incumbent_party
        
        return group
    
    df = df.groupby(group_cols, group_keys=False).apply(compute_incumbent_fields)
    
    # Clean up helper columns
    df.drop(columns=['rank_within_race'], inplace=True)
    
    # Optionally drop duplicates
    if drop_duplicates:
        rows_before = df.shape[0]
        df.drop_duplicates(subset=['standardized_district'], inplace=True, ignore_index=True)
        rows_after = df.shape[0]
        print(f'Dropped {rows_before - rows_after} rows determined to be duplicates, resulting in {rows_after} rows.')

    return df


# ------------------------------------------------------------------
# Example usage: apply to your CSV
# ------------------------------------------------------------------
if __name__ == "__main__":
    input_file = r'/Users/aspencage/Documents/Data/input/ballotpedia_paid/2024_state_legislative_candidates_for_Working_Families.csv'
    df = process_ballotpedia_data(input_file,district_pattern='/sldu:')
    
    # Show top rows, or do further analysis
    print(df.head(20))
