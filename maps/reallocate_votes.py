import pandas as pd
import numpy as np
import time 
import os 

def reallocate_missing_geometry_votes(
    df,
    county_col='County',
    precinct_col='precinct',
    geometry_col='geometry',
    vote_cols=['votes_dem', 'votes_rep'],
    update_total_col=None
):
    """
    Reallocate votes from precincts that lack geometry to those that do have geometry,
    in proportion to each precinct's share of the total votes in each vote column.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing county, precinct, geometry, and vote columns.
    county_col : str, optional
        Name of the column identifying the county (default: 'County').
    precinct_col : str, optional
        Name of the column identifying the precinct (default: 'precinct').
    geometry_col : str, optional
        Name of the column containing geometry data (default: 'geometry').
        Precincts with null or empty geometry are considered "missing geometry."
    vote_cols : list of str, optional
        Names of the columns containing vote totals (default: ['votes_dem', 'votes_rep']).
    update_total_col : str or None, optional
        Name of a column to recalculate at the end (by summing the final values in all
        vote_cols for each row). If None (default), this step is skipped.

    Returns
    -------
    pd.DataFrame
        A DataFrame in which any votes from missing-geometry precincts have been 
        proportionally added to the precincts that do have geometry (on a per-county basis).
        If update_total_col is not None, that column is updated to reflect the sum of
        all vote_cols (per row).
    """
    
    # Work on a copy so as not to mutate the original DataFrame.
    df = df.copy()
    
    # We will accumulate a list of DataFrames (one for each county)
    # and then concatenate them in the end.
    reallocated_list = []
    
    # Identify all unique counties in the DataFrame
    all_counties = df[county_col].unique()
    
    for county_val in all_counties:
        # Subset DataFrame to this single county
        county_df = df.loc[df[county_col] == county_val].copy()
        
        # Identify rows that are missing geometry (NaN, None, or empty string)
        no_geom_mask = county_df[geometry_col].isnull() | (county_df[geometry_col] == '')
        missing_geom_df = county_df.loc[no_geom_mask]
        
        # Sum the missing-geometry precinct votes (if any)
        missing_votes = missing_geom_df[vote_cols].sum()
        
        # DataFrame of precincts that do have geometry
        with_geom_df = county_df.loc[~no_geom_mask].copy()
        
        # If there are no precincts with geometry, we cannot reallocate.
        # (If the entire county has no geometry, do nothing, or keep them as is.)
        if with_geom_df.empty:
            reallocated_list.append(county_df)
            continue
        
        # Reallocate the missing votes in proportion to each precinct's share
        # for each vote column
        for vcol in vote_cols:
            total_with_geom = with_geom_df[vcol].sum()
            if total_with_geom == 0:
                continue  # Avoid division by zero

            # Each precinct’s fraction of the county total for this vote col
            fraction_col = f"{vcol}_fraction"
            with_geom_df[fraction_col] = with_geom_df[vcol] / total_with_geom
            
            # Add the allocated portion of missing votes to each precinct
            with_geom_df[vcol] += with_geom_df[fraction_col] * missing_votes[vcol]
            
            # Drop the fraction column (no longer needed)
            with_geom_df.drop(columns=[fraction_col], inplace=True)
        
        # Append updated precincts (with geometry) to the list
        reallocated_list.append(with_geom_df)
    
    # Combine the processed counties back into a single DataFrame
    reallocated_df = pd.concat(reallocated_list, ignore_index=True)
    
    # Optionally update (or create) a total votes column
    if update_total_col is not None:
        reallocated_df[update_total_col] = reallocated_df[vote_cols].sum(axis=1)
    
    return reallocated_df

if __name__ == "__main__":
    
    os.chdir(r'/Users/aspencage/Documents/Data/output/post_2024/2020_2024_pres_compare')
    
    fp = r'/Users/aspencage/Documents/Data/input/post_g2024/comparative_presidential_performance/adjusted/MI_SELECT_COUNTIES_ALL_cleaned__250408.csv'
    df = pd.read_csv(fp)
    df_out = reallocate_missing_geometry_votes(
        df,
        county_col='COUNTY',
        precinct_col='precinct',
        geometry_col='geometry',
        vote_cols=['votes_dem', 'votes_rep','votes_third'],
        update_total_col='votes_total'
    )
    
    time_str = time.strftime("%Y%m%d_%H%M%S")
    df_out.to_csv(
        f'MI_SELECT_COUNTIES_ALL_cleaned_reallocated_votes_{time_str}.csv',
        index=False
    )
