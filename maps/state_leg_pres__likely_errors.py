import pandas as pd 
import numpy as np
import geopandas as gpd 


def state_leg_pres__likely_errors(df):

    # create warning flag columns 
    df['warning__10_pt_swing_btwn_years'] = np.where(df["pres_dem_share_two_way_diff"].abs() > 0.10,True,False)
    df['warning__county_discrepancy'] = np.where(df["pct_error__btw_county_data_ported_to_districts_abs"] > 0.05,True,False)
    df['warning__pres_v_state_leg_7_pt_difference'] = np.where(
            (df["2024_two_way_dem_pres_minus_state_leg"].abs() > 0.07) &
            (df["dem_two_way_vote_share__state_leg_2024"] < 0.99) &
            (df["dem_two_way_vote_share__state_leg_2024"] > 0.01),
            True,
            False
        )
    df['warning__low_coverage'] = np.where((df["coverage_percentage"] > 0) & (df["coverage_percentage"] < 0.80),True,False)
    df['warning__precinct_splitting'] = np.where(df["split_coverage_percentage"] > 0.50,True,False)

    df['num_warnings'] = df.filter(regex='.*warning.*',axis=1).sum(axis=1)

    df['num_vote_warnings'] = sum([
        df['warning__10_pt_swing_btwn_years'],
        df['warning__county_discrepancy'],
        df['warning__pres_v_state_leg_7_pt_difference']
        ])

    # create text warning column
    df['likely_error_detected'] = np.where(
      df['warning__10_pt_swing_btwn_years'],
      "Greater than 10 pp swing in Presidential race between 2020 and 2024; ",
      ""
    )

    df['likely_error_detected'] = np.where(
        df['warning__county_discrepancy'],
        df["likely_error_detected"] + "Greater than 5% discrepancy from ported over county comparison; ",
        df["likely_error_detected"]
    )

    df['likely_error_detected'] = np.where(
        df['warning__pres_v_state_leg_7_pt_difference'],
        df["likely_error_detected"] + "Greater than 7 pp swing between presidential and state legislative data; ",
        df["likely_error_detected"]
    )   

    df['likely_error_detected'] = np.where(
        df['warning__low_coverage'],
        df["likely_error_detected"] + "Less than 80% of district is covered in precinct data; ",
        df["likely_error_detected"]
    )

    df['likely_error_detected'] = np.where(
        df['warning__precinct_splitting'],
        df["likely_error_detected"] + "Greater than 50% of district is split in precinct data; ",
        df["likely_error_detected"]
    )

    return df


if __name__ == '__main__':
    fp = (
        r'/Users/aspencage/Documents/Data/output/post_2024/2020_2024_pres_compare/'
        r'pres_in_district__20_v_24_comparison__20250330_095443__w_county_and_leg_races.csv'
        )
    df = pd.read_csv(fp)

    df_out = state_leg_pres__likely_errors(df)