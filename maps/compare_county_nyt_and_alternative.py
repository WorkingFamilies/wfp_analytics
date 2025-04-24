from prepare_precinct_data import * 
from prepare_district_data import *
from mapping_utilities import load_generic_file
import geopandas as gpd
import time 
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Load the county-level data
    county_aux_data_fp = (
        r'/Users/aspencage/Documents/Data/input/'
        r'post_g2024/comparative_presidential_performance/'
        r'G2024/County/2024_Presidential_by_County_Merged__most_recent_and_fips.csv'
        )

    county_comparison = prepare_precinct_data_multirow(
        data_input=county_aux_data_fp,
        precinct_col="fips",
        office_col="office",
        party_col="party",
        vote_col="votes",
        offices=["President "],
        dem_party="D",
        rep_party="R",
        keep_geometry=False,
        has_header=True,
        encoding="utf-8",
        col_names=None,
        dtypes={"votes":int},
        standardize_precinct=False,
        state=None,
        unique_when_combined=["fips","office","candidate"]
    )

    county_comparison.rename(columns={"D_President ": "D_President", "R_President ": "R_President"}, inplace=True)
    # torturous way to convert fips to accurate string
    county_comparison['fips'] = pd.to_numeric(county_comparison['fips'], errors='coerce').astype(int) 
    county_comparison['fips'] = county_comparison['fips'].astype(str).str.zfill(5)
    county_comparisons = state_from_county_fips(county_comparison, "fips")

    county_comparison['D_President'] = county_comparison.D_President.astype(int)
    county_comparison['R_President'] = county_comparison.R_President.astype(int)
    county_comparison["Twoway_Total_President"] = county_comparison["D_President"] + county_comparison["R_President"]
    county_comparison["D_President_Twoway_Share"] = county_comparison["D_President"] / county_comparison["Twoway_Total_President"]

    # Import the output from nyt_2024_diagnostics_county_level.py
    county_level_nyt_fp = (
        r'/Users/aspencage/Documents/Data/output/post_2024/'
        r'2020_2024_pres_compare/nyt_pres_2024_by_county_20250314_183151.gpkg'
    )

    nyt_county_gpkg_fp = county_level_nyt_fp
    nyt_counties = load_generic_file(nyt_county_gpkg_fp)
    nyt_counties['fips'] = nyt_counties['fips'].astype(str).str.zfill(5)

    merged = pd.merge(
      nyt_counties, 
      county_comparison, 
      on="fips", 
      how="outer")
    merged['pres_dem_share_two_way_2024__diff'] = merged['pres_dem_share_two_way_2024'] - merged['D_President_Twoway_Share']
    merged['pres_dem_share_two_way_2024__diff_abs'] = merged['pres_dem_share_two_way_2024__diff'].abs()
    merged['votes_two_way_by_area_2024__diff'] = merged['votes_two_way_by_area_2024'] - merged['Twoway_Total_President']
    merged['votes_two_way_by_area_2024__diff_abs'] = merged['votes_two_way_by_area_2024__diff'].abs()
    merged['votes_dem_by_area_2024__diff'] = merged['votes_dem_by_area_2024'] - merged['D_President']
    merged['votes_dem_by_area_2024__diff_abs'] = merged['votes_dem_by_area_2024__diff'].abs()

    time_str = time.strftime("%Y%m%d_%H%M%S")
    outfile_base = f'/Users/aspencage/Documents/Data/output/post_2024/2020_2024_pres_compare/nyt_pres_2024_by_county_comparison_' + time_str

    #merged.to_file(outfile_base + ".gpkg", driver="GPKG")
    df_no_geom = merged.drop(columns="geometry")
    #df_no_geom.to_csv(outfile_base + ".csv", index=False)

    print(f"Comparison saved to {outfile_base}.csv and {outfile_base}.gpkg")

    fig, ax = plt.subplots(figsize=(10, 6))
    merged.plot(column='pres_dem_share_two_way_2024__diff', cmap='coolwarm', legend=True, ax=ax, edgecolor="black")
    ax.set_title("Difference in Democratic Two-Way Vote Share (2024)")
    plt.show()