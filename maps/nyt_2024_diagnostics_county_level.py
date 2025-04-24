import matplotlib.pyplot as plt
import time 
import os

from political_geospatial import * 
from mapping_utilities import get_all_mapfiles

if __name__ == "__main__":
    # -------------------------------
    # File paths 
    # -------------------------------
    # Output of state_leg__fix_precinct_testing
    fp_precincts_fixed_2024 = (
        r'/Users/aspencage/Documents/Data/output/post_2024/'
        r'2020_2024_pres_compare/precincts__2024_pres__fixed__20250314_181817.gpkg'
    ) # currently PA and WI are overwritten with the official data

    county_shapefile = (
        r"/Users/aspencage/Documents/Data/input/post_g2024/"
        r"comparative_presidential_performance/tl_2019_us_county"
    )
    time_string = time.strftime("%Y%m%d_%H%M%S")
    outfile_base = (
        f"/Users/aspencage/Documents/Data/output/post_2024/2020_2024_pres_compare/nyt_pres_2024_by_county_{time_string}"
    )

    # -------------------------------------------------------------------------
    # A) LOAD AND PREPARE DATA
    # -------------------------------------------------------------------------
    state_col = 'State'
    selected_states = ["NJ","VA","PA","NC","AZ","MI","WI"]
    
    counties = load_and_prepare_counties(county_shapefile, crs="EPSG:2163")

    if selected_states is not None:
        counties = counties[counties[state_col].isin(selected_states)]
    # precincts_2024 already loaded
    

    # precinct level data 
    precincts_2024 = load_and_prepare_precincts(
        fp_precincts_fixed_2024, 
        crs="EPSG:2163",
        drop_na_columns=["votes_dem","votes_rep"], # previously included 'votes_total' and 'geometry'
        print_=True
    )

    if selected_states is not None:
        precincts_2024 = precincts_2024.loc[precincts_2024[state_col].isin(selected_states)]

    # -------------------------------------------------------------------------
    # CALCULATE COVERAGE AND SPLIT PRECINCTS
    # -------------------------------------------------------------------------
    county_coverage = calculate_coverage(precincts_2024, counties, district_col='County')
    county_coverage_split = calculate_split_precinct_coverage(precincts_2024, county_coverage, district_col='County')

    # -------------------------------------------------------------------------
    # PROCESS AREA-WEIGHTED VOTE TOTALS (e.g., 2024)
    # -------------------------------------------------------------------------
    # Assume precincts_2024 has 'votes_dem', 'votes_rep', 'votes_total', etc.
    print("\nCalculating area-weighted vote totals for 2024 in counties...")
    county_coverage_split.rename(columns={"GEOID":"fips"},inplace=True)
    county_diagnostics_gdf = process_votes_area_weighted(
        gdf_precinct=precincts_2024,
        gdf_districts=county_coverage_split, 
        year="2024",
        district_col="County",
        extra_district_cols=[
        "coverage_percentage",
        "split_coverage_percentage",
        "fips"
        ]
    )

    # -------------------------------------------------------------------------
    # SAVE AND/OR PLOT RESULTS
    # -------------------------------------------------------------------------
    print(f"\nSaving final coverage & vote data to {outfile_base} as .gpkg and .csv...")
    county_diagnostics_gdf.to_file(outfile_base+".gpkg", layer="sldl_coverage", driver="GPKG")
    df_no_geom = county_diagnostics_gdf.drop(columns="geometry")
    df_no_geom.to_csv(outfile_base + ".csv", index=False)
    print("Saved successfully.\n")

    # Basic coverage plots
    fig, ax = plt.subplots(1, 2, figsize=(18, 8))

    county_diagnostics_gdf.plot(
        column="coverage_percentage",
        cmap="viridis",
        legend=True,
        edgecolor="black",
        ax=ax[0]
    )
    ax[0].set_title("Precinct Coverage (%)")

    county_diagnostics_gdf.plot(
        column="split_coverage_percentage",
        cmap="magma",
        legend=True,
        edgecolor="black",
        ax=ax[1]
    )
    ax[1].set_title("Split Precinct Coverage (%)")

    plt.tight_layout()
    plt.show()
