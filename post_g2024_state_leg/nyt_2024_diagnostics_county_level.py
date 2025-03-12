import matplotlib.pyplot as plt
import time 
import os

from nyt_2024_process_and_overlay import * 
from mapping_utilities import get_all_mapfiles

if __name__ == "__main__":
    # -------------------------------
    # File paths 
    # -------------------------------
    fp_precinct_geojson_2024 = (
        r"/Users/aspencage/Documents/Data/output/post_2024/"
        r"2020_2024_pres_compare/nyt_pres_2024_simplified.geojson"
    ) # NOTE - this is the NYT TopoJSON file, adjusting with a command line utility to GeoJSON
    fp_precincts__by_state_2024 = (
        r"/Users/aspencage/Documents/Data/input/post_g2024/nyt_2024_prez_data/geojsons-by-state"
    ) # NOTE this is the NYT GeoJSONs for each state, downloaded one-by-one

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

    districts = load_and_prepare_counties(county_shapefile, crs="EPSG:3857")
    
    # NOTE - instead of loading from NYT precinct, where data is fixed from NYT precinct, we should use that gpkg instead
    nyt_geojson_fps = get_all_mapfiles(fp_precincts__by_state_2024, extension=".geojson",print_=False)
    precincts_2024 = load_and_prepare_precincts(
        fp_precinct_geojson_2024, crs="EPSG:3857"
    )

    # -------------------------------------------------------------------------
    # B) CALCULATE COVERAGE AND SPLIT PRECINCTS
    # -------------------------------------------------------------------------
    coverage_gdf = calculate_coverage(precincts_2024, districts, district_col='County')
    coverage_split_gdf = calculate_split_precinct_coverage(precincts_2024, coverage_gdf, district_col='County')
    coverage_stats_by_state = create_state_stats(coverage_split_gdf)

    # -------------------------------------------------------------------------
    # C) PROCESS AREA-WEIGHTED VOTE TOTALS (e.g., 2024)
    # -------------------------------------------------------------------------
    # Assume precincts_2024 has 'votes_dem', 'votes_rep', 'votes_total', etc.
    print("\nCalculating area-weighted vote totals for 2024...")
    coverage_split_gdf.rename(columns={"GEOID":"fips"},inplace=True)
    final_2024_gdf = process_votes_area_weighted(
        gdf_precinct=precincts_2024,
        gdf_districts=coverage_split_gdf, 
        year="2024",
        district_col="County",
        extra_district_cols=[
          "coverage_percentage",
          "split_coverage_percentage",
          "fips"
        ]
    )


    # -------------------------------------------------------------------------
    # D) SAVE AND/OR PLOT RESULTS
    # -------------------------------------------------------------------------
    print(f"\nSaving final coverage & vote data to {outfile_base} as .gpkg and .csv...")
    final_2024_gdf.to_file(outfile_base+".gpkg", layer="sldl_coverage", driver="GPKG")
    df_no_geom = final_2024_gdf.drop(columns="geometry")
    df_no_geom.to_csv(outfile_base + ".csv", index=False)
    print("Saved successfully.\n")

    # Basic coverage plots
    fig, ax = plt.subplots(1, 2, figsize=(18, 8))

    final_2024_gdf.plot(
        column="coverage_percentage",
        cmap="viridis",
        legend=True,
        edgecolor="black",
        ax=ax[0]
    )
    ax[0].set_title("Precinct Coverage (%)")

    final_2024_gdf.plot(
        column="split_coverage_percentage",
        cmap="magma",
        legend=True,
        edgecolor="black",
        ax=ax[1]
    )
    ax[1].set_title("Split Precinct Coverage (%)")

    plt.tight_layout()
    plt.show()
