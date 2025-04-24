
import matplotlib.pyplot as plt
import time 

from political_geospatial import *
from prepare_precinct_data import *
from prepare_district_data import * 

if __name__ == "__main__":
    # -------------------------------
    # File paths (adjust as needed)
    # -------------------------------
    fp_precinct_geojson_2024 = (
        r"/Users/aspencage/Documents/Data/output/post_2024/"
        r"2020_2024_pres_compare/nyt_pres_2024_simplified.geojson"
    )
    sldl_directory = (
        r"/Users/aspencage/Documents/Data/input/post_g2024/"
        r"comparative_presidential_performance/TIGER2023_SLDL"
    )
    time_string = time.strftime("%Y%m%d_%H%M%S")
    output_gpkg = (
        f"/Users/aspencage/Documents/Data/output/post_2024/2020_2024_pres_compare/nyt_pres_2024_by_district_{time_string}.gpkg"
    )

    # -------------------------------------------------------------------------
    # A) LOAD AND PREPARE DATA
    # -------------------------------------------------------------------------
    precincts_2024 = load_and_prepare_precincts(
        fp_precinct_geojson_2024, crs="EPSG:2163"
    )
    districts = load_and_prepare_state_leg_districts(sldl_directory, crs="EPSG:2163")

    # -------------------------------------------------------------------------
    # B) CALCULATE COVERAGE AND SPLIT PRECINCTS
    # -------------------------------------------------------------------------
    coverage_gdf = calculate_coverage(precincts_2024, districts)
    coverage_split_gdf = calculate_split_precinct_coverage(precincts_2024, coverage_gdf)
    coverage_stats_by_state = create_state_stats(coverage_split_gdf)

    # -------------------------------------------------------------------------
    # C) PROCESS AREA-WEIGHTED VOTE TOTALS (e.g., 2024)
    # -------------------------------------------------------------------------
    # Assume precincts_2024 has 'votes_dem', 'votes_rep', 'votes_total', etc.
    print("\nCalculating area-weighted vote totals for 2024...")
    final_2024_gdf = process_votes_area_weighted(
        gdf_precinct=precincts_2024,
        gdf_districts=coverage_split_gdf, 
        year="2024",
        district_col="District",
        extra_district_cols=[
            "coverage_percentage",
            "split_coverage_percentage"
        ]
    )

    # -------------------------------------------------------------------------
    # D) SAVE AND/OR PLOT RESULTS
    # -------------------------------------------------------------------------
    print(f"\nSaving final coverage & vote data to {output_gpkg} ...")
    final_2024_gdf.to_file(output_gpkg, layer="sldl_coverage", driver="GPKG")
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
