import os
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import time 

from mapping_utilities import (
    process_district_data, 
    map_fips_and_state, 
    concatenate_geodata,
    fix_invalid_geometries
)

from political_geospatial import create_state_stats

###############################################################################
# 1. HELPER FUNCTIONS FOR LOADING / PREPARATION
###############################################################################

def load_and_prepare_precincts(fp_precinct_geojson, crs="EPSG:2163",print_=False):
    """
    Loads precinct GeoJSON (or a list of them) via 'concatenate_geodata',
    fixes invalid geometries, reprojects, calculates area, and 
    renames 'GEOID' -> 'precinct' if present.
    """
    if print_:
        if isinstance(fp_precinct_geojson, list):
            print(f"Loading precinct(s) from list of {len(fp_precinct_geojson)} files.")
        elif isinstance(fp_precinct_geojson, str):
            print(f"Loading precinct(s) from: {fp_precinct_geojson}")
    precincts = concatenate_geodata(fp_precinct_geojson, crs=crs)
    precincts = fix_invalid_geometries(precincts)

    # Rename GEOID -> precinct if needed
    if "GEOID" in precincts.columns:
        precincts.rename(columns={"GEOID": "precinct"}, inplace=True)
    else:
        # If there's no GEOID column, ensure we have some precinct identifier
        if "precinct" not in precincts.columns:
            precincts["precinct"] = precincts.index.astype(str)

    # Calculate precinct area
    precincts["precinct_area"] = precincts.geometry.area
    return precincts

def load_and_prepare_state_leg_districts(sldl_directory, crs="EPSG:2163"):
    """
    Recursively collects shapefiles from a directory and uses 'concatenate_geodata' 
    to load them. Then processes them via 'process_district_data', fixes invalid geometries, 
    reprojects to `crs`, maps STATEFP -> State abbreviations, and calculates district area.
    """
    # Collect all .shp files from subdirectories
    district_files = []
    for root, _, files in os.walk(sldl_directory):
        for file in files:
            if file.lower().endswith('.shp'):
                district_files.append(os.path.join(root, file))
                
    if not district_files:
        raise ValueError(f"No .shp files found in {sldl_directory}")

    # Use concatenate_geodata to read and reproject
    print(f"Loading district shapefiles from directory: {sldl_directory}")
    districts = concatenate_geodata(district_files, crs=crs, print_=False)
    districts = fix_invalid_geometries(districts)

    # Process data (from your mapping_utilities)
    districts, _ = process_district_data(districts, state=None)

    # Map STATEFP to state abbreviations
    if "STATEFP" in districts.columns:
        districts["State"] = districts["STATEFP"].astype(str).apply(map_fips_and_state)
    else:
        raise ValueError("No 'STATEFP' column found in the loaded district shapefiles.")

    # Drop rows missing crucial data
    before_drop = len(districts)
    districts = districts.dropna(subset=["State", "District"])
    after_drop = len(districts)
    print(f"Dropped {before_drop - after_drop} rows with missing State or District data.")

    # Ensure District IDs are unique across states
    districts["District"] = districts["State"] + "-" + districts["District"].astype(str)

    # Compute each district's area
    districts["dist_area"] = districts.geometry.area
    return districts


def load_and_prepare_counties(fp_county_shapefile, crs="EPSG:2163"):
    """
    Loads county shapefile(s) via 'concatenate_geodata', fixes invalid geometries,
    reprojects, and calculates area. Returns the prepared GeoDataFrame.
    """
    print(f"Loading county shapefile(s) from: {fp_county_shapefile}")
    counties = concatenate_geodata(fp_county_shapefile, crs=crs)
    counties = fix_invalid_geometries(counties)
    
    # Map STATEFP to state abbreviations
    if "STATEFP" in counties.columns:
        counties["State"] = counties["STATEFP"].astype(str).apply(map_fips_and_state)
    else:
        raise ValueError("No 'STATEFP' column found in the loaded district shapefiles.")

    # Drop rows missing crucial data
    before_drop = len(counties)
    counties = counties.dropna(subset=["State", "NAMELSAD"])
    after_drop = len(counties)
    print(f"Dropped {before_drop - after_drop} rows with missing State or County data.")

    counties["County"] = counties["NAMELSAD"] + ", " + counties["State"]

    # Calculate county area
    counties["dist_area"] = counties.geometry.area
    
    return counties


###############################################################################
# 2. COVERAGE + SPLIT PRECINCT CALCULATIONS
###############################################################################

def calculate_coverage(precincts, districts, district_col="District"):
    """
    Calculates the fraction of each district covered by the given precincts.
    Returns a new GeoDataFrame with 'coverage_fraction' and 
    'coverage_percentage' columns.
    """
    print("Calculating district coverage (intersection areas)...")
    intersections = gpd.overlay(precincts, districts, how="intersection")
    intersections["intersection_area"] = intersections.geometry.area
    
    # Group intersection areas by District
    district_intersection_areas = (
        intersections.groupby(district_col)["intersection_area"].sum().reset_index()
    )

    # Merge onto districts
    merged = districts.merge(district_intersection_areas, on=district_col, how="left")
    merged["intersection_area"] = merged["intersection_area"].fillna(0)
    merged["coverage_fraction"] = merged["intersection_area"] / merged["dist_area"]
    merged["coverage_percentage"] = merged["coverage_fraction"]

    return merged

def calculate_split_precinct_coverage(precincts, coverage_gdf, district_col="District"):
    """
    Calculates how much of each district is comprised of 'split' precincts
    (where the intersection fraction of the precinct is between 3% and 97%).
    Returns a new GeoDataFrame with 'split_coverage_fraction' and 
    'split_coverage_percentage' columns.
    """
    print("Calculating 'split precinct' coverage...")
    intersections = gpd.overlay(precincts, coverage_gdf, how="intersection")
    intersections["intersection_area"] = intersections.geometry.area

    '''
    # Bring in precinct name
    intersections = intersections.merge(
        precincts[["precinct", "precinct_area"]], on="precinct", how="left"
    )
    '''

    # Fraction of each precinct that is in a district
    try:
        intersections["precinct_fraction"] = (
            intersections["intersection_area"] / intersections["precinct_area"]
        )
    except KeyError:
        raise KeyError(f"Missing 'precinct_area' column in precincts data. Columns: {intersections.columns}")

    # Mark 'split' if 3% < fraction < 97%
    # Note: "inclusive" parameter usage varies by pandas version
    split_mask = intersections["precinct_fraction"].between(0.03, 0.97, inclusive="both")
    intersections["is_split"] = np.where(split_mask, True, False)

    # For 'split' areas, keep the intersection area; otherwise 0
    intersections["split_area"] = intersections["intersection_area"].where(
        intersections["is_split"], 0
    )

    # Sum the 'split_area' in each district
    district_split_areas = (
        intersections.groupby(district_col)["split_area"].sum().reset_index(name="split_area_sum")
    )

    # Merge back
    merged = coverage_gdf.merge(district_split_areas, on=district_col, how="left")
    merged["split_area_sum"] = merged["split_area_sum"].fillna(0)

    # Compute fraction and percentage
    merged["split_coverage_fraction"] = merged["split_area_sum"] / merged["dist_area"]
    merged["split_coverage_percentage"] = merged["split_coverage_fraction"]

    return merged


###############################################################################
# 3. VOTE AGGREGATION / AREA-WEIGHTED FUNCTIONS
###############################################################################

def process_votes_area_weighted(
    gdf_precinct, 
    gdf_districts, 
    year="2024",
    district_col="District",
    extra_district_cols=None
):
    """
    Given precinct-level vote columns (rep, dem, total) and a district GDF,
    performs area-weighted overlay to compute aggregated vote totals + shares
    by district. Returns a new GeoDataFrame containing these aggregations
    merged with `gdf_districts`.
    """
    # Ensure CRS match
    if gdf_precinct.crs != gdf_districts.crs:
        gdf_precinct = gdf_precinct.to_crs(gdf_districts.crs)

    # Copy to avoid in-place editing
    gdf_precinct = gdf_precinct.copy()

    # Compute extra columns in precincts
    gdf_precinct["votes_third_party"] = (
        gdf_precinct["votes_total"] 
        - gdf_precinct["votes_rep"] 
        - gdf_precinct["votes_dem"]
    )
    gdf_precinct["votes_two_way"] = gdf_precinct["votes_dem"] + gdf_precinct["votes_rep"]

    # Spatial intersection
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # common overlay warnings
        subset = gpd.overlay(gdf_precinct, gdf_districts, how="intersection")

    # Area-based weighting
    subset["split_area"] = subset.geometry.area
    subset["area_fraction"] = subset["split_area"] / subset["precinct_area"]

    # Multiply vote columns by fraction
    subset[f"votes_rep_by_area_{year}"] = subset["votes_rep"] * subset["area_fraction"]
    subset[f"votes_dem_by_area_{year}"] = subset["votes_dem"] * subset["area_fraction"]
    subset[f"votes_third_party_by_area_{year}"] = subset["votes_third_party"] * subset["area_fraction"]
    subset[f"votes_total_by_area_{year}"] = subset["votes_total"] * subset["area_fraction"]
    subset[f"votes_two_way_by_area_{year}"] = subset["votes_two_way"] * subset["area_fraction"]

    # Group by district
    grouped = subset.groupby(district_col).agg({
        f"votes_rep_by_area_{year}": "sum",
        f"votes_dem_by_area_{year}": "sum",
        f"votes_third_party_by_area_{year}": "sum",
        f"votes_total_by_area_{year}": "sum",
        f"votes_two_way_by_area_{year}": "sum",
    }).reset_index()

    # Compute shares
    grouped[f"pres_dem_share_total_{year}"] = (
        grouped[f"votes_dem_by_area_{year}"] 
        / grouped[f"votes_total_by_area_{year}"]
    )
    grouped[f"pres_dem_share_two_way_{year}"] = (
        grouped[f"votes_dem_by_area_{year}"]
        / grouped[f"votes_two_way_by_area_{year}"]
    )
    grouped[f"third_party_vote_share_{year}"] = (
        grouped[f"votes_third_party_by_area_{year}"]
        / grouped[f"votes_total_by_area_{year}"]
    )

    # Merge with original district geometry
    keep_cols = [district_col, "geometry"]
    if extra_district_cols:
        keep_cols += extra_district_cols
    print(gdf_districts.columns)
    
    districts_subset = gdf_districts[keep_cols].drop_duplicates(subset=[district_col])
    final = districts_subset.merge(grouped, on=district_col, how="left")
    final = gpd.GeoDataFrame(final, geometry="geometry", crs=gdf_districts.crs)

    return final

###############################################################################
# 4. MAIN EXECUTION BLOCK
###############################################################################

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
