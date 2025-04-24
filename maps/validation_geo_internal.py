
import geopandas as gpd 
import numpy as np

# This is a home for functions that are used to validate the internal consistency of the data.

###############################################################################
# COVERAGE + SPLIT PRECINCT CALCULATIONS
###############################################################################

def calculate_coverage(precincts, districts, district_col="District", state_col="State", states_to_excl=None):
    """
    Calculates the fraction of each district covered by the given precincts.
    Returns a new GeoDataFrame with 'coverage_fraction' and 'coverage_percentage' columns.

    Parameters:
    - precincts (GeoDataFrame): Precinct-level geodata.
    - districts (GeoDataFrame): District-level geodata.
    - district_col (str): The column that identifies districts.
    - state_col (str): The column that identifies states.
    - states_to_excl (list): List of states to exclude from calculations (default: None).
    """
    if states_to_excl is None:
        states_to_excl = []

    print("Calculating district coverage (intersection areas)...")

    # Drop district column from precincts to avoid error from intersection with identically-named districts in both gpds
    if district_col in precincts.columns:
        precincts = precincts.copy().drop(columns=district_col)

    # Filter out states to exclude
    precincts_filtered = precincts[~precincts[state_col].isin(states_to_excl)]
    districts_filtered = districts[~districts[state_col].isin(states_to_excl)]

    # Perform intersection only on included states
    intersections = gpd.overlay(precincts_filtered, districts_filtered, how="intersection")
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

    # Ensure excluded states remain but with null coverage values
    merged.loc[merged[state_col].isin(states_to_excl), ["coverage_fraction", "coverage_percentage"]] = np.nan

    return merged


def calculate_split_precinct_coverage(precincts, coverage_gdf, district_col="District", state_col="State", states_to_excl=None, calc_missing_areas=True):
    """
    Calculates how much of each district is comprised of 'split' precincts
    (where the intersection fraction of the precinct is between 3% and 97%).
    Returns a new GeoDataFrame with 'split_coverage_fraction' and 
    'split_coverage_percentage' columns.

    Parameters:
    - precincts (GeoDataFrame): Precinct-level geodata.
    - coverage_gdf (GeoDataFrame): Dataframe containing precomputed coverage information.
    - district_col (str): The column that identifies districts.
    - state_col (str): The column that identifies states.
    - states_to_excl (list): List of states to exclude from calculations (default: None).
    """
    if states_to_excl is None:
        states_to_excl = []

    print("Calculating 'split precinct' coverage...")

    if district_col in precincts.columns:
        precincts = precincts.copy().drop(columns=district_col)

    # Filter out states to exclude
    precincts_filtered = precincts[~precincts[state_col].isin(states_to_excl)]
    coverage_filtered = coverage_gdf[~coverage_gdf[state_col].isin(states_to_excl)]

    if calc_missing_areas:
        # Calculate missing areas for precincts and coverage
        precincts_filtered["precinct_area"] = precincts_filtered.geometry.area
        coverage_filtered["dist_area"] = coverage_filtered.geometry.area

    # Perform intersection only on included states
    intersections = gpd.overlay(precincts_filtered, coverage_filtered, how="intersection")
    intersections["intersection_area"] = intersections.geometry.area

    # Calculate precinct fraction
    try:
        intersections["precinct_fraction"] = (
            intersections["intersection_area"] / intersections["precinct_area"]
        )
    except KeyError:
        raise KeyError(f"Missing 'precinct_area' column in precincts data. Columns: {intersections.columns}")

    # Identify 'split' precincts (3% < fraction < 97%)
    split_mask = intersections["precinct_fraction"].between(0.03, 0.97, inclusive="both")
    intersections["is_split"] = np.where(split_mask, True, False)

    # Assign 'split_area' only to split precincts
    intersections["split_area"] = intersections["intersection_area"].where(intersections["is_split"], 0)

    # Sum 'split_area' in each district
    district_split_areas = (
        intersections.groupby(district_col)["split_area"].sum().reset_index(name="split_area_sum")
    )

    # Merge back
    merged = coverage_gdf.merge(district_split_areas, on=district_col, how="left")
    merged["split_area_sum"] = merged["split_area_sum"].fillna(0)

    # Compute fraction and percentage
    merged["split_coverage_fraction"] = merged["split_area_sum"] / merged["dist_area"]
    merged["split_coverage_percentage"] = merged["split_coverage_fraction"]

    # Ensure excluded states remain but with null split coverage values
    merged.loc[merged[state_col].isin(states_to_excl), ["split_coverage_fraction", "split_coverage_percentage"]] = np.nan

    return merged
