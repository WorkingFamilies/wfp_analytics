import os
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd

from mapping_utilities import (
    process_district_data, 
    map_fips_and_state, 
    concatenate_geodata,
    fix_invalid_geometries,
    map_code_to_value
)
from mapping_dicts import pa_counties

# Description: Functions to prepare district data for use in district based analysis.

def normalize_district(district_value, chamber='sldl'):
    """
    Normalizes a district name by prefixing with HD- (lower) or SD- (upper)
    and zero-padding if the district value is numeric.

    Parameters:
        district_value: str or number (e.g., '1', 'Chittenden', '1.0', 'A1')
        chamber: 'lower' or 'upper' to designate HD- or SD- prefix

    Returns:
        A standardized district string like 'HD-001' or 'SD-Chittenden'
    """
    if pd.isnull(district_value):
        return ""

    prefix = 'HD-' if chamber == 'sldl' else 'SD-' if chamber == 'sldu' else ''
    
    if prefix == '':
        return ""

    s = str(district_value).strip()

    # Normalize numeric-like strings like "1.0"
    if s.replace(".", "").isdigit():
        try:
            numeric_val = int(float(s))
            district = f"{numeric_val:03d}"
        except ValueError:
            district = s
    else:
        district = s

    return f"{prefix}{district}"



def normalize_district_col(gdf, district_col="District", state_col="State", chamber='sldl'):
    # Normalize the district column

    if chamber not in ['sldl', 'sldu']:
        raise ValueError("Chamber must be 'sldl' or 'sldu'.")
    if district_col not in gdf.columns:
        raise ValueError(f"'{district_col}' column not found in GeoDataFrame.")
    if state_col not in gdf.columns:
        raise ValueError(f"'{state_col}' column not found in GeoDataFrame.")

    gdf[district_col] = gdf[district_col].apply(lambda x: normalize_district(x, chamber=chamber))

    # Prefix with "State-..." so "001" becomes "MI-001", etc.
    if state_col in gdf.columns:
        gdf[district_col] = (
            gdf[state_col] + " " + gdf[district_col]
        )
    else:
        print(f"[WARNING] '{state_col}' column not found; cannot prefix district IDs.")

        # Prepend state abbreviation, e.g. "MI-1"
        gdf[district_col] = (
            gdf[state_col] + " " + gdf[district_col]
        )

    return gdf


def overwrite_single_state(
    master_gdf, 
    override_gdf, 
    state_abbrev, 
    state_col="State"
):
    """
    Overwrite all rows for `state_abbrev` in `master_gdf` with the rows in 
    `override_gdf` (which should only contain that state's data).

    Parameters
    ----------
    master_gdf : GeoDataFrame
        The full dataset of districts.
    override_gdf : GeoDataFrame
        The override dataset containing new rows for a single state.
    state_abbrev : str
        The two-letter abbreviation (or any string) used to identify the state 
        you want to overwrite.
    state_col : str, default "State"
        The name of the column in both DataFrames that identifies the state.

    Returns
    -------
    GeoDataFrame
        A new combined GeoDataFrame with `state_abbrev` replaced by `override_gdf`.
    """
    # Filter out old rows for this state
    mask_keep = (master_gdf[state_col] != state_abbrev)
    out_gdf = master_gdf.loc[mask_keep].copy()

    # Append the override data
    out_gdf = pd.concat([out_gdf, override_gdf], ignore_index=True)
    return out_gdf


def load_and_prepare_state_leg_districts(sldl_directory, crs="EPSG:2163",chamber='sldl'):
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

    # Ensure District IDs are unique across states by normalizing the district column
    districts = normalize_district_col(districts, district_col="District", state_col="State", chamber=chamber)

    # Compute each district's area
    districts["dist_area"] = districts.geometry.area
    return districts


def load_and_prepare_districts_w_overrides(
    base_directory,
    override_configs=None,
    crs="EPSG:2163",
    state_col="State",
    district_col="District",
    main_source_name="bulk_main",
    chamber='sldl'
):
    """
    Loads all state legislative districts from ``base_directory`` using 
    ``load_and_prepare_state_leg_districts``, then applies overrides for 
    particular states as described by ``override_configs``.

    Parameters
    ----------
    base_directory : str
        Path to a directory containing bulk SLD shapefiles (possibly in subfolders).

    override_configs : dict or None, optional
        A dictionary keyed by state abbreviation (e.g. "MI", "FL"), with each
        value being another dictionary describing how to override that state's
        district geometry.

        The typical structure is:

        .. code-block:: python

            override_configs = {
                "MI": {
                    "path": "/path/to/michigan_sldl_2024.shp",
                    "col_map": {"DIST_NUM": "District"},
                    "process_override": True,
                    "keep_all_columns": False,
                    "geo_source": "Motown_Shapes_2024"
                },
                "FL": {
                    "path": gpd.GeoDataFrame(...),
                    "col_map": {"OLD_DIST": "District"},
                    "process_override": True,
                    "keep_all_columns": True,
                    "geo_source": "FL_AlternativeData_2024"
                }
            }

        Each override entry may contain:

        - **"path"** (required): A filepath (string) or an in-memory ``GeoDataFrame`` 
          containing district geometry for the override.
        - **"col_map"** (optional, default = {}): Dict mapping old column names 
          to new names (e.g. ``{"LEG_DIST": "District"}``).
        - **"process_override"** (optional, default = True): Whether to run 
          post-processing steps (fix invalid geometries, convert numeric district 
          codes, compute district area, etc.).
        - **"keep_all_columns"** (optional, default = True): If ``False``, only 
          keep the minimal set of columns (i.e., ``state_col``, ``district_col``, 
          ``geometry``, and ``dist_area`` if present). If ``True``, all columns 
          in the override data are retained.
        - **"geo_source"** (optional, default = ``f"{state_abbrev}_override"``): 
          A string label denoting the data source of that override. This label 
          will be assigned to the ``geo_source`` column for every row in the 
          override dataset.

    crs : str, default "EPSG:2163"
        The coordinate reference system to ensure for both the master and override data.

    state_col : str, default "State"
        The column name in both the master and override data that identifies the state.

    district_col : str, default "District"
        The column name in both the master and override data that identifies the district.

    main_source_name : str, default "bulk_main"
        A label to assign to the ``geo_source`` column for all rows in the *master* dataset 
        loaded from ``base_directory``.

    Returns
    -------
    GeoDataFrame
        A combined GeoDataFrame of districts, where any state specified in
        ``override_configs`` is replaced by its override geometry and attributes.

    Notes
    -----
    - If an override dataset has no column matching ``state_col``, its entire 
      dataset is labeled with the override's state abbreviation.
    - If the override dataset's district identifier is numeric (e.g., "1.0"), 
      it's converted to an integer (e.g., 1) before being combined into the 
      final district name (e.g., "MI-1").
    - The ``geo_source`` column tracks the origin of each row (e.g., the main 
      bulk data vs. a custom override). 
    """
    # 1) Load the master dataset from the bulk directory
    master_gdf = load_and_prepare_state_leg_districts(base_directory, crs=crs, chamber=chamber)

    # 2) Add geo_source to master if not present; label with main_source_name
    if "geo_source" not in master_gdf.columns:
        master_gdf["geo_source"] = main_source_name
    else:
        master_gdf["geo_source"] = main_source_name

    # 3) If no overrides, just return the master
    if not override_configs:
        return master_gdf

    for state_abbrev, override_info in override_configs.items():
        override_data = override_info.get("path")
        col_map = override_info.get("col_map", {})
        do_process = override_info.get("process_override", True)
        keep_all_columns = override_info.get("keep_all_columns", True)
        # If override has no explicit geo_source, use default e.g. "MI_override"
        override_source_name = override_info.get("geo_source", f"{state_abbrev}_override")

        if override_data is None:
            print(f"No override path/data provided for {state_abbrev}; skipping.")
            continue

        # --- 4a) Load or copy the override data ---
        if isinstance(override_data, str):
            print(f"\nLoading override for {state_abbrev} from: {override_data}")
            override_gdf = gpd.read_file(override_data)
            if override_gdf.crs is None:
                override_gdf.set_crs(crs, inplace=True)
            else:
                override_gdf = override_gdf.to_crs(crs)
        elif isinstance(override_data, gpd.GeoDataFrame):
            print(f"\nUsing in-memory override for {state_abbrev}.")
            override_gdf = override_data.copy()
            if override_gdf.crs is None:
                override_gdf.set_crs(crs, inplace=True)
            else:
                override_gdf = override_gdf.to_crs(crs)
        else:
            raise TypeError(
                f"Override for {state_abbrev} must be a path or GeoDataFrame, got: {type(override_data)}"
            )

        # --- 4b) Rename columns if provided ---
        if col_map:
            override_gdf.rename(columns=col_map, inplace=True)

        # --- 4c) If there's no 'state_col', mass-label the entire dataset ---
        if state_col not in override_gdf.columns:
            print(f"  [INFO] No '{state_col}' column in override for {state_abbrev}; "
                  f"mass-labeling all rows as '{state_abbrev}'.")
            override_gdf[state_col] = state_abbrev

        if district_col in override_gdf.columns:

            override_gdf = normalize_district_col(override_gdf, district_col=district_col,chamber=chamber)

            # Compute district area
            override_gdf["dist_area"] = override_gdf.geometry.area

        # --- 4e) Assign geo_source to override rows ---
        override_gdf["geo_source"] = override_source_name

        # --- 4f) Decide which columns to keep if keep_all_columns=False ---
        if not keep_all_columns:
            essential_cols = {state_col, district_col, "geometry", "geo_source"}
            if "dist_area" in override_gdf.columns:
                essential_cols.add("dist_area")

            actual_cols = set(override_gdf.columns)
            columns_to_keep = essential_cols.intersection(actual_cols)
            override_gdf = override_gdf[list(columns_to_keep)].copy()

        # --- 4g) Filter override to state_abbrev ---
        mask = (override_gdf[state_col] == state_abbrev)
        override_gdf = override_gdf.loc[mask].copy()
        if override_gdf.empty:
            print(f"  [WARNING] Override for {state_abbrev} is empty after filtering "
                  f"to {state_col}={state_abbrev}. Skipping.")
            continue

        # --- 4h) Overwrite master rows for this state ---
        master_gdf = overwrite_single_state(
            master_gdf, 
            override_gdf,
            state_abbrev=state_abbrev,
            state_col=state_col
        )

    return master_gdf


def state_from_county_fips(gdf, fips_col="COUNTYFP",state_col="State"):
    """
    Given a GeoDataFrame with a column of county FIPS codes, adds a new column 
    'State' with the corresponding state abbreviation.
    """
    gdf[state_col] = gdf[fips_col].astype(str).str[:2].apply(map_fips_and_state)
    return gdf


def load_and_prepare_counties(
        fp_county_shapefile, 
        crs="EPSG:2163", 
        state_fips_col="STATEFP",
        county_fips_col="GEOID"
        ):
    """
    Loads county shapefile(s) via 'concatenate_geodata', fixes invalid geometries,
    reprojects, and calculates area. Returns the prepared GeoDataFrame.
    """
    print(f"Loading county shapefile(s) from: {fp_county_shapefile}")
    counties = concatenate_geodata(fp_county_shapefile, crs=crs)
    counties = fix_invalid_geometries(counties)
    
    # Map state_fips_col or county_fips_col to state abbreviations
    if state_fips_col is not None:
        if state_fips_col in counties.columns:
            counties["State"] = counties[state_fips_col].astype(str).apply(map_fips_and_state)
        else:
            raise ValueError(f"No '{state_fips_col}' column found in the loaded district shapefiles.")
    
    if county_fips_col is not None:
        if county_fips_col in counties.columns:
            counties[county_fips_col] = counties[county_fips_col].astype(str).str.zfill(5)
            counties = state_from_county_fips(counties, fips_col=county_fips_col)
        else:
            raise ValueError(f"No '{county_fips_col}' column found in the loaded district shapefiles.")

    # Drop rows missing crucial data
    before_drop = len(counties)
    counties = counties.dropna(subset=["State", "NAMELSAD"])
    after_drop = len(counties)
    print(f"Dropped {before_drop - after_drop} rows with missing State or County data.")

    counties["County"] = counties["NAMELSAD"] + ", " + counties["State"]

    # Calculate county area
    counties["dist_area"] = counties.geometry.area
    
    return counties
