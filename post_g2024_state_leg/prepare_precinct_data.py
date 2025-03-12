import pandas as pd
import numpy as np
import os 
import geopandas as gpd  # If you eventually want geometry
from mapping_utilities import load_generic_file
from dataset_utilities import drop_duplicates_qualify


def standardize_precinct_data(
    input_data,
    # --- Column name parameters ---
    precinct_col="precinct",
    dem_col="votes_dem",
    rep_col="votes_rep",
    total_col="votes_total",
    # --- If total_col is None, compute from dem_col + rep_col (+ optional others) ---
    compute_total_if_missing=True,
    compute_third_party_if_possible=True,
    geometry_col="geometry",
    fix_invalid_geometries=True,
    target_crs="EPSG:3857",
    rename_map=None,
    retain_addl_cols:list=False
):
    """
    Transforms precinct-level data into a standardized GeoDataFrame
    suitable for coverage and vote aggregation analyses.
    
    Parameters
    ----------
    input_data : str or GeoDataFrame
        A path to a geospatial file (GeoJSON, Shapefile, etc.) 
        or an already-loaded GeoDataFrame containing precinct data.
    precinct_col : str, optional
        Column name to be used as the precinct identifier. 
        Will be renamed to 'precinct'. Defaults to 'precinct'.
    dem_col : str, optional
        Column in the input data representing Democratic votes. Defaults to 'votes_dem'.
    rep_col : str, optional
        Column in the input data representing Republican votes. Defaults to 'votes_rep'.
    total_col : str or None, optional
        Column in the input data representing total votes. If None or not found,
        and compute_total_if_missing=True, the function will attempt to compute 
        it as (dem_col + rep_col). Defaults to 'votes_total'.
    compute_total_if_missing : bool, optional
        If True, compute votes_total = votes_dem + votes_rep if total_col is not 
        provided or found. Defaults to True.
    compute_third_party_if_possible : bool, optional
        If True, and a valid total is present (or computed), compute 
        votes_third_party = votes_total - votes_dem - votes_rep.
        Defaults to False.
    geometry_col : str, optional
        Column name of the geometry in the GeoDataFrame. Defaults to 'geometry'.
    fix_invalid_geometries : bool, optional
        Whether to attempt to fix invalid geometries using buffer(0). Defaults to True.
    target_crs : str, optional
        The target CRS for reprojecting. Defaults to 'EPSG:3857' (Web Mercator).
    rename_map : dict, optional
        A dictionary for additional column renaming. For instance, 
        if your input has 'dem_2024' and you want that to become 'votes_dem', 
        you could pass {'dem_2024': 'votes_dem'}. 
        This is applied before the main renaming steps, 
        allowing ultimate flexibility with your column naming.

    Returns
    -------
    GeoDataFrame
        A standardized GeoDataFrame with columns:
        'precinct', 'votes_dem', 'votes_rep', 'votes_total', 'geometry',
        and (optionally) 'votes_third_party' if compute_third_party_if_possible=True.
    """
    # ---------------------------------------------------------------------
    # 1. Load the data if input_data is a path; otherwise assume it's a GDF
    # ---------------------------------------------------------------------
    if isinstance(input_data, str):
        cols = [precinct_col, dem_col, rep_col, total_col, geometry_col]
        if retain_addl_cols:
            cols.extend(retain_addl_cols)
        gdf = gpd.read_file(input_data,columns=cols)
    else:
        gdf = input_data.copy()

    # Ensure it's a GeoDataFrame if it's a DataFrame
    if not isinstance(gdf, gpd.GeoDataFrame):
        if geometry_col not in gdf.columns:
            raise ValueError(
                "input_data is a DataFrame with no valid geometry column. "
                f"Ensure '{geometry_col}' is present or provide a GeoDataFrame."
            )
        gdf = gpd.GeoDataFrame(gdf, geometry=gdf[geometry_col], crs=gdf.crs or "EPSG:4326")

    # ---------------------------------------------------------------------
    # 2. (Optional) Apply a rename map for custom columns
    # ---------------------------------------------------------------------
    if rename_map:
        gdf.rename(columns=rename_map, inplace=True)

    # ---------------------------------------------------------------------
    # 3. Standardize precinct identifier column 
    # ---------------------------------------------------------------------
    if precinct_col not in gdf.columns:
        if "GEOID" in gdf.columns:
            gdf.rename(columns={"GEOID": "precinct"}, inplace=True)
        else:
            # Create a precinct column from the index
            gdf["precinct"] = gdf.index.astype(str)
    else:
        if precinct_col != "precinct":
            gdf.rename(columns={precinct_col: "precinct"}, inplace=True)

    # ---------------------------------------------------------------------
    # 4. Standardize vote columns
    # ---------------------------------------------------------------------
    # votes_dem
    if dem_col in gdf.columns and dem_col != "votes_dem":
        gdf.rename(columns={dem_col: "votes_dem"}, inplace=True)
    elif dem_col not in gdf.columns:
        print(f"Column '{dem_col}' not found. Creating 'votes_dem' = 0.")
        gdf["votes_dem"] = 0

    # votes_rep
    if rep_col in gdf.columns and rep_col != "votes_rep":
        gdf.rename(columns={rep_col: "votes_rep"}, inplace=True)
    elif rep_col not in gdf.columns:
        print(f"Column '{rep_col}' not found. Creating 'votes_rep' = 0.")
        gdf["votes_rep"] = 0

    # votes_total
    if total_col and total_col in gdf.columns and total_col != "votes_total":
        gdf.rename(columns={total_col: "votes_total"}, inplace=True)
    elif (not total_col or total_col not in gdf.columns) and compute_total_if_missing:
        gdf["votes_total"] = gdf["votes_dem"] + gdf["votes_rep"]
    else:
        if total_col and total_col not in gdf.columns:
            print(f"Warning: '{total_col}' not found, and compute_total_if_missing=False. Setting 'votes_total' = 0.")
            gdf["votes_total"] = 0
        elif "votes_total" not in gdf.columns:
            gdf["votes_total"] = 0

    # ---------------------------------------------------------------------
    # 5. Optionally compute third-party votes
    # ---------------------------------------------------------------------
    if compute_third_party_if_possible:
        if "votes_total" in gdf.columns and ("votes_dem" in gdf.columns) and ("votes_rep" in gdf.columns):
            # If user already has 'votes_third_party', we can overwrite or warn
            gdf["votes_third_party"] = (
                gdf["votes_total"] - gdf["votes_dem"] - gdf["votes_rep"]
            )
        else:
            print("Cannot compute third-party votes (missing total/dem/rep columns). Skipping.")

    # ---------------------------------------------------------------------
    # 6. Fix invalid geometries (if requested)
    # ---------------------------------------------------------------------
    if fix_invalid_geometries:
        invalid_count = (~gdf.is_valid).sum()
        if invalid_count > 0:
            print(f"Found {invalid_count} invalid geometries. Attempting to fix via buffer(0).")
            gdf["geometry"] = gdf["geometry"].buffer(0)

    # ---------------------------------------------------------------------
    # 7. Reproject to target CRS (if specified) and calculate precinct area
    # ---------------------------------------------------------------------
    if target_crs:
        if gdf.crs is None:
            print("Input data has no CRS. Assuming 'EPSG:4326' before reprojecting.")
            gdf.set_crs("EPSG:4326", inplace=True)
        gdf = gdf.to_crs(target_crs)
    
    # Calculate precinct area
    gdf["precinct_area"] = gdf.geometry.area

    # ---------------------------------------------------------------------
    # 8. Final check: ensure key columns exist
    # ---------------------------------------------------------------------
    
    if compute_third_party_if_possible:
        needed_cols = ["precinct", "votes_total", "votes_dem", "votes_rep", "votes_third_party", "precinct_area", "geometry"]
    else:
        needed_cols = ["precinct", "votes_total", "votes_dem", "votes_rep", "precinct_area", "geometry"]

    for col in needed_cols:
        if col not in gdf.columns:
            print(f"Warning: Column '{col}' is missing in the final DataFrame.")

    # Reorder columns (optional convenience)
    # Put standardized columns first
    final_cols = [c for c in needed_cols if c in gdf.columns]  # only those that exist
    rest_cols = [c for c in gdf.columns if c not in final_cols]
    gdf = gdf[final_cols + rest_cols]

    return gdf


def prepare_precinct_data_single_row(
    data_input,
    precinct_col="PrecinctName",
    offices=["LEG", "PRES"],
    dem_prefix="DEM_",
    rep_prefix="REP_",
    keep_geometry=True,
    has_header=True,
    encoding="utf-8",
    col_names=None,
    dtypes=None
):
    """
    Loads data where each row is already a single precinct with vote totals for each office.
    For each office in the provided list, the function expects two columns:
        - {dem_prefix}{office} (e.g., DEM_LEG)
        - {rep_prefix}{office} (e.g., REP_LEG)
    
    The function renames the precinct column to 'precinct', validates that all expected
    columns are present, and converts vote total columns to numeric types.
    
    Can handle CSV, Shapefile, GeoJSON, or an in-memory DataFrame/GeoDataFrame.
    
    Parameters
    ----------
    data_input : str or DataFrame or GeoDataFrame
        The source data (file path or in-memory).
    precinct_col : str
        The column containing the precinct identifier in the input data.
    offices : list of str
        A list of office identifiers. For each office, the function expects vote totals
        in columns named {dem_prefix}{office} and {rep_prefix}{office}.
    dem_prefix : str
        Prefix for the Democratic vote total columns.
    rep_prefix : str
        Prefix for the Republican vote total columns.
    keep_geometry : bool
        Whether to keep geometry if the data has geospatial information.
    has_header : bool
        If True, the CSV has a header row (ignored for other formats).
    encoding : str
        File encoding (for CSV).
    col_names : list of str or None
        If has_header=False, you can specify column names for the CSV.
    dtypes : dict or None
        Optionally specify column data types for CSV.
    
    Returns
    -------
    pd.DataFrame or gpd.GeoDataFrame
        A DataFrame/GeoDataFrame with a standardized precinct column ('precinct') and
        vote total columns for each office as {dem_prefix}{office} and {rep_prefix}{office},
        plus geometry if keep_geometry=True and available.
    """
    # 1) Load the data using a generic loader
    df = load_generic_file(
        data_input=data_input,
        has_header=has_header,
        encoding=encoding,
        col_names=col_names,
        dtypes=dtypes,
        keep_geometry=keep_geometry
    )
    
    # 2) Build list of required columns
    required_cols = [precinct_col]
    for office in offices:
        required_cols.append(f"{dem_prefix}{office}")
        required_cols.append(f"{rep_prefix}{office}")
    
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataset.")
    
    # 3) Standardize the precinct column name
    if precinct_col != "precinct":
        df.rename(columns={precinct_col: "precinct"}, inplace=True)
    
    # 4) Ensure vote total columns are numeric, filling missing values with 0
    for office in offices:
        for col in [f"{dem_prefix}{office}", f"{rep_prefix}{office}"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    
    return df


def prepare_precinct_data_multirow(
    data_input,
    precinct_col="PrecinctName",
    office_col="OfficeCode",
    party_col="PartyCode",
    vote_col="VoteTotal",
    offices=["STH", "USP"],
    dem_party="DEM",
    rep_party="REP",
    keep_geometry=True,
    has_header=True,
    encoding="utf-8",
    col_names=None,
    dtypes=None,
    standardize_precinct=True,
    state=None,
    unique_when_combined=None
):
    """
    Loads data where each precinct can appear in multiple rows, one row per candidate/party performance.
    Filters for each office specified in the 'offices' list, then pivots to create separate columns
    for each party's vote totals in each office.

    For each office in the list, two new columns are created:
         DEM_<office> and REP_<office>

    Can handle CSV, Shapefile, GeoJSON, or in-memory DataFrame/GeoDataFrame.

    Returns a wide DataFrame (or GeoDataFrame) with columns:
         precinct_col, plus for each office in 'offices': DEM_<office> and REP_<office>
         + geometry if present and keep_geometry=True.
    """
    # 1) Load data via the generic loader
    df = load_generic_file(
        data_input=data_input,
        has_header=has_header,
        encoding=encoding,
        col_names=col_names,
        dtypes=dtypes,
        keep_geometry=keep_geometry
    )
    if unique_when_combined is not None:
        df = drop_duplicates_qualify(
            df, 
            unique_when_combined,
            verbose=True
        )

    # Rename the precinct column to a standard name if needed
    if standardize_precinct and precinct_col != "precinct":
        df.rename(columns={precinct_col: "precinct"}, inplace=True)
        precinct_col = "precinct"
    
    # Optional: if working with PA data, prepend CountyCode to the precinct name
    if state == 'PA' and 'CountyCode' in df.columns:
        df[precinct_col] = df['CountyCode'].astype(str) + '-' + df[precinct_col].astype(str)
    
    merged = None
    
    # 2) Loop over each office provided
    for office in offices:
        # Filter rows for the current office and for the parties of interest
        df_office = df[(df[office_col] == office) & (df[party_col].isin([dem_party, rep_party]))]
        
        # Aggregate vote totals by precinct and party
        df_office_agg = df_office.groupby([precinct_col, party_col], as_index=False)[vote_col].sum()
        
        # Pivot the aggregated data to wide format: one column per party vote total
        wide_office = df_office_agg.pivot(index=precinct_col, columns=party_col, values=vote_col).fillna(0)
        wide_office = wide_office.rename(columns={
            dem_party: f"{dem_party}_{office}",
            rep_party: f"{rep_party}_{office}"
        }).reset_index()
        
        # Merge results across offices using an outer join on the precinct column
        if merged is None:
            merged = wide_office
        else:
            merged = pd.merge(merged, wide_office, on=precinct_col, how="outer")

    # 3) Reattach geometry if it existed originally
    if keep_geometry and isinstance(df, gpd.GeoDataFrame) and "geometry" in df.columns:
        # Get a unique mapping of precinct to geometry
        geo_mapping = df.drop_duplicates(subset=[precinct_col])[[precinct_col, "geometry"]]
        merged = pd.merge(merged, geo_mapping, on=precinct_col, how="left")
        merged = gpd.GeoDataFrame(merged, geometry="geometry", crs=df.crs)
    
    # 4) Ensure missing vote columns for any office are filled with 0
    for office in offices:
        for col in [f"{dem_party}_{office}", f"{rep_party}_{office}"]:
            if col not in merged.columns:
                merged[col] = 0
    
    if merged[precinct_col].nunique() != len(merged):
        raise ValueError("Precinct column is not unique.")

    return merged

if __name__ == "__main__":
    
    county_aux_data_fp = r"/Users/aspencage/Documents/Data/input/post_g2024/comparative_presidential_performance/G2024/2024 Presidential by County with FIPS - 2024.csv"

    df = prepare_precinct_data_multirow(
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
        dtypes=None,
        standardize_precinct=False,
        state=None
    )
