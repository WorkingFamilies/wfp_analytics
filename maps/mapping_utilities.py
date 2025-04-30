import os
import re
import pandas as pd
import geopandas as gpd
import warnings
from shapely import wkt, wkb
import geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry import GeometryCollection

import subprocess
from pathlib import Path
from typing import Union


def repair_shapefile_with_ogr(
    src: Union[str, Path],
    dst: Union[str, Path] | None = None,
    *,
    layer: str | None = None,
    promote_to_multi: bool = True,
    overwrite: bool = True,
    quiet: bool = True,
) -> Path:
    """
    Create a *repaired* copy of a Shapefile using ``ogr2ogr`` + ST_MakeValid.

    Parameters
    ----------
    src, dst : str or Path
        Path to the input Shapefile (.shp).  ``dst`` is the output file; if
        omitted, ``<src stem>__valid.shp`` is written alongside *src*.
    layer : str, optional
        Source layer name.  Usually the filename without extension; leave
        ``None`` to let ogr2ogr pick the first (and only) layer.
    promote_to_multi : bool, default True
        Add ``-nlt PROMOTE_TO_MULTI`` so singleparts become multiparts and stay
        valid after repair.
    overwrite : bool, default True
        Pass ``-overwrite`` so reruns replace the previous output.
    quiet : bool, default True
        Suppress ogr2ogr chatter (``-q``).

    Returns
    -------
    Path
        Path to the repaired Shapefile (.shp).

    Raises
    ------
    subprocess.CalledProcessError
        If ogr2ogr exits with a non-zero status (e.g. GDAL missing).
    """
    src = Path(src).with_suffix(".shp")
    if dst is None:
        dst = src.with_name(f"{src.stem}__valid.shp")
    else:
        dst = Path(dst).with_suffix(".shp")

    # Build the SQL that makes every feature valid
    lyr = layer or src.stem
    sql = f'SELECT ST_MakeValid(geometry) AS geometry, * FROM "{lyr}"'

    cmd = [
        "ogr2ogr",
        "-f",
        "ESRI Shapefile",
        str(dst),
        str(src),
        "-dialect",
        "SQLite",
        "-sql",
        sql,
    ]
    if promote_to_multi:
        cmd.extend(["-nlt", "PROMOTE_TO_MULTI"])
    if overwrite:
        cmd.append("-overwrite")
    if quiet:
        cmd.append("-q")

    subprocess.run(cmd, check=True)
    return dst


def dissolve_groups(
    gdf: gpd.GeoDataFrame,
    *,
    group_col: str,
    geometry_col: str = "geometry"
) -> gpd.GeoDataFrame:
    """
    Collapse a GeoDataFrame so that all rows sharing ``group_col`` are reduced
    to a single row whose geometry is the union of their geometries.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        The input GeoDataFrame.
    group_col : str
        Name of the column that defines which rows should be merged together.
    geometry_col : str, default "geometry"
        Name of the geometry column inside ``gdf`` (set this if your geometry
        lives in a non-default column).

    Returns
    -------
    geopandas.GeoDataFrame
        A new GeoDataFrame with one row per unique value in ``group_col`` and a
        dissolved (unioned) geometry for each group.  All non-geometry columns
        are aggregated with their *first* non-null value; edit the ``aggfunc``
        dict inside the function if you need different behaviour.

    Example
    -------
    >>> merged = dissolve_groups(county_precincts,
    ...                           group_col="CountyFIPS",
    ...                           geometry_col="geom")
    """
    if group_col not in gdf.columns:
        raise KeyError(f"'{group_col}' not found in GeoDataFrame columns.")

    if geometry_col not in gdf.columns:
        raise KeyError(f"'{geometry_col}' not found in GeoDataFrame columns.")

    # Work on a copy so we don't mutate the caller's GDF
    gdf = gdf.copy()

    original_count = len(gdf)
    unique_groups = gdf[group_col].nunique()

    # Make sure the desired geometry column is active
    gdf = gdf.set_geometry(geometry_col)

    # Dissolve: union geometry, keep the first non-null value for other cols
    dissolved = (
        gdf.dissolve(
            by=group_col,
            as_index=False,             # keep group column as a regular column
            aggfunc="first"             # change to dict if you need custom aggs
        )
        .set_geometry("geometry")       # ensures 'geometry' is recognised
    )

    # Rename geometry column back if the user passed a non-default name
    if geometry_col != "geometry":
        dissolved = dissolved.rename(columns={"geometry": geometry_col})
        dissolved = dissolved.set_geometry(geometry_col)

    # Print summary
    print(f"[dissolve_groups] Combined {original_count:,} rows into {unique_groups:,} grouped geometries.")

    return dissolved

def remove_geometry(gdf,geometry_col="geometry"):
    """
    Remove geometry from a GeoDataFrame and return a DataFrame.
    """
    if geometry_col not in gdf.columns:
        print(f"Warning: '{geometry_col}' not found in GeoDataFrame columns.")
        return gdf.copy()  # Return a copy of the original GeoDataFrame
    else:
        df = pd.DataFrame(gdf.drop(columns=geometry_col))
    return df

def fix_invalid_geometries(gdf):
    """
    Fix invalid geometries in a GeoDataFrame via buffer(0).
    Returns a GeoDataFrame with repaired geometries.
    """
    if gdf.empty:
        print("Warning: Received an empty GeoDataFrame. No geometries to fix.")
        return gdf

    invalid_count = (~gdf.is_valid).sum()
    if invalid_count > 0:
        print(f"Found {invalid_count} invalid geometries; attempting to fix...")
        gdf['geometry'] = gdf['geometry'].buffer(0)
    return gdf


def check_mutual_exclusivity(
    gdf: gpd.GeoDataFrame,
    *,
    area_tol: float = 1e-8,
    pct_tol: float = 1e-4,
    verbose: bool = True,
    compact: bool = False,
):
    """Assess whether geometries in *gdf* are mutually exclusive (non‑overlapping).

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input GeoDataFrame whose ``geometry`` column has *projected* coordinates so
        that ``area`` is meaningful.
    area_tol : float, default 1e-8
        Absolute area threshold – intersections smaller than this value are ignored.
    pct_tol : float, default 1e-4 (0.01 %)
        Relative threshold expressed as *fraction of the smaller* geometry’s area
        below which the overlap is ignored.  For example, ``pct_tol=0.05`` allows
        up to 5 % overlap of the smaller polygon.
    verbose : bool, default True
        If *True* print a human‑readable report.  Set to *False* for silent mode.
    compact : bool, default False
        If *True* the report omits the per‑pair details and shows only the summary.

    Returns
    -------
    is_exclusive : bool
        ``True`` if all geometries are exclusive within the given tolerances.
    overlaps : geopandas.GeoDataFrame
        One row per overlapping pair with columns
        ``["idx1", "idx2", "overlap_area", "pct_smaller", "pct_larger"]``.
    """

    if gdf.crs is None:
        raise ValueError(
            "GeoDataFrame must have a projected CRS (e.g. EPSG:3857, EPSG:326xx) "
            "so that area calculations make sense."
        )

    # Prepare a spatial index for fast bounding‑box queries
    sindex = gdf.sindex
    overlaps_records = []

    for i, geom_i in enumerate(gdf.geometry):
        if geom_i is None or geom_i.is_empty:
            continue

        # Candidate neighbours whose bounding boxes intersect
        candidate_ids = list(sindex.intersection(geom_i.bounds))
        for j in candidate_ids:
            if j <= i:  # avoid duplicate & self comparisons
                continue
            geom_j = gdf.geometry.iloc[j]
            if geom_j is None or geom_j.is_empty:
                continue
            # Quick rejection – if geometries don’t intersect there’s no overlap
            if not geom_i.intersects(geom_j):
                continue

            inter = geom_i.intersection(geom_j)
            if inter.is_empty:
                continue

            area = inter.area
            if area < area_tol:
                continue  # too small to care

            pct_smaller = area / min(geom_i.area, geom_j.area)
            pct_larger = area / max(geom_i.area, geom_j.area)
            if pct_smaller < pct_tol:
                continue  # within tolerance

            overlaps_records.append(
                {
                    "idx1": gdf.index[i],
                    "idx2": gdf.index[j],
                    "overlap_area": area,
                    "pct_smaller": pct_smaller,
                    "pct_larger": pct_larger,
                }
            )

    overlaps_gdf = gpd.GeoDataFrame(overlaps_records)
    is_exclusive = overlaps_gdf.empty

    if verbose:
        if is_exclusive:
            print(
                f"\N{WHITE HEAVY CHECK MARK}  Geometries are mutually exclusive "
                f"within tolerances (area_tol={area_tol}, pct_tol={pct_tol:.4%})."
            )
        else:
            total_pairs = len(overlaps_gdf)
            max_pct = overlaps_gdf["pct_smaller"].max()
            mean_pct = overlaps_gdf["pct_smaller"].mean()
            total_overlap_area = overlaps_gdf["overlap_area"].sum()
            print(
                f"\N{WARNING SIGN}  Found {total_pairs} overlapping pair(s).\n"
                f"   • Total overlap area: {total_overlap_area:,.2f} (units of CRS)\n"
                f"   • Largest % overlap of smaller geometry: {max_pct:.2%}\n"
                f"   • Mean % overlap of smaller geometry: {mean_pct:.2%}"
            )
            if not compact:
                pd_opt = {
                    "overlap_area": "{:,.2f}".format,
                    "pct_smaller": "{:.2%}".format,
                    "pct_larger": "{:.2%}".format,
                }
                try:
                    import pandas as pd  # local import to avoid mandatory dep
                    print(overlaps_gdf.to_string(index=False, formatters=pd_opt))
                except ImportError:
                    # fallback if pandas isn’t available (unlikely in geopandas env)
                    print(overlaps_gdf)

    return is_exclusive, overlaps_gdf


def load_generic_file(
    data_input,
    has_header=True,
    col_names=None,
    encoding="utf-8",
    dtypes=None,
    keep_geometry=True
):
    """
    Loads data from various file formats (CSV, Shapefile, GeoJSON, GPKG, etc.)
    or accepts an already-loaded (GeoDataFrame/DataFrame).

    Parameters
    ----------
    data_input : str or DataFrame or GeoDataFrame
        - If str, the path to a file to load.
          Automatically detects extension:
              .csv / .txt => read via pandas
              else        => read via geopandas
        - If DataFrame/GeoDataFrame, returns a copy of that in the same format.
    has_header : bool
        If True and using pandas, the file has a header row to be read. 
        If False and using pandas, the file does not have a header and 
        'header=None' is used.
    col_names : list of str or None
        If the file has no header, you can specify a list of column names 
        to be used for the DataFrame. This only applies to CSV or TXT 
        when has_header=False.
    encoding : str
        File encoding (pandas).
    dtypes : dict or None
        Optionally specify column data types for pandas (e.g., {'CountyCode': str}).
    keep_geometry : bool
        If False, drops geometry columns even if it's a GeoDataFrame. 
        If True, retains geometry (where available).

    Returns
    -------
    DataFrame or GeoDataFrame
        - For CSV/TXT, returns a DataFrame (unless user merges geometry later).
        - For other geospatial formats, returns a GeoDataFrame unless 
          keep_geometry=False, in which case it returns a DataFrame.
        - If data_input was already a DataFrame/GeoDataFrame, returns a copy.
    """
    # 1) If already a DataFrame or GeoDataFrame, just return a copy
    if isinstance(data_input, (pd.DataFrame, gpd.GeoDataFrame)):
        df = data_input.copy()
        if not keep_geometry and isinstance(df, gpd.GeoDataFrame):
            # Drop geometry if not needed
            df = pd.DataFrame(df.drop(columns="geometry"))
        return df

    # 2) Otherwise it's a path; detect the extension
    filepath = data_input
    _, ext = os.path.splitext(filepath.lower())

    # 3) CSV or text-based => load with pandas
    if ext in [".csv", ".txt"]:
        read_csv_kwargs = {
            "filepath_or_buffer": filepath,
            "encoding": encoding,
        }
        # Handle header / column names
        if has_header:
            read_csv_kwargs["header"] = 0
        else:
            read_csv_kwargs["header"] = None
            read_csv_kwargs["names"] = col_names

        # Attempt reading with the provided dtypes
        if dtypes:
            # First try with user-supplied dtypes
            try:
                df = pd.read_csv(**read_csv_kwargs, dtype=dtypes)
            except (ValueError, TypeError) as e:
                # Warn the user we had to fallback
                warnings.warn(
                    f"Could not load with specified dtypes. Will load without dtypes "
                    f"and then attempt to coerce columns as possible.\n  Original error: {e}",
                    UserWarning
                )
                # Read again without dtype
                df = pd.read_csv(**read_csv_kwargs, dtype=None)
                # Attempt to coerce columns to the desired dtypes
                for col, desired_type in dtypes.items():
                    if col not in df.columns:
                        continue
                    try:
                        if pd.api.types.is_numeric_dtype(desired_type):
                            # Use to_numeric for numeric types to coerce invalid entries to NaN
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                            # If user specifically asked for int, you can optionally cast again
                            # to integer type (e.g., pd.Int64Dtype()) if you want to maintain
                            # integer semantics with potential NaNs. Example:
                            # df[col] = df[col].astype("Int64")
                            if desired_type is int:
                                # One possible approach: cast to a pandas integer type:
                                df[col] = df[col].astype("Int64")
                        else:
                            # Try standard astype for other types (string, category, etc.)
                            df[col] = df[col].astype(desired_type)
                    except Exception as col_err:
                        warnings.warn(
                            f"Could not convert column '{col}' to {desired_type}. "
                            f"Coercion left invalid values as NaN or unchanged.\n"
                            f"  Original error: {col_err}",
                            UserWarning
                        )
        else:
            # No dtypes given: just read normally
            df = pd.read_csv(**read_csv_kwargs, dtype=None)

        return df

    # 4) Otherwise, treat as a geospatial format => geopandas
    else:
        gdf = gpd.read_file(filepath)
        if not keep_geometry:
            # Convert to plain DataFrame by dropping geometry
            gdf = pd.DataFrame(gdf.drop(columns="geometry"))
        return gdf


def process_district_data(gdf, state, crs="EPSG:2163"):
    
    gdf = gdf.to_crs(crs)

    gdf["State"] = state

    # Add the "Chamber" column based on the presence of specific columns
    if "SLDUST" in gdf.columns:
        gdf["Chamber"] = "Upper"
        gdf["District"] = gdf["SLDUST"]
        district_col = "SLDUST"
    elif "SLDLST" in gdf.columns:
        gdf["Chamber"] = "Lower"
        gdf["District"] = gdf["SLDLST"]
        district_col = "SLDLST"
    else:
        gdf["Chamber"] = "Unknown"
        gdf["District"] = None  # Assign None if neither column exists
        district_col = None
    
    # Drop the original SLDUST and SLDLST columns
    gdf = gdf.drop(columns=district_col, errors="ignore")

    return gdf, district_col


def get_all_mapfiles(directory_paths, extension=".geojson", filename_regex=None, print_=False):
    """
    Recursively find all map files with the specified extension in one or more directories
    and their subdirectories, optionally filtering by a filename pattern.

    Parameters:
        directory_paths (list or str): A list of directory paths or a single directory path to search.
        extension (str): The file extension to search for (e.g., ".shp", ".geojson", ".gpkg"). Defaults to ".geojson".
        filename_regex (str, optional): A regular expression to filter filenames. 
            Only files with names matching the regex will be included. Defaults to None (no filtering).
        print_ (bool): Whether to print debugging information. Defaults to False.

    Returns:
        list: A list of full paths to the files with the specified extension and matching the regex.
    """
    if isinstance(directory_paths, str):  # Convert single path to list
        directory_paths = [directory_paths]

    mapfiles = []
    for directory_path in directory_paths:
        if print_:
            print(f"Searching in: {directory_path}")

        if not os.path.exists(directory_path):
            if print_:
                print(f"ERROR: Directory {directory_path} does not exist!")
            continue

        for root, _, files in os.walk(directory_path):
            if print_:
                print(f"Checking folder: {root}, found {len(files)} files")

            for file in files:
                if print_:
                    print(f"Examining file: {file}")

                if file.endswith(extension):
                    if filename_regex is None or re.search(filename_regex, file):
                        full_path = os.path.join(root, file)
                        if print_:
                            print(f"Found matching file: {full_path}")
                        mapfiles.append(full_path)

    return mapfiles


def concatenate_geodata(data_paths, crs="EPSG:2163", print_=False):
    """
    Reads geospatial data from a single path or a list of paths and concatenates them into a single GeoDataFrame.
    
    Parameters:
        data_path (str or list): A single file path or a list of file paths to geospatial data.
        
    Returns:
        GeoDataFrame: A single concatenated GeoDataFrame.
    """

    def load_gdf(path,crs):
        try:
            gdf = gpd.read_file(path)
            gdf = gdf.to_crs(crs)
            return gdf
        except Exception as e:
            print(f"Error reading {path}: {e}")
            return None

    if isinstance(data_paths, str):  # Single path, read directly
        gdf = load_gdf(data_paths,crs)
    elif isinstance(data_paths, list):  # List of paths, read and concatenate
        gdf_list = []
        for path in data_paths:
            gdf_list.append(load_gdf(path,crs))
            if print_ >= 1:
                print(f"Read: {path}")
            if print_ >= 2:
                print(f"Columns: {gdf_list[-1].columns}")
                print()
              
        if gdf_list:
            gdf = pd.concat(gdf_list, ignore_index=True)
        else:
            gdf = gpd.GeoDataFrame()  # Return empty GeoDataFrame if no files could be read
    else:
        raise ValueError("data_path must be a string or a list of strings.")
    
    return gdf


# Function to get state abbreviation or FIPS code
def map_fips_and_state(value):
    
    # FIPS to state abbreviation mapping
    fips_to_state = {
        '01': 'AL', '02': 'AK', '04': 'AZ', '05': 'AR', '06': 'CA',
        '08': 'CO', '09': 'CT', '10': 'DE', '11': 'DC', '12': 'FL',
        '13': 'GA', '15': 'HI', '16': 'ID', '17': 'IL', '18': 'IN',
        '19': 'IA', '20': 'KS', '21': 'KY', '22': 'LA', '23': 'ME',
        '24': 'MD', '25': 'MA', '26': 'MI', '27': 'MN', '28': 'MS',
        '29': 'MO', '30': 'MT', '31': 'NE', '32': 'NV', '33': 'NH',
        '34': 'NJ', '35': 'NM', '36': 'NY', '37': 'NC', '38': 'ND',
        '39': 'OH', '40': 'OK', '41': 'OR', '42': 'PA', '44': 'RI',
        '45': 'SC', '46': 'SD', '47': 'TN', '48': 'TX', '49': 'UT',
        '50': 'VT', '51': 'VA', '53': 'WA', '54': 'WV', '55': 'WI',
        '56': 'WY', '72': 'PR'  # Puerto Rico
    }

    # Create reverse mapping
    state_to_fips = {state: fips for fips, state in fips_to_state.items()}

    if value in fips_to_state:
        return fips_to_state[value]  # FIPS to state
    elif value in state_to_fips:
        return state_to_fips[value]  # State to FIPS
    else:
        return "Invalid input"
    

def map_code_to_value(input,map_dict):
    map_reversed = {v: k for k, v in map_dict.items()}

    if input in map_dict:
        return map_dict[input]
    elif input in map_reversed:
        return map_reversed[input]
    else:
        return "Invalid input"
    

def extract_fips_from_path(path):
    # Extract the filename
  filename = os.path.basename(path)

  # Regex to extract the FIPS code from the filename
  match = re.search(r'(\d{2})_(sldl|sldu)', filename)

  if match:
      fips_code = match.group(1)
      return fips_code
  else:
      print(f"No FIPS sfound in: {filename}.")
    


def generate_state_file_table(directory_path, filter_incomplete=False):
    """
    Generates a table of states with separate columns for SLDU and SLDL files found in a directory.
    
    Parameters:
        directory_path (str): Path to the directory to search for files.
        filter_incomplete (bool): If True, only include rows where SLDU or SLDL is False.
        
    Returns:
        DataFrame: A Pandas DataFrame with columns 'State', 'SLDU', and 'SLDL'.
    """
    # List of all state abbreviations
    all_states = [
        'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 
        'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 
        'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 
        'VA', 'WA', 'WV', 'WI', 'WY'
    ]

    # Initialize dictionary with all states, setting SLDU and SLDL to False
    state_files = {state: {'State': state, 'SLDL': False, 'SLDU': False} for state in all_states}

    # Regex patterns for SLDU and SLDL
    sldu_regex = r"([A-Z]{2})_by_SLDU"
    sldl_regex = r"([A-Z]{2})_by_SLDL"

    # Walk through the directory and get file names
    for root, _, files in os.walk(directory_path):
        for file in files:
            sldu_match = re.search(sldu_regex, file)
            sldl_match = re.search(sldl_regex, file)

            if sldu_match or sldl_match:
                state_abbr = sldu_match.group(1) if sldu_match else sldl_match.group(1)

                # Update state entry to mark presence of SLDU or SLDL file
                if state_abbr in state_files:
                    if sldu_match:
                        state_files[state_abbr]['SLDU'] = True
                    if sldl_match:
                        state_files[state_abbr]['SLDL'] = True

    # Convert dictionary to DataFrame
    state_table = pd.DataFrame.from_dict(state_files, orient='index')
    state_table.reset_index(drop=True, inplace=True)
    state_table.sort_values("State", inplace=True)

    # Filter rows where one of SLDU or SLDL is False, if specified
    if filter_incomplete:
        state_table = state_table[(state_table['SLDU'] == False) | (state_table['SLDL'] == False)]

    return state_table



def filter_shapefile_paths(shapefile_paths, state_table, print_=False):
    """
    Filters out shapefile paths for states and chambers represented as `True` in the state table.
    
    Parameters:
        shapefile_paths (list): List of shapefile paths to be filtered.
        state_table (DataFrame): A DataFrame with columns 'State', 'SLDU', and 'SLDL' indicating the presence of files.
        
    Returns:
        list: A list of filtered shapefile paths.
    """
    filtered_paths = []

    # Iterate over all shapefile paths
    for path in shapefile_paths:
        # Extract the FIPS code from the path
        fips_code = extract_fips_from_path(path)
        if not fips_code:
            continue

        # Map the FIPS code to a state abbreviation
        state_abbr = map_fips_and_state(fips_code)

        # Check if the state and both chambers (SLDU and SLDL) are marked as `True` in the state table
        state_row = state_table[state_table["State"] == state_abbr]
        if not state_row.empty and state_row.iloc[0]["SLDU"] and state_row.iloc[0]["SLDL"]:
            if print_:
                print(f"Removing: {path} (State: {state_abbr})")
        else:
            filtered_paths.append(path)

    return filtered_paths


def load_as_gdf(
    csv_path,
    geometry_col='geometry',
    crs="EPSG:2163",
    guess_format='auto'
):
    """
    Load a CSV file and convert a geometry column into a GeoDataFrame.
    Automatically attempts to parse geometry in WKT or WKB format 
    unless 'guess_format' is specifically set to 'wkt' or 'wkb'.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.
    geometry_col : str, optional
        Name of the column that contains geometry data (default: 'geometry').
    crs : str or dict, optional
        The coordinate reference system to set on the resulting GeoDataFrame.
        (e.g. 'EPSG:4326'). If None, no CRS is set.
    guess_format : {'auto', 'wkt', 'wkb'}, optional
        Whether to auto-detect geometry format, or force parse as WKT or WKB.
        (Default: 'auto').

    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame with parsed geometry.
    """

    # 1. Read the CSV with pandas
    df = pd.read_csv(csv_path)

    # 2. Check that the geometry column exists
    if geometry_col not in df.columns:
        raise ValueError(f"Specified geometry_col '{geometry_col}' not found in CSV columns.")

    # 3. If guess_format == 'auto', attempt to detect WKT vs. WKB
    if guess_format == 'auto':
        # We'll sample the first non-null row of the geometry column
        sample_val = df[geometry_col].dropna().iloc[0] if not df[geometry_col].dropna().empty else None
        
        if sample_val is None:
            # No geometry data at all
            # Return an empty GDF or raise an error, whichever suits your workflow
            return gpd.GeoDataFrame(df, geometry=None, crs=crs)
        
        # If the sample is a string that starts with POINT/POLYGON/etc., assume WKT.
        # Otherwise, assume WKB (often stored as hex strings).
        # This is very simplistic logic; adjust as needed for your data.
        upper_val = str(sample_val).strip().upper()
        if any(upper_val.startswith(tok) for tok in ("POINT", "LINESTRING", "POLYGON", "MULTIPOLYGON", "MULTILINESTRING")):
            guess_format = 'wkt'
        else:
            guess_format = 'wkb'

    # 4. Parse geometry
    if guess_format == 'wkt':
        # WKT parsing
        df[geometry_col] = df[geometry_col].apply(lambda x: wkt.loads(x) if pd.notnull(x) else None)
    elif guess_format == 'wkb':
        # WKB parsing (assuming geometry data is a hex string). 
        # If your data is base64-encoded or actual binary, adjust accordingly.
        df[geometry_col] = df[geometry_col].apply(lambda x: wkb.loads(bytes.fromhex(x)) if pd.notnull(x) else None)
    else:
        raise ValueError("guess_format must be 'auto', 'wkt', or 'wkb'.")

    # 5. Convert the DataFrame to a GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=geometry_col, crs=crs)

    return gdf
