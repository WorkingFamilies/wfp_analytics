import os
import re
import pandas as pd
import geopandas as gpd
import warnings




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
