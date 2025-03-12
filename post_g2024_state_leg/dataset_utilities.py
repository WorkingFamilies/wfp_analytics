import pandas as pd
import numpy as np
import geopandas as gpd


import pandas as pd
import pandas as pd
import numpy as np

import numpy as np
import pandas as pd

def prettify_dataset(
    df, 
    round_decimals=2, 
    int_columns=None, 
    percent_columns=None, 
    sort_by=None, 
    fillna_value=0,
    round_all_numeric=True,
    round_columns=None
):
    """
    Prettify a Pandas DataFrame by (optionally) rounding numeric columns, 
    converting specified columns to integers, formatting percentages, 
    and sorting by specified columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    round_decimals : int, default=2
        Number of decimal places to round numeric columns.
    int_columns : list of str, optional
        List of columns to convert to integers after rounding. 
        Columns that do not exist in df will be skipped.
    percent_columns : list of str, optional
        List of columns to convert to percentage format (multiplies by 100).
        Columns that do not exist or are non-numeric will be skipped.
    sort_by : str or list of str, optional
        Column(s) to sort the DataFrame by.
    fillna_value : int or float, default=0
        Value to replace NaNs (and infinities) before converting to integers.
    round_all_numeric : bool, default=True
        If True, round all numeric columns to 'round_decimals'.
        If False, only columns in 'round_columns' get rounded.
    round_columns : list of str, optional
        Explicit list of columns to round. 
        Only used if round_all_numeric=False. 
        Defaults to None.
    
    Returns
    -------
    pd.DataFrame
        The prettified DataFrame.
    """

    df = df.copy()  # Avoid modifying the original DataFrame in-place

    # ----------------------------------------------------------------
    # 1) Validate and normalize parameters
    # ----------------------------------------------------------------
    # Ensure int_columns and percent_columns are lists
    int_columns = list(int_columns or [])
    percent_columns = list(percent_columns or [])

    # Validate round_decimals
    if not isinstance(round_decimals, int) or round_decimals < 0:
        raise ValueError("round_decimals must be a non-negative integer.")

    # ----------------------------------------------------------------
    # 2) (Optional) Round columns
    # ----------------------------------------------------------------
    # Decide which columns we are actually rounding
    if round_all_numeric:
        # Gather all numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    else:
        # User-specified columns only
        round_columns = round_columns or []
        # Filter out columns that aren’t in df or aren’t numeric
        round_columns = [
            c for c in round_columns 
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
        ]
        numeric_cols = round_columns

    # Actually round them
    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].round(round_decimals)

    # ----------------------------------------------------------------
    # 3) Convert specified columns to int
    # ----------------------------------------------------------------
    for col in int_columns:
        if col not in df.columns:
            print(f"Warning: int_columns: '{col}' not in df; skipping.")
            continue

        # Skip non-numeric columns
        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"Warning: column '{col}' is not numeric; cannot convert to int.")
            continue

        # Replace NaN and +/- inf, then cast to int
        df[col] = df[col].fillna(fillna_value).replace([np.inf, -np.inf], fillna_value)
        try:
            df[col] = df[col].round(0).astype(int)
        except ValueError as e:
            print(f"Error converting column '{col}' to int: {e}")

    # ----------------------------------------------------------------
    # 4) Format specified columns as percentages
    # ----------------------------------------------------------------
    for col in percent_columns:
        if col not in df.columns:
            print(f"Warning: percent_columns: '{col}' not in df; skipping.")
            continue

        # Ensure it's numeric
        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"Warning: column '{col}' is not numeric; skipping percent format.")
            continue

        # Multiply by 100, round, then add '%' 
        # (change .round(0) to .round(N) if you want decimals)
        df[col] = (df[col] * 100).round(0).astype(int).astype(str) + "%"

    # ----------------------------------------------------------------
    # 5) Sort by specified column(s)
    # ----------------------------------------------------------------
    if sort_by:
        # If you want a hard error on missing columns, remove the try/except
        try:
            df = df.sort_values(by=sort_by)
        except KeyError as e:
            print(f"Warning: Sorting failed due to missing column(s): {e}")

    return df


def reorder_columns(df, cols_to_front):
    """
    Reorder the columns of a Pandas DataFrame or GeoPandas GeoDataFrame.
    
    Parameters:
    df (pd.DataFrame or gpd.GeoDataFrame): The input dataframe.
    cols_to_front (list): A list of column names to move to the front.
    
    Returns:
    pd.DataFrame or gpd.GeoDataFrame: A new dataframe with reordered columns.
    """
    # Ensure input is a DataFrame or GeoDataFrame
    if not isinstance(df, (pd.DataFrame, gpd.GeoDataFrame)):
        raise TypeError("Input must be a Pandas DataFrame or a GeoPandas GeoDataFrame")
    
    # Get the list of remaining columns in their original order
    remaining_cols = [col for col in df.columns if col not in cols_to_front]
    
    # Reorder the columns
    new_col_order = cols_to_front + remaining_cols
    
    return df[new_col_order]


def filter_by_quantile(s, lower_quantile=None, upper_quantile=None):
    """
    Filter rows in a Series to only those above a lower quantile
    and/or below an upper quantile (inclusive).

    Parameters
    ----------
    s : pd.Series
        The Series to filter.
    lower_quantile : float, optional
        The lower quantile (between 0 and 1). If provided, rows
        with values below this quantile are dropped.
    upper_quantile : float, optional
        The upper quantile (between 0 and 1). If provided, rows
        with values above this quantile are dropped.

    Returns
    -------
    pd.Series
        The filtered Series containing only rows within the specified
        quantile range.
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> s = pd.Series(np.random.randn(1000))
    >>> # Keep only rows >= 0.05 quantile and <= 0.95 quantile
    >>> filtered = filter_by_quantile(s, lower_quantile=0.05, upper_quantile=0.95)
    >>> len(filtered)
    900   # (approximately, depends on distribution)
    """
    # Copy so we don't alter the original
    s = s.copy()

    mask = pd.Series([True]*len(s), index=s.index)

    if lower_quantile is not None:
        # Compute the lower quantile value
        q_lower = s.quantile(lower_quantile)
        # Keep rows whose value is >= that threshold
        mask &= s >= q_lower

    if upper_quantile is not None:
        # Compute the upper quantile value
        q_upper = s.quantile(upper_quantile)
        # Keep rows whose value is <= that threshold
        mask &= s <= q_upper

    return s[mask]


def drop_duplicates_qualify(
    df,
    subset_cols,
    order_by=None,
    ascending=True,
    nulls_first=True,
    verbose=False
):
    """
    Drops duplicates so that each unique combination of `subset_cols` remains
    with exactly one row. If `order_by` is provided, the single row kept
    is whichever row ranks first (lowest or highest) by `order_by`, with
    optional control over whether NULLs come first or last.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.

    subset_cols : list of str
        The column(s) that define uniqueness (akin to SQL PARTITION BY).

    order_by : str, optional
        The column used to order rows within each group. If None, we simply
        use pandas' drop_duplicates(subset=subset_cols, keep='first').

    ascending : bool, default=True
        If True, keep the row with the smallest `order_by` value in each group.
        If False, keep the row with the largest `order_by` value.

    nulls_first : bool, default=True
        Determines how NULL (NaN) values in `order_by` are ranked.
        If True, NaNs sort before all non-null values (like "NULLS FIRST" in SQL).
        If False, NaNs sort after all non-null values (like "NULLS LAST").

    verbose : bool, default=False
        If True, prints the number of rows dropped.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with one row per group of `subset_cols`,
        chosen by the ordering rules if `order_by` is provided.
    """
    original_count = len(df)

    if order_by is None:
        # Simple drop_duplicates on subset_cols
        out_df = df.drop_duplicates(subset=subset_cols, keep='first')
    else:
        df = df.copy()

        # Flag for null ordering
        if nulls_first:
            df['_null_flag'] = df[order_by].notna().astype(int)
        else:
            df['_null_flag'] = df[order_by].isna().astype(int)

        # Sort by nulls first/last, then by order_by ascending/descending
        df = df.sort_values(
            by=['_null_flag', order_by],
            ascending=[True, ascending]
        )

        # Drop duplicates, keep first row in each group
        out_df = df.drop_duplicates(subset=subset_cols, keep='first')
        out_df.drop(columns=['_null_flag'], inplace=True)

    final_count = len(out_df)
    dropped_rows = original_count - final_count

    if verbose:
        print(f"Dropped {dropped_rows} rows from {original_count} rows (remaining: {final_count}).")

    return out_df.reset_index(drop=True)
