import pandas as pd
import numpy as np
import geopandas as gpd  # If you eventually want geometry
import matplotlib.pyplot as plt
import os 
from scipy.optimize import curve_fit
from scipy.stats import norm
from mapping_utilities import load_generic_file

def prepare_precinct_data_single_row(
    data_input,
    precinct_col="PrecinctName",
    dem_leg_col="DEM_LEG",
    rep_leg_col="REP_LEG",
    dem_pres_col="DEM_PRES",
    rep_pres_col="REP_PRES",
    keep_geometry=True,
    has_header=True,
    encoding="utf-8",
    col_names=None,
    dtypes=None
):
    """
    Loads data where each row is already a single precinct with direct columns 
    for legislative DEM/REP and presidential DEM/REP. Can handle CSV/GeoJSON/Shapefile 
    or an in-memory DataFrame/GeoDataFrame.

    The function returns a DataFrame (or GeoDataFrame) that always has columns:
      'PrecinctName', 'DEM_LEG', 'REP_LEG', 'DEM_PRES', 'REP_PRES'
    (plus geometry if keep_geometry=True and it's available in the source).

    Parameters
    ----------
    data_input : str or DataFrame or GeoDataFrame
        The source data (file path or in-memory).
    precinct_col : str
        The column containing the precinct identifier in the input data.
    dem_leg_col, rep_leg_col, dem_pres_col, rep_pres_col : str
        The column names for legislative DEM, legislative REP, 
        presidential DEM, presidential REP in the input data.
    keep_geometry : bool
        Whether to keep geometry if it's a geospatial file or GeoDataFrame.
    has_header : bool
        If True, the CSV has a header row (ignored for other formats).
    encoding : str
        File encoding (for CSV).
    col_names : list of str or None
        If has_header=False, you can specify column names for the CSV.
    dtypes : dict or None
        Optionally specify column data types (pandas) for CSV.

    Returns
    -------
    pd.DataFrame or gpd.GeoDataFrame
        A DataFrame/GeoDataFrame containing:
          'PrecinctName', 'DEM_LEG', 'REP_LEG', 'DEM_PRES', 'REP_PRES'
        plus geometry if keep_geometry=True and available.
    """
    # -------------------------------------------------------------------
    # 1) Load via the generic function
    # -------------------------------------------------------------------
    df = load_generic_file(
        data_input=data_input,
        has_header=has_header,
        encoding=encoding,
        col_names=col_names,
        dtypes=dtypes,
        keep_geometry=keep_geometry
    )

    # -------------------------------------------------------------------
    # 2) Check for required columns in the user-provided names
    # -------------------------------------------------------------------
    required_cols = [precinct_col, dem_leg_col, rep_leg_col, dem_pres_col, rep_pres_col]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in single-row dataset.")

    # -------------------------------------------------------------------
    # 3) Rename user-provided vote columns to the final standard column names
    # -------------------------------------------------------------------
    rename_map = {}
    
    # rename legislative columns
    if dem_leg_col != "DEM_LEG":
        rename_map[dem_leg_col] = "DEM_LEG"
    if rep_leg_col != "REP_LEG":
        rename_map[rep_leg_col] = "REP_LEG"
    
    # rename presidential columns
    if dem_pres_col != "DEM_PRES":
        rename_map[dem_pres_col] = "DEM_PRES"
    if rep_pres_col != "REP_PRES":
        rename_map[rep_pres_col] = "REP_PRES"
    
    # rename precinct col (if not already called 'PrecinctName')
    if precinct_col != "PrecinctName":
        rename_map[precinct_col] = "precinct"

    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    # -------------------------------------------------------------------
    # 4) Ensure numeric for the final columns
    # -------------------------------------------------------------------
    numeric_cols = ["DEM_LEG", "REP_LEG", "DEM_PRES", "REP_PRES"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # -------------------------------------------------------------------
    # 5) Return final DataFrame/GeoDataFrame
    # -------------------------------------------------------------------
    return df

def prepare_precinct_data_multirow(
    data_input,
    precinct_col="PrecinctName",
    office_col="OfficeCode",
    party_col="PartyCode",
    vote_col="VoteTotal",
    leg_office="STH",
    pres_office="USP",
    dem_party="DEM",
    rep_party="REP",
    keep_geometry=True,
    has_header=True,
    encoding="utf-8",
    col_names=None,
    dtypes=None,
    state='PA'
):
        """
        Loads data where each precinct can appear in multiple rows, 
        one row per candidate or party performance. Filters for legislative 
        and presidential offices, then pivots to get columns for DEM/REP 
        in each race.
    
        Can handle CSV, Shapefile, GeoJSON, or in-memory DataFrame/GeoDataFrame.
    
        Returns a wide DataFrame (or GeoDataFrame) with columns:
            precinct_col, DEM_LEG, REP_LEG, DEM_PRES, REP_PRES
            + geometry if present and keep_geometry=True
        """
        # 1) Load via the generic function
        df = load_generic_file(
            data_input=data_input,
            has_header=has_header,
            encoding=encoding,
            col_names=col_names,
            dtypes=dtypes,
            keep_geometry=keep_geometry
        )

        if precinct_col != "precinct":
            df.rename(columns={precinct_col:"precinct"}, inplace=True)
            precinct_col = "precinct"
        
        # NOTE hacky 
        if state == 'PA':
            df[precinct_col] = df['CountyCode'].astype(str)+ '-' + df['precinct'].astype(str)
    
        # 2) Filter for legislative + presidential offices, DEM/REP
        df_leg = df[(df[office_col] == leg_office) & (df[party_col].isin([dem_party, rep_party]))]
        df_pres = df[(df[office_col] == pres_office) & (df[party_col].isin([dem_party, rep_party]))]
    
        # 3) Aggregate & pivot legislative
        df_leg_agg = df_leg.groupby([precinct_col, party_col], as_index=False)[vote_col].sum()
        wide_leg = df_leg_agg.pivot(index=precinct_col, columns=party_col, values=vote_col).fillna(0)
        wide_leg = wide_leg.rename(columns={
            dem_party: "DEM_LEG",
            rep_party: "REP_LEG"
        }).reset_index()
    
        # 4) Aggregate & pivot presidential
        df_pres_agg = df_pres.groupby([precinct_col, party_col], as_index=False)[vote_col].sum()
        wide_pres = df_pres_agg.pivot(index=precinct_col, columns=party_col, values=vote_col).fillna(0)
        wide_pres = wide_pres.rename(columns={
            dem_party: "DEM_PRES",
            rep_party: "REP_PRES"
        }).reset_index()
    
        # 5) Merge two pivoted frames
        merged = pd.merge(wide_leg, wide_pres, on=precinct_col, how="outer")
    
        # 6) If geometry existed, it would be lost due to pivoting (since pivot 
        #    reindexes by precinct_col). If you want to re-merge geometry, you'd 
        #    do so from 'df'. For example:
        if keep_geometry and isinstance(df, gpd.GeoDataFrame) and "geometry" in df.columns:
            # Original geometry was repeated across many rows. We need a unique mapping 
            # from precinct_col -> geometry. Let's build that from the original.
            geo_mapping = df.drop_duplicates(subset=[precinct_col])[[precinct_col, "geometry"]]
            # Merge that in
            merged = pd.merge(merged, geo_mapping, on=precinct_col, how="left")
            # Convert back to GeoDataFrame
            merged = gpd.GeoDataFrame(merged, geometry="geometry", crs=df.crs)
    
        # 7) Fill any missing vote columns with 0
        for c in ["DEM_LEG","REP_LEG","DEM_PRES","REP_PRES"]:
            if c not in merged.columns:
                merged[c] = 0
    
        return merged


def compute_leg_vs_pres_precinct(
    df,
    dem_leg_col="DEM_LEG",
    rep_leg_col="REP_LEG",
    dem_pres_col="DEM_PRES",
    rep_pres_col="REP_PRES"
):
    """
    Given a DataFrame with columns for precinct, DEM/REP legislative votes,
    and DEM/REP presidential votes, calculates:
      - two-way Dem share in legislative race,
      - two-way Dem share in presidential race,
      - share difference = (pres - leg).
    
    Returns a new DataFrame with these columns added:
      'DemShare_LEG', 'DemShare_PRES', 'ShareDifference'
    """
    df = df.copy()  # avoid in-place modifications

    # Convert vote columns to numeric if not already
    for col in [dem_leg_col, rep_leg_col, dem_pres_col, rep_pres_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Compute two-way shares
    df["DemShare_LEG"] = df[dem_leg_col] / (df[dem_leg_col] + df[rep_leg_col])
    df["DemShare_PRES"] = df[dem_pres_col] / (df[dem_pres_col] + df[rep_pres_col])

    df[["DemShare_LEG","DemShare_PRES"]] = df[["DemShare_LEG","DemShare_PRES"]].fillna(0)

    # Difference: how much better Dem did in the Presidential vs. Legislative
    df["ShareDifference"] = df["DemShare_PRES"] - df["DemShare_LEG"]

    return df

def logistic_function(x, a, b):
    """
    LOGISTIC:
      y = 1 / (1 + exp(-(a + b*x)))
    """
    return 1.0 / (1.0 + np.exp(-(a + b * x)))


def probit_function(x, a, b):
    """
    PROBIT:
      y = Phi(a + b*x), where Phi is the standard normal CDF
    """
    return norm.cdf(a + b * x)


def gompertz_function(x, alpha, beta):
    """
    GOMPERTZ:
      y = exp(-alpha * exp(beta * x))
    """
    return np.exp(-alpha * np.exp(beta * x))


def richards_function(x, a, b, c, d):
    """
    RICHARDS (4-parameter):
      y = d / [1 + c * exp(-b * (x - a))]^(1/c)
    A typical usage is that 0 < y < d, with d often ~1. 
    We'll keep it general.
    """
    return d / np.power(1.0 + c * np.exp(-b * (x - a)), 1.0/c)


def fit_curve(curve_func, xdata, ydata, p0):
    """
    Attempts to fit 'curve_func' to xdata, ydata with initial guess p0.
    Returns (popt, pcov, success), where:
      popt: best-fit parameters
      pcov: covariance matrix (from curve_fit)
      success: bool, whether it converged
    """
    try:
        popt, pcov = curve_fit(curve_func, xdata, ydata, p0=p0, maxfev=10000)
        return popt, pcov, True
    except RuntimeError:
        # curve_fit failed to converge
        return None, None, False


def calculate_sse(y_true, y_pred):
    """
    Sum of Squared Errors
    """
    residuals = y_true - y_pred
    return np.sum(residuals**2)

def calculate_r2(y_true, y_pred):
    """
    Simple R^2 measure
    """
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - (ss_res / ss_tot)


def try_multiple_s_curves(x, y, verbose=True):
    """
    Given arrays x and y (both 0<...<1) tries several S-shaped functions:
      - logistic
      - probit
      - gompertz
      - richards
    and returns a DataFrame summarizing the results:
      columns: ['model_name', 'popt', 'SSE', 'R2', 'converged']
    Also returns a dictionary of the curve functions for reference.
    
    The user can then pick the best or plot them.
    """
    # 1) We'll store each model in a list, with: 
    #    (model_name, function, initial_guess)
    #    You might want to tailor these guesses to your typical data range.
    models = [
        ("logistic", logistic_function, [0.0, 1.0]),
        ("probit",   probit_function,   [0.0, 1.0]),
        ("gompertz", gompertz_function, [1.0, -1.0]),
        # For Richards, let's guess a=0.5, b=5.0, c=1.0, d=1.0
        # that should produce a typical shape between 0..1
        ("richards", richards_function, [0.5, 5.0, 1.0, 1.0]),
    ]
    
    results = []
    # 2) Fit each model
    for (name, func, p0) in models:
        popt, pcov, success = fit_curve(func, x, y, p0)
        if success:
            # compute predictions
            y_pred = func(x, *popt)
            sse = calculate_sse(y, y_pred)
            r2  = calculate_r2(y, y_pred)
        else:
            sse = np.nan
            r2  = np.nan
        
        results.append({
            "model_name": name,
            "popt": popt if success else None,
            "SSE": sse,
            "R2": r2,
            "converged": success
        })
    
    df_results = pd.DataFrame(results)
    if verbose:
        print("=== Fitting Results ===")
        print(df_results)
    return df_results


def plot_s_curve_fit(x, y, model_name, popt, curve_func, 
                     title=None, x_label="X", y_label="Y"):
    """
    Plots the data points along with the specified S-curve.
    'popt' are the fitted parameters for 'curve_func'.
    """
    plt.figure(figsize=(8,6))
    plt.scatter(x, y, alpha=0.6, edgecolor="k", label="Data")
    
    # define a smooth x range
    x_plot = np.linspace(x.min(), x.max(), 200)
    y_plot = curve_func(x_plot, *popt)
    
    plt.plot(x_plot, y_plot, color="red", linewidth=2,
             label=f"{model_name} fit")
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if title is None:
        title = f"S-curve fit: {model_name}"
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def select_best_s_curve(gdf,col_x='DemShare_PRES',col_y='DemShare_LEG'):
    fits_df = try_multiple_s_curves(gdf[col_x], gdf[col_y], verbose=True)

    # e.g., best by lowest SSE:
    best_index_sse = fits_df["SSE"].idxmin()
    best_model_name_sse = fits_df.loc[best_index_sse, "model_name"]
    print(f"Best model by SSE: {best_model_name_sse}")
      
    best_index_r2 = fits_df["R2"].idxmax()
    best_model_name_r2 = fits_df.loc[best_index_r2, "model_name"]
    print(f"Best model by R^2: {best_model_name_r2}")

    if best_model_name_sse != best_model_name_r2:
        print("Warning: best models differ by SSE vs. R^2.")
    else: 
        best_model_name = best_model_name_sse

    # 3) Retrieve function
    model_dict = {
        "logistic": {
            "func":logistic_function,
            "p0":[0.0, 1.0]
            },
        "probit": {
            "func":probit_function,
            "p0":[0.0, 1.0]
        },
        "gompertz": {
            "func":gompertz_function,
            "p0":[1.0, -1.0]
        },
        "richards": {
            "func":richards_function,
            "p0":[0.5, 5.0, 1.0, 1.0]
        }
    }
    best_model_dict = model_dict[best_model_name]

    return best_model_dict

def plot_two_way_voteshare_comparison(
    df,
    xcol='DemShare_Y',
    ycol='DemShare_X',
    label_col='DistrictName',
    title="Comparison of Two-Way Dem Vote Share",
    xlabel='Right Office Dem Share',
    ylabel='Left Office Dem Share',
    vert_line_label='50% (Y)',
    horiz_line_label='50% (X)',
    label_points=True,
    color_col=None,
    plot_s_curve=False,
    plot_model_dict=None
):
    """
    Plots a scatter comparing two two-way Democratic vote share columns, 
    adding lines for 50%, means, penalty/bonus difference. Optionally fits 
    an S-curve of your choosing (logistic, probit, Gompertz, Richards, etc.) 
    to the data using curve_fit, ignoring points where x or y is 0 or 1.

    Parameters
    ----------
    df : DataFrame
        The input DataFrame containing xcol and ycol columns.
    xcol : str
        Column for the X-axis (e.g., 'DemShare_PRES').
    ycol : str
        Column for the Y-axis (e.g., 'DemShare_LEG').
    label_col : str
        Column whose values label each point (if label_points=True).
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    vert_line_label : str
        Label for the horizontal 50% line.
    horiz_line_label : str
        Label for the vertical 50% line.
    label_points : bool
        If True, label each scatter point with the value in `label_col`.
    color_col : str or None
        If None, use a single color for all dots.
        If not None, color points by unique values in this column.
        - If numeric, use a continuous colormap.
        - If categorical, assign discrete colors.
    plot_s_curve : bool
        If True, attempt to fit the function from `plot_fn` (or default logistic) 
        ignoring rows where x=0/1 or y=0/1, and plot that curve.
    plot_fn : None, dict, or callable
        - If None, defaults to a 2-parameter logistic function.
        - If dict, it should have:
              {"func": some_callable, "p0": [initial_guesses]}
          e.g. Richards or Gompertz with >2 params.
        - If just a function, we assume 2 params with initial guesses [0,1].
        You can modify this logic to handle more parameters.

    Returns
    -------
    None (displays the plot)
    """

    # 1) Checks
    if xcol not in df.columns or ycol not in df.columns:
        raise ValueError("DataFrame must contain columns for xcol and ycol.")

    # 2) Determine color mapping for the scatter
    scatter_kwargs = dict(alpha=0.7, edgecolor='k', linewidth=0.5)
    if color_col is None:
        scatter_kwargs['c'] = 'steelblue'
    else:
        if color_col not in df.columns:
            raise ValueError(f"Column '{color_col}' not found in DataFrame for coloring.")
        if pd.api.types.is_numeric_dtype(df[color_col]):
            scatter_kwargs['c'] = df[color_col]
            scatter_kwargs['cmap'] = 'viridis'
        else:
            # Categorical => discrete colors
            categories = df[color_col].astype('category')
            cat_codes = categories.cat.codes
            scatter_kwargs['c'] = cat_codes
            scatter_kwargs['cmap'] = 'tab10'

    # 3) Compute means, differences
    # Filter out x or y == 0 or 1
    filtered = df[
          (df[xcol] > 0) & (df[xcol] < 1) &
          (df[ycol] > 0) & (df[ycol] < 1)
      ].copy()
    mean_x = filtered[xcol].mean()
    mean_y = filtered[ycol].mean()
    avg_x_v_y_diff = mean_x - mean_y
    x_v_y_type = "Penalty" if avg_x_v_y_diff > 0 else "Bonus"

    # 4) Create figure & scatter
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(df[xcol], df[ycol], **scatter_kwargs)
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)

    # 5) If color_col is categorical, build a manual legend
    if color_col and not pd.api.types.is_numeric_dtype(df[color_col]):
        categories = df[color_col].astype('category')
        handles = []
        for cat_val in categories.cat.categories:
            cat_idx = categories.cat.categories.get_loc(cat_val)
            # if only 1 category, cat_idx=0 => fraction=0
            if len(categories.cat.categories) > 1:
                fraction = cat_idx / (len(categories.cat.categories) - 1)
            else:
                fraction = 0
            handles.append(
                plt.Line2D(
                    [], [], marker='o',
                    color=scatter.cmap(fraction),
                    label=str(cat_val),
                    linestyle=''
                )
            )
        plt.legend(handles=handles, title=color_col, loc='best')

    # 6) Reference lines at 50%, means, penalty/bonus offset
    plt.axhline(y=0.5, color='red', linestyle='-', alpha=0.5, label=vert_line_label)
    plt.axvline(x=0.5, color='blue', linestyle='-', alpha=0.5, label=horiz_line_label)
    plt.axvline(
        x=0.5 + avg_x_v_y_diff,
        color='purple', linestyle='--', alpha=0.9,
        label=f'{x_v_y_type} to Y Var ({abs(avg_x_v_y_diff):.2f})'
    )
    plt.axvline(x=mean_x, color='blue', linestyle='--', alpha=0.4, linewidth=1, label=f'Avg {xcol} ({mean_x:.2f})')
    plt.axhline(y=mean_y, color='red', linestyle='--', alpha=0.4, linewidth=1, label=f'Avg {ycol} ({mean_y:.2f})')

    # 7) Diagonal lines (y=x and offset)
    min_val = min(df[xcol].min(), df[ycol].min())
    max_val = max(df[xcol].max(), df[ycol].max())
    x_vals = np.linspace(min_val, max_val, 200)
    line_exact = x_vals  # y = x
    line_offset = x_vals - avg_x_v_y_diff  # y = x - diff
    plt.plot(x_vals, line_exact, color='orange', linestyle='--', alpha=0.6,
             label='Identical voteshare in both')
    plt.plot(x_vals, line_offset, color='gold', linestyle='--', alpha=0.8,
             label=f'{x_v_y_type} Line (y = x + {avg_x_v_y_diff:.2f})')

    # 8) Optionally label each point
    if label_points:
        if label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found in df.")
        for _, row in df.iterrows():
            plt.text(row[xcol], row[ycol], str(row[label_col]), fontsize=8, alpha=0.7)

    # 9) Optionally fit & plot an S-curve from plot_fn
    if plot_s_curve:
        if filtered.shape[0] < 5:
            print("Not enough data to fit S-curve (need >= 5 points with 0<x<1, 0<y<1).")
        else:
            X = filtered[xcol].values
            Y = filtered[ycol].values

            # 9a) Decide which function & initial guess to use
            if plot_model_dict is None:
                # Default to logistic with 2 params
                func = logistic_function
                p0 = [0.0, 1.0]
            elif isinstance(plot_model_dict, dict):
                # Expect {"func": some_callable, "p0": [initial_params]}
                func = plot_model_dict.get("func", logistic_function)
                p0   = plot_model_dict.get("p0", [0.0, 1.0])
            else:
                # Assume it's just a function needing 2 params
                func = plot_model_dict
                p0   = [0.0, 1.0]

            try:
                popt, pcov = curve_fit(func, X, Y, p0=p0, maxfev=10000)
                # Create smooth x for plotting curve
                x_smooth = np.linspace(0, 1, 200)  
                y_smooth = func(x_smooth, *popt)
                plt.plot(x_smooth, y_smooth, color='pink', linewidth=2,
                         label="Fitted S-curve")
                param_str = ", ".join(f"{v:.3f}" for v in popt)
                print(f"S-curve parameters: {param_str}")
            except RuntimeError:
                print("Could not fit curve (curve_fit failed).")

    # 10) Final legend & show
    plt.legend()
    plt.show()


# One attempt to investigate 
def plot_dem_turnout_difference(
    df,
    dem_pres_col="DEM_PRES",
    rep_pres_col="REP_PRES",
    dem_leg_col="DEM_LEG",
    rep_leg_col="REP_LEG",
    x_axis_col="PresDemShare",
    color_col=None,
    label_col=None,
    figsize=(8,6),
    alpha=0.7,
    edgecolor='k',
    linewidth=0.5,
    title="Dem Over/Under Performance (State House vs. Pres)",
    xlabel="Presidential Dem Share",
    ylabel="Dem Vote Difference (Leg - Expected)",
    label_points=True,
    filter_out_uncontested=True
):
    """
    Plots how many Dem votes the State House race got above/below what you'd
    "expect" if the same fraction of the electorate turned out as in the
    Presidential race.

    Steps:
      1) share_pres_dem = DEM_PRES / (DEM_PRES + REP_PRES)
      2) total_leg = DEM_LEG + REP_LEG
      3) expected_dem_leg = share_pres_dem * total_leg
      4) dem_leg_diff = DEM_LEG - expected_dem_leg

    By default, the X-axis is 'PresDemShare' (df["share_pres_dem"]),
    but you can override with x_axis_col (any column).
    
    Optional: filter_out_uncontested=True to remove rows where x or y is exactly 0 or 1.

    Parameters
    ----------
    df : DataFrame
        Must contain columns: dem_pres_col, rep_pres_col, dem_leg_col, rep_leg_col.
    dem_pres_col, rep_pres_col : str
        Columns for Presidential DEM and REP votes.
    dem_leg_col, rep_leg_col : str
        Columns for Legislative (State House) DEM and REP votes.
    x_axis_col : str
        Which column to place on the X-axis. Default is "PresDemShare".
    color_col : str or None
        If None, use a single color. If numeric, use a continuous colormap.
        If categorical, assign discrete colors.
    label_col : str or None
        If provided and label_points=True, we'll label each point with that column's value.
    filter_out_uncontested : bool
        If True, remove rows where the X or Y value is exactly 0 or 1 
        (useful to ignore extreme shares or differences).
        
    Returns
    -------
    None (displays a matplotlib plot)
    """

    # 1) Check required columns
    required_cols = [dem_pres_col, rep_pres_col, dem_leg_col, rep_leg_col]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in DataFrame.")
    
    df = df.copy()  # avoid in-place modifications

    # 2) Compute the Presidential Dem share
    pres_total = df[dem_pres_col] + df[rep_pres_col]
    df["share_pres_dem"] = np.where(pres_total != 0, df[dem_pres_col] / pres_total, np.nan)

    # 3) Compute the total in Leg race & expected Dem votes
    leg_total = df[dem_leg_col] + df[rep_leg_col]
    df["expected_dem_leg"] = df["share_pres_dem"] * leg_total
    df["dem_leg_diff"] = df[dem_leg_col] - df["expected_dem_leg"]

    # 4) Ensure x_axis_col exists (default: 'PresDemShare')
    #    If user gave a custom x_axis_col but it's not in df, fallback or raise
    if x_axis_col not in df.columns:
        # fallback: use our "share_pres_dem"
        df[x_axis_col] = df["share_pres_dem"]

    # 5) If user wants to filter out rows where x=0 or 1 or y=0 or 1
    x_data = df[x_axis_col]
    y_data = df["dem_leg_diff"]
    if filter_out_uncontested:
        # drop rows where x or y is exactly 0 or 1
        mask = ~(
            ((x_data == 0) | (x_data == 1)) 
            | ((y_data == 0) | (y_data == 1))
        )
        df = df[mask]

    # 6) Build scatter plot
    plt.figure(figsize=figsize)

    scatter_kwargs = dict(
        x=x_axis_col,
        y="dem_leg_diff",
        data=df,
        alpha=alpha,
        edgecolor=edgecolor,
        linewidth=linewidth
    )

    # 7) Handle color
    if color_col is None:
        scatter_kwargs["c"] = "steelblue"
    else:
        if color_col not in df.columns:
            raise ValueError(f"Column '{color_col}' not found in DataFrame.")
        if pd.api.types.is_numeric_dtype(df[color_col]):
            scatter_kwargs["c"] = df[color_col]
            scatter_kwargs["cmap"] = "viridis"
        else:
            categories = df[color_col].astype('category')
            cat_codes = categories.cat.codes
            scatter_kwargs["c"] = cat_codes
            scatter_kwargs["cmap"] = "tab10"

    plt.scatter(**scatter_kwargs)
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)

    # 8) If color_col is categorical, build a legend
    if color_col and not pd.api.types.is_numeric_dtype(df[color_col]):
        categories = df[color_col].astype('category')
        handles = []
        for i, cat_val in enumerate(categories.cat.categories):
            fraction = i / (len(categories.cat.categories) - 1) if len(categories.cat.categories) > 1 else 0
            handles.append(
                plt.Line2D(
                    [], [], marker='o',
                    color=plt.cm.get_cmap('tab10')(fraction),
                    label=str(cat_val),
                    linestyle=''
                )
            )
        plt.legend(handles=handles, title=color_col, loc='best')

    # 9) Optionally label each point
    if label_points and label_col:
        if label_col not in df.columns:
            raise ValueError(f"label_col '{label_col}' not found in DataFrame.")
        for _, row in df.iterrows():
            plt.text(row[x_axis_col], row["dem_leg_diff"], str(row[label_col]),
                     fontsize=8, alpha=0.7)

    # Horizontal line at 0 => No difference
    plt.axhline(0, color='red', linestyle='--', alpha=0.7, label="No difference")

    plt.legend()
    plt.show()

# another approach to investigate 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plot_downballot_dropoff_comparison(
    df,
    dem_leg_col="DEM_LEG",
    rep_leg_col="REP_LEG",
    dem_pres_col="DEM_PRES",
    rep_pres_col="REP_PRES",
    label_col="DistrictName",
    color_col=None,
    title="Downballot Dropoff Comparison",
    xlabel="Republican Downballot Dropoff (%)",
    ylabel="Democratic Downballot Dropoff (%)",
    label_points=True,
    figsize=(8, 6),
    alpha=0.7,
    edgecolor='k',
    linewidth=0.5,
    filter_out_uncontested=True,
    color_by_demshare_pres=False
):
    """
    Plots the percentage downballot dropoff for Democrats (vertical axis) vs.
    Republicans (horizontal axis). The dropoff is calculated as:
    
      dropoff_dem = 100 * (DEM_PRES - DEM_LEG) / DEM_PRES
      dropoff_rep = 100 * (REP_PRES - REP_LEG) / REP_PRES

    A y = x reference line is added to help visualize whether the dropoff is
    more pronounced for Democrats than Republicans.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing vote columns for both races.
    dem_leg_col, rep_leg_col : str
        Column names for the legislative (downballot) Democratic and Republican votes.
    dem_pres_col, rep_pres_col : str
        Column names for the presidential Democratic and Republican votes.
    label_col : str
        Column name used to label points (if label_points is True).
    color_col : str or None
        If provided, color points by the values in this column. If numeric, a continuous
        colormap is used; if categorical, discrete colors are assigned.
    title : str
        Plot title.
    xlabel : str
        Label for the x-axis.
    ylabel : str
        Label for the y-axis.
    label_points : bool
        If True, label each point with the value in `label_col`.
    figsize : tuple
        Figure size in inches (width, height).
    alpha : float
        Transparency for scatter points.
    edgecolor : str
        Edge color for scatter points.
    linewidth : float
        Edge line width for scatter points.
    filter_out_uncontested : bool, default True
        If True, remove rows where any of the vote columns are zero, to avoid extreme (uncontested) cases.
    color_by_demshare_pres : bool
        If True, ignore color_col and instead color points continuously by the
        presidential Democratic share (DEM_PRES/(DEM_PRES+REP_PRES)) using a red-to-blue spectrum.
    
    Returns
    -------
    None (displays the plot)
    """
    # 1) Check that required columns exist
    required_cols = [dem_leg_col, rep_leg_col, dem_pres_col, rep_pres_col]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    if filter_out_uncontested:
        df = df[
            (df[dem_leg_col] != 0) & (df[dem_pres_col] != 0) &
            (df[rep_leg_col] != 0) & (df[rep_pres_col] != 0)
        ].copy()
    else:
        df = df.copy()

    # 2) Calculate downballot dropoff percentages for Democrats and Republicans
    df["dropoff_dem"] = np.where(
        df[dem_pres_col] != 0,
        100.0 * (df[dem_pres_col] - df[dem_leg_col]) / df[dem_pres_col],
        np.nan
    )
    df["dropoff_rep"] = np.where(
        df[rep_pres_col] != 0,
        100.0 * (df[rep_pres_col] - df[rep_leg_col]) / df[rep_pres_col],
        np.nan
    )

    # 3) Prepare scatter plot keyword arguments.
    # Here we want to plot x="dropoff_rep" and y="dropoff_dem"
    scatter_kwargs = dict(
        x="dropoff_rep",
        y="dropoff_dem",
        data=df,
        alpha=alpha,
        edgecolor=edgecolor,
        linewidth=linewidth
    )

    # 4) Determine color mapping:
    if color_by_demshare_pres:
        # Compute presidential Dem share if not already present.
        if "DemShare_PRES" not in df.columns:
            pres_total = df[dem_pres_col] + df[rep_pres_col]
            df["DemShare_PRES"] = np.where(pres_total != 0, df[dem_pres_col] / pres_total, np.nan)
        scatter_kwargs['c'] = df["DemShare_PRES"]
        # Create a red-to-blue colormap (red = lower, blue = higher)
        red_blue_cmap = mcolors.LinearSegmentedColormap.from_list("RedBlue", ["red", "blue"])
        scatter_kwargs['cmap'] = red_blue_cmap
    elif color_col is None:
        scatter_kwargs['c'] = 'steelblue'
    else:
        if color_col not in df.columns:
            raise ValueError(f"Column '{color_col}' not found in DataFrame for coloring.")
        if pd.api.types.is_numeric_dtype(df[color_col]):
            scatter_kwargs['c'] = df[color_col]
            scatter_kwargs['cmap'] = 'viridis'
        else:
            categories = df[color_col].astype('category')
            cat_codes = categories.cat.codes
            scatter_kwargs['c'] = cat_codes
            scatter_kwargs['cmap'] = 'tab10'

    # 5) Create figure & scatter plot
    plt.figure(figsize=figsize)
    plt.scatter(**scatter_kwargs)
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)

    # 6) If color_col is categorical, build a manual legend
    if (color_by_demshare_pres is False) and color_col and not pd.api.types.is_numeric_dtype(df[color_col]):
        categories = df[color_col].astype('category')
        handles = []
        for cat_val in categories.cat.categories:
            idx = categories.cat.categories.get_loc(cat_val)
            fraction = idx / (len(categories.cat.categories) - 1) if len(categories.cat.categories) > 1 else 0
            handles.append(
                plt.Line2D(
                    [], [], marker='o',
                    color=plt.cm.get_cmap('tab10')(fraction),
                    label=str(cat_val),
                    linestyle=''
                )
            )
        plt.legend(handles=handles, title=color_col, loc='best')

    # 7) Optionally label each point with label_col
    if label_points:
        if label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found in DataFrame.")
        for _, row in df.iterrows():
            plt.text(row["dropoff_rep"], row["dropoff_dem"], str(row[label_col]),
                     fontsize=8, alpha=0.7)

    # 8) Plot a y = x reference line.
    valid_dropoff = df[["dropoff_rep", "dropoff_dem"]].dropna()
    if not valid_dropoff.empty:
        min_val = min(valid_dropoff["dropoff_rep"].min(), valid_dropoff["dropoff_dem"].min())
        max_val = max(valid_dropoff["dropoff_rep"].max(), valid_dropoff["dropoff_dem"].max())
        x_vals = np.linspace(min_val, max_val, 200)
        plt.plot(x_vals, x_vals, color='orange', linestyle='--', alpha=0.8, label="y = x")
        plt.legend()

    plt.show()

    return df 

def assign_quadrant(x, y):
    """
    Given numeric values x and y, return a quadrant label:
      - "Q1" if x > 0 and y > 0
      - "Q2" if x < 0 and y > 0
      - "Q3" if x < 0 and y < 0
      - "Q4" if x > 0 and y < 0
      - "On Axis" otherwise (if either x or y is exactly 0)
    """
    if x > 0 and y > 0:
        return "Q1"
    elif x < 0 and y > 0:
        return "Q2"
    elif x < 0 and y < 0:
        return "Q3"
    elif x > 0 and y < 0:
        return "Q4"
    else:
        return "On Axis"

def count_precincts_by_quadrant(df, xcol, ycol, groupby=None, drop_axis=False):
    """
    Counts the number of precincts (rows) falling into each quadrant based on two numeric columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    xcol : str
        Name of the column to use as the x-value (e.g. Republican dropoff).
    ycol : str
        Name of the column to use as the y-value (e.g. Democratic dropoff).
    groupby : str or list of str, optional
        A column name or list of column names to group by (e.g., "State"). 
        If provided, counts are calculated within each group.
    drop_axis : bool, default False
        If True, rows where xcol or ycol equals exactly 0 will be dropped 
        (not counted in any quadrant); otherwise, they are labeled as "On Axis".
    
    Returns
    -------
    pd.DataFrame
        A DataFrame with counts for each quadrant. If groupby is provided, the DataFrame
        is indexed (or has columns) for each group and quadrant.
    """
    df = df.copy()

    # Optionally drop rows on an axis
    if drop_axis:
        df = df[(df[xcol] != 0) & (df[ycol] != 0)]
    
    # Assign quadrant for each row using the provided x and y values.
    df["Quadrant"] = df.apply(lambda row: assign_quadrant(row[xcol], row[ycol]), axis=1)
    
    # If no grouping is desired, simply count quadrant frequencies.
    if groupby is None:
        counts = df["Quadrant"].value_counts().reset_index()
        counts.columns = ["Quadrant", "Count"]
    else:
        # Group by the grouping variable(s) and quadrant, then count
        counts = df.groupby(groupby + ["Quadrant"]).size().reset_index(name="Count")
    
    return counts


if __name__ == "__main__":

    # Example 1: Single-row format from a CSV
    single_row_path = r'/Users/aspencage/Documents/Data/input/post_g2024/2024_precinct_level_data/wi/2024_Election_Data_with_2025_Wards_-5879223691586298781.geojson'  # e.g. .csv
    wi_gdf = prepare_precinct_data_single_row(
        data_input=single_row_path, 
        precinct_col="GEOID",
        dem_leg_col="WSADEM24",
        rep_leg_col="WSAREP24",
        dem_pres_col="PREDEM24",
        rep_pres_col="PREREP24",
        keep_geometry=False  
    )
    print("Wisconsin single-row shape:", wi_gdf.shape)

    # Example 2: Multi-row format from a shapefile
    multi_row_path = r'/Users/aspencage/Documents/Data/input/post_g2024/2024_precinct_level_data/pa/erstat_2024_g_268768_20250110 copy.csv'  # e.g. .shp
        # Example: keep 'CountyCode' and 'PrecinctCode' as strings
    
    pa_col_names = [
        "ElectionYear",
        "ElectionType",
        "CountyCode",
        "PrecinctCode",
        "CandidateOfficeRank",
        "CandidateDistrict",
        "CandidatePartyRank",
        "CandidateBallotPosition",
        "CandidateOfficeCode",
        "CandidatePartyCode",
        "CandidateNumber",
        "CandidateLastName",
        "CandidateFirstName",
        "CandidateMiddleName",
        "CandidateSuffix",
        "VoteTotal",
        "YesVoteTotal",
        "NoVoteTotal",
        "USCongressionalDistrict",
        "StateSenatorialDistrict",
        "StateHouseDistrict",
        "MunicipalityTypeCode",
        "MunicipalityName",
        "MunicipalityBreakdownCode1",
        "MunicipalityBreakdownName1",
        "MunicipalityBreakdownCode2",
        "MunicipalityBreakdownName2",
        "BiCountyCode",
        "MCDCode",
        "FIPSCode",
        "VTDCode",
        "BallotQuestion",
        "RecordType",
        "PreviousPrecinctCode",
        "PreviousUSCongressionalDist",
        "PreviousStateSenatorialDist",
        "PreviousStateHouseDist"
    ]

    pa_dtypes = {
        'CountyCode': str,
        'PrecinctCode': str,
        "StateSenatorialDistrict":str,
        "StateHouseDistrict":str,
    }
    
    pa_gdf = prepare_precinct_data_multirow(
        data_input=multi_row_path,
        precinct_col="PrecinctCode",
        office_col="CandidateOfficeCode",
        party_col="CandidatePartyCode",
        vote_col="VoteTotal",
        leg_office="STH",   
        pres_office="USP",
        dem_party="DEM",
        rep_party="REP",
        keep_geometry=False,
        has_header=False,
        col_names=pa_col_names,
        dtypes=pa_dtypes,
        state='PA'
    )
    print("Pennsylvania multi-row shape:", pa_gdf.shape)

    # Now each DataFrame has columns: PrecinctName, DEM_LEG, REP_LEG, DEM_PRES, REP_PRES, [geometry if kept]
    wi_gdf["state"] = "WI"
    pa_gdf["state"] = "PA"
    cols_standard = ["state", "precinct", "DEM_LEG", "REP_LEG", "DEM_PRES", "REP_PRES"]
    gdf = pd.concat([wi_gdf[cols_standard], pa_gdf[cols_standard]], ignore_index=True)
    gdf = compute_leg_vs_pres_precinct(gdf)
    
    # Select the best S-curve
    best_model_dict = select_best_s_curve(gdf)

    state_leg_str = "State House"

    plot_two_way_voteshare_comparison(
        gdf,
        xcol='DemShare_PRES',
        ycol='DemShare_LEG',
        label_col=None,
        title=f"G2024 Presidential vs. {state_leg_str} Two-Way Vote Share (PA,WI)",
        xlabel='Presidential Dem Two-Way Vote Share',
        ylabel=f'{state_leg_str} Dem Two-Way Vote Share',
        vert_line_label=f'50% {state_leg_str}',
        horiz_line_label='50% (Presidential)',
        label_points=False,
        color_col='state',
        plot_s_curve=True,
        plot_model_dict=best_model_dict
        )

    plot_two_way_voteshare_comparison(
        gdf.loc[gdf["state"] == "PA"],
        xcol='DemShare_PRES',
        ycol='DemShare_LEG',
        label_col=None,
        title=f"G2024 Presidential vs. {state_leg_str} Two-Way Vote Share (PA)",
        xlabel='Presidential Dem Two-Way Vote Share',
        ylabel=f'{state_leg_str} Dem Two-Way Vote Share',
        vert_line_label=f'50% {state_leg_str}',
        horiz_line_label='50% (Presidential)',
        label_points=False,
        color_col='state',
        plot_s_curve=True,
        plot_model_dict=select_best_s_curve(gdf.loc[gdf["state"] == "PA"])
        )

    plot_two_way_voteshare_comparison(
        gdf.loc[gdf["state"] == "WI"],
        xcol='DemShare_PRES',
        ycol='DemShare_LEG',
        label_col=None,
        title=f"G2024 Presidential vs. {state_leg_str} Two-Way Vote Share (WI)",
        xlabel='Presidential Dem Two-Way Vote Share',
        ylabel=f'{state_leg_str} Dem Two-Way Vote Share',
        vert_line_label=f'50% {state_leg_str}',
        horiz_line_label='50% (Presidential)',
        label_points=False,
        color_col='state',
        plot_s_curve=True,
        plot_model_dict=select_best_s_curve(gdf.loc[gdf["state"] == "WI"])
        )

    # NOTE hacky 
    gdf["DemShare_PRES"] = np.where(gdf["DEM_PRES"] > gdf["REP_PRES"], 1, 0)

    turnout = plot_downballot_dropoff_comparison(
          gdf,
          dem_leg_col="DEM_LEG",
          rep_leg_col="REP_LEG",
          dem_pres_col="DEM_PRES",
          rep_pres_col="REP_PRES",
          label_col=None,
          #color_col="pres_winner",
          color_by_demshare_pres=True,
          title=f"Party Downballot Dropoff Comparison: {state_leg_str} & Presidential (PA,WI)",
          xlabel="Republican Downballot Dropoff (%)",
          ylabel="Democratic Downballot Dropoff (%)",
          label_points=False,
          figsize=(8, 6),
          alpha=0.7,
          edgecolor='k',
          linewidth=0.5
      )

    plot_downballot_dropoff_comparison(
          gdf.loc[gdf["state"] == "PA"],
          dem_leg_col="DEM_LEG",
          rep_leg_col="REP_LEG",
          dem_pres_col="DEM_PRES",
          rep_pres_col="REP_PRES",
          label_col=None,
          color_by_demshare_pres=True,
          title=f"Party Downballot Dropoff Comparison: {state_leg_str} & Presidential (PA)",
          xlabel="Republican Downballot Dropoff (%)",
          ylabel="Democratic Downballot Dropoff (%)",
          label_points=False,
          figsize=(8, 6),
          alpha=0.7,
          edgecolor='k',
          linewidth=0.5
      )
    
    plot_downballot_dropoff_comparison(
          gdf.loc[gdf["state"] == "WI"],
          dem_leg_col="DEM_LEG",
          rep_leg_col="REP_LEG",
          dem_pres_col="DEM_PRES",
          rep_pres_col="REP_PRES",
          label_col=None,
          color_by_demshare_pres=True,
          title=f"Party Downballot Dropoff Comparison: {state_leg_str} & Presidential (WI)",
          xlabel="Republican Downballot Dropoff (%)",
          ylabel="Democratic Downballot Dropoff (%)",
          label_points=False,
          figsize=(8, 6),
          alpha=0.7,
          edgecolor='k',
          linewidth=0.5
      )
    
    turnout["pres_winner"] = np.where(turnout["DEM_PRES"] > turnout["REP_PRES"], 'harris', 'trump')
    turnout['greater_downballot_dropoff'] = np.where(turnout['dropoff_dem'] > turnout['dropoff_rep'], 'dem', 'rep')
    precinct_count = count_precincts_by_quadrant(turnout, "dropoff_rep", "dropoff_dem", groupby=["state",'pres_winner'], drop_axis=True)