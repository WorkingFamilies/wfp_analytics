from dataset_utilities import verbose_merge
from prepare_precinct_data import prepare_precinct_data_multirow
from nyt_2024_process_and_overlay import load_and_prepare_precincts
import pandas as pd

'''
NOTE - VA data has neither geospatial coordinates (ideal) nor is it mapped definitively 
onto State Legislative districts. Because State Leg is off cycle from the Presidential
years, it cannot be inferred from within this dataset. Past years datasets can be used,
but this requires consistent naming and presumes that boundaries remain unchanged. 

Conclusion: Only about half of the precincts merge correctly with minor adjustments 
More adjustments may bring this up, but it appears to be at least quasi-manual 
in terms of spacing, capitalization, leading 0 handling, etc. 
'''

fp_precincts_fixed = (
    r'/Users/aspencage/Documents/Data/output/post_2024/'
    r'2020_2024_pres_compare/precincts__2024_pres__fixed__20250313_121550.gpkg'
)

va_2024_precincts_fp = (
    r'/Users/aspencage/Documents/Data/input/post_g2024/'
    r'2024_precinct_level_data/va/va_2024_precinct__ret_241218_utf.csv'
)

precincts_2024 = load_and_prepare_precincts(
    fp_precincts_fixed, 
    crs="EPSG:2163",
    drop_na_columns=["votes_dem","votes_rep","votes_total",'geometry'],
    print_=True
)

precincts_2024 = precincts_2024.loc[precincts_2024["state"] == 'VA']

va_2024_precincts = pd.read_csv(va_2024_precincts_fp)

va_2024_precincts = prepare_precinct_data_multirow(
  va_2024_precincts_fp,
  precinct_col='PrecinctName',
  office_col='OfficeTitle',
  party_col='Party',
  vote_col='TOTAL_VOTES',
  offices=['President and Vice President'],
  dem_party='Democratic',
  rep_party='Republican',
  keep_geometry=False,
  has_header=True
)

precincts_2024["precinct"] = precincts_2024["precinct"].str.replace(r"^\d+-", "", regex=True)

merged = verbose_merge(
    precincts_2024,
    va_2024_precincts,
    left_on='precinct',
    right_on='precinct',
    how='outer'
)
