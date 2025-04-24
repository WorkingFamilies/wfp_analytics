# examine__pa_counties.py

from prepare_precinct_data import load_and_prepare_precincts
from prepare_district_data import load_and_prepare_counties

# purpose - more rapidly debugging PA counties (where matched on field rather than geometry)

fp_precincts_fixed = (
    r'/Users/aspencage/Documents/Data/output/post_2024/'
    r'2020_2024_pres_compare/'
    r'precincts__2024_pres__fixed__20250314_211335.gpkg'
) 

county_shapefile = (
    r"/Users/aspencage/Documents/Data/input/post_g2024/"
    r"comparative_presidential_performance/tl_2019_us_county"
)

selected_states = ['PA']

state_col = 'State'

# precinct level data 
precincts_2024 = load_and_prepare_precincts(
    fp_precincts_fixed, 
    crs="EPSG:2163",
    drop_na_columns=["votes_dem","votes_rep"], # previously included 'votes_total' and 'geometry'
    print_=True
)
if selected_states is not None:
    precincts_2024 = precincts_2024.loc[precincts_2024[state_col].isin(selected_states)]

counties = load_and_prepare_counties(county_shapefile, crs="EPSG:2163")

if selected_states is not None:
    counties = counties[counties[state_col].isin(selected_states)]