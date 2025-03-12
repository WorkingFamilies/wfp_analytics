from state_leg__fix_specific_state import fix_precinct_data_for_states
from prepare_precinct_data import standardize_precinct_data, prepare_precinct_data_multirow, prepare_precinct_data_single_row
from nyt_2024_process_and_overlay import load_and_prepare_precincts
import os 

if __name__ == "__main__":
    
    out_dir = r'/Users/aspencage/Documents/Data/output/post_2024/2020_2024_pres_compare'
    os.chdir(out_dir)

    nyt_data = (
        r'/Users/aspencage/Documents/Data/output/'
        r'post_2024/2020_2024_pres_compare/'
        r'nyt_pres_2024_simplified.geojson'
      )
    nyt_precincts = load_and_prepare_precincts(nyt_data, crs="EPSG:2163")
    nyt_precincts['source'] = 'NYT Data'

    wi_precinct_fp = (
        r'/Users/aspencage/Documents/Data/input/post_g2024/'
        r'2024_precinct_level_data/wi/'
        r'2024_Election_Data_with_2025_Wards_-5879223691586298781.geojson'
    )

    wi_gdf_precincts = standardize_precinct_data(
        input_data=wi_precinct_fp,
        precinct_col="GEOID",          # rename GEOID -> 'precinct'
        dem_col="PREDEM24",      # rename -> 'votes_dem'
        rep_col="PREREP24",      # rename -> 'votes_rep'
        total_col='PRETOT24',                # no total col in data, so compute from dem+rep
        rename_map=None,               # no extra renaming
        fix_invalid_geometries=True,
        target_crs="EPSG:2163",
        retain_addl_cols=False
    )
    wi_gdf_precincts['state'] = 'WI'
    wi_gdf_precincts['official_bounary'] = True
    wi_gdf_precincts['source'] = 'WI SOS'

    # 4) Run the function, telling it which columns correspond to precinct, Dem votes, etc.
    combined_precincts = fix_precinct_data_for_states(
        original_precinct_data=nyt_precincts,
        states_to_fix=['WI'],
        improved_precincts_list=[wi_gdf_precincts],
        state_col="state",           # Column that identifies each row's state
        target_crs="EPSG:2163",
        export_fixed_file=True,      # If True, writes out a corrected GPKG
        output_dir=".",
        output_prefix="precincts__2024_pres__fixed_",
    )

    # then compare older fixed dataset (WI)
    ## TODO add PA, VA? 
    ## TODO add diagnostic function 

    # export to use this for run of show 