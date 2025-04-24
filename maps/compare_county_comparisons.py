import pandas as pd 
import time

fp_topojson_based = (
    r'/Users/aspencage/Documents/Data/output/post_2024/2020_2024_pres_compare/'
    r'nyt_pres_2024_by_county_comparison_20250304_160216.csv'
)
fp_state_geojsons_based = (
    r'/Users/aspencage/Documents/Data/output/post_2024/2020_2024_pres_compare/'
    r'nyt_state_by_state_jsons/nyt_pres_2024_by_county_comparison_20250304_164648.csv'
)
topojson_based = pd.read_csv(fp_topojson_based)
state_geojsons_based = pd.read_csv(fp_state_geojsons_based)
topojson_based['fips'] = topojson_based['fips'].astype(int)
state_geojsons_based['fips'] = state_geojsons_based['fips'].astype(int)
merged = pd.merge(topojson_based, state_geojsons_based, on="fips", how="outer", suffixes=('_topojson', '_state_geojsons'))
merged['pres_dem_share_two_way_2024__diff_abs__diff'] = merged['pres_dem_share_two_way_2024__diff_abs_topojson'] - merged['pres_dem_share_two_way_2024__diff_abs_state_geojsons']

time_str = time.strftime("%Y%m%d_%H%M%S")
outfile_base = (
    r'/Users/aspencage/Documents/Data/output/post_2024/2020_2024_pres_compare/'
    f'nyt_pres_2024_by_county_comparison_comparison_{time_str}'
)
merged.to_csv(outfile_base + ".csv", index=False)
print("Done!")