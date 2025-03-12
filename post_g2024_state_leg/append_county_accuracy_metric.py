from political_geospatial import process_area_weighted_metric, create_state_stats
from mapping_utilities import load_generic_file
from dataset_utilities import filter_by_quantile
from state_leg__fix_specific_state import standardize_precinct_data, fix_bad_states_data
import time 
import os 
import matplotlib.pyplot as plt


if __name__ == '__main__':
    
    main_gpkg_path = r"/Users/aspencage/Documents/Data/output/post_2024/2020_2024_pres_compare/pres_in_state_leg__20_v_24_comparison_250226-111450.gpkg"

    # NOTE importing here and tiering while debugging. also fix bad data should really happen in a separate script.  

    tier_1_states = ["NJ","VA","PA","NC","AZ","MI"]

    main_gdf = load_generic_file(main_gpkg_path)
    main_gdf = main_gdf[main_gdf["State"].isin(tier_1_states)]

    # Load your counties and districts
    print("Loading county data...")
    county_gpkg_path = (
        r'/Users/aspencage/Documents/Data/output/post_2024/'
        r'2020_2024_pres_compare/nyt_pres_2024_by_county_comparison_20250310_140158.gpkg')
    counties = load_generic_file(county_gpkg_path)

    counties["State"] = counties["County"].str.extract(r', ([A-Z]{2})$')
    counties = counties[counties["State"].isin(tier_1_states)]

    # Run the area-weighted aggregation
    counties.rename(columns=
                    {
                        "votes_dem_by_area_2024": "votes_dem_nyt_county",
                        'votes_rep_by_area_2024': 'votes_rep_nyt_county',
                        'D_President': 'votes_dem_county_aux',
                        'R_President': 'votes_rep_county_aux',
                        'votes_two_way_by_area_2024':'votes_two_way_nyt_county',
                        'Twoway_Total_President':'votes_two_way_county_aux',
                     }, inplace=True)
    print("Running area-weighted aggregation from one dataset into the other...")
    cols_to_add = [
        'votes_dem_nyt_county',
        'votes_rep_nyt_county',
        'votes_dem_county_aux',
        'votes_rep_county_aux',
        'votes_two_way_nyt_county',
        'votes_two_way_county_aux',
        'votes_two_way_by_area_2024__diff',
        'County'
        ]
    suffix = "__from_county_comp"  

    gdf_with_county_comp = process_area_weighted_metric(
        gdf_source=counties,
        gdf_target=main_gdf,
        source_cols=cols_to_add,
        target_id_col="District",
        suffix=suffix,
        agg_dict=None,
        print_warnings=True,
        return_intersection=False
    )

    gdf_with_county_comp["ratio__county_aux_twoway_to_precinct_twoway"] = gdf_with_county_comp["votes_two_way_county_aux" + suffix] / gdf_with_county_comp["votes_two_way_by_area_2024"]
    gdf_with_county_comp["ratio__nyt_county_two_way_to_precinct_two_way"] = gdf_with_county_comp["votes_two_way_nyt_county" + suffix] / gdf_with_county_comp["votes_two_way_by_area_2024"]
    gdf_with_county_comp["ratio__nyt_county_two_way_to_county_aux_two_way"] = gdf_with_county_comp["votes_two_way_nyt_county" + suffix] / gdf_with_county_comp["votes_two_way_county_aux" + suffix]
    gdf_with_county_comp["pct_error__btw_county_data_ported_to_districts"] = gdf_with_county_comp["votes_two_way_by_area_2024__diff" + suffix] / gdf_with_county_comp["votes_two_way_nyt_county" + suffix]
    gdf_with_county_comp["pct_error__btw_county_data_ported_to_districts_abs"] = gdf_with_county_comp["pct_error__btw_county_data_ported_to_districts"].abs()

    gdf = gdf_with_county_comp.copy() 

    state_stats = create_state_stats(gdf, cols=["pct_error__btw_county_data_ported_to_districts_abs","coverage_percentage","split_coverage_percentage"])

    # Save to disk 
    time_str = time.strftime("%Y%m%d_%H%M%S")
    os.chdir(r'/Users/aspencage/Documents/Data/output/post_2024/2020_2024_pres_compare')
    outfile_base = f"pres_in_district__20_v_24_comparison__{time_str}"

    gdf_with_county_comp.to_file(outfile_base + ".gpkg", driver="GPKG")
    df_no_geom = gdf_with_county_comp.drop(columns="geometry")
    df_no_geom.to_csv(outfile_base + ".csv", index=False)

    state_stats.to_csv(outfile_base + "_state_stats.csv", index=False)