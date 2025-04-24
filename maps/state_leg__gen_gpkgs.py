import pandas as pd
import numpy as np 
import geopandas as gp
import os
import re 
import warnings

from mapping_utilities import *


def process_district_data(gdf, state, crs="EPSG:3857"):
    
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

def state_specific_precinct_processing(precincts, state):
    
    # remove Lake Eerie
    if state == "PA":
        precincts = precincts.loc[~(precincts.precinct == "000000")]

    return precincts

def process_precinct_numbers_into_districts(precincts_2016, precincts_2020, districts, district_col, state=None, crs="EPSG:3857", output_path=None):

    def process_vest_precinct_data(gdf, crs, rep_regex="G20PRER.*", dem_regex="G20PRED.*", generic_party_regex="G20PRE.*", precinct_regex="VTD.*"):
        """
        Processes precinct-level election returns by renaming columns, calculating totals, 
        and adding new attributes such as precinct area.
        
        Parameters:
            gdf (GeoDataFrame): Input GeoDataFrame with precinct data.
            crs (str or dict): The CRS to which the GeoDataFrame should be transformed.
            rep_regex (str): Regex pattern to identify the Republican vote column.
            dem_regex (str): Regex pattern to identify the Democratic vote column.
            generic_party_regex (str): Regex pattern to identify all party vote columns.
            precinct_regex (str): Regex pattern to identify the precinct identifier column.
        
        Returns:
            GeoDataFrame: Processed GeoDataFrame with renamed columns and new attributes.
        """
        # Convert to specified CRS
        gdf = gdf.to_crs(crs)
        
        # Identify columns using regex
        rep_col = next((col for col in gdf.columns if re.match(rep_regex, col)), None)
        dem_col = next((col for col in gdf.columns if re.match(dem_regex, col)), None)
        precinct_col = next((col for col in gdf.columns if re.match(precinct_regex, col)), None)
        
        # Identify all party-related columns using the generic regex
        party_columns = [col for col in gdf.columns if re.match(generic_party_regex, col)]
        
        # Exclude Republican and Democratic columns from the third-party calculation
        third_party_columns = [col for col in party_columns if col not in {rep_col, dem_col}]
        
        # Rename columns based on identified matches
        rename_map = {}
        if rep_col:
            rename_map[rep_col] = "pres_rep"
        if dem_col:
            rename_map[dem_col] = "pres_dem"
        if precinct_col:
            rename_map[precinct_col] = "precinct"
        
        gdf.rename(columns=rename_map, inplace=True)
        
        # Calculate third-party votes as the sum of the remaining party columns
        if third_party_columns:
            gdf["pres_third_party"] = gdf[third_party_columns].sum(axis=1)
        else:
            gdf["pres_third_party"] = 0  # Default to 0 if no third-party columns exist
        
        # Add calculated columns
        if "pres_rep" in gdf and "pres_dem" in gdf:
            gdf["pres_total"] = gdf["pres_rep"] + gdf["pres_dem"] + gdf["pres_third_party"]
            gdf["pres_two_way"] = gdf["pres_rep"] + gdf["pres_dem"]
        
        # Add precinct area
        gdf['precinct_area'] = gdf.geometry.area
        
        # Drop rows where the sum of rep_col and dem_col is 0
        gdf = gdf.loc[gdf["pres_total"] != 0]

        if state is not None:
          gdf = state_specific_precinct_processing(gdf, state)

        return gdf
    
    # Process precinct data for 2016 and 2020
    precincts_2016 = process_vest_precinct_data(precincts_2016, crs, rep_regex="G16PRER.*", dem_regex="G16PRED.*", generic_party_regex="G16PRE.*", precinct_regex="PRECINCT")
    precincts_2020 = process_vest_precinct_data(precincts_2020, crs, rep_regex="G20PRER.*", dem_regex="G20PRED.*", generic_party_regex="G20PRE.*", precinct_regex="VTDST")

    # Function to process data for a given year
    def process_year_data(gdf_precinct, year):
        # Perform spatial intersection with legislative districts
        with warnings.catch_warnings():
          warnings.simplefilter("ignore", UserWarning)
          subset = gp.overlay(gdf_precinct, districts, how='intersection')
        subset['split_area'] = subset.geometry.area
        subset['area_fraction'] = subset['split_area'] / subset['precinct_area']
        subset['pres_dem_fraction'] = subset['pres_dem'] * subset['area_fraction']
        subset['pres_rep_fraction'] = subset['pres_rep'] * subset['area_fraction']
        subset['pres_third_party_fraction'] = subset['pres_third_party'] * subset['area_fraction']
        subset['pres_total_fraction'] = subset['pres_total'] * subset['area_fraction']
        subset['pres_two_way_fraction'] = subset['pres_two_way'] * subset['area_fraction']
        grouped = subset.groupby(district_col).agg({
            'pres_dem_fraction': 'sum',
            'pres_rep_fraction': 'sum',
            'pres_third_party_fraction': 'sum',
            'pres_total_fraction': 'sum',
            'pres_two_way_fraction': 'sum'
        }).reset_index()
        grouped[f'pres_dem_share_total_{year}'] = grouped['pres_dem_fraction'] / grouped['pres_total_fraction']
        grouped[f'pres_dem_share_two_way_{year}'] = grouped['pres_dem_fraction'] / grouped['pres_two_way_fraction']
        grouped[f'third_party_vote_share_{year}'] = grouped['pres_third_party_fraction'] / grouped['pres_total_fraction']
        return grouped

    # Process data for 2016 and 2020
    grouped_2016 = process_year_data(precincts_2016, '2016')
    grouped_2020 = process_year_data(precincts_2020, '2020')

    # Merge the results on district_col
    grouped = grouped_2016.merge(grouped_2020, on=district_col)
    grouped['pres_dem_share_total_diff'] = grouped['pres_dem_share_total_2020'] - grouped['pres_dem_share_total_2016']
    grouped['pres_dem_share_two_way_diff'] = grouped['pres_dem_share_two_way_2020'] - grouped['pres_dem_share_two_way_2016']
    grouped['third_party_vote_share_diff'] = grouped['third_party_vote_share_2020'] - grouped['third_party_vote_share_2016']
    grouped['potential_error_detected'] = np.where(grouped['pres_dem_share_two_way_diff'] > abs(0.2),"Greater than 20 pp swing",None)

    # Merge with legislative districts to get geometry
    output_gdf = districts.merge(grouped, on=district_col)

    # Save the results if output_path is provided
    if output_path:
        output_gdf.to_file(output_path, layer="weighted_averages", driver="GPKG")
        print(f"Saved output to {output_path}")
    
    return output_gdf


def bulk_process(district_parent_directories, output_path,precinct_parent_directory_2016,precinct_parent_directory_2020):

    district_shapefile_paths = get_all_mapfiles(district_parent_directories)
    state_table = generate_state_file_table(output_path)
    district_shapefile_paths = filter_shapefile_paths(district_shapefile_paths, state_table)
    print(f"Processing {len(district_shapefile_paths)} shapefiles.")
    print(district_shapefile_paths)


    for district_path in district_shapefile_paths:
        print(f"Processing: {os.path.basename(district_path)}")
        
        fips = extract_fips_from_path(district_path)

        try:
            state = map_fips_and_state(fips)
            if not state or state == "Invalid input":
                raise ValueError(f"Invalid state mapping for FIPS: {fips}")
        except Exception as e:
            print(f"Error processing state for FIPS {fips}: {e}")
            continue
        
        try:
            if state == "PR":
                print(f"Skipping Puerto Rico (PR) as it is not supported.")
                continue

            districts = gp.read_file(district_path)
            districts, district_col = process_district_data(districts, state)
            
            name = f"{state.lower()}_2016"
            path = os.path.join(precinct_parent_directory_2016,name,name+".shp")
            if not os.path.exists(path):
              print(f"File does not exist: {path}")
              continue
            precincts_2016 = gp.read_file(path)

            name = f"{state.lower()}_2020"
            path = os.path.join(precinct_parent_directory_2020,name,name+".shp")
            if not os.path.exists(path):
              print(f"File does not exist: {path}")
              continue
            precincts_2020 = gp.read_file(path)      

            # Example usage
            output_gdf = process_precinct_numbers_into_districts(
              precincts_2016=precincts_2016,
              precincts_2020=precincts_2020,
              districts=districts,
              district_col="District",
              state=state,
              output_path=os.path.join(output_path,f"{state}_by_{district_col}_test.gpkg")
            )

        except Exception as e:
          print(f"Error processing {district_path}: {e}")
          continue


if __name__ == "__main__":

    district_parent_directories = [r"/Users/aspencage/Documents/Data/input/post_g2024/comparative_presidential_performance/TIGER2023_SLDL", r"/Users/aspencage/Documents/Data/input/post_g2024/comparative_presidential_performance/TIGER2023_SLDU"]
    output_path = r"/Users/aspencage/Documents/Data/output/g2024/2020_2016_pres_state_leg"
    precinct_parent_directory_2016 = r"/Users/aspencage/Documents/Data/input/post_g2024/comparative_presidential_performance/2016_precinct_level_election_returns_dataverse_files"
    precinct_parent_directory_2020 = r"/Users/aspencage/Documents/Data/input/post_g2024/comparative_presidential_performance/2020_precinct_level_election_returns_dataverse_files"

    bulk_process(district_parent_directories, output_path,precinct_parent_directory_2016,precinct_parent_directory_2020)

    state_table = generate_state_file_table(output_path, filter_incomplete=True)
    print(state_table)