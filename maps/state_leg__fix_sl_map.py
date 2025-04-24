from nyt_2024_process_and_overlay import (
    load_and_prepare_districts_w_overrides
  )

if __name__ == "__main__":

    sldl_directory = (
        r"/Users/aspencage/Documents/Data/input/post_g2024/"
        r"comparative_presidential_performance/TIGER2023_SLDL"
      )

    mi_state_house_fp = (
        r'/Users/aspencage/Documents/Data/'
        r'input/post_g2024/comparative_presidential_performance/'
        r'G2024/Redistricted/State House/'
        r'mi_sldl_2024_Motown_Sound_FC_E1_Shape_Files/91b9440a853443918ad4c8dfdf52e495.shp'
        )
    
    wi_state_house_fp = (
        r'/Users/aspencage/Documents/Data/input/post_g2024/comparative_presidential_performance/'
        r'G2024/Redistricted/State House/nc_sldl_adopted_2023/'
        r'SL 2023-149 House - Shapefile/SL 2023-149.shp'
    )

    district_col = 'District'
    override_configs = {
        "MI": {
              "path" : mi_state_house_fp,
              "col_map" : {'DISTRICTNO':district_col}, 
              "geo_source": "Michigan SOS: 2024 Motown Sound FC E1 Shape File",
              "process_override" : True,
              "keep_all_columns" : False
            },
        "WI": {
              "path" : wi_state_house_fp,
              "col_map" : {'DISTRICT':district_col}, 
              "geo_source": "Wisconsin LRB: 2023-149 State House",
              "process_override" : True,
              "keep_all_columns" : False
      }
    }

    districts_corrected = load_and_prepare_districts_w_overrides(
      sldl_directory, 
      crs="EPSG:2163",
      override_configs=override_configs,
      district_col=district_col
    )