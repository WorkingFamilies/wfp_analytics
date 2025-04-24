import folium
import geopandas as gpd
from mapping_utilities import load_generic_file


def display_interactive_map(
    layer_configs,
    start_location=None,
    start_zoom=None,
    default_tiles="OpenStreetMap",
    tolerance_meters=500
):
    """
    Create an interactive Folium map with toggleable layers, 
    each loaded from various data sources or in-memory GeoDataFrames.
    This version attaches each Choropleth directly to the map so 
    we avoid "Choropleth must be added to a Map object" errors.

    Parameters
    ----------
    layer_configs : list of dict
        Each element describes a layer:
            {
                "data_input": <str or GeoDataFrame>,
                "layer_name": <str>,
                "layer_in_file": <optional str>,
                "choropleth_col": <optional str>,
                "null_choropleth": <bool>,
                "style_function": <optional callable>,
                "show": <bool>
            }
    ...
    """
    # Create the base map
    m = folium.Map(location=[0, 0], zoom_start=2, tiles=default_tiles)

    # Track overall bounds for auto-fitting
    overall_bounds = [float("inf"), float("inf"), float("-inf"), float("-inf")]

    for cfg in layer_configs:
        data_input = cfg["data_input"]
        layer_name = cfg.get("layer_name", "Unnamed Layer")
        layer_in_file = cfg.get("layer_in_file", None)
        choropleth_col = cfg.get("choropleth_col", None)
        null_choropleth = cfg.get("null_choropleth", False)
        style_func = cfg.get("style_function", None)
        show_layer = cfg.get("show", True)

        # 1) Load the data (using your helper or direct geopandas)
        if layer_in_file:
            gdf = gpd.read_file(data_input, layer=layer_in_file)
        else:
            gdf = load_generic_file(data_input, keep_geometry=True)

        if not isinstance(gdf, gpd.GeoDataFrame):
            print(f"[WARNING] {layer_name} is not a GeoDataFrame.")
            continue

        # 2) Simplify geometry in EPSG:3857, then reproject to 4326
        if tolerance_meters and gdf.crs:
            gdf_merc = gdf.to_crs(epsg=3857)
            gdf_merc["geometry"] = gdf_merc.geometry.simplify(
                tolerance=tolerance_meters, preserve_topology=True
            )
            gdf = gdf_merc.to_crs(epsg=4326)
        elif gdf.crs and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)

        # Update bounding box
        minx, miny, maxx, maxy = gdf.total_bounds
        overall_bounds[0] = min(overall_bounds[0], minx)
        overall_bounds[1] = min(overall_bounds[1], miny)
        overall_bounds[2] = max(overall_bounds[2], maxx)
        overall_bounds[3] = max(overall_bounds[3], maxy)

        # 3) Choropleth or simple GeoJson?
        if choropleth_col and choropleth_col in gdf.columns:
            # Ensure we have a real "index" column
            gdf.reset_index(drop=False, inplace=True)

            # Add a Choropleth directly to the map
            choropleth = folium.Choropleth(
                geo_data=gdf.__geo_interface__,
                name=layer_name,
                data=gdf,
                columns=["index", choropleth_col],
                key_on="feature.properties.index",
                fill_color="YlOrRd",
                fill_opacity=0.7,
                line_opacity=0.5,
                legend_name=f"{layer_name} - {choropleth_col}",
                highlight=True
            ).add_to(m)

            # Also add the underlying geojson portion to the map
            choropleth.geojson.add_to(m)

            # If user wants nulls highlighted, create a separate layer
            if null_choropleth:
                gdf_nulls = gdf[gdf[choropleth_col].isnull()]
                if not gdf_nulls.empty:
                    folium.GeoJson(
                        data=gdf_nulls.__geo_interface__,
                        name=f"{layer_name} (NULLs)",
                        style_function=lambda feat: {
                            "fillColor": "gray",
                            "color": "black",
                            "fillOpacity": 0.6,
                            "dashArray": "5,5"
                        }
                    ).add_to(m)

        else:
            # Simple GeoJson
            folium.GeoJson(
                data=gdf.__geo_interface__,
                name=layer_name,
                style_function=style_func
            ).add_to(m)

        # If user wants it hidden by default (show=False),
        # we cannot directly do that with Folium.Choropleth or Folium.GeoJson.
        # See note below for a "hide by default" workaround.

    # Add layer control
    folium.LayerControl().add_to(m)

    # Auto-fit if start_location or start_zoom not specified
    if start_location is None or start_zoom is None:
        if overall_bounds != [float("inf"), float("inf"), float("-inf"), float("-inf")]:
            m.fit_bounds([[overall_bounds[1], overall_bounds[0]],
                          [overall_bounds[3], overall_bounds[2]]])
        else:
            m.location = [0, 0]
            m.zoom_start = 2
    else:
        if start_location is not None:
            m.location = start_location
        if start_zoom is not None:
            m.zoom_start = start_zoom

    return m


if __name__ == "__main__":
    import time
    import os

    # How I realized I was using NC data lol

    wd = r"/Users/aspencage/Documents/Data/output/post_2024/2020_2024_pres_compare"
    os.chdir(wd)

    wi_state_house_districts_fp = (
        r"/Users/aspencage/Documents/Data/input/post_g2024/"
        r"comparative_presidential_performance/G2024/Redistricted/"
        r"State House/wi_sldl_adopted_2023/"
        r"AssemblyDistricts_2023WIAct94/AssemblyDistricts_2023WIAct94.shp"
    )
    
    wi_2024_precincts_fp = (
        r"/Users/aspencage/Documents/Data/input/post_g2024/"
        r"2024_precinct_level_data/wi/"
        r"2024_Election_Data_with_2025_Wards_-5879223691586298781.geojson"
    )

    from state_leg__2020_2024 import * 

    fp_precincts_fixed = (
        r'/Users/aspencage/Documents/Data/output/post_2024/'
        r'2020_2024_pres_compare/precincts__2024_pres__fixed__20250313_121550.gpkg'
    ) # NOTE - currently the output of state_leg__fix_precinct_testing

    sldl_directory = (
        r"/Users/aspencage/Documents/Data/input/post_g2024/"
        r"comparative_presidential_performance/TIGER2023_SLDL"
    )

    # filepaths to districts where we want to override the initial data 
    mi_state_house_fp = (
        r'/Users/aspencage/Documents/Data/'
        r'input/post_g2024/comparative_presidential_performance/'
        r'G2024/Redistricted/State House/'
        r'mi_sldl_2024_Motown_Sound_FC_E1_Shape_Files/91b9440a853443918ad4c8dfdf52e495.shp'
        )

    nc_data_i_thought_was_wi = (
        r'/Users/aspencage/Documents/Data/input/post_g2024/comparative_presidential_performance/'
        r'G2024/Redistricted/State House/nc_sldl_adopted_2023/'
        r'SL 2023-149 House - Shapefile/SL 2023-149.shp' # NOTE - this is the NC file!!
    )

    selected_states = ["MI","WI","MN"]

    # -------------------------------------------------------------------------
    # A) LOAD AND PREPARE DATA
    # -------------------------------------------------------------------------

    # precinct level data 
    precincts_2024 = load_and_prepare_precincts(
        fp_precincts_fixed, 
        crs="EPSG:2163",
        drop_na_columns=["votes_dem","votes_rep","votes_total",'geometry'],
        print_=True
    )

    if selected_states is not None:
        precincts_2024 = precincts_2024.loc[precincts_2024["state"].isin(selected_states)]

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
                "path" : nc_data_i_thought_was_wi,
                "col_map" : {'DISTRICT':district_col}, 
                "geo_source": "Wisconsin LRB: 2023-149 State House",
                "process_override" : True,
                "keep_all_columns" : False
            },
    }

    districts = load_and_prepare_districts_w_overrides(
        sldl_directory, 
        crs="EPSG:2163",
        override_configs=override_configs,
        district_col=district_col
    )
    if selected_states is not None:
        print("flag")
        districts = districts.loc[districts["State"].isin(selected_states)]

    layer_configs = [
        {
            "data_input": districts,
            "layer_name": "Districts modified",
            "style_function": None
        },
        {
            "data_input": precincts_2024,
            "layer_name": "Precincts",
            "choropleth_col": "votes_dem",
            "null_choropleth": True
        },
        {
            "data_input": wi_state_house_districts_fp,
            "layer_name": "WI districts direct",
            "style_function": None
        },
    ]

    m = display_interactive_map(layer_configs,tolerance_meters=2000)
    ts = time.strftime("%Y%m%d_%H%M%S")
    outfile = f"tristate_interactive_map_{ts}.html"
    m.save(outfile)
    print(f"Map saved to {outfile}")