
from maps.mapping_utilities import fix_invalid_geometries
from typing import Union, List
import geopandas as gpd
import pandas as pd

import os
import time
import warnings
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.errors import TopologicalError
from typing import Union, List

# for now, including all functions into one import for ease of use 
from maps.prepare_precinct_data import *
from maps.prepare_district_data import *
from maps.include_historical_data import (
    prepare_2016_v_2020_data,
    merge_in_2020_data
)
from maps.validation_geo_internal import * 
from maps.area_weighted_metrics import *
from maps.fix_specific_state import *


def create_state_stats(districts, cols=["coverage_percentage", "split_coverage_percentage"]):
    """
    Summarizes statistics by state, returning aggregated stats.
    """
    print("Calculating summary stats by State...")
    agg_dict = {col: ["mean", "median", "min", "max"] for col in cols}
    
    state_stats = districts.groupby("State").agg(agg_dict).reset_index()
    
    # Flatten MultiIndex columns
    state_stats.columns = ["_".join(col).strip("_") for col in state_stats.columns.to_flat_index()]
    
    return state_stats
