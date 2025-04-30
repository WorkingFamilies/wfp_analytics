import geopandas as gpd
import os 

from maps.mapping_utilities import (
  repair_shapefile_with_ogr,
  check_mutual_exclusivity
  )

os.chdir(r'/Users/aspencage/Documents/Data/projects/ohio_precinct_level_vote')
fp = (
  r'input/May 06 2025 Active Precinct Portions_region/'
  r'May 06 2025 Active Precinct Portions_region.shp'
)

# repair shapefile with ogr
precincts = gpd.read_file(fp)
precincts = repair_shapefile_with_ogr(
  src =fp,
  dst = fp.replace('.shp', '_repaired.shp'),
  quiet=False
)

# check mutual exclusivity
precincts = gpd.read_file(fp.replace('.shp', '_repaired.shp'))
check_mutual_exclusivity(
  precincts
)
