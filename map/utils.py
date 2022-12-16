from functools import lru_cache
from typing import List
import warnings

def suppress_geopandas_warning():
    warnings.filterwarnings('ignore', '.*The Shapely GEOS version .* is incompatible with the GEOS version PyGEOS.*')

from map.nuplan_map.nuplan_map import NuPlanMap
from map.maps_db.gpkg_mapsdb import GPKGMapsDB


# some map consts
ALL_CITIES = ['sg-one-north', 'us-ma-boston', 'us-nv-las-vegas-strip', 'us-pa-pittsburgh-hazelwood']
VERSIONS = {'sg-one-north': '9.17.1964', 
            'us-ma-boston': '9.12.1817', 
            'us-nv-las-vegas-strip': '9.15.1915', 
            'us-pa-pittsburgh-hazelwood': '9.17.1937'}

@lru_cache(maxsize=2)
def get_maps_db(map_root: str, map_version: str) -> GPKGMapsDB:
    """
    Get a maps_db from disk.
    :param map_root: The root folder for the map data.
    :param map_version: The version of the map to load.
    :return; The loaded MapsDB object.
    """
    return GPKGMapsDB(map_root=map_root, map_version=map_version)


@lru_cache(maxsize=32)
def get_maps_api(map_root: str, map_version: str, map_name: str) -> NuPlanMap:
    """
    Get a NuPlanMap object corresponding to a particular set of parameters.
    :param map_root: The root folder for the map data.
    :param map_version: The map version to load.
    :param map_name: The map name to load.
    :return: The loaded NuPlanMap object.
    """
    maps_db = get_maps_db(map_root, map_version)
    return NuPlanMap(maps_db, map_name.replace(".gpkg", ""))

def get_all_maps_api(map_root: str, map_names: List[str]) -> NuPlanMap:
    maps_api = {}
    for name in map_names:
        map_version = VERSIONS[name]
        maps_db = get_maps_db(map_root, map_version)
        maps_api[name] = NuPlanMap(maps_db, name)
    return maps_api


def parquet_city_name_to_map_api_name(parquet_city_name: str) -> str:
    if "vegas" in parquet_city_name.lower():        # las_vegas in dataset
        return "us-nv-las-vegas-strip"
    elif "sg" in parquet_city_name.lower():         # normal -- does not need this, but do so just for sure
        return "sg-one-north"
    elif "boston" in parquet_city_name.lower():     # normal -- does not need this, but do so just for sure
        return "us-ma-boston"
    elif "pittsburgh" in parquet_city_name.lower(): # normal -- does not need this, but do so just for sure
        return "us-pa-pittsburgh-hazelwood"