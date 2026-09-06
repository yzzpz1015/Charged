# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2025/4/1 10:20
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2025/4/1 10:20

"""
Script for updating and aggregating site information across multiple cities.

This script processes site data from different cities, computes cluster metrics,
and generates aggregated information files. It handles both regular and zero-removed
datasets, providing comprehensive site clustering and metric calculations.

Key features:
    - Geographic coordinate conversion and projection
    - Cluster perimeter and area calculations
    - Site aggregation based on file type
    - Support for multiple cities and data formats
"""

import json
import os
from tqdm import tqdm
import pandas as pd
from geopy.distance import geodesic
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.ops import transform
import pyproj

# City mapping dictionary with Chinese names and corresponding English info
city_dict = {
    '约翰内斯堡': ['Johannesburg', 'SouthAfrica', 'JHB', 40.291345471713605],
    '圣保罗': ['SaoPaulo', 'Brazil', 'SPO', 34.96669500667566],
    '洛杉矶': ['LosAngeles', 'UnitedStates', 'LOA', 86.70261425391611],
    '墨尔本': ['Melbourne', 'Australia', 'MEL', 32.9575438893481],
    '阿姆斯特丹': ['Amsterdam', 'Netherlands', 'AMS', 47.36294378245436],
    '深圳': ['Shenzhen', 'China', 'SZH', 41.95321426823552],
}

# Weather condition mapping for categorical encoding
weather_conditions = {
    'Clear': 0,
    'Overcast': 1,
    'Partially cloudy': 2,
    'Rain': 3,
    'Rain, Fog': 4,
    'Rain, Overcast': 5,
    'Rain, Partially cloudy': 6,
    'Snow': 7,
    'Snow, Fog': 8,
    'Snow, Partially cloudy': 9,
    'Snow, Rain': 10,
    'Snow, Rain, Overcast': 11,
    'Snow, Rain, Partially cloudy': 12
}


def convert_geometry_to_meters(geom):
    """
    Convert geometry from WGS84 (EPSG:4326) to Web Mercator (EPSG:3857) projection.
    
    Args:
        geom: Shapely geometry object in WGS84 coordinates
        
    Returns:
        Shapely geometry object in Web Mercator coordinates
    """
    project = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform
    return transform(project, geom)

def compute_cluster_metrics(cluster_points, r):
    """
    Compute perimeter and area metrics for a cluster of points.
    
    Args:
        cluster_points (list): List of (longitude, latitude) tuples
        r (float): Radius for buffer calculation in kilometers
        
    Returns:
        tuple: (perimeter, area) in meters and square meters respectively
    """
    if len(cluster_points) == 0:
        return 0, 0
    circles = [Point(lon, lat).buffer(r) for lon, lat in cluster_points]
    union_geom = unary_union(circles)
    proj_geom = convert_geometry_to_meters(union_geom)

    return proj_geom.length, proj_geom.area


def filter_zero_clusters(aggregated_dict):
    """
    Find clusters that have zero values across all dataframes.
    
    Args:
        aggregated_dict (dict): Dictionary of aggregated dataframes
        
    Returns:
        set: Set of cluster IDs that have zero values in all dataframes
    """
    zero_clusters = None
    for key, df in aggregated_dict.items():
        clusters_zero = {col for col in df.columns if df[col].min() == 0 and df[col].max() == 0}
        if zero_clusters is None:
            zero_clusters = clusters_zero
        else:
            zero_clusters = zero_clusters.intersection(clusters_zero)
    return zero_clusters

def read_and_aggregate(path, site_cluster):
    """
    Read data file and aggregate by site clusters.
    
    Args:
        path (str): Path to the data file
        site_cluster (dict): Site clustering information
        
    Returns:
        pd.DataFrame: Aggregated data grouped by clusters
    """
    data = pd.read_csv(path, index_col=0)
    data.index = pd.to_datetime(data.index)

    file_name = os.path.splitext(os.path.basename(path))[0]

    site_to_cluster = {}
    for cluster_id, details in site_cluster.items():
        site_ids = details[0]
        for site_id in site_ids:
            site_to_cluster[site_id] = cluster_id

    valid_sites = [site_id for site_id in data.columns if site_id in site_to_cluster]
    data = data[valid_sites]

    data.columns = [site_to_cluster[site_id] for site_id in data.columns]

    # Determine aggregation rule based on filename
    if 'duration' in file_name or 'volume' in file_name:
        aggregation_func = 'sum'
    elif 'occupancy' in file_name or 'fee' in file_name:
        aggregation_func = 'mean'

    aggregated_data = data.groupby(data.columns, axis=1).agg(aggregation_func)

    return aggregated_data


if __name__ == '__main__':
    for remove_zero_clusters in [False,True]:
    # for remove_zero_clusters in [False]:
        for city_cn, city_info in tqdm(city_dict.items(), desc="Processing city data"):
            city_en = city_info[0]
            city_country = city_info[1]
            city_abr = city_info[2]
            r = city_info[3]
            data_path = "path_to_data"
            ori_path = "path_to_data"
            if remove_zero_clusters:
                save_path = "path_to_save_data"
            else:
                save_path = "path_to_save_data"
            os.makedirs(save_path, exist_ok=True)

            lanlon = pd.read_csv(f'{data_path}/lanlon.csv')
            lanlon['site_id'] = lanlon['site_id'].astype(str)
            lanlon = lanlon[['site_id', 'longitude', 'latitude']]
            lanlon = lanlon.groupby('site_id', as_index=True).mean()

            with open(os.path.join(data_path, 'site_cluster.json'), 'r', encoding='utf-8') as f:
                site_cluster = json.load(f)

            inf_list = []
            for cluster_id, details in site_cluster.items():
                site_ids = details[0]
                cluster_lon = details[1]
                cluster_lat = details[2]
                site_ids = [str(sid) for sid in site_ids]
                cluster_data = lanlon.loc[lanlon.index.intersection(site_ids)]
                charger_num = len(cluster_data)
                points = list(cluster_data[['longitude', 'latitude']].itertuples(index=False, name=None))
                perimeter, area = compute_cluster_metrics(points, r/111/1000)
                inf_list.append({
                    'ID': cluster_id,
                    'longitude': cluster_lon,
                    'latitude': cluster_lat,
                    'charger_num': charger_num,
                    'perimeter': perimeter,
                    'area': area
                })
            inf_df = pd.DataFrame(inf_list)
            inf_df.to_csv(os.path.join(save_path, 'inf.csv'))
