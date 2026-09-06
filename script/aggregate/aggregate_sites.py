# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2025/4/1 10:20
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2025/4/1 10:20

"""
Script for clustering charging sites based on geographic proximity.

This script uses DBSCAN clustering algorithm to group charging sites
that are within a specified distance threshold. It processes site data
from multiple cities and generates cluster information including site IDs
and cluster centroids.

Key features:
    - Geographic clustering using DBSCAN algorithm
    - Support for multiple cities with different distance thresholds
    - Automatic centroid calculation for each cluster
    - JSON output format for cluster information
"""

import json
import os
from tqdm import tqdm
import pandas as pd
from sklearn.cluster import DBSCAN

# City mapping dictionary with Chinese names and corresponding English info
city_dict = {
    '阿姆斯特丹': ['Amsterdam', 'Netherlands', 'AMS',47.36294378245436],
    '约翰内斯堡': ['Johannesburg', 'SouthAfrica', 'JHB',40.291345471713605],
    '洛杉矶'  : ['LosAngeles', 'UnitedStates', 'LOA',86.70261425391611],
    '墨尔本'  : ['Melbourne', 'Australia', 'MEL',32.9575438893481],
    '圣保罗'  : ['SaoPaulo', 'Brazil', 'SPO',34.96669500667566],
    '深圳'   : ['Shenzhen', 'China', 'SZH',41.95321426823552],
}

def cluster_site(data, distance):
    """
    Cluster sites based on geographic proximity using DBSCAN.
    
    Args:
        data (pd.DataFrame): DataFrame containing site data with 'latitude',
                           'longitude', and 'site_id' columns
        distance (float): Distance threshold in kilometers for clustering
        
    Returns:
        dict: Dictionary with cluster information where each key is a cluster ID
              and value is [site_ids, centroid_longitude, centroid_latitude]
    """
    # Convert distance from km to degrees (approximate conversion)
    dbscan = DBSCAN(eps=distance / 111 / 1000, min_samples=1)
    X = data[['latitude', 'longitude']]
    data['cluster'] = dbscan.fit_predict(X)
    
    return_dict = dict()
    for i in set(data['cluster']):
        return_dict[i] = [[], 0, 0]
    
    # Aggregate site IDs and calculate sum of coordinates for centroid
    for index, row in data.iterrows():
        return_dict[int(row['cluster'])][0].append(row['site_id'])
        return_dict[int(row['cluster'])][1] += row['longitude']
        return_dict[int(row['cluster'])][2] += row['latitude']
    
    # Calculate centroid coordinates by averaging
    for i in set(data['cluster']):
        return_dict[i][1] = return_dict[i][1] / len(return_dict[i][0])
        return_dict[i][2] = return_dict[i][2] / len(return_dict[i][0])
    
    return return_dict

if __name__ == '__main__':
    for city_cn,city_info in tqdm(city_dict.items(), desc="Processing city data"):
        city_en=city_info[0]
        city_country=city_info[1]
        city_abr=city_info[2]
        r=city_info[3]
        data_path="path_to_data"
        ori_path="path_to_data"
        save_path="path_to_save_data"

        data = pd.read_csv(f'{data_path}/lanlon.csv')

        site_cluster=cluster_site(data,r)
        result_file = os.path.join(data_path, 'site_cluster.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(site_cluster, f, ensure_ascii=False, indent=4)


