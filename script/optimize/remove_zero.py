# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2025/04/10 23:10
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2025/04/10 23:10

"""
Script for removing zero-value clusters from aggregated charging data.

This script identifies and removes clusters (sites) that have zero values
across all time periods for duration and volume metrics. It also updates
related data files including pricing, distance, and site information.

Features:
    - Zero cluster detection across multiple metrics
    - Comprehensive data cleanup
    - Related file updates
    - Site and charger information updates
"""

from tqdm import tqdm
import pandas as pd

# City mapping dictionary with Chinese names and corresponding English info
city_dict = {
    '圣保罗': ['SaoPaulo', 'Brazil', 'SPO', 34.96669500667566],
    '约翰内斯堡': ['Johannesburg', 'SouthAfrica', 'JHB', 40.291345471713605],
    '洛杉矶': ['LosAngeles', 'UnitedStates', 'LOA', 86.70261425391611],
    '墨尔本': ['Melbourne', 'Australia', 'MEL', 32.9575438893481],
    '阿姆斯特丹': ['Amsterdam', 'Netherlands', 'AMS', 47.36294378245436],
    '深圳': ['Shenzhen', 'China', 'SZH', 41.95321426823552],
}


def filter_zero_clusters(aggregated_dict):
    """
    Identify clusters that have zero values across all time periods.
    
    Args:
        aggregated_dict (dict): Dictionary containing DataFrames with cluster data.
        
    Returns:
        set: Set of cluster IDs that have zero values across all metrics.
    """
    zero_clusters = None
    for key, df in aggregated_dict.items():
        # Find columns where min and max are both zero
        clusters_zero = {col for col in df.columns if df[col].min() == 0 and df[col].max() == 0}
        if zero_clusters is None:
            zero_clusters = clusters_zero
        else:
            zero_clusters = zero_clusters.intersection(clusters_zero)
    return zero_clusters


if __name__ == '__main__':
    # Process each city
    for city_cn, city_info in tqdm(city_dict.items(), desc="Processing city data"):
        city_en = city_info[0]
        city_country = city_info[1]
        city_abr = city_info[2]
        r = city_info[3]
        data_path = "path_to_data"

        # Load duration and volume data
        duration = pd.read_csv(f'{data_path}duration.csv', index_col=0)
        volume = pd.read_csv(f'{data_path}volume.csv', index_col=0)

        # Create dictionary of aggregated results
        aggregated_results = {
            'duration': duration,
            'volume': volume,
        }
        
        # Identify zero clusters
        zero_clusters = filter_zero_clusters(aggregated_results)
        
        # Remove zero clusters from main metrics
        for key in aggregated_results:
            aggregated_results[key] = aggregated_results[key].drop(columns=zero_clusters, errors='ignore')
            aggregated_results[key].to_csv(f'{data_path}{key}.csv', index=True)

        # Update electricity price data
        e_price = pd.read_csv(f'{data_path}e_price.csv', index_col=0)
        e_price = e_price.drop(columns=zero_clusters, errors='ignore')
        e_price.to_csv(f'{data_path}e_price.csv', index=True)
        
        # Update service price data
        s_price = pd.read_csv(f'{data_path}s_price.csv', index_col=0)
        s_price = e_price.drop(columns=zero_clusters, errors='ignore')
        s_price.to_csv(f'{data_path}s_price.csv', index=True)

        # Update distance matrix
        distance = pd.read_csv(f'{data_path}distance.csv', index_col=0)
        distance.index = distance.index.astype(str)
        distance = distance.drop(index=zero_clusters, errors='ignore')
        distance = distance.drop(columns=zero_clusters, errors='ignore')
        distance.to_csv(f'{data_path}distance.csv', index=True)

        # Update cluster information
        inf = pd.read_csv(f'{data_path}inf.csv', index_col=0)
        inf['ID'] = inf['ID'].astype(str)
        inf = inf[~inf['ID'].isin(zero_clusters)]
        inf.to_csv(f'{data_path}inf.csv')

        # Update general information
        info = pd.read_csv(f'{data_path}info.csv', index_col=None)

        # Update charger information
        chargers = pd.read_csv(f'{data_path}chargers.csv', index_col=None)
        chargers['site_id'] = chargers['site_id'].astype(str)
        chargers = chargers[~chargers['site_id'].isin(zero_clusters)]
        chargers.to_csv(f'{data_path}chargers.csv', index=False)

        # Update site information
        sites = pd.read_csv(f'{data_path}sites.csv', index_col=None)
        sites['site_id'] = sites['site_id'].astype(str)
        sites = sites[~sites['site_id'].isin(zero_clusters)]
        sites.to_csv(f'{data_path}sites.csv', index=False)

        # Update summary statistics in info file
        total_chargers = chargers.shape[0]
        total_sites = sites.shape[0]
        info['total_chargers'] = total_chargers
        info['total_sites'] = total_sites
        avg_power = chargers['avg_power'].mean()
        info['avg_power'] = avg_power
        info.to_csv(f'{data_path}info.csv', index=False)

