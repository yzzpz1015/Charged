# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2025/3/30 20:44
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2025/3/30 20:44

"""
Script for extracting Points of Interest (POI) data from OpenStreetMap.

This script uses OSMnx to fetch POI data for multiple cities, including
amenities, buildings, offices, shops, and other relevant locations.
The data is extracted based on city boundaries and saved to CSV format.

Features:
    - Multi-city POI extraction
    - Comprehensive POI categories
    - Geographic boundary-based filtering
    - CSV output with type and coordinates
"""

import os
import osmnx as ox
import geopandas as gpd
import csv
from tqdm import tqdm


def get_file(path, suffix):
    """
    Get all files with specified suffix from a directory.
    
    Args:
        path (str): Directory path to search in.
        suffix (str): File suffix to filter by.
        
    Returns:
        list: List of filenames matching the suffix.
    """
    input_template_All = []
    f_list = os.listdir(path)
    for i in f_list:
        if os.path.splitext(i)[1] == suffix:
            input_template_All.append(i)
    return input_template_All


# City mapping dictionary with Chinese names and corresponding English info
city_dict = {
    '约翰内斯堡': ['Johannesburg', 'SouthAfrica', 'JHB', 40.291345471713605],
    '圣保罗': ['SaoPaulo', 'Brazil', 'SPO', 34.96669500667566],
    '洛杉矶': ['LosAngeles', 'UnitedStates', 'LOA', 86.70261425391611],
    '墨尔本': ['Melbourne', 'Australia', 'MEL', 32.9575438893481],
    '阿姆斯特丹': ['Amsterdam', 'Netherlands', 'AMS', 47.36294378245436],
    '深圳': ['Shenzhen', 'China', 'SZH', 41.95321426823552],
}

if __name__ == '__main__':
    # Configure OSMnx settings for data extraction
    ox.settings.overpass_settings = '[out:json][timeout:36000][date:"2024-01-02T00:00:00Z"]'
    ox.settings.requests_timeout = 36000

    # Define POI tags to extract from OpenStreetMap
    tags = {
        'amenity': True,      # Public amenities (restaurants, schools, etc.)
        'building': True,     # Buildings
        'craft': True,        # Craft businesses
        'office': True,       # Office buildings
        'landuse': True,      # Land use classifications
        'tourism': True,      # Tourist attractions
        'shop': True,         # Commercial establishments
        'leisure': True       # Leisure facilities
    }

    # Process each city
    for cn_name, city_info in tqdm(city_dict.items(), desc="Processing city data"):
        try:
            city_en = city_info[0]
            city_country = city_info[1]
            city_abr = city_info[2]
            r = city_info[3]
            
            # Load city boundary data
            file_path = get_file("path_to_boundary", '.shp')
            city_gdf = gpd.read_file("path_to_boundary" + file_path[0])
            city_gdf = city_gdf.to_crs(epsg=4326)  # Convert to WGS84

            # Extract polygon for POI query
            polygon = city_gdf.geometry.iloc[0]

            # Fetch POI data from OpenStreetMap
            pois = ox.features.features_from_polygon(polygon, tags)

            # Save POI data to CSV
            csv_filename = "path_to_save_data"
            with open(csv_filename, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["type", "longitude", "latitude"])
                
                # Process each POI
                for idx, row in tqdm(pois.iterrows(), desc=f"{cn_name} POIs", total=len(pois), leave=False):
                    # Determine POI type from available tags
                    for key in ["amenity", "tourism", "shop", "leisure", "office", "man_made", "craft", "landuse"]:
                        if row.get(key) is not None and row.get(key) == row.get(key):
                            poi_type = row.get(key)
                            if key in ['shop', 'office', 'craft']:
                                poi_type = key
                            else:
                                poi_type = row.get(key)
                            break
                        poi_type = "other"
                    
                    # Fallback POI type determination
                    poi_type = row.get("amenity") or row.get("tourism") or row.get("shop") or row.get("leisure") or row.get("office") or row.get("man_made") or row.get("craft") or row.get("landuse")
                    if not poi_type:
                        poi_type = "other"
                    if str(poi_type) == 'nan':
                        poi_type = "other"
                    
                    # Extract coordinates from geometry
                    geom = row.geometry
                    if geom.geom_type == "Point":
                        lon, lat = geom.x, geom.y
                    else:
                        lon, lat = geom.centroid.x, geom.centroid.y
                    
                    writer.writerow([poi_type, lon, lat])
                    
        except Exception as e:
            print(f"Error processing {cn_name}: {e}")