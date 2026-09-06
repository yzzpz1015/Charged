# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2025/4/1 10:20
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2025/4/1 10:20

"""
Script for temporal aggregation of electric vehicle charging data.

This script processes raw charging session data and aggregates it by hour
and site to create time series datasets. It handles large datasets
using chunked processing and calculates key metrics including duration,
energy consumption, and occupancy rates.

Features:
    - Chunked processing for large datasets
    - Temporal aggregation by hour
    - Site-wise aggregation
    - Occupancy rate calculation
    - Multiple output formats (duration, volume, occupancy)
"""

import os
import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':
    # Define cities to process
    cities = ['Sao Paulo', 'Melbourne', 'Los Angeles', 'Shenzhen', 'Johannesburg', 'Amsterdam']
    chunksize = 10000000  # Process data in 10M row chunks

    # Process each city
    for city in tqdm(cities, desc="Processing city data"):
        file_path = "path_to_data"
        agg_list = []

        # Process data in chunks to handle large files
        for chunk in tqdm(pd.read_csv(file_path, chunksize=chunksize), desc=f"{city} chunk processing", leave=False):
            # Convert time column to datetime
            chunk['time_new'] = pd.to_datetime(chunk['time_new'])
            
            # Filter for 2023 data only
            chunk = chunk[(chunk['time_new'] >= '2023-01-01') & (chunk['time_new'] < '2024-01-01')]
            if chunk.empty:
                continue

            # Create hourly time bins and non-zero indicators
            chunk['time_hour'] = chunk['time_new'].dt.floor('H')
            chunk['nonzero'] = (chunk['duration'] != 0).astype(int)

            # Aggregate by hour and site
            agg_chunk = chunk.groupby(['time_hour', 'site_id'], as_index=False).agg(
                duration=('duration', 'sum'),      # Total charging duration
                kwh=('kwh', 'sum'),               # Total energy consumption
                count=('nonzero', 'sum')          # Number of charging sessions
            )
            agg_list.append(agg_chunk)

        # Check if any data was processed
        if not agg_list:
            print(f"{city} has no data for 2023")
            continue

        # Combine all chunks and perform final aggregation
        agg_df = pd.concat(agg_list, ignore_index=True)
        agg_df = agg_df.groupby(['time_hour', 'site_id'], as_index=False).agg({
            'duration': 'sum',
            'kwh': 'sum',
            'count': 'sum'
        })

        # Calculate occupancy rate (assuming 12 sessions per hour capacity)
        expected_count = 12
        agg_df['occupancy'] = (agg_df['count'] / expected_count).clip(upper=1)

        # Create pivot tables for different metrics
        pivot_duration = agg_df.pivot(index='time_hour', columns='site_id', values='duration')
        pivot_kwh = agg_df.pivot(index='time_hour', columns='site_id', values='kwh')
        pivot_occupancy = agg_df.pivot(index='time_hour', columns='site_id', values='occupancy')

        # Sort by time index
        pivot_duration.sort_index(inplace=True)
        pivot_kwh.sort_index(inplace=True)
        pivot_occupancy.sort_index(inplace=True)

        # Fill missing values with zeros
        pivot_duration.fillna(0, inplace=True)
        pivot_kwh.fillna(0, inplace=True)
        pivot_occupancy.fillna(0, inplace=True)

        # Reset index for CSV output
        pivot_duration.reset_index(inplace=True)
        pivot_kwh.reset_index(inplace=True)
        pivot_occupancy.reset_index(inplace=True)

        # Clean column names (remove time column name)
        duration_cols = pivot_duration.columns.tolist()
        duration_cols[0] = ""
        pivot_duration.columns = duration_cols

        kwh_cols = pivot_kwh.columns.tolist()
        kwh_cols[0] = ""
        pivot_kwh.columns = kwh_cols

        occupancy_cols = pivot_occupancy.columns.tolist()
        occupancy_cols[0] = ""
        pivot_occupancy.columns = occupancy_cols

        # Create output directory and save files
        os.makedirs("path_to_save_data", exist_ok=True)

        pivot_duration.to_csv(os.path.join("path_to_save_data", "duration.csv"), index=False)
        pivot_kwh.to_csv(os.path.join("path_to_save_data", "volume.csv"), index=False)
        pivot_occupancy.to_csv(os.path.join("path_to_save_data", "occupancy.csv"), index=False)
