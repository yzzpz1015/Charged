# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2025/3/31 14:59
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2025/3/31 14:59

"""
Weather data collection script for electric vehicle charging demand analysis.

This script fetches historical weather data for multiple cities using the
Visual Crossing Weather API. It collects hourly weather data for the entire
year 2023, including temperature, precipitation, visibility, and other
meteorological parameters that may influence EV charging patterns.

The script handles API rate limiting, error recovery, and data formatting
to create standardized CSV files for each city.

Requirements:
    - Visual Crossing Weather API key
    - requests library for HTTP requests
    - pandas library for data processing
"""

# Standard library imports
import calendar
import time

# Third-party imports
import requests
import pandas as pd


if __name__ == '__main__':
    # API configuration
    API_KEY = "Visual Crossing API Key"  # Replace with actual API key
    base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"

    # City mapping: Chinese names to English API locations
    cities = {
        "Los Angeles"  : "Los Angeles, California, United States",
        "Sao Paulo"    : "Sao Paulo, Sao Paulo, Brazil",
        "Melbourne"    : "Melbourne, Victoria, Australia",
        "Shenzhen"     : "Shenzhen, Guangdong, China",
        "Johannesburg" : "Johannesburg, Gauteng, South Africa",
        "Amsterdam"    : "Amsterdam, North Holland, Netherlands"
    }

    # API request parameters
    unitGroup = "metric"      # Use metric units (Celsius, mm, etc.)
    include = "hours"         # Include hourly data
    contentType = "json"      # Request JSON response format

    # Iterate through each city to collect weather data
    for city_cn, city_en in cities.items():
        hourly_data = []  # List to store hourly weather records
        
        # Collect data for each month of 2023
        for month in range(1, 13):
            # Calculate date range for current month
            start_date = f"2023-{month:02d}-01"  # First day of month
            last_day = calendar.monthrange(2023, month)[1]  # Last day of month
            end_date = f"2023-{month:02d}-{last_day:02d}"  # Last day of month

            # Construct API URL with parameters
            url = f"{base_url}{city_en}/{start_date}/{end_date}?key={API_KEY}&unitGroup={unitGroup}&include={include}&contentType={contentType}"
            print(f"Fetching data for {city_cn} from {start_date} to {end_date}...")

            # Implement retry logic for API rate limiting
            attempt = 0
            max_attempts = 5
            while attempt < max_attempts:
                response = requests.get(url)  # Make API request
                if response.status_code == 200:
                    break  # Success, exit retry loop
                elif response.status_code == 429:
                    # Rate limit exceeded, wait with exponential backoff
                    wait_time = 10 * (attempt + 1)
                    print(f"Rate limit exceeded (429), waiting {wait_time} seconds before retry (attempt {attempt + 1})...")
                    time.sleep(wait_time)
                    attempt += 1
                else:
                    # Other error, log and break
                    print(f"Request error, status code: {response.status_code}")
                    break

            # Check if request was successful
            if response.status_code != 200:
                print(f"Failed to fetch data for {city_cn} from {start_date} to {end_date}, status code: {response.status_code}")
                continue

            # Parse JSON response
            data = response.json()

            # Extract hourly weather data from response
            for day in data.get("days", []):
                for hour in day.get("hours", []):
                    # Create timestamp for current hour
                    dt = f"{day.get('datetime')} {hour.get('datetime')}"
                    record = {"time": dt}  # Initialize record with timestamp
                    
                    # Add all weather parameters except datetime
                    for key, value in hour.items():
                        if key != "datetime":
                            record[key] = value
                    
                    hourly_data.append(record)  # Add record to collection

            # Rate limiting: wait between requests to avoid API limits
            time.sleep(2)

        # Convert collected data to DataFrame and save to CSV
        df = pd.DataFrame(hourly_data)  # Create pandas DataFrame
        csv_filename = f"{city_cn}.csv"  # Generate filename
        df.to_csv(csv_filename, index=False)  # Save to CSV without index
        print(f"Data for {city_cn} saved as {csv_filename}")
