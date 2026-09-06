# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2025/4/1 10:20
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2025/4/1 10:20

"""
Script for detecting and imputing anomalies in time series charging data.

This script implements data quality improvement techniques for electric vehicle
charging time series data. It detects and fixes continuous zero sequences and
outliers using statistical methods and interpolation techniques.

Features:
    - Continuous zero sequence detection and imputation
    - Outlier detection using IQR method
    - Time-based interpolation
    - Forward/backward fill for missing values
    - Comprehensive data preprocessing pipeline
"""

from tqdm import tqdm
import pandas as pd
import numpy as np

# City mapping dictionary with Chinese names and corresponding English info
city_dict = {
    '圣保罗'  : ['SaoPaulo', 'Brazil', 'SPO',34.96669500667566],
    '阿姆斯特丹': ['Amsterdam', 'Netherlands', 'AMS',47.36294378245436],
    '约翰内斯堡': ['Johannesburg', 'SouthAfrica', 'JHB',40.291345471713605],
    '洛杉矶'  : ['LosAngeles', 'UnitedStates', 'LOA',86.70261425391611],
    '墨尔本'  : ['Melbourne', 'Australia', 'MEL',32.9575438893481],
    '深圳'   : ['Shenzhen', 'China', 'SZH',41.95321426823552],
}


def detect_and_fix_zeros(series, threshold):
    """
    Detect and fix continuous zero sequences in time series data.
    
    This function identifies sequences of zeros that exceed a threshold length
    and replaces them with interpolated values using time-based interpolation.
    
    Args:
        series (pd.Series): Time series data to process.
        threshold (int): Minimum length of zero sequence to consider as anomaly.
        
    Returns:
        pd.Series: Processed time series with fixed zero sequences.
    """
    # Identify zero values
    zero_mask = series == 0
    
    # Group consecutive zeros
    group = (zero_mask != zero_mask.shift()).cumsum()
    group_size = zero_mask.groupby(group).transform('sum')

    # Replace long zero sequences with NaN for interpolation
    series.loc[(zero_mask) & (group_size >= threshold)] = np.nan

    # Apply time-based interpolation
    series = series.interpolate(method='time')
    
    # Fill remaining gaps with forward/backward fill
    series = series.fillna(method='ffill', limit=threshold)
    series = series.fillna(method='bfill', limit=threshold)
    
    return series


def detect_and_repair(df, weight=4):
    """
    Detect and repair outliers in time series data using IQR method.
    
    This function identifies outliers using the Interquartile Range (IQR) method
    and replaces them with interpolated values.
    
    Args:
        df (pd.DataFrame): DataFrame containing time series data.
        weight (float): Multiplier for IQR to define outlier bounds (default: 4).
        
    Returns:
        pd.DataFrame: DataFrame with outliers replaced by interpolated values.
    """
    df_fixed = df.copy()
    
    # Process each column
    for col in tqdm(df.columns, desc="Processing columns"):
        s = df[col]
        
        # Calculate IQR and bounds
        Q1 = s.quantile(0.25)
        Q3 = s.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = max(0, Q1 - weight * IQR)  # Ensure lower bound is non-negative
        upper_bound = Q3 + weight * IQR

        # Identify outliers
        is_outlier = (s < lower_bound) | (s > upper_bound)

        # Replace outliers with interpolated values
        if is_outlier.any():
            s[is_outlier] = np.nan
            s.interpolate(method='linear', inplace=True, limit_direction='both')
            s.fillna(method='ffill', inplace=True)
            s.fillna(method='bfill', inplace=True)

        df_fixed[col] = s

    return df_fixed


def read_and_preprocess(path):
    """
    Read and preprocess time series data with comprehensive cleaning.
    
    This function applies the complete data preprocessing pipeline including
    zero sequence detection, outlier detection, and final gap filling.
    
    Args:
        path (str): Path to the CSV file containing time series data.
        
    Returns:
        pd.DataFrame: Cleaned and preprocessed time series data.
    """
    # Load data
    data = pd.read_csv(path, index_col=0)
    data.index = pd.to_datetime(data.index)
    
    # Fix continuous zeros in each column
    for col in tqdm(data.columns, desc="Fixing continuous zeros"):
        data[col] = detect_and_fix_zeros(data[col], threshold=24)
        data[col].fillna(0, inplace=True)
    
    # Detect and repair outliers
    data = detect_and_repair(data)
    
    # Final gap filling for any remaining missing values
    if data.isnull().values.any():
        data = data.fillna(method='ffill', limit=24).fillna(method='bfill', limit=24)
        data.fillna(0, inplace=True)
    
    return data