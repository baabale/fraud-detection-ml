"""
Utility functions for data loading, preprocessing, and feature engineering.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def load_data_from_parquet(file_path):
    """
    Load data from a Parquet file.
    
    Args:
        file_path (str): Path to the Parquet file
        
    Returns:
        DataFrame: Pandas DataFrame containing the data
    """
    return pd.read_parquet(file_path)

def load_data_from_csv(file_path, **kwargs):
    """
    Load data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        **kwargs: Additional arguments to pass to pd.read_csv
        
    Returns:
        DataFrame: Pandas DataFrame containing the data
    """
    return pd.read_csv(file_path, **kwargs)

def identify_column_types(df):
    """
    Identify numeric, categorical, and datetime columns in a DataFrame.
    
    Args:
        df (DataFrame): Input DataFrame
        
    Returns:
        tuple: Lists of numeric, categorical, and datetime column names
    """
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    return numeric_cols, categorical_cols, datetime_cols

def create_preprocessing_pipeline(numeric_cols, categorical_cols):
    """
    Create a scikit-learn preprocessing pipeline for numeric and categorical features.
    
    Args:
        numeric_cols (list): List of numeric column names
        categorical_cols (list): List of categorical column names
        
    Returns:
        ColumnTransformer: Preprocessing pipeline
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    return preprocessor

def extract_time_features(df, timestamp_col):
    """
    Extract time-based features from a timestamp column.
    
    Args:
        df (DataFrame): Input DataFrame
        timestamp_col (str): Name of the timestamp column
        
    Returns:
        DataFrame: DataFrame with additional time features
    """
    df = df.copy()
    
    # Convert to datetime if not already
    if df[timestamp_col].dtype != 'datetime64[ns]':
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # Extract time components
    df[f'{timestamp_col}_hour'] = df[timestamp_col].dt.hour
    df[f'{timestamp_col}_day'] = df[timestamp_col].dt.day
    df[f'{timestamp_col}_dayofweek'] = df[timestamp_col].dt.dayofweek
    df[f'{timestamp_col}_month'] = df[timestamp_col].dt.month
    df[f'{timestamp_col}_year'] = df[timestamp_col].dt.year
    df[f'{timestamp_col}_quarter'] = df[timestamp_col].dt.quarter
    
    # Create is_weekend feature
    df[f'{timestamp_col}_is_weekend'] = df[f'{timestamp_col}_dayofweek'].isin([5, 6]).astype(int)
    
    # Create time of day category
    hour = df[f'{timestamp_col}_hour']
    conditions = [
        (hour >= 0) & (hour < 6),
        (hour >= 6) & (hour < 12),
        (hour >= 12) & (hour < 18),
        (hour >= 18) & (hour < 24)
    ]
    categories = ['night', 'morning', 'afternoon', 'evening']
    df[f'{timestamp_col}_time_of_day'] = np.select(conditions, categories)
    
    return df

def create_amount_features(df, amount_col):
    """
    Create features based on transaction amount.
    
    Args:
        df (DataFrame): Input DataFrame
        amount_col (str): Name of the amount column
        
    Returns:
        DataFrame: DataFrame with additional amount features
    """
    df = df.copy()
    
    # Log transform (handle zeros/negatives by adding a small constant)
    df[f'{amount_col}_log'] = np.log1p(df[amount_col].clip(lower=0))
    
    # Binning
    df[f'{amount_col}_bin'] = pd.qcut(
        df[amount_col].clip(lower=0), 
        q=10, 
        labels=False, 
        duplicates='drop'
    )
    
    return df

def create_aggregated_features(df, group_cols, agg_cols, windows=None):
    """
    Create aggregated features by grouping on specified columns.
    
    Args:
        df (DataFrame): Input DataFrame
        group_cols (list): Columns to group by (e.g., ['user_id', 'merchant'])
        agg_cols (list): Columns to aggregate (e.g., ['amount'])
        windows (list, optional): Time windows for rolling aggregations in days
        
    Returns:
        DataFrame: DataFrame with additional aggregated features
    """
    df = df.copy()
    result = df.copy()
    
    # Sort by timestamp if available
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp')
    
    # Static aggregations
    for group_col in group_cols:
        for agg_col in agg_cols:
            # Group by and calculate aggregations
            aggs = df.groupby(group_col)[agg_col].agg(['mean', 'std', 'min', 'max', 'count'])
            aggs.columns = [f'{group_col}_{agg_col}_{agg}' for agg in aggs.columns]
            
            # Merge back to the original dataframe
            result = result.merge(aggs, left_on=group_col, right_index=True, how='left')
    
    # Time-window based aggregations
    if windows and 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        for window in windows:
            for group_col in group_cols:
                for agg_col in agg_cols:
                    # Create a copy with timestamp as index
                    temp_df = df.set_index('timestamp')
                    
                    # Group by and calculate rolling aggregations
                    rolling_aggs = temp_df.groupby(group_col)[agg_col].rolling(f'{window}D').agg(['mean', 'std', 'count'])
                    rolling_aggs.columns = [f'{group_col}_{agg_col}_{window}d_{agg}' for agg in rolling_aggs.columns]
                    
                    # Reset index and merge back
                    rolling_aggs = rolling_aggs.reset_index()
                    result = result.merge(
                        rolling_aggs, 
                        on=[group_col, 'timestamp'], 
                        how='left'
                    )
    
    return result

def create_velocity_features(df, entity_col, time_col, amount_col=None, windows=[1, 7, 30]):
    """
    Create velocity features (transaction frequency and amount) for entities.
    
    Args:
        df (DataFrame): Input DataFrame
        entity_col (str): Column representing the entity (e.g., 'user_id')
        time_col (str): Column containing timestamps
        amount_col (str, optional): Column containing transaction amounts
        windows (list): Time windows in days
        
    Returns:
        DataFrame: DataFrame with additional velocity features
    """
    df = df.copy()
    
    # Ensure timestamp column is datetime
    df[time_col] = pd.to_datetime(df[time_col])
    
    # Sort by entity and time
    df = df.sort_values([entity_col, time_col])
    
    # Calculate time difference between consecutive transactions for each entity
    df[f'{entity_col}_time_diff'] = df.groupby(entity_col)[time_col].diff().dt.total_seconds() / 3600  # in hours
    
    result = df.copy()
    
    # Calculate transaction frequency and amount velocity for different time windows
    for window in windows:
        window_hours = window * 24  # Convert days to hours
        
        # Create a window for each transaction
        df['window_end'] = df[time_col]
        df['window_start'] = df[time_col] - pd.Timedelta(days=window)
        
        # For each entity and transaction, count transactions in the window
        for entity in df[entity_col].unique():
            entity_df = df[df[entity_col] == entity]
            
            # For each transaction, count previous transactions in the window
            for idx, row in entity_df.iterrows():
                window_txns = entity_df[
                    (entity_df[time_col] >= row['window_start']) & 
                    (entity_df[time_col] < row['window_end'])
                ]
                
                # Update the result DataFrame
                result.loc[idx, f'{entity_col}_txn_count_{window}d'] = len(window_txns)
                
                # Calculate amount velocity if amount column is provided
                if amount_col:
                    result.loc[idx, f'{entity_col}_amount_sum_{window}d'] = window_txns[amount_col].sum()
                    result.loc[idx, f'{entity_col}_amount_mean_{window}d'] = window_txns[amount_col].mean()
    
    # Drop temporary columns
    if 'window_start' in result.columns:
        result = result.drop(['window_start', 'window_end'], axis=1)
    
    return result

def handle_class_imbalance(X, y, method='undersample', ratio=1.0):
    """
    Handle class imbalance in the dataset.
    
    Args:
        X (array): Feature matrix
        y (array): Target vector
        method (str): Method to use ('undersample', 'oversample', or 'smote')
        ratio (float): Desired ratio of minority to majority class
        
    Returns:
        tuple: Balanced X and y
    """
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.over_sampling import RandomOverSampler, SMOTE
    
    if method == 'undersample':
        sampler = RandomUnderSampler(sampling_strategy=ratio, random_state=42)
    elif method == 'oversample':
        sampler = RandomOverSampler(sampling_strategy=ratio, random_state=42)
    elif method == 'smote':
        sampler = SMOTE(sampling_strategy=ratio, random_state=42)
    else:
        raise ValueError(f"Unknown sampling method: {method}")
    
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    
    return X_resampled, y_resampled
