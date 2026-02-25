#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("augment_log.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('augment')

# ===== Fonctions de création des caractéristiques =====
def create_time_features(df):
    df = df.copy()
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_month'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['quarter'] = df['timestamp'].dt.quarter
    df['year'] = df['timestamp'].dt.year
    df['is_night'] = df['hour'].apply(lambda x: 1 if x < 6 or x >= 20 else 0)
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    return df

def create_rolling_features(df, window_sizes=[5, 10, 30], group_by='equipment_id'):
    df = df.copy().sort_values(by=[group_by, 'timestamp'])
    numeric_cols = ['temperature', 'vibration', 'pressure', 'current']
    for window in window_sizes:
        for col in numeric_cols:
            df[f'{col}_rolling_mean_{window}'] = df.groupby(group_by)[col].transform(lambda x: x.rolling(window, min_periods=1).mean())
            df[f'{col}_rolling_std_{window}'] = df.groupby(group_by)[col].transform(lambda x: x.rolling(window, min_periods=1).std())
            df[f'{col}_rolling_min_{window}'] = df.groupby(group_by)[col].transform(lambda x: x.rolling(window, min_periods=1).min())
            df[f'{col}_rolling_max_{window}'] = df.groupby(group_by)[col].transform(lambda x: x.rolling(window, min_periods=1).max())
    return df

def create_lag_features(df, lag_periods=[1, 3, 5, 10], group_by='equipment_id'):
    df = df.copy().sort_values(by=[group_by, 'timestamp'])
    numeric_cols = ['temperature', 'vibration', 'pressure', 'current']
    for lag in lag_periods:
        for col in numeric_cols:
            df[f'{col}_lag_{lag}'] = df.groupby(group_by)[col].shift(lag)
            df[f'{col}_change_{lag}'] = df[col] - df[f'{col}_lag_{lag}']
            df[f'{col}_pct_change_{lag}'] = df.groupby(group_by)[col].pct_change(periods=lag)
    return df

def add_failure_indicators(sensor_df, failure_df, time_window=24):
    sensor_df = sensor_df.copy()
    sensor_df['failure_soon'] = 0
    sensor_df['time_to_failure'] = pd.NA
    sensor_df['next_failure_type'] = pd.NA
    for _, failure in failure_df.iterrows():
        equipment_id = failure['equipment_id']
        failure_time = failure['failure_timestamp']
        failure_type = failure['failure_type']
        window_mask = (
            (sensor_df['equipment_id'] == equipment_id) &
            (sensor_df['timestamp'] <= failure_time) &
            (sensor_df['timestamp'] >= failure_time - pd.Timedelta(hours=time_window))
        )
        if window_mask.any():
            sensor_df.loc[window_mask, 'failure_soon'] = 1
            sensor_df.loc[window_mask, 'time_to_failure'] = (
                (failure_time - sensor_df.loc[window_mask, 'timestamp']).dt.total_seconds() / 3600
            )
            sensor_df.loc[window_mask, 'next_failure_type'] = failure_type
    return sensor_df

def create_component_health_features(sensor_df, failure_df):
    sensor_df = sensor_df.copy()
    sensor_df['days_since_last_failure'] = np.inf
    sensor_df['failures_count_last_30days'] = 0
    for equipment_id in sensor_df['equipment_id'].unique():
        equip_failures = failure_df[failure_df['equipment_id'] == equipment_id]
        if len(equip_failures) == 0:
            continue
        equip_mask = sensor_df['equipment_id'] == equipment_id
        equip_sensors = sensor_df[equip_mask]
        for idx, sensor_row in equip_sensors.iterrows():
            current_time = sensor_row['timestamp']
            prev_failures = equip_failures[equip_failures['failure_timestamp'] < current_time]
            if len(prev_failures) > 0:
                last_failure_time = prev_failures['failure_timestamp'].max()
                sensor_df.loc[idx, 'days_since_last_failure'] = (current_time - last_failure_time).days
                sensor_df.loc[idx, 'failures_count_last_30days'] = prev_failures[prev_failures['failure_timestamp'] >= current_time - pd.Timedelta(days=30)].shape[0]
    return sensor_df

def feature_scaling(df, method='standard', exclude_cols=None):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if exclude_cols:
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Méthode non reconnue. Utilisez 'standard' ou 'minmax'.")
    # Remplacer inf et NaN par 0 avant scaling
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0))
    return df

def create_interaction_features(df):
    df = df.copy()
    base_cols = ['temperature', 'vibration', 'pressure', 'current']
    for i, col1 in enumerate(base_cols):
        for col2 in base_cols[i+1:]:
            df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
    for col1 in base_cols:
        for col2 in base_cols:
            if col1 != col2:
                df[f'{col1}_div_{col2}'] = df[col1] / df[col2].replace(0, np.nan)
    return df

def plot_feature_importances(df, target_col='failure_soon', output_path=None):
    # Sélection uniquement des colonnes numériques
    numeric_df = df.select_dtypes(include=['number'])
    if target_col not in numeric_df.columns:
        logger.warning(f"{target_col} n'est pas numérique. Impossible de tracer les corrélations.")
        return
    corr_with_target = numeric_df.corr()[target_col].sort_values(ascending=False)
    top_corr = corr_with_target.drop(target_col)[0:20]
    plt.figure(figsize=(12, 8))
    sns.barplot(x=top_corr.values, y=top_corr.index)
    plt.title(f'Top 20 des caractéristiques corrélées avec {target_col}')
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

# ===== Fonction principale =====
def augment_data(
    input_dir=r"E:\CTéléchargements\Maintenance_Predictive\ML project sprint (1)\data (1)\processed (1)\cleaned_data (1)",
    output_dir=r"E:\CTéléchargements\Maintenance_Predictive\ML project sprint (1)\data (1)\processed (1)\augmented_data (1)"
):
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        viz_dir = os.path.join(output_dir, 'visualizations')
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)

        sensor_data_path = os.path.join(input_dir, 'clean_sensor_data.CSV')
        failure_data_path = os.path.join(input_dir, 'clean_failure_data.CSV')

        sensor_df = pd.read_csv(sensor_data_path, parse_dates=['timestamp'])
        failure_df = pd.read_csv(failure_data_path, parse_dates=['failure_timestamp'])

        # Création des caractéristiques
        sensor_df = create_time_features(sensor_df)
        sensor_df = create_rolling_features(sensor_df)
        sensor_df = create_lag_features(sensor_df)
        sensor_df = add_failure_indicators(sensor_df, failure_df)
        sensor_df = create_component_health_features(sensor_df, failure_df)
        sensor_df = create_interaction_features(sensor_df)

        # Scaling uniquement des colonnes numériques (exclure les identifiants et dates)
        sensor_df = feature_scaling(sensor_df, exclude_cols=['timestamp', 'equipment_id', 'next_failure_type'])

        # Tracé des importances
        plot_feature_importances(sensor_df, target_col='failure_soon', output_path=os.path.join(viz_dir, 'feature_importances.png'))

        augmented_data_path = os.path.join(output_dir, 'augmented_sensor_data.parquet')
        sensor_df.to_parquet(augmented_data_path, index=False)
        logger.info(f"Données enrichies sauvegardées sous {augmented_data_path}")

        return sensor_df
    except Exception as e:
        logger.error(f"Erreur lors de l'augmentation: {e}")
        raise

if __name__ == "__main__":
    augmented_data = augment_data()
    print("\nRésumé des données augmentées:")
    print(augmented_data.describe())


# In[ ]:




