#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("clean_log.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('clean')

def plot_distribution(df, column, output_path):
    """Crée un graphique de distribution pour identifier les anomalies"""
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], kde=True)
    plt.title(f'Distribution de {column}')
    plt.savefig(output_path)
    plt.close()

def detect_outliers(df, column, method='zscore', threshold=3):
    """Détecte les valeurs aberrantes dans une colonne"""
    if method == 'zscore':
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        return z_scores > threshold
    elif method == 'iqr':
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return (df[column] < lower_bound) | (df[column] > upper_bound)
    else:
        raise ValueError("Méthode non reconnue. Utilisez 'zscore' ou 'iqr'.")

def clean_data(input_dir='data/raw', output_dir='data/cleaned'):
    """Nettoie les données extraites et les sauvegarde dans un nouveau format"""
    try:
        # Création du répertoire de sortie
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Répertoire créé: {output_dir}")

        # Sous-répertoire pour visualisations
        viz_dir = os.path.join(output_dir, 'visualizations')
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)

        # Chemins fichiers extraits
        sensor_data_path = os.path.join(input_dir, 'sensor_data.parquet')
        failure_data_path = os.path.join(input_dir, 'failure_data.parquet')

        # Chargement des données
        logger.info(f"Chargement des données capteurs depuis {sensor_data_path}")
        sensor_df = pd.read_parquet(sensor_data_path)
        logger.info(f"Chargement des données de défaillance depuis {failure_data_path}")
        failure_df = pd.read_parquet(failure_data_path)

        # --- Nettoyage données capteurs ---
        original_len = len(sensor_df)
        duplicates = sensor_df.duplicated().sum()
        missing_values_sensor = sensor_df.isnull().sum()
        logger.info(f"Valeurs manquantes dans les données capteurs:\n{missing_values_sensor}")

        # Remplacer inf et NaN par 0
        sensor_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        sensor_df.fillna(0, inplace=True)

        # Supprimer doublons
        sensor_df = sensor_df.drop_duplicates()
        logger.info(f"Nombre de doublons supprimés: {duplicates}")
        logger.info(f"Lignes supprimées pour valeurs manquantes: {original_len - len(sensor_df)}")

        # Détection et traitement des valeurs aberrantes
        numeric_columns = ['temperature', 'vibration', 'pressure', 'current']
        for column in numeric_columns:
            if column in sensor_df.columns:
                plot_path = os.path.join(viz_dir, f'{column}_distribution.png')
                plot_distribution(sensor_df, column, plot_path)
                outliers_mask = detect_outliers(sensor_df, column, method='iqr')
                outliers_count = outliers_mask.sum()
                logger.info(f"Valeurs aberrantes détectées dans {column}: {outliers_count}")
                if outliers_count > 0:
                    sensor_df[f'{column}_outlier'] = outliers_mask
                    # Optionnel: remplacer les outliers par la médiane
                    # median_val = sensor_df[~outliers_mask][column].median()
                    # sensor_df.loc[outliers_mask, column] = median_val

        # Trier par équipement et timestamp
        sensor_df = sensor_df.sort_values(by=['equipment_id', 'timestamp'])

        # --- Nettoyage données défaillance ---
        missing_values_failure = failure_df.isnull().sum()
        logger.info(f"Valeurs manquantes dans les données de défaillance:\n{missing_values_failure}")

        # Imputation repair_duration et repair_cost par médiane par type de défaillance
        for column in ['repair_duration', 'repair_cost']:
            if column in failure_df.columns:
                if failure_df[column].isnull().sum() > 0:
                    medians = failure_df.groupby('failure_type')[column].median()
                    for failure_type in failure_df['failure_type'].unique():
                        mask = (failure_df['failure_type'] == failure_type) & (failure_df[column].isnull())
                        failure_df.loc[mask, column] = medians[failure_type]
                    failure_df[column] = failure_df[column].fillna(failure_df[column].median())

        # Supprimer doublons défaillances
        duplicates_failure = failure_df.duplicated().sum()
        failure_df = failure_df.drop_duplicates()
        logger.info(f"Nombre de doublons supprimés dans les défaillances: {duplicates_failure}")

        # Vérification cohérence temporelle
        valid_equipment_ids = sensor_df['equipment_id'].unique()
        invalid_ids = failure_df[~failure_df['equipment_id'].isin(valid_equipment_ids)]
        if len(invalid_ids) > 0:
            logger.warning(f"Défaillances pour des équipements inexistants: {len(invalid_ids)}")
            failure_df = failure_df[failure_df['equipment_id'].isin(valid_equipment_ids)]

        # Sauvegarde CSV
        sensor_df.to_csv(os.path.join(output_dir, 'clean_sensor_data.csv'), index=False)
        failure_df.to_csv(os.path.join(output_dir, 'clean_failure_data.csv'), index=False)
        logger.info(f"Données nettoyées sauvegardées dans {output_dir}")

        # Rapport nettoyage
        cleaning_report = {
            "date_nettoyage": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "nb_enregistrements_capteurs_initial": original_len,
            "nb_enregistrements_capteurs_final": len(sensor_df),
            "nb_enregistrements_defaillances_initial": len(failure_df) + len(invalid_ids),
            "nb_enregistrements_defaillances_final": len(failure_df),
            "nb_valeurs_aberrantes_detectees": {col: detect_outliers(sensor_df, col).sum() for col in numeric_columns if col in sensor_df.columns}
        }
        pd.DataFrame([cleaning_report]).to_csv(os.path.join(output_dir, 'cleaning_report.csv'), index=False)

        return sensor_df, failure_df

    except Exception as e:
        logger.error(f"Erreur lors du nettoyage des données: {str(e)}")
        raise

if __name__ == "__main__":
    clean_sensor_df, clean_failure_df = clean_data()
    print("\nRésumé des données capteurs nettoyées:")
    print(clean_sensor_df.describe())
    print("\nRésumé des données de défaillance nettoyées:")
    print(clean_failure_df.describe())


# In[ ]:




