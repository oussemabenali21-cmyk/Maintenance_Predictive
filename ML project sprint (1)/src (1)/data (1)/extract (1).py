#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import os
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("extract_log.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('extract')

def extract_data(sensor_file_path, failure_file_path, output_dir='data/raw'):

    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Répertoire créé: {output_dir}")

        # ===== Lecture données capteurs =====
        logger.info(f"Extraction capteurs depuis {sensor_file_path}")
        sensor_data = pd.read_csv(sensor_file_path)

        # Conversion timestamp
        sensor_data['timestamp'] = pd.to_datetime(sensor_data['timestamp'])

        # ===== Lecture données défaillance =====
        logger.info(f"Extraction défaillances depuis {failure_file_path}")
        failure_data = pd.read_csv(failure_file_path)

        # Conversion timestamp défaillance
        failure_data['failure_timestamp'] = pd.to_datetime(failure_data['failure_timestamp'])

        # ===== Sauvegarde parquet =====
        sensor_data.to_parquet(os.path.join(output_dir, "sensor_data.parquet"), index=False)
        failure_data.to_parquet(os.path.join(output_dir, "failure_data.parquet"), index=False)

        logger.info("Données sauvegardées en parquet")

        # ===== Rapport =====
        report = {
            "date_extraction": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "nombre_equipements": sensor_data['equipment_id'].nunique(),
            "types_equipement": sensor_data['equipment_type'].unique().tolist(),
            "periode_debut": sensor_data['timestamp'].min(),
            "periode_fin": sensor_data['timestamp'].max(),
            "nombre_enregistrements_capteurs": len(sensor_data),
            "nombre_defaillances": len(failure_data),
            "types_defaillance": failure_data['failure_type'].unique().tolist()
        }

        pd.DataFrame([report]).to_csv(
            os.path.join(output_dir, 'extraction_report.csv'),
            index=False
        )

        logger.info("Rapport d'extraction créé")

        return sensor_data, failure_data

    except Exception as e:
        logger.error(f"Erreur lors de l'extraction: {str(e)}")
        raise
        
if __name__ == "__main__":
    # Chemins des fichiers (à ajuster selon votre environnement)
    SENSOR_FILE = r"E:\CTéléchargements\ML project sprint (1)-20250422T132820Z-001\ML project sprint (1)\data (1)\raw (1)\predictive_maintenance_sensor_data (1).csv"
    FAILURE_FILE = r"E:\CTéléchargements\ML project sprint (1)-20250422T132820Z-001\ML project sprint (1)\data (1)\raw (1)\predictive_maintenance_failure_logs (1).csv"
    
    # Exécution de la fonction d'extraction
    sensor_df, failure_df = extract_data(SENSOR_FILE, FAILURE_FILE)
    
    # Affichage des premières lignes pour vérification
    print("\nAperçu des données capteurs:")
    print(sensor_df.head())
    
    print("\nAperçu des données de défaillance:")
    print(failure_df.head())


# In[ ]:




