import pandas as pd
import os
import logging
from datetime import datetime

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("extract_log.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('extract')

def extract_data(sensor_file_path, failure_file_path, output_dir='extracted_data'):
    """
    Extrait les données des fichiers CSV et les sauvegarde dans un format structuré.
    
    Args:
        sensor_file_path (str): Chemin vers le fichier de données capteurs
        failure_file_path (str): Chemin vers le fichier de journal des défaillances
        output_dir (str): Répertoire de sortie pour les données extraites
    
    Returns:
        tuple: (DataFrame des capteurs, DataFrame des défaillances)
    """
    try:
        # Création du répertoire de sortie s'il n'existe pas
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Répertoire créé: {output_dir}")
        
        # Extraction des données des capteurs
        logger.info(f"Extraction des données de capteurs depuis {sensor_file_path}")
        sensor_data = pd.read_csv(sensor_file_path)
        
        # Convertir timestamp en datetime pour les données capteurs
        
        # Extraction des données de défaillance
        
        # Convertir failure_timestamp en datetime
        
        # Sauvegarde des données extraites dans le format parquet pour une meilleure efficacité
        
        
        logger.info(f"Données extraites et sauvegardées dans {output_dir}")
        logger.info(f"Forme des données de capteurs: {sensor_data.shape}")
        logger.info(f"Forme des données de défaillance: {failure_data.shape}")
        
        # Création d'un rapport d'extraction
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
        
        # Sauvegarde du rapport sous forme de DataFrame
        pd.DataFrame([report]).to_csv(os.path.join(output_dir, 'extraction_report.csv'), index=False)
        logger.info("Rapport d'extraction créé")
        
        return sensor_data, failure_data
    
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction des données: {str(e)}")
        raise
        
if __name__ == "__main__":
    # 1. On donne les chemins exacts vers tes fichiers de 27 Mo
    SENSOR_FILE = "data/raw/predictive_maintenance_sensor_data (1).csv"
    FAILURE_FILE = "data/raw/predictive_maintenance_failure_logs (1).csv"
    
    # 2. On définit où créer les fichiers Parquet pour le nettoyage
    OUTPUT_DIR = "data/processed/extracted_data"
    
    # 3. On lance l'extraction avec ces nouveaux paramètres
    sensor_df, failure_df = extract_data(SENSOR_FILE, FAILURE_FILE, output_dir=OUTPUT_DIR)
    
    # 4. Vérification visuelle dans Jupyter
    print("\n✅ Extraction réussie !")
    print(f"Nombre de lignes capteurs : {len(sensor_df)}")
    print(sensor_df.head())