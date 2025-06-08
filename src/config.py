import os
from pathlib import Path

#: Project base directory and subdirectories for data and models.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "models"
MLFLOW_TRACKING_URI = "http://localhost:5000"  # Change if using a remote MLflow server

# Create directories if not exist
for path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR]:
    os.makedirs(path, exist_ok=True)
