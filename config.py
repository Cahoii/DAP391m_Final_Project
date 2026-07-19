from pathlib import Path

# ==============================
# Project Configuration
# ==============================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "car_price_dataset.csv"

MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "best_model.pkl"

SCALER_PATH = MODEL_DIR / "scaler.pkl"

FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.pkl"

RESULTS_DIR = BASE_DIR / "results"

METRICS_PATH = RESULTS_DIR / "model_metrics.csv"

PREDICTIONS_PATH = RESULTS_DIR / "predictions.csv"

MILEAGE_MAX = 300000
