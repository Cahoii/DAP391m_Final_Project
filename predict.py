"""
=========================================================
Prediction Module
=========================================================

Load trained model
↓

Feature Engineering
↓

Preprocessing

↓

Prediction

=========================================================
"""

import joblib
import pandas as pd

from config import (
    MODEL_PATH,
    SCALER_PATH,
    FEATURE_COLUMNS_PATH,
    MILEAGE_MAX,
)

from utils.feature_engineering import feature_engineering
from utils.preprocessing import preprocess_data
from utils.eda import grab_col_names


# ==========================================================
# Load Artifacts
# ==========================================================

print("Loading trained artifacts...")

model = joblib.load(MODEL_PATH)

# Verify artifact files exist (scaler & feature_columns
# will be loaded inside preprocess_data when needed)
if not SCALER_PATH.exists():
    raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}")
if not FEATURE_COLUMNS_PATH.exists():
    raise FileNotFoundError(
        f"Feature columns not found: {FEATURE_COLUMNS_PATH}")

print("Artifacts loaded successfully.")


# ==========================================================
# Prediction
# ==========================================================

def predict_price(user_input):
    """
    Predict car price.
    """
    # ------------------------------------
    # Clamp Mileage to training range
    # ------------------------------------

    if "Mileage" in user_input:
        user_input["Mileage"] = min(user_input["Mileage"], MILEAGE_MAX)

    # ------------------------------------
    # Create DataFrame
    # ------------------------------------

    df = pd.DataFrame([user_input])

    # ------------------------------------
    # Feature Engineering
    # ------------------------------------
    df = feature_engineering(df)
    # ------------------------------------
    # Detect column types
    # ------------------------------------

    cat_cols, num_cols, cat_but_car, _ = grab_col_names(df)

    cat_cols = cat_cols + cat_but_car

    # ------------------------------------
    # Preprocessing
    # ------------------------------------

    df = preprocess_data(
        df,
        cat_cols,
        num_cols,
        training=False
    )

    # ------------------------------------
    # Prediction
    # ------------------------------------

    prediction = model.predict(df)

    return round(float(prediction[0]), 2)
