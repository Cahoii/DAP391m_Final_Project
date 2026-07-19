"""
=========================================================
Training Pipeline
=========================================================

Pipeline:

Load Dataset
↓

Cleaning
↓

Feature Engineering
↓

Detect Feature Types
↓

Encoding
↓

Scaling
↓

Ready for Training

=========================================================
"""

from utils.cleaning import clean_data
from utils.feature_engineering import feature_engineering
from utils.preprocessing import preprocess_data
from utils.eda import grab_col_names


def prepare_dataset(df):
    """
    Chuẩn bị dữ liệu cho Machine Learning.
    """

    print("\nPreparing dataset...")

    # Cleaning
    df = clean_data(df)

    # Feature Engineering
    df = feature_engineering(df)

    # Detect columns
    cat_cols, num_cols, cat_but_car, _ = grab_col_names(df)

    # Encode luôn các cột cardinal
    cat_cols = cat_cols + cat_but_car

    # Preprocessing
    df = preprocess_data(
        df,
        cat_cols,
        num_cols,
        training=True
    )


    print("Dataset ready.")

    return df