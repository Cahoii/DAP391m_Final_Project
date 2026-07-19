"""
=========================================================
Data Preprocessing
=========================================================

Module này chịu trách nhiệm chuẩn bị dữ liệu trước khi
đưa vào mô hình Machine Learning.

Pipeline gồm:

1. One-Hot Encoding
2. Feature Alignment
3. Robust Scaling
4. Lưu Scaler
5. Lưu danh sách Feature

Các file được sinh ra:

models/
    scaler.pkl
    feature_columns.pkl
    best_model.pkl

=========================================================
"""

import os
import joblib
import pandas as pd

from sklearn.preprocessing import RobustScaler

from config import (
    MODEL_DIR,
    FEATURE_COLUMNS_PATH,
    SCALER_PATH,
)


# ==========================================================
# One Hot Encoding
# ==========================================================

def one_hot_encoding(df, categorical_columns):
    """
    Chuyển các cột dạng Category
    thành dạng số.

    Ví dụ

    Brand

    Toyota
    BMW
    Ford

    ↓

    Brand_Toyota
    Brand_BMW
    Brand_Ford
    """

    print("Applying One-Hot Encoding...")

    df = pd.get_dummies(
        df,
        columns=categorical_columns,
        drop_first=False,
        dtype=int
    )

    return df


# ==========================================================
# Save Feature Columns
# ==========================================================

def save_feature_columns(columns):

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        columns,
        FEATURE_COLUMNS_PATH
    )

    print("Feature columns saved.")


# ==========================================================
# Load Feature Columns
# ==========================================================

def load_feature_columns():

    return joblib.load(
        FEATURE_COLUMNS_PATH
    )


# ==========================================================
# Scale Data
# ==========================================================

def scale_data(df, numeric_columns):

    print("Scaling numerical features...")

    scaler = RobustScaler()

    df[numeric_columns] = scaler.fit_transform(
        df[numeric_columns]
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        scaler,
        SCALER_PATH
    )

    print("Scaler saved.")

    return df


# ==========================================================
# Align Columns
# ==========================================================

def align_columns(df):
    """
    Khi dự đoán trên Flask,
    dữ liệu người dùng nhập có thể
    thiếu một số cột.

    Hàm này sẽ tự thêm các cột còn thiếu
    để khớp với lúc train.
    """

    if not FEATURE_COLUMNS_PATH.exists():
        return df

    feature_columns = load_feature_columns()

    for col in feature_columns:

        if col not in df.columns:

            df[col] = 0

    df = df[feature_columns]

    return df


# ==========================================================
# Main Preprocessing
# ==========================================================

def preprocess_data(
        df,
        cat_cols,
        num_cols,
        training=True):
    """
    Pipeline tiền xử lý.

    Parameters
    ----------

    training=True

        Train model

    training=False

        Predict trên Flask
    """

    df = one_hot_encoding(
        df,
        cat_cols
    )

    if training:

        feature_columns = df.drop(
            columns=["Price"],
            errors="ignore"
        ).columns.tolist()

        save_feature_columns(
            feature_columns
        )

    else:

        df = align_columns(df)

    # Lấy toàn bộ cột số sau khi Feature Engineering
    scale_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    if "Price" in scale_columns:
        scale_columns.remove("Price")

    if training:

        df = scale_data(
            df,
            scale_columns
        )

    else:

        scaler = joblib.load(
            SCALER_PATH
        )

        df[scale_columns] = scaler.transform(
            df[scale_columns]
        )

    print("Preprocessing completed.")

    return df
