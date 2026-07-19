"""
=========================================================
Model Training Module
=========================================================

Module này chịu trách nhiệm:

1. Train nhiều mô hình
2. So sánh kết quả
3. Chọn mô hình tốt nhất
4. Lưu model

Để đơn giản và phù hợp, project chỉ sử dụng 3 mô hình:

- Linear Regression
- Random Forest
- XGBoost

=========================================================
"""

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from utils.evaluation import (
    evaluate_model,
    save_metrics,
)

from config import (
    MODEL_DIR,
    MODEL_PATH,
    RESULTS_DIR,
    PREDICTIONS_PATH,
)


# ==========================================================
# Model List
# ==========================================================

MODELS = {

    "Linear Regression":

        LinearRegression(),

    "Random Forest":

        RandomForestRegressor(

            n_estimators=200,

            random_state=42,

            n_jobs=-1

        ),

    "XGBoost":

        XGBRegressor(

            n_estimators=300,

            learning_rate=0.05,

            max_depth=5,

            random_state=42,

            objective="reg:squarederror"

        )

}


# ==========================================================
# Train
# ==========================================================

def train_models(df):
    """
    Train toàn bộ model.

    Trả về:

    Best Model
    """

    print("\nTraining models...\n")

    X = df.drop("Price", axis=1)

    y = df["Price"]

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.2,

        random_state=42

    )

    results = []

    best_model = None

    best_rmse = float("inf")

    best_name = ""

    for name, model in MODELS.items():

        print("=" * 50)

        print(f"Training {name}")

        model.fit(

            X_train,

            y_train

        )

        metric = evaluate_model(

            model,

            X_test,

            y_test

        )

        result = {

            "Model": name,

            "RMSE": metric["RMSE"],

            "MAE": metric["MAE"],

            "R2": metric["R2"]

        }

        results.append(result)

        print(result)

        if metric["RMSE"] < best_rmse:

            best_rmse = metric["RMSE"]

            best_model = model

            best_name = name

    print("\n")

    print("=" * 60)

    print("Best Model :", best_name)

    print("=" * 60)

    save_metrics(results)

    save_best_model(best_model)

    save_prediction(

        best_model,

        X_test,

        y_test

    )

    return best_model


# ==========================================================
# Save Model
# ==========================================================

def save_best_model(model):

    MODEL_DIR.mkdir(

        parents=True,

        exist_ok=True

    )

    joblib.dump(

        model,

        MODEL_PATH

    )

    print("Best model saved.")


# ==========================================================
# Load Model
# ==========================================================

def load_best_model():

    return joblib.load(

        MODEL_PATH

    )


# ==========================================================
# Prediction Result
# ==========================================================

def save_prediction(

    model,

    X_test,

    y_test

):

    prediction = model.predict(

        X_test

    )

    result = pd.DataFrame({

        "Actual Price": y_test,

        "Predicted Price": prediction

    })

    RESULTS_DIR.mkdir(

        parents=True,

        exist_ok=True

    )

    result.to_csv(

        PREDICTIONS_PATH,

        index=False

    )

    print("Prediction file saved.")


# ==========================================================
# Check Existing Model
# ==========================================================

def model_exists():
    """
    Kiểm tra toàn bộ artifact cần thiết.

    Chỉ khi đủ cả 3 file mới bỏ qua bước train.
    """

    from config import SCALER_PATH, FEATURE_COLUMNS_PATH

    return (
        MODEL_PATH.exists()
        and SCALER_PATH.exists()
        and FEATURE_COLUMNS_PATH.exists()
    )
