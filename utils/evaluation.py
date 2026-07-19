"""
=========================================================
Model Evaluation
=========================================================

Đánh giá mô hình bằng các chỉ số:

- RMSE
- MAE
- R2 Score

Đồng thời lưu kết quả thành metrics.csv
=========================================================
"""

import os
import pandas as pd

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from config import (
    RESULTS_DIR,
    METRICS_PATH,
)


def evaluate_model(model, X_test, y_test):
    """
    Đánh giá một mô hình.
    """

    predictions = model.predict(X_test)

    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }


def save_metrics(results):
    """
    Lưu bảng so sánh các mô hình.
    """

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.DataFrame(results)

    df.to_csv(
        METRICS_PATH,
        index=False
    )

    print("Metrics saved.")
