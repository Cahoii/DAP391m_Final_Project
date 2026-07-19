"""
=========================================================
Car Price Prediction

Main Training Script
=========================================================

Quy trình:

1. Load Dataset
2. Data Cleaning
3. Feature Engineering
4. Encoding & Scaling
5. Train Multiple Models
6. Compare Performance
7. Save Best Model
8. Save Scaler
9. Save Feature Columns

Nếu model đã tồn tại thì sẽ load lại,
không train lại nhằm tiết kiệm thời gian.
=========================================================
"""

from utils.data_loader import load_dataset
from utils.pipeline import prepare_dataset

from utils.model import (
    train_models,
    load_best_model,
    model_exists,
)


def main():

    print("=" * 70)
    print("CAR PRICE PREDICTION")
    print("=" * 70)

    print("\nStep 1 - Loading Dataset...")
    df = load_dataset()

    print("\nStep 2 - Preparing Dataset...")
    df = prepare_dataset(df)

    print("\nStep 3 - Checking Existing Model...")

    if model_exists():

        print("\nModel artifacts found.")

        print("Loading trained model...")

        model = load_best_model()

        print("Model loaded successfully.")

    else:

        print("\nNo trained model found.")

        print("Start training...")

        model = train_models(df)

        print("\nTraining completed successfully.")

    print("\nPipeline Finished Successfully.")

    print("=" * 70)


if __name__ == "__main__":
    main()