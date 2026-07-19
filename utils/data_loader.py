"""
=========================================================
Dataset Loader
=========================================================

Đọc dữ liệu từ thư mục data.
"""

import pandas as pd

from config import DATA_PATH


def load_dataset():

    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    print(f"Dataset loaded: {df.shape}")

    return df