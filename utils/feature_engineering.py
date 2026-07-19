"""
=========================================================
Feature Engineering
=========================================================

Author:
Car Price Prediction Project

Description
-----------
Module này tạo thêm các đặc trưng (features) từ dữ liệu gốc.

Việc tạo feature giúp mô hình Machine Learning học được
nhiều thông tin hơn thay vì chỉ sử dụng dữ liệu ban đầu.

Notebook tạo ra:

1. Engine_Size_Group
2. Mileage_Group
3. Year_Group

Module này đóng gói toàn bộ quá trình đó thành các hàm
để có thể tái sử dụng trong Flask và quá trình dự đoán.
=========================================================
"""

import pandas as pd


# ==========================================================
# Engine Size Group
# ==========================================================

def create_engine_group(df):
    """
    Chia dung tích động cơ thành 3 nhóm.

    Ví dụ

    1.0L  -> Small

    2.0L  -> Medium

    4.0L  -> Large

    Mục đích:
    Giúp mô hình học theo nhóm thay vì chỉ nhìn
    giá trị số liên tục.
    """

    df["Engine_Size_Group"] = pd.cut(
        df["Engine_Size"],
        bins=3,
        labels=[
            "Small",
            "Medium",
            "Large"
        ]
    )

    return df


# ==========================================================
# Mileage Group
# ==========================================================

def create_mileage_group(df):
    """
    Chia quãng đường đã đi thành

    Low

    Medium

    High

    Xe chạy ít thường có giá cao hơn.
    """

    df["Mileage_Group"] = pd.cut(
        df["Mileage"],
        bins=3,
        labels=[
            "Low",
            "Medium",
            "High"
        ]
    )

    return df


# ==========================================================
# Year Group
# ==========================================================

def create_year_group(df):
    """
    Chia năm sản xuất thành các nhóm.


    Điều này giúp giảm nhiễu
    so với việc dùng từng năm riêng lẻ.
    """

    df["Year_Group"] = pd.cut(
        df["Year"],
        bins=3,
        labels=[
            "2000s",
            "2010s",
            "2020s"
        ]
    )

    return df


# ==========================================================
# Car Age
# ==========================================================

def create_car_age(df):
    """
    Tạo tuổi của xe.

    Car_Age = 2026 - Year

    Trong thực tế,
    tuổi xe thường có ý nghĩa hơn
    so với chỉ dùng năm sản xuất.
    """

    CURRENT_YEAR = 2026

    df["Car_Age"] = CURRENT_YEAR - df["Year"]

    return df


# ==========================================================
# Mileage Per Year
# ==========================================================

def create_mileage_per_year(df):
    """
    Feature mới.

    Ví dụ

    Xe A

    100000 km

    20 năm

    = 5000 km/năm

    Xe B

    100000 km

    5 năm

    = 20000 km/năm

    Đây là đặc trưng khá hữu ích.
    """

    age = 2026 - df["Year"]

    age = age.replace(0, 1)

    df["Mileage_Per_Year"] = df["Mileage"] / age

    return df


# ==========================================================
# Engine Efficiency
# ==========================================================

def create_engine_efficiency(df):
    """
    Feature mới.

    Tỷ lệ

    Mileage

    /

    Engine Size

    Chỉ nhằm cung cấp thêm
    thông tin cho mô hình.
    """
    engine = df["Engine_Size"].replace(0, 0.1)

    df["Engine_Efficiency"] = (
        df["Mileage"] /
        df["Engine_Size"]
    )

    return df


# ==========================================================
# Feature Engineering Pipeline
# ==========================================================

def feature_engineering(df):
    """
    Thực hiện toàn bộ quá trình
    Feature Engineering.

    Chỉ cần gọi một lần:

    df = feature_engineering(df)
    """

    print("\nCreating new features...")

    df = create_engine_group(df)

    df = create_mileage_group(df)

    df = create_year_group(df)

    df = create_car_age(df)

    df = create_mileage_per_year(df)

    df = create_engine_efficiency(df)

    print("Feature Engineering completed.")

    return df