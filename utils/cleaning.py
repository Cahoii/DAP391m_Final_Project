"""
=========================================================
Data Cleaning Module

Author:

Description
-----------
Module thực hiện các bước làm sạch dữ liệu trước khi
Feature Engineering và Model Training.

Bao gồm:

1. Missing Value Analysis
2. Duplicate Removal
3. Rare Category Analysis
4. Rare Encoding
5. Dataset Validation

=========================================================
"""

import numpy as np
import pandas as pd


# =========================================================
# DATASET INFORMATION
# =========================================================

def dataset_info(df):
    """
    Hiển thị thông tin tổng quan của dataset.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    None
    """

    print("=" * 70)
    print("DATASET INFORMATION")
    print("=" * 70)

    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    print("\nData Types")
    print(df.dtypes)

    print("\nMemory Usage")
    print(df.memory_usage(deep=True))


# =========================================================
# DUPLICATE
# =========================================================

def remove_duplicates(df):
    """
    Xóa các bản ghi trùng lặp.

    Returns
    -------
    Clean dataframe
    """

    duplicate_count = df.duplicated().sum()

    print(f"\nDuplicate Rows : {duplicate_count}")

    if duplicate_count > 0:

        df = df.drop_duplicates()

        print("Duplicate rows removed.")

    return df


# =========================================================
# MISSING VALUE
# =========================================================

def missing_values_table(df,
                         return_columns=False):
    """
    Hiển thị bảng Missing Values.

    Parameters
    ----------
    df : dataframe

    return_columns : bool

    Returns
    -------
    list
        Danh sách cột bị thiếu dữ liệu.
    """

    na_columns = [
        col
        for col in df.columns
        if df[col].isnull().sum() > 0
    ]

    if len(na_columns) == 0:

        print("No Missing Values Found.")

        return []

    n_miss = (
        df[na_columns]
        .isnull()
        .sum()
        .sort_values(ascending=False)
    )

    ratio = (
        (
            df[na_columns]
            .isnull()
            .sum()
            / len(df)
        ) * 100
    ).sort_values(ascending=False)

    missing_df = pd.concat(
        [
            n_miss,
            ratio
        ],
        axis=1,
        keys=[
            "Missing Count",
            "Ratio (%)"
        ]
    )

    print(missing_df)

    if return_columns:

        return na_columns


# =========================================================
# MISSING VALUE IMPUTATION
# =========================================================

def fill_missing_values(df):
    """
    Điền Missing Value.

    Numeric
        Median

    Category
        Mode
    """

    for col in df.columns:

        if df[col].isnull().sum() == 0:
            continue

        if df[col].dtype == "O":

            df[col] = df[col].fillna(df[col].mode()[0])

        else:

            df[col] = df[col].fillna(df[col].median())

    return df


# =========================================================
# RARE ANALYSIS
# =========================================================

def rare_analyser(df,
                  target,
                  cat_cols):
    """
    Phân tích các nhóm dữ liệu hiếm.
    """

    for col in cat_cols:

        print("=" * 60)

        print(col)

        print("=" * 60)

        summary = pd.DataFrame({

            "COUNT":
                df[col].value_counts(),

            "RATIO":
                df[col].value_counts() / len(df),

            "TARGET_MEAN":
                df.groupby(col)[target].mean()

        })

        print(summary)

        print()


# =========================================================
# RARE ENCODER
# =========================================================

def rare_encoder(df,
                 rare_perc=0.01):
    """
    Gom các nhóm xuất hiện ít hơn rare_perc
    thành nhãn Rare.

    Parameters
    ----------
    rare_perc : float

    Example
    -------
    Toyota

    BMW

    Ferrari

    Lamborghini

    nếu Lamborghini chỉ chiếm
    <1%

    sẽ đổi thành

    Rare
    """

    temp_df = df.copy()

    rare_columns = [

        col

        for col in temp_df.columns

        if temp_df[col].dtype == "O"

        and (
            (
                temp_df[col]
                .value_counts()
                / len(temp_df)
            ) < rare_perc
        ).any()

    ]

    for col in rare_columns:

        frequencies = (
            temp_df[col]
            .value_counts()
            / len(temp_df)
        )

        rare_labels = frequencies[
            frequencies < rare_perc
        ].index

        temp_df[col] = np.where(

            temp_df[col].isin(rare_labels),

            "Rare",

            temp_df[col]

        )

    return temp_df


# =========================================================
# DATA VALIDATION
# =========================================================

def validate_dataset(df):
    """
    Kiểm tra dataset trước khi train.

    Returns
    -------
    True
        Dataset hợp lệ.

    False
        Dataset lỗi.
    """

    if df.empty:

        print("Dataset is Empty.")

        return False

    if df.isnull().sum().sum() > 0:

        print("Dataset still contains Missing Values.")

        return False

    print("Dataset validation successful.")

    return True


# =========================================================
# COMPLETE CLEANING PIPELINE
# =========================================================

def clean_data(df):
    """
    Thực hiện toàn bộ quy trình làm sạch dữ liệu.

    Pipeline

    Remove Duplicate

    ↓

    Missing Value

    ↓

    Rare Encoding

    ↓

    Outlier Capping

    ↓

    Validation
    """

    from utils.outlier import cap_outliers

    print("\nCleaning Dataset ...")

    df = remove_duplicates(df)

    missing_values_table(df)

    df = fill_missing_values(df)

    df = rare_encoder(df)

    # Outlier capping cho các cột số
    numeric_cols = df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()
    if "Price" in numeric_cols:
        numeric_cols.remove("Price")
    cap_outliers(df, numeric_cols)

    validate_dataset(df)

    print("Cleaning Completed.\n")

    return df
