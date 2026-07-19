"""
=========================================================
Outlier Detection Module

Author:

Description
-----------
Module xử lý Outlier cho Car Price Prediction Project.

Bao gồm:

1. Tính ngưỡng IQR
2. Kiểm tra Outlier
3. Hiển thị Outlier
4. Capping Outlier
5. Xóa Outlier

=========================================================
"""

import pandas as pd


# =========================================================
# OUTLIER THRESHOLD
# =========================================================

def outlier_thresholds(df,
                       column,
                       q1=0.25,
                       q3=0.75):
    """
    Tính ngưỡng Outlier theo IQR.

    Parameters
    ----------
    df : DataFrame

    column : str

    q1 : float

    q3 : float

    Returns
    -------
    lower_limit

    upper_limit
    """

    quartile1 = df[column].quantile(q1)

    quartile3 = df[column].quantile(q3)

    iqr = quartile3 - quartile1

    lower = quartile1 - 1.5 * iqr

    upper = quartile3 + 1.5 * iqr

    return lower, upper


# =========================================================
# CHECK OUTLIER
# =========================================================

def check_outlier(df,
                  column):
    """
    Kiểm tra cột có Outlier hay không.
    """

    lower, upper = outlier_thresholds(df, column)

    return df[
        (df[column] < lower)
        | (df[column] > upper)
    ].shape[0] > 0


# =========================================================
# COUNT OUTLIER
# =========================================================

def count_outlier(df,
                  column):

    lower, upper = outlier_thresholds(df, column)

    return df[
        (df[column] < lower)
        | (df[column] > upper)
    ].shape[0]


# =========================================================
# SHOW OUTLIERS
# =========================================================

def show_outliers(df,
                  column,
                  head=5):

    lower, upper = outlier_thresholds(df, column)

    outliers = df[
        (df[column] < lower)
        | (df[column] > upper)
    ]

    print(outliers.head(head))

    return outliers


# =========================================================
# REPLACE OUTLIERS
# =========================================================

def replace_with_thresholds(df,
                            column):
    """
    Winsorization (Capping)

    Giá trị lớn hơn Upper
    ↓
    Upper

    Giá trị nhỏ hơn Lower
    ↓
    Lower
    """

    lower, upper = outlier_thresholds(df, column)

    # Convert column to float if it's integer to avoid
    # TypeError when assigning float thresholds (pandas 3.x)
    if df[column].dtype == "int64":
        df[column] = df[column].astype("float64")

    df.loc[
        df[column] < lower,
        column
    ] = lower

    df.loc[
        df[column] > upper,
        column
    ] = upper


# =========================================================
# REMOVE OUTLIERS
# =========================================================

def remove_outliers(df,
                    column):

    lower, upper = outlier_thresholds(df, column)

    return df[
        (df[column] >= lower)
        &
        (df[column] <= upper)
    ]


# =========================================================
# SUMMARY
# =========================================================

def outlier_summary(df,
                    numeric_columns):

    print("=" * 70)
    print("OUTLIER SUMMARY")
    print("=" * 70)

    for col in numeric_columns:

        status = check_outlier(df, col)

        total = count_outlier(df, col)

        print(f"{col:<25} {status} ({total})")


# =========================================================
# CAPPING ALL
# =========================================================

def cap_outliers(df,
                 numeric_columns):

    print("Applying Outlier Capping...")

    for col in numeric_columns:

        replace_with_thresholds(df, col)

    print("Completed.")

    return df
