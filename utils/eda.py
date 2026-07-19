"""
=========================================================
Exploratory Data Analysis Module

Mục đích:
- Phân tích dữ liệu trước khi huấn luyện.
- Kiểm tra dữ liệu đầu vào.
- Phân loại biến.
- Phân tích tương quan.

=========================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def check_dataframe(df, head=5):
    """
    Hiển thị tổng quan dataset.

    Parameters
    ----------
    df : pandas.DataFrame

    head : int
        số dòng hiển thị.

    Returns
    -------
    None
    """

    print("=" * 70)
    print("DATASET OVERVIEW")
    print("=" * 70)

    print("\nShape")
    print(df.shape)

    print("\nData Types")
    print(df.dtypes)

    print("\nHead")
    print(df.head(head))

    print("\nTail")
    print(df.tail(head))

    print("\nMissing Values")
    print(df.isnull().sum())

    print("\nQuantiles")

    numeric_df = df.select_dtypes(include=np.number)

    print(
        numeric_df.quantile(
            [
                0,
                0.05,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
                1,
            ]
        ).T
    )


def grab_col_names(dataframe,
                   cat_th=10,
                   car_th=31):
    """
    Chia cột thành:

    cat_cols

    num_cols

    cat_but_car

    num_but_cat
    """

    # Các cột dạng object hoặc category
    cat_cols = dataframe.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # Chỉ chuyển numeric -> categorical khi dataset đủ lớn
    if len(dataframe) > 1:

        num_but_cat = [
            col
            for col in dataframe.select_dtypes(
                include=np.number
            ).columns
            if dataframe[col].nunique() < cat_th
        ]

    else:

        num_but_cat = []

    # Các cột category có quá nhiều giá trị
    cat_but_car = [
        col
        for col in cat_cols
        if dataframe[col].nunique() > car_th
    ]

    # Category cuối cùng
    cat_cols = cat_cols + num_but_cat

    cat_cols = [
        col
        for col in cat_cols
        if col not in cat_but_car
    ]

    # Numerical cuối cùng
    num_cols = [
        col
        for col in dataframe.select_dtypes(
            include=np.number
        ).columns
        if col not in num_but_cat
    ]

    print("=" * 70)

    print(f"Observations : {dataframe.shape[0]}")
    print(f"Variables    : {dataframe.shape[1]}")
    print(f"Categorical  : {len(cat_cols)}")
    print(f"Numerical    : {len(num_cols)}")
    print(f"Cat but Car  : {len(cat_but_car)}")
    print(f"Num but Cat  : {len(num_but_cat)}")

    print("=" * 70)

    return (
        cat_cols,
        num_cols,
        cat_but_car,
        num_but_cat,
    )

def cat_summary(df,
                column,
                plot=False):
    """
    Thống kê biến phân loại.
    """

    summary = pd.DataFrame(
        {
            column: df[column].value_counts(),
            "Ratio":
                100 *
                df[column].value_counts()
                / len(df)
        }
    )

    print(summary)

    if plot:

        plt.figure(figsize=(8, 4))

        sns.countplot(
            x=df[column],
            palette="Set2"
        )

        plt.xticks(rotation=45)

        plt.tight_layout()

        plt.show()


def num_summary(df,
                column,
                plot=False):
    """
    Thống kê biến số.
    """

    quantiles = [
        .05,
        .10,
        .20,
        .30,
        .40,
        .50,
        .60,
        .70,
        .80,
        .90,
        .95,
        .99,
    ]

    print(
        df[column].describe(
            quantiles
        )
    )

    if plot:

        plt.figure(figsize=(6, 4))

        sns.histplot(
            df[column],
            kde=True
        )

        plt.show()


def correlation_analysis(df,
                         plot=True):

    corr = df.select_dtypes(
        include=np.number
    ).corr()

    if plot:

        plt.figure(figsize=(10, 8))

        sns.heatmap(
            corr,
            cmap="RdBu",
            annot=True
        )

        plt.show()

    return corr


def high_correlations(df,
                      threshold=0.70):

    corr = (
        df.select_dtypes(
            include=np.number
        )
        .corr()
        .abs()
    )

    upper = corr.where(
        np.triu(
            np.ones(corr.shape),
            k=1
        ).astype(bool)
    )

    pairs = []

    for col in upper.columns:

        for row in upper.index:

            value = upper.loc[row, col]

            if pd.notna(value):

                if value > threshold:

                    pairs.append(
                        (
                            row,
                            col,
                            value,
                        )
                    )

    return pairs