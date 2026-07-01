import pandas as pd


def load_data(file_path):
    """
    Đọc dữ liệu từ file csv
    """
    df = pd.read_csv(file_path)
    return df


def clean_data(df):
    """
    Hàm làm sạch dữ liệu.
    Tạm thời chỉ trả lại dataframe.
    Sau sẽ chuyển từng bước từ notebook vào đây.
    """

    return df