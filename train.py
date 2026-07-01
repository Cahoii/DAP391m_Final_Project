from utils.cleaning import load_data, clean_data

DATA_PATH = "data/car_price_dataset.csv"


def main():

    print("Loading dataset...")

    df = load_data(DATA_PATH)

    print(df.head())

    print("\nDataset Shape:")

    print(df.shape)

    df = clean_data(df)

    print("\nCleaning completed!")



if __name__ == "__main__":
    main()