import os

import pandas as pd
import click
from sklearn.preprocessing import StandardScaler


@click.command()
def main():
    """
    Data Normalization: As you may notice, the data varies widely in scale, so normalization is necessary.
    You can use existing functions to construct this script.
    As output, this script will create two new datasets (X_train_scaled, X_test_scaled) which you will also save in data/processed.
    """

    split_data_folder = (os.path.join(os.getcwd(), 'data/split'))
    processed_data_folder = (os.path.join(os.getcwd(), 'data/processed'))

    if not os.path.exists(processed_data_folder):
        os.makedirs(processed_data_folder)

    normalize_data(split_data_folder, processed_data_folder)


def normalize_data(split_data_folder, processed_data_folder):
    X_train = pd.read_csv(f"{split_data_folder}/X_train.csv")
    X_test = pd.read_csv(f"{split_data_folder}/X_test.csv")

    # Normalize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save the datasets
    pd.DataFrame(X_train_scaled).to_csv(f"{processed_data_folder}/X_train_scaled.csv", index=False)
    pd.DataFrame(X_test_scaled).to_csv(f"{processed_data_folder}/X_test_scaled.csv", index=False)

    print("Data normalized successfully.")


if __name__ == '__main__':
    main()
