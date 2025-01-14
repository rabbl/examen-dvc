import os
import pandas as pd
import click
from sklearn.model_selection import train_test_split


@click.command()
def main():
    """
    Data Splitting: Split the data into training and testing sets.
    Our target variable is silica_concentrate, located in the last column of the dataset.
    This script will produce 4 datasets (X_test, X_train, y_test, y_train).

    The input data file has to be in the data/raw folder.
    The output preprocessed data file will be saved in the output folder.
    """

    raw_data_file = os.path.join(os.getcwd(), 'data/raw/raw.csv')
    output_filepath = os.path.join(os.getcwd(), 'data/split')

    if not os.path.exists(output_filepath):
        os.makedirs(output_filepath)

    split_data(raw_data_file, output_filepath)


def split_data(raw_data_file, output_filepath):
    # Load the raw data
    df = pd.read_csv(raw_data_file, index_col=0)

    # Split the data into training and testing sets
    X = df.drop(columns=['silica_concentrate'])
    y = df['silica_concentrate']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save the datasets
    X_train.to_csv(f"{output_filepath}/X_train.csv", index=False)
    X_test.to_csv(f"{output_filepath}/X_test.csv", index=False)
    y_train.to_csv(f"{output_filepath}/y_train.csv", index=False)
    y_test.to_csv(f"{output_filepath}/y_test.csv", index=False)

    print("Data split successfully.")


if __name__ == '__main__':
    main()
