import pandas as pd
import click
import logging
from sklearn.model_selection import train_test_split


@click.command()
@click.argument('input_filepath', type=click.Path(exists=False), required=0)
@click.argument('output_filepath', type=click.Path(exists=False), required=0)
def main(input_filepath, output_filepath):
    """
    Data Splitting: Split the data into training and testing sets.
    Our target variable is silica_concentrate, located in the last column of the dataset.
    This script will produce 4 datasets (X_test, X_train, y_test, y_train).

    The input data file has to be in the data/raw folder.
    The output preprocessed data file will be saved in the output folder.
    """

    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    input_filepath = click.prompt('Enter the file path for the input data', type=click.Path(exists=True))
    output_filepath = click.prompt('Enter the file path for the output preprocessed data', type=click.Path())

    raw_data_file = f"{input_filepath}/raw.csv"
    split_data(raw_data_file, output_filepath)


def split_data(raw_data_file, output_filepath):
    # Load the raw data
    df = pd.read_csv(raw_data_file)

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
