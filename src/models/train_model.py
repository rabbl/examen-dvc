import os

import pandas as pd
import click
from sklearn.ensemble import RandomForestRegressor
import joblib


@click.command()
def main():
    """
    # Model Training: Using the parameters found through GridSearch, we will train the model and save the trained model in the models directory.
    """

    processed_data_folder = os.path.join(os.getcwd(), 'data/processed_data')
    model_folder = os.path.join(os.getcwd(), 'models')
    train(processed_data_folder, model_folder)


def train(processed_data_folder, model_folder):
    X_train = pd.read_csv(f"{processed_data_folder}/X_train_scaled.csv")
    y_train = pd.read_csv(f"{processed_data_folder}/y_train.csv").values.ravel()

    best_params = joblib.load(f"{model_folder}/best_params.pkl")
    rf = RandomForestRegressor(**best_params)

    rf.fit(X_train, y_train)

    joblib.dump(rf, f"{model_folder}/trained_model.pkl")

    print("Model trained and saved successfully.")


if __name__ == '__main__':
    main()
