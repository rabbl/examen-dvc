import os

import pandas as pd
import click
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import joblib


@click.command()
def main():
    """
    GridSearch for Best Parameters: Decide on the regression model to implement and the parameters to test.
    At the end of this script, we will have the best parameters saved as a .pkl file in the models directory.
    """

    split_data_folder = os.path.join(os.getcwd(), 'data/split')
    processed_data_folder = os.path.join(os.getcwd(), 'data/processed')
    model_folder = os.path.join(os.getcwd(), 'models')
    grid_search(split_data_folder, processed_data_folder, model_folder)


def grid_search(split_data_folder, processed_data_folder, model_folder):
    X_train = pd.read_csv(f"{processed_data_folder}/X_train_scaled.csv")
    y_train = pd.read_csv(f"{split_data_folder}/y_train.csv").values.ravel()

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    rf = RandomForestRegressor()

    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    joblib.dump(best_params, f"{model_folder}/best_params.pkl")


if __name__ == '__main__':
    main()
