import os

import pandas as pd
import click
import json
import joblib
from sklearn.metrics import mean_squared_error, r2_score


@click.command()
def main():
    """
    Model Evaluation: Finally, using the trained model, we will evaluate its performance and make predictions.
    At the end of this script, we will have a new dataset in data containing the predictions,
    along with a scores.json file in the metrics directory that will capture evaluation metrics of our model (e.g., MSE, R2).
    """

    split_data_folder = os.path.join(os.getcwd(), 'data/split')
    processed_data_folder = os.path.join(os.getcwd(), 'data/processed')
    model_folder = os.path.join(os.getcwd(), 'models')
    metrics_folder = os.path.join(os.getcwd(), 'metrics')

    if not os.path.exists(metrics_folder):
        os.makedirs(metrics_folder)

    evaluate(split_data_folder, processed_data_folder, model_folder, metrics_folder)


def evaluate(split_data_folder, processed_data_folder, model_folder, metrics_folder):
    X_test = pd.read_csv(f"{processed_data_folder}/X_test_scaled.csv")
    y_test = pd.read_csv(f"{split_data_folder}/y_test.csv").values.ravel()

    rf = joblib.load(f"{model_folder}/trained_model.pkl")

    predictions = rf.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    scores = {
        'mse': mse,
        'r2': r2
    }

    with open(f"{metrics_folder}/scores.json", 'w') as f:
        json.dump(scores, f)

    predictions_df = pd.DataFrame(predictions, columns=['silica_concentrate'])
    predictions_df.to_csv(f"{processed_data_folder}/predictions.csv", index=False)

    print("Model evaluated successfully.")


if __name__ == '__main__':
    main()
