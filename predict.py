import os
import pickle
from sklearn.pipeline import Pipeline
import numpy as np
import joblib


PIPELINE_FILE = 'pipeline.pkl'
DATA_TEST_FILE = 'test_data.pkl'

def load_pipeline():
    """
load the trained pipeline object if it exist
    """
    trained_pipeline = joblib.load("pipeline.pkl")
    return trained_pipeline


def load_test_dataset():
    """
load test dataset
    """
    with open('test_data.pkl', 'rb') as f:
        test_dataset = pickle.load(f)
        return test_dataset


def get_predictions(test_dataset: dict, trained_pipeline: Pipeline):
    """
Use the trained pipeline to make prediction with the test dataset and write \
the result
    """
    predictions = trained_pipeline.predict(test_dataset["X"])

    for index, value in enumerate(predictions):
        print(f"predicted [{value}] - real [{test_dataset['y'][index]}]")
    accuracy = np.mean(predictions == test_dataset["y"])
    print(f"Prediction accuracy on training dataset : {accuracy:.2f}" )


def main():
    trained_pipeline = load_pipeline()
    test_dataset = load_test_dataset()
    get_predictions(test_dataset=test_dataset,
                    trained_pipeline=trained_pipeline)


if __name__ == "__main__":
    try:
        assert os.path.isfile(DATA_TEST_FILE) and os.path.isfile(PIPELINE_FILE), "You must train your model before prediction"
        main()
    except Exception as msg:
        print(f"Error: {msg}")
