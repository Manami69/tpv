import os
import pickle
import joblib
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split,
                                     ShuffleSplit)
from sklearn.pipeline import make_pipeline, Pipeline
from utils import load_filter_dataset, get_filtered_events
from mne.decoding import CSP
import argparse
import numpy as np

tasks = {
    1: [3, 7, 11],
    2: [4, 8, 12],
    3: [5, 9, 13],
    4: [6, 10, 14]
    }
PIPELINE_FILE = 'pipeline.pkl'
DATA_TEST_FILE = 'test_data.pkl'


def load_dataset(subject, task):
    """
Load events data and label for a given subject and a given task.
Split the dataset and save test datas in `test_data.csv` for future prediction.

Returns training datas only.
    """
    raw = load_filter_dataset(subject=subject, runs=tasks[task])
    # (epochs_data, labels) = get_filtered_events(raw)
    # X_train, X_test, y_train, y_test = train_test_split(
    #      epochs_data, labels, test_size=0.2, random_state=42)
    # test_data = {"X": X_test, "y": y_test}
    # with open(DATA_TEST_FILE, 'wb') as file:
    #     pickle.dump(test_data, file)
    # return (epochs_data, labels)
    return raw


def select_train_pipeline(raw):
    """
Use cross validation in training dataset to choose the best classifier
algorithm and parameters for this dataset.
    """
    cv_scores_by_frequency = []
    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)  # TODO:
    for freq_min in range(7, 29, 2):
        freq_max = min(freq_min + 2, 30)
        print(f"start freq {freq_min} - {freq_max}")
        (X, y) = get_filtered_events(raw, tmin=-1.0, tmax=4.0, freq_min=freq_min, freq_max=freq_max)
        X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.2, random_state=5)
        # LDA
        clf_lda_pip = make_pipeline(csp, LinearDiscriminantAnalysis())
        cv_scores_by_frequency.append({
            "score": cross_val_score(clf_lda_pip, X_train, y_train).mean(),
            "X": {
                "train": X_train,
                "test": X_test
                },
            "y": {
                "train": y_train,
                "test": y_test
                },
            "pipeline": clf_lda_pip
        })
        print("TOP")
    best_dataset = max(cv_scores_by_frequency, key=lambda x: x["score"])
    best_dataset['pipeline'].fit(best_dataset["X"]["train"], best_dataset["y"]["train"])
    test_data = {"X": best_dataset["X"]["test"], "y": best_dataset["y"]["test"]}
    print(f"Best cross val score is {best_dataset['score']}")
    print(f"Prediction accuracy on training dataset : {best_dataset['pipeline'].score(best_dataset['X']['train'], best_dataset['y']['train']):.2f}")
    with open(DATA_TEST_FILE, 'wb') as file:
        pickle.dump(test_data, file)
    joblib.dump(best_dataset['pipeline'], PIPELINE_FILE, compress=True)


def main(subject, task):
    if os.path.isfile(PIPELINE_FILE):
        os.remove(PIPELINE_FILE)
    if os.path.isfile(DATA_TEST_FILE):
        os.remove(DATA_TEST_FILE)
    raw = load_dataset(subject, task)
    select_train_pipeline(raw)


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description='Train a model that \
classify events from eeg datas for a given task and a given subject. \
Using physionet EEG Motor Movement/Imagery Dataset.')
        parser.add_argument('-s', '--subject',
                            nargs=1,
                            metavar=('num'),
                            help="subject number (1 .. 109), default=1",
                            default=[1],
                            type=int,
                            choices=range(1, 110))
        parser.add_argument('-t', '--task',
                            nargs=1,
                            metavar=('num'),
                            help="task number (1 .. 4), default=1",
                            default=[1],
                            type=int,
                            choices=range(1, 5))
        args = parser.parse_args()
        print(args)
        main(args.subject[0], args.task[0])
    except Exception as msg:
        print(f"Error: {msg}")
