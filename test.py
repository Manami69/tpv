import os
import pickle
import joblib
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (GridSearchCV, KFold, StratifiedKFold,
                                     cross_val_score, train_test_split,
                                     ShuffleSplit)
from sklearn.pipeline import make_pipeline, Pipeline
from utils import load_filter_dataset, get_filtered_events
from csp import CustomCSP
import argparse
import numpy as np

tasks = {
    1: [3, 7, 11],
    2: [4, 8, 12],
    3: [5, 9, 13],
    4: [6, 10, 14]
    }
DIR= "train_datas"

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


def select_best_channels_for_subject(X, y):
    scores = []
    classifier = svm.SVC()
    cv = KFold(n_splits=10)
    for idx in range(64):
        X_chn = X[:, idx]
        score = cross_val_score(classifier, X_chn, y, cv=cv)
        scores.append((idx, score.mean()))
    sorted_best = sorted(scores, key=lambda s: s[1], reverse=True)[:30]
    return [i[0] for i in sorted_best]


def select_train_pipeline(raw, subject, task):
    """
Use cross validation in training dataset to choose the best classifier
algorithm and parameters for this dataset.
    """
    # cv_scores_by_frequency = []
    csp = CustomCSP(n_components=3)  # TODO:
    cv = KFold(n_splits=10)
    (X, y) = get_filtered_events(raw, tmin=1, tmax=2, freq_min=7, freq_max=30)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    chnls_filter = select_best_channels_for_subject(X_train, y_train)
    fX_train = np.take(X_train, chnls_filter, 1)
    fX_test = np.take(X_test, chnls_filter, 1)
    pipeline = make_pipeline(csp, LinearDiscriminantAnalysis())
    score = cross_val_score(pipeline, fX_train, y_train, cv=cv)
    print(f"cross val score :{score.mean()}")
    pipeline.fit(fX_train, y_train)
    print(f"Prediction accuracy on training dataset : {pipeline.score(fX_train, y_train):.2f}")
    test_data = {"X": fX_test, "y": y_test}
    with open(f'{DIR}/test_data_s{subject}_t{task}.pkl', 'wb') as file:
        pickle.dump(test_data, file)
    joblib.dump(pipeline, f'{DIR}/pipeline_s{subject}_t{task}.pkl', compress=True)


def main(subject, task):
    if os.path.isfile(f'{DIR}/pipeline_s{subject}_t{task}.pkl'):
        os.remove(f'{DIR}/pipeline_s{subject}_t{task}.pkl')
    if os.path.isfile(f'{DIR}/test_data_s{subject}_t{task}.pkl'):
        os.remove(f'{DIR}/test_data_s{subject}_t{task}.pkl')
    print("O", subject, task)
    raw = load_dataset(subject, task)
    print("a")
    select_train_pipeline(raw, subject, task)


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description='Train a model that \
classify events from eeg datas for a given task and a given subject. \
Using physionet EEG Motor Movement/Imagery Dataset.')
        parser.add_argument('-a', '--all', action="store_true", help="compute all subjects for all task (must take some time)")
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
        if not os.path.exists(DIR):
            os.makedirs(DIR)
        if args.all:
            for subject in range(1, 110):
                for task in range(1, 5):
                    main(subject, task)
        else:
            main(args.subject[0], args.task[0])
    except Exception as msg:
        print(f"Error: {msg}")
