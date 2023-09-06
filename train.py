import os
import pickle
import joblib
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import (KFold,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import make_pipeline
from utils import load_filter_dataset, get_filtered_events
from csp import CustomCSP
import argparse

tasks = {
    1: [3, 7, 11],  # open and close left or right fist
    2: [4, 8, 12],  # imagine opening and closing left or right fist
    3: [5, 9, 13],  # open and close both fists or both feet
    4: [6, 10, 14],  # imagine opening and closing both fists or both feet
    5: [3, 7, 11, 4, 8, 12],  # task 1 + task 2
    6: [5, 9, 13, 6, 10, 14]  # task 3 + task 4
    }


DIR = "train_datas"


def load_split_dataset(subject, task):
    """
Load events data and label for a given subject and a given task.
Split the dataset and save test datas in `test_data.csv` for future prediction.

Returns training datas only.
    """
    raw = load_filter_dataset(subject=subject, runs=tasks[task])
    (X, y) = get_filtered_events(raw, tmin=1, tmax=4, freq_min=7, freq_max=20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    test_data = {"X": X_test, "y": y_test}
    with open(f'{DIR}/test_data_s{subject}_t{task}.pkl', 'wb') as file:
        pickle.dump(test_data, file)
    return (X_train, y_train)


def train_model(X, y, subject, task):
    """
Use cross validation in training dataset to choose the best classifier
algorithm and parameters for this dataset.
    """
    csp = CustomCSP(n_components=4)
    cv = KFold(n_splits=10)
    pipeline = make_pipeline(csp, LinearDiscriminantAnalysis())
    score = cross_val_score(pipeline, X, y, cv=cv)
    print(f"cross val score : {score.mean():.2f}")
    pipeline.fit(X, y)
    print(f"Prediction accuracy on training dataset : \
{pipeline.score(X, y):.2f}")
    
    joblib.dump(pipeline, f'{DIR}/pipeline_s{subject}_t{task}.pkl',
                compress=True)


def main(subject, task):
    if os.path.isfile(f'{DIR}/pipeline_s{subject}_t{task}.pkl'):
        os.remove(f'{DIR}/pipeline_s{subject}_t{task}.pkl')
    if os.path.isfile(f'{DIR}/test_data_s{subject}_t{task}.pkl'):
        os.remove(f'{DIR}/test_data_s{subject}_t{task}.pkl')
    print(f"Making Model for subject {subject} on task {task}")
    (X, y) = load_split_dataset(subject, task)
    train_model(X, y, subject, task)


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description='Train a model that \
classify events from eeg datas for a given task and a given subject. \
Using physionet EEG Motor Movement/Imagery Dataset.')
        parser.add_argument('-a', '--all', action="store_true", 
                            help="compute all \
subjects for all task (must take some time)")
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
                            help="task number (1 .. 6), default=1",
                            default=[1],
                            type=int,
                            choices=range(1, 7))
        args = parser.parse_args()
        if not os.path.exists(DIR):
            os.makedirs(DIR)
        if args.all:
            for subject in range(1, 110):
                for task in range(1, 7):
                    main(subject, task)
        else:
            main(args.subject[0], args.task[0])
    except Exception as msg:
        print(f"Error: {msg}")
