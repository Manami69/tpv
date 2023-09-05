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
    (epochs_data, labels) = get_filtered_events(raw)
    X_train, X_test, y_train, y_test = train_test_split(
         epochs_data, labels, test_size=0.2, random_state=42)
    test_data = {"X": X_test, "y": y_test}
    with open(DATA_TEST_FILE, 'wb') as file:
        pickle.dump(test_data, file)
    return (X_train, y_train)


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



def select_train_pipeline(X, y):
    """
Use cross validation in training dataset to choose the best classifier
algorithm and parameters for this dataset.
    """
    pipelines_by_cv_score = []
    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)  # TODO:

    # SVM classifier test
    clf_svm_pip = make_pipeline(csp, svm.SVC(random_state=42))
    parameters = {'svc__kernel': ['linear', 'rbf', 'sigmoid'],
                  'svc__C': [0.1, 1, 10]}
    gs_cv_svm = GridSearchCV(clf_svm_pip,
                             parameters,
                             scoring='accuracy',
                             cv=StratifiedKFold(n_splits=5),
                             return_train_score=True)
    gs_cv_svm.fit(X, y)
    pipelines_by_cv_score.append({
        "name": "Support Vector Machine",
        "score": gs_cv_svm.best_score_,
        "pipeline": gs_cv_svm
    })

    # Logistic regression test
    clf_lr_pip = make_pipeline(csp, LogisticRegression(random_state=0))
    parameters = {'logisticregression__penalty': ['l1', 'l2']}
    gs_cv_lr = GridSearchCV(clf_lr_pip, parameters, scoring='accuracy')
    gs_cv_lr.fit(X, y)
    pipelines_by_cv_score.append({
        "name": "Logistic Regression",
        "score": gs_cv_lr.best_score_,
        "pipeline": gs_cv_lr
    })

    # LDA
    cv = ShuffleSplit(n_splits=5, test_size=0.1, random_state=0)
    clf_lda_pip = make_pipeline(csp, LinearDiscriminantAnalysis())
    pipelines_by_cv_score.append({
        "name": "Linear Discriminant Analysis",
        "score": cross_val_score(clf_lda_pip, X, y, cv=cv).mean(),
        "pipeline": clf_lda_pip
    })
    

    best_match = max(pipelines_by_cv_score, key=lambda x: x["score"])
    if best_match["name"] == "Linear Discriminant Analysis":
        best_match["pipeline"].fit(X, y)
    
    print(f"Using {best_match['name']} algorithm with a cross_val_score \
of {best_match['score']:.2f}")
    print(f"Prediction accuracy on training dataset : {best_match['pipeline'].score(X, y):.2f}")
    joblib.dump(best_match["pipeline"], PIPELINE_FILE, compress=True)
    


def main(subject, task):
    if os.path.isfile(PIPELINE_FILE):
        os.remove(PIPELINE_FILE)
    if os.path.isfile(DATA_TEST_FILE):
        os.remove(DATA_TEST_FILE)
    (X_train, y_train) = load_dataset(subject, task)
    select_train_pipeline(X_train, y_train)


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
