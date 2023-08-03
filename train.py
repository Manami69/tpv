import pickle
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import make_pipeline, Pipeline
from utils import load_filter_dataset, get_filtered_events
from mne.decoding import CSP
import argparse

tasks = {1: [3, 7, 11], 2: [4, 8, 12], 3: [5, 9, 13], 4: [6, 10, 14]}


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
    with open("test_data.plk", 'wb') as file:
        pickle.dump(test_data, file)
    return (X_train, y_train)


def select_pipeline(X, y):
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
    clf_lr_pip = make_pipeline(csp, LogisticRegression(random_state=42))
    parameters = {'logisticregression__penalty': ['l1', 'l2']}
    gs_cv_lr = GridSearchCV(clf_lr_pip, parameters, scoring='accuracy', cv=5)
    gs_cv_lr.fit(X, y)
    pipelines_by_cv_score.append({
        "name": "Logistic Regression",
        "score": gs_cv_lr.best_score_,
        "pipeline": gs_cv_lr
    })

    # LDA
    clf_lda_pip = make_pipeline(csp, LinearDiscriminantAnalysis())
    pipelines_by_cv_score.append({
        "name": "Linear Discriminant Analysis",
        "score": cross_val_score(clf_lda_pip, X, y, cv=5).mean(),
        "pipeline": clf_lda_pip
    })
    best_match = max(pipelines_by_cv_score, key=lambda x: x["score"])
    print(f"Using {best_match['name']} algorithm with a cross_val_score \
of {best_match['score']}")
    return best_match["pipeline"]


def train_model(X_train, y_train, pipeline: Pipeline):
    """
train model and save it for prediction
    """
    pipeline.fit(X_train, y_train)
    with open("pipeline.plk", 'wb') as file:
        pickle.dump(pipeline, file)


def main(subject, task):
    (X_train, y_train) = load_dataset(subject, task)
    pipeline = select_pipeline(X_train, y_train)
    train_model(X_train, y_train, pipeline)


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description='Train a model that \
classify events from eeg datas for a given task and a given subject. \
Using physionet EEG Motor Movement/Imagery Dataset.')
        parser.add_argument('-s', '--subject',
                            nargs=1,
                            metavar=('num'),
                            help="subject number (1 .. 109), default=1",
                            default=1,
                            type=int,
                            choices=range(1, 110))
        parser.add_argument('-t', '--task',
                            nargs=1,
                            metavar=('num'),
                            help="task number (1 .. 4), default=1",
                            default=1,
                            type=int,
                            choices=range(1, 5))
        args = parser.parse_args()
        main(args.subject, args.task)
    except Exception as msg:
        print(f"Error: {msg}")
