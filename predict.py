import argparse
import os
import pickle
from sklearn.pipeline import Pipeline
import numpy as np
import joblib

DIR = "train_datas"


def load_pipeline(subject, task):
    """
load the trained pipeline object if it exist
    """
    pipeline_file = f'{DIR}/pipeline_s{subject}_t{task}.pkl'
    assert os.path.isfile(pipeline_file), \
        f"No datas found for subject {subject} and task {task}"
    trained_pipeline = joblib.load(pipeline_file)
    return trained_pipeline


def load_test_dataset(subject, task):
    """
load test dataset
    """
    test_data_file = f'{DIR}/test_data_s{subject}_t{task}.pkl'
    assert os.path.isfile(test_data_file), \
        f"No datas found for subject {subject} and task {task}"
    with open(test_data_file, 'rb') as f:
        test_dataset = pickle.load(f)
        return test_dataset


def get_predictions(subject, task, test_dataset: dict, trained_pipeline: Pipeline):
    """
Use the trained pipeline to make prediction with the test dataset and write \
the result
    """
    predictions = trained_pipeline.predict(test_dataset["X"])

    for index, value in enumerate(predictions):
        print(f"predicted [{value}] - real [{test_dataset['y'][index]}]")
    accuracy = np.mean(predictions == test_dataset["y"])
    print(f"Prediction accuracy for subject {subject} on task {task} : {accuracy:.2f}" )




def main(all, subject, task):
    if all:
        tasks_scores = np.empty((6, 109))
        for s in range(1, 110):
            subject_scores = np.empty(6)
            for t in range(1, 7):
                trained_pipeline = load_pipeline(s, t)
                test_dataset = load_test_dataset(s, t)
                predictions = trained_pipeline.predict(test_dataset["X"])
                score = np.mean(predictions == test_dataset["y"])
                subject_scores[t - 1] = score
                tasks_scores[t-1][s-1] = score
            print(f"prediction accuracy for subject {s} is {subject_scores.mean():.2f}")
        for t in range(4):
            print(f"task {t} accuracy score is {tasks_scores[t].mean():.2f}")
        print(f"Total accuracy score is {tasks_scores.mean(axis=0).mean():.2f}")

    else:
        trained_pipeline = load_pipeline(subject, task)
        test_dataset = load_test_dataset(subject, task)
        get_predictions(subject, task, test_dataset=test_dataset,
                        trained_pipeline=trained_pipeline)


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description='Predict events from \
unknown eeg datas for a given task and a given subject. \
Using physionet EEG Motor Movement/Imagery Dataset.')
        parser.add_argument('-a', '--all', action="store_true",
                            help="Gives result for all tasks and subjects")
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
        assert os.path.exists(DIR), "You must train your model first"
        main(args.all, args.subject[0], args.task[0])
    except Exception as msg:
        print(f"Error: {msg}")
