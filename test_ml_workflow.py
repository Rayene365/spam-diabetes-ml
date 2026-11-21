import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.svm import LinearSVC

from ml_workflow import (
    evaluate_classifier,
    split_dataset,
    train_logistic_baseline,
)


def test_split_dataset_preserves_class_proportions():
    X = pd.DataFrame({"feature": range(100)})
    y = pd.Series([0] * 90 + [1] * 10)

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    assert y_train.value_counts().to_dict() == {0: 72, 1: 8}
    assert y_test.value_counts().to_dict() == {0: 18, 1: 2}
    assert X_train.index.equals(y_train.index)
    assert X_test.index.equals(y_test.index)
    assert X_train.index.intersection(X_test.index).empty


def test_split_dataset_is_reproducible():
    X = pd.DataFrame({"feature": range(100)})
    y = pd.Series([0] * 90 + [1] * 10)

    first_split = split_dataset(X, y, random_state=67)
    second_split = split_dataset(X, y, random_state=67)
    different_split = split_dataset(X, y, random_state=42)

    for first, second in zip(first_split, second_split):
        assert first.equals(second)
    assert not first_split[1].index.equals(different_split[1].index)


def test_logistic_baseline_handles_missing_values():
    X_train = pd.DataFrame({"feature": [-3.0, -2.0, float("nan"), 2.0, 3.0, 4.0]})
    y_train = pd.Series([0, 0, 0, 1, 1, 1])
    X_test = pd.DataFrame({"feature": [-4.0, float("nan"), 5.0]})

    model = train_logistic_baseline(X_train, y_train)
    predictions = model.predict(X_test)

    assert len(predictions) == len(X_test)
    assert set(predictions) <= {0, 1}
    assert predictions[[0, 2]].tolist() == [0, 1]


def test_evaluate_classifier_with_no_positive_predictions():
    X = pd.DataFrame({"feature": [0.0, 1.0, 2.0, 3.0]})
    y = pd.Series([0, 0, 0, 1])
    model = DummyClassifier(strategy="most_frequent").fit(X, y)

    scores = evaluate_classifier(model, X, y)

    assert scores == pytest.approx({
        "accuracy": 0.75,
        "balanced_accuracy": 0.5,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "roc_auc": 0.5,
        "average_precision": 0.25,
    })


def test_evaluate_classifier_with_svm_decision_scores():
    X_train = pd.DataFrame({"feature": [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]})
    y_train = pd.Series([0, 0, 0, 1, 1, 1])
    X_test = pd.DataFrame({"feature": [-2.0, -1.0, 1.0, 2.0]})
    y_test = pd.Series([0, 1, 0, 1])
    model = LinearSVC(dual=False, random_state=42).fit(X_train, y_train)

    scores = evaluate_classifier(model, X_test, y_test)

    # The predicted labels are [0, 0, 1, 1], and decision scores increase with X.
    assert scores == pytest.approx({
        "accuracy": 0.5,
        "balanced_accuracy": 0.5,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "roc_auc": 0.75,
        "average_precision": 5 / 6,
    })
