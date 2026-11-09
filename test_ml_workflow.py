import pandas as pd

from ml_workflow import split_dataset


def test_split_dataset_preserves_class_proportions():
    X = pd.DataFrame({"feature": range(100)})
    y = pd.Series([0] * 90 + [1] * 10)

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    assert y_train.value_counts().to_dict() == {0: 72, 1: 8}
    assert y_test.value_counts().to_dict() == {0: 18, 1: 2}
    assert X_train.index.equals(y_train.index)
    assert X_test.index.equals(y_test.index)
    assert X_train.index.intersection(X_test.index).empty