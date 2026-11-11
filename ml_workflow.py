from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / "data"


def load_spambase() -> tuple[pd.DataFrame, pd.Series]:
    """Return the Spambase features and spam labels."""
    dataset_dir = DATA_DIR / "spambase"
    lines = (dataset_dir / "spambase.names").read_text(encoding="utf-8").splitlines()
    feature_names = [
        line.split(":", 1)[0].strip()
        for line in lines
        if line.rstrip().endswith("continuous.")
    ]

    data = pd.read_csv(dataset_dir / "spambase.data", header=None)
    data.columns = feature_names + ["spam"]

    return data.drop(columns="spam"), data["spam"]

def load_diabetes() -> tuple[pd.DataFrame, pd.Series]:
    """Return the BRFSS 2015 features and binary diabetes labels."""
    filename = "diabetes_binary_health_indicators_BRFSS2015.csv"
    data = pd.read_csv(DATA_DIR / "diabetes" / filename)
    target = "Diabetes_binary"

    return data.drop(columns=target), data[target].astype(int)

def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the data while keeping similar class proportions."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test

def train_logistic_baseline(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Pipeline:
    """Fit a logistic regression pipeline on the training split."""
    model = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )
    return model.fit(X_train, y_train)

def evaluate_classifier(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Evaluate a fitted classifier with binary labels 0 and 1."""
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score),
        "average_precision": average_precision_score(y_test, y_score),
    }