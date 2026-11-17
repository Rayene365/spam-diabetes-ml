from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


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
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    class_weight: str | None = None,
) -> Pipeline:
    """Fit a logistic regression pipeline with optional class weighting."""
    model = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight=class_weight,
                ),
            ),
        ]
    )
    return model.fit(X_train, y_train)


def evaluate_classifier(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Evaluate binary labels 0 and 1 using probabilities or decision scores."""
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score),
        "average_precision": average_precision_score(y_test, y_score),
    }


def cross_validate_classifier(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return validation metrics for each stratified fold of the training data."""
    folds = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=folds,
        scoring=evaluate_classifier,
        error_score="raise",
    )
    validation_scores = {
        key.removeprefix("test_"): values
        for key, values in scores.items()
        if key.startswith("test_")
    }
    return pd.DataFrame(
        validation_scores,
        index=range(1, n_splits + 1),
    ).rename_axis("fold")


def build_models(random_state: int = 42) -> dict[str, Pipeline]:
    """Build the classifiers used in the model comparison."""
    classifiers = {
        "Logistic regression": LogisticRegression(max_iter=1000),
        "Logistic regression (balanced)": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        ),
        "Gaussian Naive Bayes": GaussianNB(),
        "Linear SVM (balanced)": LinearSVC(
            C=1.0,
            class_weight="balanced",
            dual=False,
            max_iter=5000,
            random_state=random_state,
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(32,),
            batch_size=256,
            max_iter=300,
            early_stopping=True,
            n_iter_no_change=15,
            random_state=random_state,
        ),
    }
    return {
        name: Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("classifier", classifier),
            ]
        )
        for name, classifier in classifiers.items()
    }


def compare_models(
    models: dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return mean/std metrics and per-fold scores for each model."""
    fold_scores = {}
    for name, model in models.items():
        print(f"Cross-validating {name}", flush=True)
        fold_scores[name] = cross_validate_classifier(
            model,
            X_train,
            y_train,
            n_splits=n_splits,
            random_state=random_state,
        )

    per_fold = pd.concat(fold_scores, names=["model", "fold"])
    summary = per_fold.groupby(level="model", sort=False).agg(["mean", "std"])
    return summary, per_fold
