from pathlib import Path
from sklearn.model_selection import train_test_split

import pandas as pd


DATA_DIR = Path(file).resolve().parent / "data"


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