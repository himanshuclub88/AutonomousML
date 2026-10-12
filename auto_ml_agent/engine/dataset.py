from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

def inspect_csv(path: str) -> tuple[pd.DataFrame, list[str]]:
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"CSV not found: {file_path}")
    frame = pd.read_csv(file_path)
    if frame.empty or len(frame.columns) < 2:
        raise ValueError("CSV must contain rows and at least two columns (features plus target).")
    return frame, list(frame.columns)

def generate_synthetic(problem_type: str, target: str, features: list[str], rows: int) -> pd.DataFrame:
    n_features = max(2, len(features))
    if problem_type == "regression":
        X, y = make_regression(n_samples=rows, n_features=n_features, n_informative=min(n_features, 8), noise=15.0, random_state=42)
    else:
        X, y = make_classification(n_samples=rows, n_features=n_features,
                                   n_informative=max(2, min(n_features, 6)),
                                   n_redundant=0, n_repeated=0, n_classes=2,
                                   n_clusters_per_class=1, random_state=42)
    names = list(features)
    while len(names) < n_features:
        names.append(f"feature_{len(names)+1}")
    frame = pd.DataFrame(X, columns=names[:n_features])
    frame[target] = y
    return frame

def split_data(frame: pd.DataFrame, target: str, problem_type: str):
    if target not in frame.columns:
        raise ValueError(f"Target column {target!r} is not in the dataset.")
    clean = frame.dropna(subset=[target]).copy()
    if len(clean) < 30:
        raise ValueError("Need at least 30 usable rows for this prototype.")
    X = clean.drop(columns=[target])
    y = clean[target]
    if X.empty:
        raise ValueError("No feature columns remain after removing target.")
    stratify = y if problem_type == "classification" and y.nunique() > 1 and y.value_counts().min() >= 3 else None
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=stratify
    )
    stratify_temp = y_temp if problem_type == "classification" and y_temp.nunique() > 1 and y_temp.value_counts().min() >= 2 else None
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=stratify_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
