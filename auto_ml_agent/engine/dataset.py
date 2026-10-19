
from __future__ import annotations
import json
from sklearn.model_selection import train_test_split
from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression, make_blobs

def inspect_csv(path: str) -> tuple[pd.DataFrame, list[str]]:
    """Load a CSV dataset and return its columns."""
    file_path = Path(path).expanduser().resolve()

    if not file_path.is_file():
        raise FileNotFoundError(f"CSV not found: {file_path}")

    df = pd.read_csv(file_path)

    if df.empty or len(df.columns) < 2:
        raise ValueError(
            "CSV must contain rows and at least two columns."
        )

    return df, list(df.columns)



def generate_synthetic(
    problem_type: str,
    target: str,
    features: list[str],
    rows: int,
    *,
    use_case: str = "",
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate domain-specific or generic synthetic tabular data."""

    if rows < 30:
        raise ValueError("At least 30 rows are required.")

    features = [f for f in features if f != target]

    if not features:
        raise ValueError("At least one feature is required.")

    # Generic fallback for other ML tasks.
    n_features = max(2, len(features))

    if problem_type in {"clustering", "anomaly_detection", "dimensionality_reduction"}:
        # Unsupervised workflows have no label. main.py still requires a target
        # field, so that field is generated as an ordinary input feature.
        input_names = list(dict.fromkeys(features + [target]))
        n_features = max(2, len(input_names))
        rng = np.random.default_rng(random_state)

        if problem_type == "clustering":
            centers = min(4, max(2, rows // 20))
            X, _ = make_blobs(
                n_samples=rows,
                n_features=n_features,
                centers=centers,
                cluster_std=1.8,
                random_state=random_state,
            )
        else:
            X = rng.normal(size=(rows, n_features))
            if problem_type == "anomaly_detection":
                anomaly_count = max(1, int(rows * 0.03))
                anomaly_rows = rng.choice(rows, size=anomaly_count, replace=False)
                X[anomaly_rows] += rng.normal(0, 8, size=(anomaly_count, n_features))
            else:
                # Correlated dimensions make PCA/SVD demonstrations meaningful.
                if n_features > 1:
                    X[:, 1:] = 0.7 * X[:, [0]] + 0.3 * X[:, 1:]

        while len(input_names) < n_features:
            input_names.append(f"feature_{len(input_names) + 1}")
        return pd.DataFrame(X, columns=input_names[:n_features])

    if problem_type == "regression":
        X, y = make_regression(
            n_samples=rows,
            n_features=n_features,
            n_informative=min(n_features, 8),
            noise=15,
            random_state=random_state,
        )

    elif problem_type == "classification":
        X, y = make_classification(
            n_samples=rows,
            n_features=n_features,
            n_informative=max(2, min(n_features, 6)),
            n_redundant=0,
            n_classes=2,
            random_state=random_state,
        )

    else:
        raise ValueError(
            f"Unsupported problem type: {problem_type}"
        )

    names = list(features)

    while len(names) < n_features:
        names.append(f"feature_{len(names) + 1}")

    df = pd.DataFrame(X, columns=names[:n_features])
    df[target] = y

    return df


def validate_generated_data(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    expected_rows: int,
) -> None:
    """Check basic dataset integrity before training."""

    required = features + [target]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing generated columns: {missing}"
        )

    if len(df) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows; got {len(df)}."
        )

    if df.empty:
        raise ValueError("Dataset is empty.")

    if df[required].isna().any().any():
        raise ValueError(
            "Generated dataset contains missing values."
        )

    if df[target].nunique() < 2:
        raise ValueError(
            "Target must have at least two unique values."
        )


def split_data(
    frame: pd.DataFrame,
    target: str,
    problem_type: str,
):
    """Split data into training, validation, and test sets."""

    if target not in frame.columns:
        raise ValueError(
            f"Target column {target!r} is not in the dataset."
        )

    clean = frame.dropna(subset=[target]).copy()

    if len(clean) < 30:
        raise ValueError(
            "Need at least 30 usable rows for training."
        )

    X = clean.drop(columns=[target])
    y = clean[target]

    if X.empty:
        raise ValueError("No input features remain.")

    stratify = None

    if (
        problem_type == "classification"
        and y.nunique() > 1
        and y.value_counts().min() >= 3
    ):
        stratify = y

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=stratify,
    )

    stratify_temp = None

    if (
        problem_type == "classification"
        and y_temp.nunique() > 1
        and y_temp.value_counts().min() >= 2
    ):
        stratify_temp = y_temp

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=stratify_temp,
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )

def prepare_dataset(
    frame: pd.DataFrame,
    target: str,
    features: list[str],
    problem_type: str,
    *,
    remove_duplicates: bool = False,
    min_rows: int = 30,
) -> tuple[pd.DataFrame, dict]:
    """
    Validate and prepare tabular data before training.

    Policy:
    - Drop rows with a missing/invalid target (the target cannot be imputed safely).
    - Convert +/- infinity in numeric columns to NaN so the trainer's imputers can handle
      missing feature values using training-fold statistics.
    - Detect duplicate rows, but preserve them by default.
    - Do not silently remove constant or all-missing features; report them and fail for
      all-missing planned features because they cannot provide useful training information.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")

    if not target or target not in frame.columns:
        raise ValueError(f"Target column {target!r} is not present in the dataset.")

    features = list(dict.fromkeys(features))
    if target in features:
        features.remove(target)
    if not features:
        raise ValueError("At least one feature column is required.")

    missing_features = [column for column in features if column not in frame.columns]
    if missing_features:
        raise ValueError(f"Planned feature columns are missing from the dataset: {missing_features}")

    supported_types = {
        "classification", "regression", "clustering",
        "anomaly_detection", "dimensionality_reduction",
    }
    if problem_type not in supported_types:
        raise ValueError(f"Unsupported problem type: {problem_type!r}")

    required = features + [target]
    clean = frame[required].copy()
    rows_before = int(len(clean))

    missing_before = {
        str(column): int(count)
        for column, count in clean.isna().sum().items()
        if int(count) > 0
    }

    # Infinite numeric values are invalid data. Treat them as missing so feature
    # imputers can handle them; invalid target rows are dropped below.
    numeric_columns = clean.select_dtypes(include=[np.number]).columns.tolist()
    infinity_counts = {}
    for column in numeric_columns:
        mask = np.isinf(clean[column].to_numpy(dtype=float, na_value=np.nan))
        count = int(mask.sum())
        if count:
            infinity_counts[str(column)] = count
            clean.loc[mask, column] = np.nan

    is_unsupervised = problem_type in {
        "clustering", "anomaly_detection", "dimensionality_reduction"
    }
    invalid_target_rows = int(clean[target].isna().sum())
    if not is_unsupervised:
        clean = clean.dropna(subset=[target]).copy()

    # Regression requires a numeric target. Avoid silently converting arbitrary
    # strings into numbers or guessing how labels should be interpreted.
    if problem_type == "regression" and not pd.api.types.is_numeric_dtype(clean[target]):
        raise ValueError(
            f"Regression target {target!r} must be numeric; found dtype "
            f"{clean[target].dtype!s}."
        )

    if not is_unsupervised and clean[target].nunique(dropna=True) < 2:
        raise ValueError(
            f"Target {target!r} has fewer than two unique values after removing "
            "rows with missing/invalid targets."
        )

    columns_to_check = required if is_unsupervised else features
    all_missing_features = [
        column for column in columns_to_check if clean[column].isna().all()
    ]
    if all_missing_features:
        raise ValueError(
            "These planned feature columns contain no usable values: "
            f"{all_missing_features}. Remove them or provide valid values."
        )

    duplicate_count = int(clean.duplicated(subset=required).sum())
    duplicates_removed = 0
    if remove_duplicates and duplicate_count:
        before_dedup = len(clean)
        clean = clean.drop_duplicates(subset=required).copy()
        duplicates_removed = int(before_dedup - len(clean))

    if len(clean) < min_rows:
        raise ValueError(
            f"Only {len(clean)} usable rows remain; at least {min_rows} are required."
        )

    if problem_type == "classification" and clean[target].nunique() < 2:
        raise ValueError("Classification requires at least two target classes.")

    # Features that have only one distinct non-null value may be uninformative.
    constant_features = [
        column for column in features
        if clean[column].nunique(dropna=True) <= 1
    ]

    missing_after = {
        str(column): int(count)
        for column, count in clean[required].isna().sum().items()
        if int(count) > 0
    }

    report = {
        "status": "passed",
        "problem_type": problem_type,
        "target_column": target,
        "feature_columns": features,
        "rows_before": rows_before,
        "rows_after": int(len(clean)),
        "rows_dropped_for_missing_target": 0 if is_unsupervised else invalid_target_rows,
        "missing_values_in_target_placeholder": invalid_target_rows if is_unsupervised else 0,
        "duplicate_rows_detected": duplicate_count,
        "duplicate_rows_removed": duplicates_removed,
        "remove_duplicates_enabled": remove_duplicates,
        "missing_values_before": missing_before,
        "infinite_values_converted_to_missing": infinity_counts,
        "missing_values_after": missing_after,
        "constant_features": constant_features,
        "all_missing_features": all_missing_features,
        "target_unique_values": (
            int(clean[target].nunique(dropna=True)) if not is_unsupervised else None
        ),
        "target_is_compatibility_placeholder": bool(is_unsupervised),
        "target_class_counts": (
            {str(k): int(v) for k, v in clean[target].value_counts(dropna=False).items()}
            if problem_type == "classification"
            else None
        ),
        "notes": [
            "Missing feature values are left for the training pipeline's imputers.",
            "For unsupervised tasks, target_column is a compatibility placeholder and is treated as an input feature.",
            "Duplicate rows are reported and preserved unless remove_duplicates=True.",
            "This report describes data checks; it does not prove the dataset is unbiased or leakage-free.",
        ],
    }

    return clean, report


def write_data_quality_report(report: dict, path: str | Path) -> None:
    """Save the data quality report as readable JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

def analyze_features(
    frame: pd.DataFrame,
    target: str,
    features: list[str],
    *,
    high_cardinality_ratio: float = 0.50,
    identifier_unique_ratio: float = 0.90,
) -> dict:
    """Flag possible feature risks without automatically dropping columns."""
    if target not in frame.columns:
        raise ValueError(f"Target column {target!r} is not present in the dataset.")

    missing_features = [column for column in features if column not in frame.columns]
    if missing_features:
        raise ValueError(f"Feature columns are missing from the dataset: {missing_features}")

    id_name_pattern = re.compile(
        r"(^|[\s_\-])(id|uuid|guid|key|index|identifier|record|account|customer_id|"
        r"transaction_id|row_number)([\s_\-]|$)",
        flags=re.IGNORECASE,
    )
    leakage_name_pattern = re.compile(
        r"(target|label|outcome|actual|ground.?truth|future|post.?event|"
        r"final.?price|sale.?price|sold.?price|price.?after)",
        flags=re.IGNORECASE,
    )

    feature_reports = []
    for column in features:
        series = frame[column]
        non_null = series.dropna()
        unique_count = int(non_null.nunique())
        unique_ratio = float(unique_count / max(1, len(non_null)))
        name = str(column)
        flags = []
        recommendations = []

        if unique_count <= 1:
            flags.append("constant_feature")
            recommendations.append("Consider excluding: the feature has no variation.")

        if id_name_pattern.search(name):
            flags.append("identifier_like_name")
            recommendations.append(
                "Review whether this is an identifier rather than a meaningful predictive feature."
            )

        is_numeric = pd.api.types.is_numeric_dtype(series)

        if (
            not is_numeric
            and unique_ratio >= identifier_unique_ratio
            and len(non_null) >= 20
        ):
            flags.append("near_unique_values")
            recommendations.append(
                "Review for unique identifiers or categorical values "
                "that may not generalize."
            )

        is_categorical = (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
        )
        if is_categorical and unique_count > 20 and unique_ratio >= high_cardinality_ratio:
            flags.append("high_cardinality_categorical")
            recommendations.append(
                "Review encoding cost and generalization; consider domain-specific encoding or feature engineering."
            )

        if leakage_name_pattern.search(name):
            flags.append("possible_target_leakage_by_name")
            recommendations.append(
                "Check whether this column is derived from the target or only available after the prediction point."
            )

        feature_reports.append({
            "column": name,
            "dtype": str(series.dtype),
            "rows": int(len(series)),
            "missing_values": int(series.isna().sum()),
            "unique_values": unique_count,
            "unique_ratio": round(unique_ratio, 4),
            "flags": flags,
            "recommendations": recommendations,
        })

    flagged = [item for item in feature_reports if item["flags"]]
    return {
        "target_column": target,
        "feature_count": len(features),
        "features_flagged": len(flagged),
        "feature_analysis": feature_reports,
        "important_limitations": [
            "These are heuristic warnings, not proof that a feature is harmful or leaks the target.",
            "Columns are not automatically removed; review recommendations before changing the training features.",
            "Address/location features can be predictive, but raw high-cardinality addresses may overfit or increase encoding cost.",
        ],
    }
