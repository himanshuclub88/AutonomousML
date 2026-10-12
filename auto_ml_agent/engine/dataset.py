
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression


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


def is_house_price_problem(
    use_case: str,
    target: str,
    features: list[str],
) -> bool:
    text = " ".join([use_case, target, *features]).lower()

    keywords = [
        "house price",
        "home price",
        "property price",
        "real estate",
        "house_price",
        "sale_price",
        "housing",
    ]

    return any(word in text for word in keywords)


def generate_house_dataset(
    target: str,
    features: list[str],
    rows: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate correlated, illustrative house-price data."""

    rng = np.random.default_rng(random_state)

    # Generate underlying property characteristics.
    area = np.clip(
        rng.normal(1800, 650, rows), 450, 6000
    )

    bedrooms = np.clip(
        np.round(area / 650 + rng.normal(0.5, 0.8, rows)),
        1, 7,
    ).astype(int)

    bathrooms = np.clip(
        np.round(bedrooms * 0.65 + rng.normal(0.5, 0.6, rows)),
        1, 5,
    ).astype(int)

    year_built = rng.integers(1950, 2026, rows)
    property_age = 2026 - year_built

    lot_size = np.clip(
        area * rng.uniform(1.2, 4.5, rows),
        500, 30000,
    )

    neighborhood_quality = np.clip(
        rng.normal(6.5, 1.8, rows), 1, 10
    )

    distance = np.clip(
        rng.gamma(2, 3, rows), 0.2, 35
    )

    garage_spaces = np.clip(
        np.round(bedrooms / 2 + rng.normal(0.2, 0.7, rows)),
        0, 4,
    ).astype(int)

    has_basement = rng.binomial(1, 0.45, rows)
    has_pool = rng.binomial(1, 0.22, rows)

    known_features = {
        "square_feet": area,
        "area": area,
        "living_area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "year_built": year_built,
        "property_age": property_age,
        "age": property_age,
        "lot_size_sqft": lot_size,
        "lot_size": lot_size,
        "neighborhood_quality": neighborhood_quality,
        "location_quality": neighborhood_quality,
        "distance_to_city_center": distance,
        "garage_spaces": garage_spaces,
        "has_basement": has_basement,
        "has_pool": has_pool,
    }

    df = pd.DataFrame()

    for feature in features:
        key = re.sub(
            r"[^a-z0-9]+", "_", feature.lower()
        ).strip("_")

        if key in known_features:
            df[feature] = known_features[key]

        elif "location" in key or "neighborhood" in key:
            df[feature] = rng.choice(
                ["urban", "suburban", "rural"],
                size=rows,
                p=[0.35, 0.45, 0.20],
            )

        else:
            # Unknown features receive placeholder values.
            df[feature] = rng.normal(0, 1, rows)

    # Illustrative target formula with random noise.
    # Monetary units are arbitrary, not real market prices.
    price = (
        45000
        + area * 155
        + bedrooms * 9000
        + bathrooms * 12500
        + (neighborhood_quality - 5) * 14000
        - property_age * 650
        + lot_size * 1.8
        - distance * 1700
        + garage_spaces * 4500
        + has_basement * 7000
        + has_pool * 11000
        + rng.normal(0, 25000, rows)
    )

    df[target] = np.maximum(price, 25000).round(2)

    return df


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

    if (
        problem_type == "regression"
        and is_house_price_problem(use_case, target, features)
    ):
        return generate_house_dataset(
            target, features, rows, random_state
        )

    # Generic fallback for other ML tasks.
    n_features = max(2, len(features))

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
