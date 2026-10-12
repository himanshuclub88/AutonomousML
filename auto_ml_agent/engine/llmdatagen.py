from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from ..llm import generate_reply


class GeneratedRows(BaseModel):
    """Structured response expected from the LLM."""

    rows: list[dict[str, Any]] = Field(
        description="Synthetic records using exactly the existing dataset columns."
    )


SYSTEM_PROMPT = """
Generate synthetic tabular records based on the provided dataset schema and profile.

Rules:
- Return exactly the requested number of records.
- Use exactly the provided column names. Do not add, rename, or omit columns.
- Follow the provided data types, numeric ranges, common categories, and relationships where possible.
- Do not copy example records verbatim.
- Return structured records only, without explanations.
- The records are synthetic and are not verified real-world observations.
"""


def build_dataset_profile(frame: pd.DataFrame) -> dict[str, Any]:
    """Build a compact schema/profile; never send the full DataFrame."""
    columns: list[dict[str, Any]] = []

    for name in frame.columns:
        series = frame[name]
        item: dict[str, Any] = {
            "name": str(name),
            "dtype": str(series.dtype),
            "missing_count": int(series.isna().sum()),
            "unique_count": int(series.nunique(dropna=True)),
        }

        if pd.api.types.is_numeric_dtype(series):
            values = pd.to_numeric(series, errors="coerce").dropna()
            if not values.empty:
                item.update(
                    minimum=float(values.min()),
                    maximum=float(values.max()),
                    mean=float(values.mean()),
                )
        else:
            counts = series.dropna().astype(str).value_counts().head(10)
            item["common_values"] = [
                {"value": str(value), "count": int(count)}
                for value, count in counts.items()
            ]

        columns.append(item)

    sample_frame = frame.head(5).astype(object)
    sample_frame = sample_frame.where(pd.notna(sample_frame), None)

    return {
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "columns": columns,
        "example_records": sample_frame.to_dict(orient="records"),
    }


def _cast_generated_rows(
    generated: pd.DataFrame,
    original: pd.DataFrame,
) -> pd.DataFrame:
    expected_columns = list(original.columns)
    missing = [column for column in expected_columns if column not in generated.columns]
    extra = [column for column in generated.columns if column not in expected_columns]
    if missing or extra:
        raise ValueError(
            f"Generated schema mismatch. Missing columns: {missing}; extra columns: {extra}."
        )

    generated = generated[expected_columns].copy()

    for column in expected_columns:
        dtype = original[column].dtype

        if pd.api.types.is_integer_dtype(dtype):
            values = pd.to_numeric(generated[column], errors="raise")
            if values.isna().any() or (values % 1 != 0).any():
                raise ValueError(f"Generated column {column!r} has invalid integer values.")
            generated[column] = values.astype(dtype)

        elif pd.api.types.is_float_dtype(dtype):
            generated[column] = pd.to_numeric(
                generated[column], errors="raise"
            ).astype(dtype)

        elif pd.api.types.is_bool_dtype(dtype):
            mapping = {
                True: True, False: False, 1: True, 0: False,
                "true": True, "false": False, "True": True, "False": False,
                "1": True, "0": False,
            }
            values = generated[column].map(mapping)
            if values.isna().any():
                raise ValueError(f"Generated column {column!r} has invalid boolean values.")
            generated[column] = values.astype(dtype)

        elif isinstance(dtype, pd.CategoricalDtype):
            allowed = set(original[column].dropna().unique())
            unknown = set(generated[column].dropna().unique()) - allowed
            if unknown:
                raise ValueError(
                    f"Generated column {column!r} has unknown categories: "
                    f"{list(unknown)[:5]}"
                )
            generated[column] = pd.Categorical(
                generated[column],
                categories=dtype.categories,
                ordered=dtype.ordered,
            )

        else:
            generated[column] = generated[column].astype(dtype)

    if generated.isna().any().any():
        missing_counts = generated.isna().sum()
        missing_counts = missing_counts[missing_counts > 0].to_dict()
        raise ValueError(f"Generated rows contain missing values: {missing_counts}")

    return generated


def generate_batch(
    frame: pd.DataFrame,
    rows: int,
    *,
    batch_number: int = 1,
    dataset_profile: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Generate exactly one batch. This function has no CLI input and no progress UI.

    `frame` is used for its schema and a small sample; the full dataset is not sent.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")
    if frame.empty or len(frame.columns) < 2:
        raise ValueError("frame must contain rows and at least two columns.")
    if rows < 1:
        raise ValueError("rows must be a positive integer.")

    profile = dataset_profile or build_dataset_profile(frame)
    request = {
        "requested_rows": rows,
        "batch_number": batch_number,
        "dataset_profile": profile,
    }

    with open("prompt.txt", "w", encoding="utf-8") as f:
        f.write(f"SYSTEM PROMPT:\n{SYSTEM_PROMPT}\n\n")
        f.write(f"MESSAGES:\n[{ {'role': 'user', 'content': str(request)} }]\n\n")
        f.write(f"RESPONSE MODEL:\n{GeneratedRows}")

    response = generate_reply(
        system_prompt=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": str(request)}],
        response_model=GeneratedRows,
    )

    if not isinstance(response, GeneratedRows):
        raise TypeError("The LLM returned an unexpected response type.")
    if len(response.rows) != rows:
        raise ValueError(
            f"Batch {batch_number}: requested {rows} rows, received {len(response.rows)}."
        )

    return _cast_generated_rows(pd.DataFrame(response.rows), frame)
