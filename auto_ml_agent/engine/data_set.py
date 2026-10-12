from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from ..llm import generate_reply


class GeneratedRows(BaseModel):
    """Structured LLM output: rows matching the input DataFrame schema."""

    rows: list[dict[str, Any]] = Field(
        description="Synthetic records using exactly the supplied dataset columns."
    )


SYSTEM_PROMPT = """
Generate synthetic tabular records based on the provided dataset schema and examples.

Requirements:
- Return exactly the requested number of records.
- Use exactly the existing column names; do not add or omit columns.
- Follow the supplied data types, numeric ranges, common categories, and relationships where possible.
- Do not copy the example records verbatim.
- Return structured data only, without explanations.
- The generated records are synthetic examples, not verified real-world observations.
"""


def _profile_dataset(frame: pd.DataFrame) -> dict[str, Any]:
    """Describe columns and a few examples without sending the full dataset."""
    columns = []

    for name in frame.columns:
        series = frame[name]
        column_info: dict[str, Any] = {
            "name": str(name),
            "dtype": str(series.dtype),
            "missing": int(series.isna().sum()),
        }

        if pd.api.types.is_numeric_dtype(series):
            values = pd.to_numeric(series, errors="coerce").dropna()
            if not values.empty:
                column_info.update(
                    minimum=float(values.min()),
                    maximum=float(values.max()),
                    mean=float(values.mean()),
                )
        else:
            column_info["common_values"] = (
                series.dropna().astype(str).value_counts().head(10).index.tolist()
            )

        columns.append(column_info)

    sample = frame.head(5).astype(object).where(pd.notna(frame.head(5)), None)

    return {
        "rows": len(frame),
        "columns": columns,
        "examples": sample.to_dict(orient="records"),
    }


def _cast_to_original_types(
    generated: pd.DataFrame,
    original: pd.DataFrame,
) -> pd.DataFrame:
    """Keep generated columns aligned with the original schema."""
    if list(generated.columns) != list(original.columns):
        raise ValueError("Generated columns do not match the original column names and order.")

    result = generated.copy()

    for column in original.columns:
        dtype = original[column].dtype

        if pd.api.types.is_integer_dtype(dtype):
            values = pd.to_numeric(result[column], errors="raise")
            if values.isna().any() or (values % 1 != 0).any():
                raise ValueError(f"Generated column {column!r} has invalid integer values.")
            result[column] = values.astype(dtype)

        elif pd.api.types.is_float_dtype(dtype):
            result[column] = pd.to_numeric(result[column], errors="raise").astype(dtype)

        elif isinstance(dtype, pd.CategoricalDtype):
            allowed = set(original[column].dropna().unique())
            unknown = set(result[column].dropna().unique()) - allowed
            if unknown:
                raise ValueError(
                    f"Generated column {column!r} contains unknown categories: "
                    f"{list(unknown)[:5]}"
                )
            result[column] = pd.Categorical(
                result[column],
                categories=dtype.categories,
                ordered=dtype.ordered,
            )

        elif pd.api.types.is_bool_dtype(dtype):
            mapping = {
                True: True, False: False, 1: True, 0: False,
                "true": True, "false": False, "True": True, "False": False,
                "1": True, "0": False,
            }
            values = result[column].map(mapping)
            if values.isna().any():
                raise ValueError(f"Generated column {column!r} has invalid boolean values.")
            result[column] = values.astype(dtype)

        else:
            result[column] = result[column].astype(dtype)

    if result.isna().any().any():
        raise ValueError("Generated rows contain missing values.")

    return result


def augment_dataset(
    frame: pd.DataFrame,
    additional_rows: int,
) -> pd.DataFrame:
    """
    Generate and append synthetic rows.

    This function does not ask for user input, making it suitable for CLI,
    API, or other callers. It returns a new combined DataFrame and does not
    modify the input DataFrame.
    """

    batch_size = additional_rows//200

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")
    if frame.empty:
        raise ValueError("Cannot augment an empty DataFrame.")
    if len(frame.columns) < 2:
        raise ValueError("The dataset must contain at least two columns.")
    if additional_rows < 1:
        raise ValueError("additional_rows must be a positive integer.")
    if batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")

    schema = _profile_dataset(frame)
    generated_batches: list[pd.DataFrame] = []
    remaining = additional_rows
    batch_number = 1

    while remaining > 0:
        count = min(batch_size, remaining)
        request = {
            "requested_rows": count,
            "batch_number": batch_number,
            "dataset_profile": schema,
        }

        response = generate_reply(
            system_prompt=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": str(request)}],
            response_model=GeneratedRows,
        )

        if not isinstance(response, GeneratedRows):
            raise TypeError("The LLM returned an unexpected response type.")
        if len(response.rows) != count:
            raise ValueError(
                f"Requested {count} rows but the LLM returned {len(response.rows)}."
            )

        batch = pd.DataFrame(response.rows)
        expected_columns = list(frame.columns)

        missing = [column for column in expected_columns if column not in batch.columns]
        extra = [column for column in batch.columns if column not in expected_columns]
        if missing or extra:
            raise ValueError(
                f"Generated schema mismatch. Missing columns: {missing}; "
                f"unexpected columns: {extra}."
            )

        batch = _cast_to_original_types(batch[expected_columns], frame)
        generated_batches.append(batch)
        remaining -= count
        batch_number += 1

    generated = pd.concat(generated_batches, ignore_index=True)
    return pd.concat([frame.copy(), generated], ignore_index=True)
