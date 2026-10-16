
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


ProblemType = Literal[
    "classification",
    "regression",
    "forecasting",
    "clustering",
    "anomaly_detection",
]

EvaluationMetric = Literal[
    # Classification
    "accuracy",
    "f1_weighted",
    "roc_auc",
    "average_precision",

    # Regression and forecasting
    "rmse",
    "mae",
    "r2",
    "mape",
    "smape",

    # Clustering
    "silhouette_score",
    "davies_bouldin_score",
    "calinski_harabasz_score",
]


class MLExperimentPlan(BaseModel):
    problem_type: ProblemType = Field(
        description=(
            "ML task type: classification, regression, forecasting, "
            "clustering, or anomaly_detection."
        )
    )

    target_column: str | None = Field(
        default=None,
        description=(
            "Target column for supervised tasks and forecasting. "
            "May be None for clustering or unsupervised anomaly detection."
        ),
    )

    feature_columns: list[str] = Field(
        min_length=1,
        description="Input feature columns used by the experiment.",
    )

    candidate_models: list[str] = Field(
        min_length=1,
        max_length=4,
        description=(
            "One to four supported model identifiers appropriate "
            "for the selected ML task."
        ),
    )

    evaluation_metric: EvaluationMetric = Field(
        description="Evaluation metric appropriate for the selected task.",
    )

    generate_synthetic_data: bool = Field(
        description=(
            "Whether to use generated demonstration data instead of "
            "an uploaded dataset."
        ),
    )

    synthetic_rows: int = Field(
        default=3000,
        ge=200,
        le=20000,
        description="Number of rows to generate for synthetic data.",
    )

    reasoning: str = Field(
        description="Brief explanation of why this task and models were selected.",
    )

    @model_validator(mode="after")
    def validate_task_requirements(self) -> "MLExperimentPlan":
        supervised_tasks = {
            "classification",
            "regression",
            "forecasting",
        }

        if self.problem_type in supervised_tasks and not self.target_column:
            raise ValueError(
                f"{self.problem_type} requires a target_column."
            )

        if self.target_column and self.target_column in self.feature_columns:
            raise ValueError(
                "target_column must not also appear in feature_columns."
            )

        if len(self.candidate_models) != len(set(self.candidate_models)):
            raise ValueError("candidate_models must not contain duplicates.")

        return self


class ExperimentRequest(BaseModel):
    use_case: str = Field(
        min_length=1,
        description="Natural-language description of the ML problem.",
    )

    backend: Literal["local", "colab"] = "local"

    dataset_path: str | None = None

    target_column: str | None = None
