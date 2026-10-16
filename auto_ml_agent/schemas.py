
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


ProblemType = Literal[
    "classification",
    "regression",
    "forecasting",
    "clustering",
    "anomaly_detection",
    "dimensionality_reduction",
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

    # Dimensionality reduction
    "explained_variance",
    "reconstruction_error",

    # Tasks where a conventional metric is unavailable
    "not_applicable",
]


TASK_METRICS = {
    "classification": {
        "accuracy",
        "f1_weighted",
        "roc_auc",
        "average_precision",
    },
    "regression": {"rmse", "mae", "r2"},
    "forecasting": {"rmse", "mae", "r2", "mape", "smape"},
    "clustering": {
        "silhouette_score",
        "davies_bouldin_score",
        "calinski_harabasz_score",
        "not_applicable",
    },
    "anomaly_detection": {
        "roc_auc",
        "average_precision",
        "not_applicable",
    },
    "dimensionality_reduction": {
        "explained_variance",
        "reconstruction_error",
        "not_applicable",
    },
}


class MLExperimentPlan(BaseModel):
    problem_type: ProblemType = Field(
        description=(
            "Choose classification, regression, forecasting, clustering, "
            "anomaly_detection, or dimensionality_reduction."
        )
    )

    target_column: str | None = Field(
        default=None,
        description=(
            "Target column for supervised learning and forecasting. "
            "May be None for clustering, unsupervised anomaly detection, "
            "and dimensionality reduction."
        ),
    )

    feature_columns: list[str] = Field(
        min_length=1,
        description="Input columns used by the experiment.",
    )

    candidate_models: list[str] = Field(
        min_length=1,
        max_length=4,
        description=(
            "Between one and four supported model identifiers "
            "appropriate for the selected task."
        ),
    )

    evaluation_metric: EvaluationMetric = Field(
        description="Evaluation metric appropriate for the selected task.",
    )

    generate_synthetic_data: bool = Field(
        description=(
            "Whether synthetic demonstration data should be generated. "
            "Python code must enforce the actual data-source choice."
        ),
    )

    synthetic_rows: int = Field(
        default=3000,
        ge=200,
        le=20000,
        description="Number of rows for synthetic demonstration data.",
    )

    reasoning: str = Field(
        description="Brief explanation of the selected task and models.",
    )

    @model_validator(mode="after")
    def validate_plan(self) -> "MLExperimentPlan":
        supervised_tasks = {
            "classification",
            "regression",
            "forecasting",
        }

        if self.problem_type in supervised_tasks and not self.target_column:
            raise ValueError(
                f"{self.problem_type} requires a target column."
            )

        if (
            self.target_column is not None
            and self.target_column in self.feature_columns
        ):
            raise ValueError(
                "The target column cannot also be a feature column."
            )

        if len(self.feature_columns) != len(set(self.feature_columns)):
            raise ValueError("Feature columns must not contain duplicates.")

        if len(self.candidate_models) != len(set(self.candidate_models)):
            raise ValueError("Candidate models must not contain duplicates.")

        if self.evaluation_metric not in TASK_METRICS[self.problem_type]:
            raise ValueError(
                f"Metric {self.evaluation_metric!r} is not appropriate "
                f"for {self.problem_type}."
            )

        return self


class ExperimentRequest(BaseModel):
    use_case: str = Field(min_length=1)

    backend: Literal["local", "colab"] = "local"

    dataset_path: str | None = None

    target_column: str | None = None
