from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field

ProblemType = Literal[
    "classification",
    "regression",
    "clustering",
    "anomaly_detection",
    "dimensionality_reduction",
]

class MLExperimentPlan(BaseModel):
    problem_type: ProblemType = Field(
        description=(
            "Classification, regression, clustering, anomaly detection, "
            "or dimensionality reduction."
        )
    )
    target_column: str = Field(
        description=(
            "Supervised target column. For unsupervised tasks, this is a "
            "compatibility placeholder column that is also treated as an input feature."
        )
    )
    feature_columns: list[str] = Field(min_length=1, description="Input feature names.")
    candidate_models: list[str] = Field(min_length=1, max_length=4)
    evaluation_metric: Literal[
        "accuracy", "f1_weighted", "roc_auc", "rmse", "mae", "r2",
        "silhouette_score", "outlier_fraction", "explained_variance",
    ]
    generate_synthetic_data: bool
    synthetic_rows: int = Field(default=3000, ge=200, le=20000)
    reasoning: str = Field(description="Brief explanation of the plan.")

class ExperimentRequest(BaseModel):
    use_case: str
    backend: Literal["local", "colab"] = "local"
    dataset_path: str | None = None
    target_column: str | None = None
