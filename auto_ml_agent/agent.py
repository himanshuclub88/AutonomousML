
from __future__ import annotations

from .llm import generate_reply
from .schemas import MLExperimentPlan, TASK_METRICS


USE_LLM_PLANNER = True


# Only these identifiers may be returned to the execution engine.
# The trainer must implement each identifier before it can be executed.
ALLOWED_MODELS = {
    "classification": {
        "logistic_regression",
        "random_forest",
        "extra_trees",
        "hist_gradient_boosting",
    },
    "regression": {
        "linear_regression",
        "random_forest",
        "extra_trees",
        "hist_gradient_boosting",
    },
    "forecasting": {
        "linear_regression",
        "random_forest",
        "extra_trees",
    },
    "clustering": {
        "kmeans",
        "dbscan",
        "agglomerative_clustering",
    },
    "anomaly_detection": {
        "isolation_forest",
        "one_class_svm",
        "local_outlier_factor",
    },
    "dimensionality_reduction": {
        "pca",
        "truncated_svd",
        "autoencoder",
    },
}


SYSTEM_PROMPT = """
You are an ML experiment planner.

Choose exactly one task type from:
- classification
- regression
- forecasting
- clustering
- anomaly_detection
- dimensionality_reduction

Task definitions:
- classification: predict a discrete class or category.
- regression: predict a continuous numeric value.
- forecasting: predict future values in chronological order.
- clustering: discover groups without predefined target labels.
- anomaly_detection: identify unusual observations.
- dimensionality_reduction: represent data using fewer dimensions.

Choose models ONLY from the following allowlists:

classification:
logistic_regression, random_forest, extra_trees,
hist_gradient_boosting

regression:
linear_regression, random_forest, extra_trees,
hist_gradient_boosting

forecasting:
linear_regression, random_forest, extra_trees

clustering:
kmeans, dbscan, agglomerative_clustering

anomaly_detection:
isolation_forest, one_class_svm, local_outlier_factor

dimensionality_reduction:
pca, truncated_svd, autoencoder

Use between two and four candidates when reasonable.
Do not invent model identifiers.

Metrics by task:
- classification: accuracy, f1_weighted, roc_auc, average_precision
- regression: rmse, mae, r2
- forecasting: rmse, mae, r2, mape, smape
- clustering: silhouette_score, davies_bouldin_score,
  calinski_harabasz_score, not_applicable
- anomaly_detection: roc_auc, average_precision, not_applicable
- dimensionality_reduction: explained_variance,
  reconstruction_error, not_applicable

For clustering and dimensionality reduction, target_column should be null.
For anomaly detection, target_column may be null when labels are unavailable.
Classification, regression, and forecasting require a target column.

When a dataset is supplied:
- Use only its actual column names.
- Never invent feature or target columns.
- Respect an explicitly requested target.
- Do not include the target among the features.
- For forecasting, preserve chronological order conceptually;
  do not treat future observations as ordinary random samples.

For missing datasets, synthetic data may be proposed only for demonstration.
Synthetic-data results do not prove real-world performance.
Never request execution of arbitrary generated Python.

Return only the requested structured MLExperimentPlan.
"""


def _detect_problem_type(use_case: str) -> str:
    """Conservative keyword-based fallback when LLM planning is unavailable."""
    text = use_case.lower().replace("-", "_")

    if any(word in text for word in (
        "dimensionality reduction",
        "dimension reduction",
        "reduce dimensions",
        "feature compression",
        "pca",
        "autoencoder",
    )):
        return "dimensionality_reduction"

    if any(word in text for word in (
        "anomaly detection",
        "outlier detection",
        "detect anomalies",
        "unusual transactions",
        "isolation forest",
    )):
        return "anomaly_detection"

    if any(word in text for word in (
        "clustering",
        "cluster customers",
        "customer segments",
        "group customers",
        "kmeans",
        "k-means",
    )):
        return "clustering"

    if any(word in text for word in (
        "time series",
        "time-series",
        "forecast",
        "future demand",
        "next week's demand",
        "next week demand",
    )):
        return "forecasting"

    if any(word in text for word in (
        "predict price",
        "house price",
        "delivery duration",
        "regression",
        "continuous value",
        "numeric value",
        "estimate cost",
        "predict sales",
    )):
        return "regression"

    return "classification"


def _default_metric(problem_type: str) -> str:
    return {
        "classification": "f1_weighted",
        "regression": "rmse",
        "forecasting": "rmse",
        "clustering": "silhouette_score",
        "anomaly_detection": "not_applicable",
        "dimensionality_reduction": "explained_variance",
    }[problem_type]


def _default_models(problem_type: str) -> list[str]:
    preferred = {
        "classification": [
            "logistic_regression",
            "random_forest",
            "extra_trees",
        ],
        "regression": [
            "linear_regression",
            "random_forest",
            "extra_trees",
        ],
        "forecasting": [
            "linear_regression",
            "random_forest",
            "extra_trees",
        ],
        "clustering": [
            "kmeans",
            "dbscan",
            "agglomerative_clustering",
        ],
        "anomaly_detection": [
            "isolation_forest",
            "one_class_svm",
            "local_outlier_factor",
        ],
        "dimensionality_reduction": [
            "pca",
            "truncated_svd",
        ],
    }

    return preferred[problem_type][:3]


def create_plan(
    use_case: str,
    *,
    dataset_columns: list[str] | None = None,
    requested_target: str | None = None,
) -> MLExperimentPlan:
    """
    Create an ML plan while preserving the existing function signature.

    dataset_columns=None means no dataset columns were supplied.
    An empty list is not treated as a valid dataset.
    """
    if dataset_columns is not None:
        if not dataset_columns:
            raise ValueError("The supplied dataset has no columns.")

        if len(dataset_columns) != len(set(dataset_columns)):
            raise ValueError("The supplied dataset has duplicate column names.")

        if requested_target is not None and requested_target not in dataset_columns:
            raise ValueError(
                f"Requested target column {requested_target!r} "
                "was not found in the dataset."
            )

    if not USE_LLM_PLANNER:
        return fallback_plan(use_case, dataset_columns, requested_target)

    context = {
        "use_case": use_case,
        "dataset_columns": dataset_columns,
        "requested_target_column": requested_target,
        "dataset_available": dataset_columns is not None,
    }

    try:
        result = generate_reply(
            system_prompt=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": str(context)}],
            response_model=MLExperimentPlan,
        )

        if not isinstance(result, MLExperimentPlan):
            raise TypeError("Planner returned an unexpected response type.")

        problem_type = result.problem_type
        allowed_models = ALLOWED_MODELS[problem_type]

        models = []
        for model in result.candidate_models:
            if model in allowed_models and model not in models:
                models.append(model)

        if not models:
            models = _default_models(problem_type)

        # Keep the explicitly requested target authoritative.
        target = (
            requested_target
            if requested_target is not None
            else result.target_column
        )

        features = list(result.feature_columns)

        if problem_type in {
            "clustering",
            "dimensionality_reduction",
        }:
            # These tasks do not require a target label.
            target = None

        if dataset_columns is not None:
            columns = set(dataset_columns)

            if target is not None and target not in columns:
                raise ValueError(
                    f"Target column {target!r} was not found in the dataset."
                )

            if problem_type in {
                "classification",
                "regression",
                "forecasting",
            } and target is None:
                raise ValueError(
                    f"{problem_type} requires a target column."
                )

            features = [
                column for column in features
                if column in columns and column != target
            ]

            if not features:
                features = [
                    column for column in dataset_columns
                    if column != target
                ]

            if not features:
                raise ValueError(
                    "The dataset must contain at least one feature column."
                )

        metric = result.evaluation_metric
        if metric not in TASK_METRICS[problem_type]:
            metric = _default_metric(problem_type)

        return result.model_copy(update={
            "problem_type": problem_type,
            "target_column": target,
            "feature_columns": features,
            "candidate_models": models[:4],
            "evaluation_metric": metric,
            # Preserve existing CLI behavior: a missing dataset means
            # synthetic data is proposed. The frontend's explicit choice
            # must be enforced by server.py separately.
            "generate_synthetic_data": dataset_columns is None,
        })

    except Exception as exc:
        print(
            f"LLM planner unavailable or invalid "
            f"({type(exc).__name__}: {exc}). Using fallback planner."
        )
        return fallback_plan(use_case, dataset_columns, requested_target)


def fallback_plan(
    use_case: str,
    columns: list[str] | None,
    target: str | None,
) -> MLExperimentPlan:
    """Create a conservative plan without requiring the LLM."""

    problem_type = _detect_problem_type(use_case)

    if columns is not None:
        if not columns:
            raise ValueError("The supplied dataset has no columns.")

        if target is not None and target not in columns:
            raise ValueError(
                f"Requested target column {target!r} "
                "was not found in the dataset."
            )

        if problem_type in {
            "clustering",
            "dimensionality_reduction",
        }:
            chosen_target = None
        else:
            chosen_target = target or columns[-1]

        features = [
            column for column in columns
            if column != chosen_target
        ]

        if not features:
            raise ValueError(
                "The dataset must contain at least one feature column."
            )

        if problem_type in {
            "classification",
            "regression",
            "forecasting",
        } and chosen_target is None:
            raise ValueError(
                f"{problem_type} requires a target column."
            )

        return MLExperimentPlan(
            problem_type=problem_type,
            target_column=chosen_target,
            feature_columns=features,
            candidate_models=_default_models(problem_type),
            evaluation_metric=_default_metric(problem_type),
            generate_synthetic_data=False,
            reasoning=(
                "Fallback plan based on the use-case description. "
                "Verify the task type and target before training."
            ),
        )

    # No dataset supplied: use synthetic data for a demo.
    target_name = (
        None
        if problem_type in {
            "clustering",
            "dimensionality_reduction",
        }
        else "target"
    )

    return MLExperimentPlan(
        problem_type=problem_type,
        target_column=target_name,
        feature_columns=[
            "feature_1",
            "feature_2",
            "feature_3",
            "feature_4",
        ],
        candidate_models=_default_models(problem_type),
        evaluation_metric=_default_metric(problem_type),
        generate_synthetic_data=True,
        synthetic_rows=3000,
        reasoning=(
            "Fallback plan based on keywords. Synthetic data is intended "
            "for demonstration only; verify the task before training."
        ),
    )


def review_plan_with_data_quality(
    plan: MLExperimentPlan,
    *,
    use_case: str,
    quality_report: dict,
) -> MLExperimentPlan:
    """
    Review a plan without allowing the LLM to change the task, target,
    data source, or supported-model boundary.
    """
    context = {
        "use_case": use_case,
        "initial_plan": plan.model_dump(),
        "data_quality_report": quality_report,
        "instructions": [
            "Return a revised MLExperimentPlan only.",
            "Keep problem_type and target_column unchanged.",
            "Use only existing feature columns from the initial plan.",
            "Do not invent columns.",
            "Do not claim heuristic warnings prove leakage.",
            "Choose only models from the task-specific allowlist.",
            "Choose only metrics compatible with the task.",
            "Preserve the data-source choice and synthetic row count.",
            "Explain any changes in the reasoning field.",
        ],
    }

    try:
        result = generate_reply(
            system_prompt=SYSTEM_PROMPT + """
You are reviewing an existing plan using data-quality findings.
Be conservative. Do not change its problem type, target, or data source.
Only select task-appropriate supported models, features, and metrics.
Return only the requested structured plan.
""",
            messages=[{"role": "user", "content": str(context)}],
            response_model=MLExperimentPlan,
        )

        if not isinstance(result, MLExperimentPlan):
            raise TypeError("Plan reviewer returned an unexpected response type.")

        if result.problem_type != plan.problem_type:
            raise ValueError("Reviewer attempted to change the problem type.")

        if result.target_column != plan.target_column:
            raise ValueError("Reviewer attempted to change the target column.")

        available_features = {
            item["column"]
            for item in quality_report.get(
                "feature_analysis", {}
            ).get("feature_analysis", [])
            if isinstance(item, dict) and "column" in item
        }

        revised_features = [
            column for column in result.feature_columns
            if column in plan.feature_columns
            and column in available_features
            and column != plan.target_column
        ]

        if not revised_features:
            revised_features = list(plan.feature_columns)

        allowed_models = ALLOWED_MODELS[plan.problem_type]
        revised_models = [
            model for model in result.candidate_models
            if model in allowed_models
        ]

        if not revised_models:
            revised_models = list(plan.candidate_models)

        metric = (
            result.evaluation_metric
            if result.evaluation_metric in TASK_METRICS[plan.problem_type]
            else plan.evaluation_metric
        )

        return plan.model_copy(update={
            "feature_columns": revised_features,
            "candidate_models": revised_models[:4],
            "evaluation_metric": metric,
            "generate_synthetic_data": plan.generate_synthetic_data,
            "synthetic_rows": plan.synthetic_rows,
            "reasoning": result.reasoning,
        })

    except Exception as exc:
        print(
            f"LLM plan review unavailable or invalid "
            f"({type(exc).__name__}: {exc}). Keeping the initial plan."
        )
        return plan
