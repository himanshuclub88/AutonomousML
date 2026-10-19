from __future__ import annotations

import os
from .llm import generate_reply
from .schemas import MLExperimentPlan

USE_LLM_PLANNER = True


SYSTEM_PROMPT = """
You are an autonomous ML experiment planner.

Identify the correct task type from the user's use case and dataset.

SUPPORTED TASK TYPES AND MODELS

1. Classification:
   logistic_regression, random_forest, extra_trees,
   hist_gradient_boosting

2. Regression:
   linear_regression, random_forest, extra_trees,
   hist_gradient_boosting

3. Clustering:
   kmeans, dbscan, agglomerative_clustering

4. Anomaly detection:
   isolation_forest, local_outlier_factor, one_class_svm

5. Dimensionality reduction:
   pca, truncated_svd

TASK SELECTION RULES

- Use classification for predicting categorical labels.
- Use regression for predicting continuous numerical values.
- Use clustering to discover groups in unlabeled data.
- Use anomaly_detection to identify unusual observations.
- Use dimensionality_reduction to reduce feature dimensions.

PLANNING RULES

- Select only models supported for the chosen task.
- Choose 2 to 3 candidate models when reasonable.
- Clustering, anomaly detection, and dimensionality reduction do not use a label.
- The schema requires target_column for compatibility: for unsupervised tasks,
  set it to the first available CSV column (or "target" for synthetic data).
  This column is a placeholder and will also be used as an input feature.
- For unsupervised tasks, feature_columns should contain the other actual CSV
  columns; never drop the placeholder column from the training inputs.
- Use the actual dataset columns when a dataset is provided.
- Never invent feature or target columns.
- Do not treat an identifier as a target without justification.
- Do not claim synthetic data demonstrates real-world performance.
- Do not execute arbitrary generated Python.
- Return only the requested structured plan.
"""



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
    "clustering": {
        "kmeans",
        "dbscan",
        "agglomerative_clustering",
    },
    "anomaly_detection": {
        "isolation_forest",
        "local_outlier_factor",
        "one_class_svm",
    },
    "dimensionality_reduction": {
        "pca",
        "truncated_svd",
    },
}


def _detect_advanced_task(use_case: str) -> str | None:
    """Recognize unsupervised tasks for the deterministic fallback planner."""
    text = use_case.lower().replace("_", " ").replace("-", " ")
    patterns = {
        "anomaly_detection": (
            "anomaly detection", "detect anomalies", "outlier detection",
            "identify outliers", "isolation forest", "one class svm",
            "local outlier factor",
        ),
        "dimensionality_reduction": (
            "dimensionality reduction", "reduce dimensions", "reduce dimensionality",
            "feature reduction", "principal component analysis", "truncated svd",
        ),
        "clustering": (
            "clustering", "cluster analysis", "customer segmentation",
            "segment customers", "group similar", "k means", "dbscan",
            "agglomerative clustering",
        ),
    }
    for task_type, phrases in patterns.items():
        if any(phrase in text for phrase in phrases):
            return task_type
    # Handle PCA as a standalone acronym without matching unrelated words.
    if "pca" in text.split():
        return "dimensionality_reduction"
    return None


def _advanced_defaults(task_type: str) -> tuple[list[str], str]:
    defaults = {
        "clustering": (
            ["kmeans", "dbscan", "agglomerative_clustering"],
            "silhouette_score",
        ),
        "anomaly_detection": (
            ["isolation_forest", "local_outlier_factor", "one_class_svm"],
            "outlier_fraction",
        ),
        "dimensionality_reduction": (
            ["pca", "truncated_svd"], "explained_variance",
        ),
    }
    return defaults[task_type]


def create_plan(use_case: str, *, dataset_columns: list[str] | None = None,
                requested_target: str | None = None) -> MLExperimentPlan:
    if not USE_LLM_PLANNER:
        return fallback_plan(use_case, dataset_columns, requested_target)

    context = {
        "use_case": use_case,
        "dataset_columns": dataset_columns or [],
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

        allowed = ALLOWED_MODELS[result.problem_type]
        models = [m for m in result.candidate_models if m in allowed]
        if not models:
            models = _advanced_defaults(result.problem_type)[0][:3] if result.problem_type in {
                "clustering", "anomaly_detection", "dimensionality_reduction"
            } else list(allowed)[:3]

        is_advanced = result.problem_type in {
            "clustering", "anomaly_detection", "dimensionality_reduction"
        }
        if is_advanced:
            if dataset_columns is not None:
                if len(dataset_columns) < 2:
                    raise ValueError("The dataset must contain at least two columns.")
                # main.py requires a target column. For unsupervised work this is
                # only a compatibility placeholder; trainer.py uses every column.
                target = dataset_columns[0]
                features = [c for c in dataset_columns if c != target]
            else:
                target = result.target_column or "target"
                features = [c for c in result.feature_columns if c != target]
                if not features:
                    features = ["feature_1", "feature_2", "feature_3", "feature_4"]
            metric = _advanced_defaults(result.problem_type)[1]
            return result.model_copy(update={
                "target_column": target,
                "feature_columns": features,
                "candidate_models": models[:4],
                "evaluation_metric": metric,
                "generate_synthetic_data": dataset_columns is None,
                "reasoning": (
                    result.reasoning + " Unsupervised mode: target_column is a compatibility "
                    "placeholder, and all selected columns are treated as input features."
                ),
            })

        target = requested_target or result.target_column
        features = result.feature_columns
        if dataset_columns is not None:
            if target not in dataset_columns:
                raise ValueError(f"Target column {target!r} was not found in the CSV.")
            features = [c for c in features if c in dataset_columns and c != target]
            if not features:
                features = [c for c in dataset_columns if c != target]
            if not features:
                raise ValueError("The dataset must contain at least one feature column besides the target.")
        return result.model_copy(update={
            "target_column": target,
            "feature_columns": features,
            "candidate_models": models[:4],
            "generate_synthetic_data": dataset_columns is None,
        })
    except Exception as exc:
        print(f"LLM planner unavailable or invalid ({type(exc).__name__}: {exc}). Using fallback planner.")
        return fallback_plan(use_case, dataset_columns, requested_target)


def fallback_plan(use_case: str, columns: list[str] | None, target: str | None) -> MLExperimentPlan:
    advanced_task = _detect_advanced_task(use_case)
    if advanced_task:
        models, metric = _advanced_defaults(advanced_task)
        if columns:
            if len(columns) < 2:
                raise ValueError("The dataset must contain at least two columns.")
            # Compatibility placeholder only; all columns are used by trainer.py.
            chosen_target = columns[0]
            features = [c for c in columns if c != chosen_target]
            synthetic = False
        else:
            chosen_target = "target"
            features = ["feature_1", "feature_2", "feature_3", "feature_4"]
            synthetic = True
        return MLExperimentPlan(
            problem_type=advanced_task,
            target_column=chosen_target,
            feature_columns=features,
            candidate_models=models,
            evaluation_metric=metric,
            generate_synthetic_data=synthetic,
            synthetic_rows=3000,
            reasoning=(
                f"Fallback plan for {advanced_task}. This is an unsupervised task; "
                "target_column is only a compatibility placeholder and all columns "
                "will be used as input features."
            ),
        )

    if columns:
        chosen_target = target or columns[-1]
        features = [c for c in columns if c != chosen_target]

        if not features:
            raise ValueError("CSV needs at least one feature and one target column.")

        return MLExperimentPlan(
            problem_type="classification",
            target_column=chosen_target,
            feature_columns=features,
            candidate_models=["logistic_regression", "random_forest", "extra_trees"],
            evaluation_metric="f1_weighted",
            generate_synthetic_data=False,
            reasoning="Fallback plan: classification is assumed. Verify problem type and target before accepting.",
        )

    lowered = use_case.lower()
    regression_words = ("predict price", "forecast", "predict sales", "estimate", "regression", "numeric")
    problem_type = "regression" if any(word in lowered for word in regression_words) else "classification"
    features = ["feature_1", "feature_2", "feature_3", "feature_4"]
    target_name = "target"
    models = (
        ["linear_regression", "random_forest", "extra_trees"]
        if problem_type == "regression"
        else ["logistic_regression", "random_forest", "extra_trees"]
    )
    return MLExperimentPlan(
        problem_type=problem_type,
        target_column=target_name,
        feature_columns=features,
        candidate_models=models,
        evaluation_metric="rmse" if problem_type == "regression" else "f1_weighted",
        generate_synthetic_data=True,
        synthetic_rows=3000,
        reasoning="Fallback plan based on keywords. Review the task type before training.",
    )

def review_plan_with_data_quality(
    plan: MLExperimentPlan,
    *,
    use_case: str,
    quality_report: dict,
) -> MLExperimentPlan:
    """
    Ask the configured LLM to review a valid initial plan against data-quality
    findings. Fail safely to the original plan if the review is unavailable.
    """
    context = {
        "use_case": use_case,
        "initial_plan": plan.model_dump(),
        "data_quality_report": quality_report,
        "instructions": [
            "Return a revised MLExperimentPlan only.",
            "Keep the problem_type and target column unchanged.",
            "For unsupervised tasks, target_column is only a compatibility placeholder.",
            "Use only feature columns already present in the report's feature_analysis list.",
            "Do not remove features solely because numeric values are unique.",
            "Treat identifier and leakage checks as warnings, not proof.",
            "Do not claim that synthetic data demonstrates real-world performance.",
            "Choose only supported models and metrics.",
            "Explain any feature or model changes in the reasoning field.",
        ],
    }

    try:
        result = generate_reply(
            system_prompt=(
                SYSTEM_PROMPT
                + "\nReview the supplied initial plan and data-quality report. "
                "Make conservative recommendations. Preserve the target column. "
                "Do not invent dataset columns or claim heuristic warnings prove leakage. "
                "For classification use accuracy, f1_weighted, or roc_auc; for regression use "
                "rmse, mae, or r2; for clustering use silhouette_score; for anomaly detection "
                "use outlier_fraction; for dimensionality reduction use explained_variance. "
                "Return only the requested structured plan."
            ),
            messages=[{"role": "user", "content": str(context)}],
            response_model=MLExperimentPlan,
        )

        if not isinstance(result, MLExperimentPlan):
            raise TypeError("Plan reviewer returned an unexpected response type.")

        if result.target_column != plan.target_column:
            raise ValueError("Plan reviewer attempted to change the target column.")
        if result.problem_type != plan.problem_type:
            raise ValueError("Plan reviewer attempted to change the task type.")

        available_features = {
            item["column"]
            for item in quality_report.get("feature_analysis", {}).get(
                "feature_analysis", []
            )
            if isinstance(item, dict) and "column" in item
        }

        # Keep only original, available features. The reviewer cannot introduce
        # new columns or change the data source.
        revised_features = [
            column for column in result.feature_columns
            if column in available_features and column in plan.feature_columns
            and column != plan.target_column
        ]
        if not revised_features:
            revised_features = list(plan.feature_columns)

        allowed_models = ALLOWED_MODELS[plan.problem_type]
        revised_models = [
            model for model in result.candidate_models if model in allowed_models
        ]
        if not revised_models:
            revised_models = list(plan.candidate_models)

        allowed_metrics = {
            "classification": {"accuracy", "f1_weighted", "roc_auc"},
            "regression": {"rmse", "mae", "r2"},
            "clustering": {"silhouette_score"},
            "anomaly_detection": {"outlier_fraction"},
            "dimensionality_reduction": {"explained_variance"},
        }[plan.problem_type]
        metric = (
            result.evaluation_metric
            if result.evaluation_metric in allowed_metrics
            else plan.evaluation_metric
        )

        return plan.model_copy(update={
            "problem_type": plan.problem_type,
            "target_column": plan.target_column,
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
