from __future__ import annotations

import os
from .llm import generate_reply
from .schemas import MLExperimentPlan

USE_LLM_PLANNER = True

SYSTEM_PROMPT = '''
You are an ML experiment planner. Convert the user's use case and dataset context into a conservative, valid ML plan.
Choose classification when the target is categorical and regression when the target is numeric and continuous.
Use only these candidate model identifiers:
classification: logistic_regression, random_forest, extra_trees, hist_gradient_boosting
regression: linear_regression, random_forest, extra_trees, hist_gradient_boosting
Choose 2 to 4 candidates when reasonable. Use accuracy or f1_weighted for classification; use rmse, mae, or r2 for regression.
Do not claim that synthetic data proves real-world performance. For a missing dataset, propose synthetic data only as a demonstration/test dataset.
If a dataset is provided, respect its actual target and feature columns. Never ask to execute arbitrary generated Python.
Return only the requested structured plan.
'''


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
            models = list(allowed)[:3]
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
            "Keep the target column unchanged.",
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
                "For classification use accuracy or f1_weighted; for regression use "
                "rmse, mae, or r2. Return only the requested structured plan."
            ),
            messages=[{"role": "user", "content": str(context)}],
            response_model=MLExperimentPlan,
        )

        if not isinstance(result, MLExperimentPlan):
            raise TypeError("Plan reviewer returned an unexpected response type.")

        if result.target_column != plan.target_column:
            raise ValueError("Plan reviewer attempted to change the target column.")

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

        allowed_metrics = (
            {"accuracy", "f1_weighted"}
            if plan.problem_type == "classification"
            else {"rmse", "mae", "r2"}
        )
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
