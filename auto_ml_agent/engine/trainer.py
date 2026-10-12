from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor

def _build_estimator(name: str, problem_type: str):
    if problem_type == "classification":
        mapping = {
            "logistic_regression": LogisticRegression(max_iter=1000),
            "random_forest": RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=1),
            "extra_trees": ExtraTreesClassifier(n_estimators=150, random_state=42, n_jobs=1),
            "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=42),
        }
    else:
        mapping = {
            "linear_regression": LinearRegression(),
            "random_forest": RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=1),
            "extra_trees": ExtraTreesRegressor(n_estimators=150, random_state=42, n_jobs=1),
            "hist_gradient_boosting": HistGradientBoostingRegressor(random_state=42),
        }
    if name not in mapping:
        raise ValueError(f"Model {name!r} is not supported for {problem_type}.")
    return mapping[name]

def _pipeline(frame: pd.DataFrame, estimator):
    numeric = frame.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in frame.columns if c not in numeric]
    transformers = []
    if numeric:
        num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
        transformers.append(("numeric", num_pipe, numeric))
    if categorical:
        cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                             ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
        transformers.append(("categorical", cat_pipe, categorical))
    preprocess = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline([("preprocess", preprocess), ("model", estimator)])

def _metric(problem_type: str, metric: str, y_true, y_pred) -> float:
    if problem_type == "regression":
        if metric == "mae":
            return float(mean_absolute_error(y_true, y_pred))
        if metric == "r2":
            return float(r2_score(y_true, y_pred))
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

def _score_key(problem_type: str, metric: str) -> str:
    if problem_type == "regression":
        return "r2" if metric == "r2" else "neg_error"
    return metric

def _is_better(problem_type: str, metric: str, a: float, b: float) -> bool:
    if problem_type == "regression" and metric in ("rmse", "mae"):
        return a < b
    return a > b

def train_candidates(frame: pd.DataFrame, target: str, features: list[str], problem_type: str,
                     candidates: list[str], metric: str, run_dir: Path, workers: int = 3) -> dict:
    
    run_dir.mkdir(parents=True, exist_ok=True)
    frame = frame[features + [target]].copy()
    X_train, X_val, X_test, y_train, y_val, y_test = _split(frame, target, problem_type)
    results = []
    errors = []

    def run_one(name: str):
        started = time.time()
        estimator = _build_estimator(name, problem_type)
        pipe = _pipeline(X_train, estimator)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_val)
        score = _metric(problem_type, metric, y_val, pred)
        artifact = run_dir / f"{name}.joblib"
        joblib.dump(pipe, artifact)
        return {"model": name, "validation_metric": metric, "validation_score": score,
                "duration_seconds": round(time.time()-started, 2), "artifact": str(artifact)}
    
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(candidates)))) as pool:
        futures = {pool.submit(run_one, name): name for name in candidates}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                errors.append({"model": futures[future], "error": f"{type(exc).__name__}: {exc}"})
    if not results:
        raise RuntimeError(f"All model experiments failed: {errors}")
    best = results[0]
    for result in results[1:]:
        if _is_better(problem_type, metric, result["validation_score"], best["validation_score"]):
            best = result
    # Refit the selected pipeline on train+validation, then evaluate exactly once on held-out test data.
    X_trainval = pd.concat([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val])
    best_pipe = _pipeline(X_trainval, _build_estimator(best["model"], problem_type))
    best_pipe.fit(X_trainval, y_trainval)
    test_pred = best_pipe.predict(X_test)
    test_score = _metric(problem_type, metric, y_test, test_pred)
    final_artifact = run_dir / "best_model.joblib"
    joblib.dump(best_pipe, final_artifact)

    output = {
        "problem_type": problem_type, "target_column": target, "features": features,
        "selection_metric": metric, "best_model": best["model"],
        "best_validation_score": best["validation_score"],
        "held_out_test_score": test_score, "test_metric": metric,
        "experiments": sorted(results, key=lambda x: x["duration_seconds"]),
        "errors": errors, "best_model_path": str(final_artifact),
        "rows_used": int(len(frame)),
    }
    
    (run_dir / "metrics.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    _write_training_script(run_dir / "train_model.py", target, features, problem_type, best["model"], metric)
    return output

def _split(frame, target, problem_type):
    from .dataset import split_data
    return split_data(frame, target, problem_type)

def _write_training_script(path: Path, target: str, features: list[str], problem_type: str, model: str, metric: str):
    content = f'''# Reproducible training entry point generated by Auto ML Agent.
# Review the dataset path and target before running.
from pathlib import Path
import pandas as pd
from auto_ml_agent.engine.trainer import train_candidates

CSV_PATH = Path("your_dataset.csv")
TARGET = {target!r}
FEATURES = {features!r}
PROBLEM_TYPE = {problem_type!r}
MODEL = {model!r}
METRIC = {metric!r}

if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    result = train_candidates(df, TARGET, FEATURES, PROBLEM_TYPE, [MODEL], METRIC, Path("model_output"), workers=1)
    print(result)
'''
    path.write_text(content, encoding="utf-8")
