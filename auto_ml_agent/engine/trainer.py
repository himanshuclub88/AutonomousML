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
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import silhouette_score
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

ADVANCED_MODELS = {
    "clustering": {"kmeans", "dbscan", "agglomerative_clustering"},
    "anomaly_detection": {"isolation_forest", "local_outlier_factor", "one_class_svm"},
    "dimensionality_reduction": {"pca", "truncated_svd"},
}


def _unsupervised_preprocessor(frame: pd.DataFrame):
    """Build preprocessing for target-free tabular workflows."""
    numeric = frame.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [column for column in frame.columns if column not in numeric]
    transformers = []
    if numeric:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("numeric", num_pipe, numeric))
    if categorical:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("categorical", cat_pipe, categorical))
    if not transformers:
        raise ValueError("No usable input features were provided.")
    return ColumnTransformer(transformers=transformers, remainder="drop")


def _build_advanced_estimator(name: str, task_type: str, n_features: int):
    if task_type == "clustering":
        mapping = {
            "kmeans": KMeans(n_clusters=min(3, max(2, n_features)), n_init=10, random_state=42),
            "dbscan": DBSCAN(eps=0.8, min_samples=5),
            "agglomerative_clustering": AgglomerativeClustering(
                n_clusters=min(3, max(2, n_features))
            ),
        }
    elif task_type == "anomaly_detection":
        mapping = {
            "isolation_forest": IsolationForest(contamination="auto", random_state=42),
            # LOF detects outliers in the supplied dataset; novelty=False is intentional.
            "local_outlier_factor": LocalOutlierFactor(n_neighbors=20, contamination="auto"),
            "one_class_svm": OneClassSVM(kernel="rbf", gamma="scale", nu=0.05),
        }
    elif task_type == "dimensionality_reduction":
        components = max(1, min(2, n_features))
        mapping = {
            "pca": PCA(n_components=components, random_state=42),
            "truncated_svd": TruncatedSVD(n_components=components, random_state=42),
        }
    else:
        raise ValueError(f"Unsupported advanced task type: {task_type!r}")
    if name not in mapping:
        raise ValueError(f"Model {name!r} is not supported for {task_type}.")
    return mapping[name]


def _train_advanced_candidates(
    frame: pd.DataFrame,
    target: str,
    features: list[str],
    problem_type: str,
    candidates: list[str],
    metric: str,
    run_dir: Path,
) -> dict:
    """Run unsupervised models without target-based splitting or scoring."""
    if problem_type not in ADVANCED_MODELS:
        raise ValueError(f"Unsupported advanced task type: {problem_type!r}")
    if target not in frame.columns:
        raise ValueError(f"Compatibility placeholder column {target!r} is missing.")

    # main.py still supplies target_column for compatibility. For these tasks it
    # is ordinary input data, not a label, so include every selected column.
    input_columns = list(dict.fromkeys(list(features) + [target]))
    missing = [column for column in input_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Input columns are missing: {missing}")
    X = frame[input_columns].copy()
    if len(X) < 3:
        raise ValueError("At least three rows are required for unsupervised training.")

    run_dir.mkdir(parents=True, exist_ok=True)
    results = []
    errors = []
    output_frames = {}

    for name in candidates:
        started = time.time()
        try:
            if name not in ADVANCED_MODELS[problem_type]:
                raise ValueError(f"Model {name!r} is not valid for {problem_type}.")
            preprocessor = _unsupervised_preprocessor(X)
            X_ready = preprocessor.fit_transform(X)
            if hasattr(X_ready, "toarray"):
                X_ready = X_ready.toarray()
            X_ready = np.asarray(X_ready, dtype=float)
            if X_ready.ndim != 2 or X_ready.shape[1] == 0:
                raise ValueError("Preprocessing produced no usable feature dimensions.")
            if not np.isfinite(X_ready).all():
                raise ValueError("Preprocessed data contains non-finite values.")

            estimator = _build_advanced_estimator(name, problem_type, X_ready.shape[1])
            artifact = run_dir / f"{name}.joblib"
            model_output = {}

            if problem_type == "clustering":
                labels = estimator.fit_predict(X_ready)
                unique_labels = set(labels.tolist())
                non_noise_labels = unique_labels - {-1}
                score = None
                scored_mask = np.asarray(labels) != -1 if name == "dbscan" else np.ones(len(labels), dtype=bool)
                scored_labels = np.asarray(labels)[scored_mask]
                if len(set(scored_labels.tolist())) >= 2 and int(scored_mask.sum()) > len(set(scored_labels.tolist())):
                    score = float(silhouette_score(X_ready[scored_mask], scored_labels))
                model_output = {
                    "cluster_count": int(len(non_noise_labels) if name == "dbscan" else len(unique_labels)),
                    "noise_points": int(np.sum(np.asarray(labels) == -1)) if name == "dbscan" else 0,
                    "silhouette_score": score,
                }
                output_frames[name] = pd.DataFrame({"row_index": frame.index, "cluster": labels})
                selection_score = score

            elif problem_type == "anomaly_detection":
                if name == "local_outlier_factor":
                    labels = estimator.fit_predict(X_ready)
                    scores = -estimator.negative_outlier_factor_
                else:
                    estimator.fit(X_ready)
                    labels = estimator.predict(X_ready)
                    if hasattr(estimator, "decision_function"):
                        scores = -estimator.decision_function(X_ready)
                    else:
                        scores = -estimator.score_samples(X_ready)
                is_anomaly = np.asarray(labels) == -1
                model_output = {
                    "anomaly_count": int(is_anomaly.sum()),
                    "outlier_fraction": float(is_anomaly.mean()),
                    "score_direction": "higher values indicate more anomalous observations",
                }
                detected = frame.copy()
                detected["anomaly_label"] = np.where(is_anomaly, "anomaly", "normal")
                detected["anomaly_score"] = np.asarray(scores, dtype=float)
                output_frames[name] = detected
                # No unsupervised metric can establish the best detector without labels.
                selection_score = None

            else:
                reduced = estimator.fit_transform(X_ready)
                reduced = np.asarray(reduced, dtype=float)
                names = [f"component_{i + 1}" for i in range(reduced.shape[1])]
                model_output = {"n_components": int(reduced.shape[1])}
                if hasattr(estimator, "explained_variance_ratio_"):
                    variance = np.asarray(estimator.explained_variance_ratio_, dtype=float)
                    model_output["explained_variance_ratio"] = variance.tolist()
                    model_output["total_explained_variance"] = float(variance.sum())
                    selection_score = float(variance.sum())
                else:
                    selection_score = None
                output_frames[name] = pd.DataFrame(reduced, columns=names, index=frame.index)

            joblib.dump({
                "task_type": problem_type,
                "model_name": name,
                "input_columns": input_columns,
                "preprocessor": preprocessor,
                "model": estimator,
            }, artifact)
            results.append({
                "model": name,
                "task_type": problem_type,
                "metrics": model_output,
                "selection_score": selection_score,
                "duration_seconds": round(time.time() - started, 2),
                "artifact": str(artifact),
            })
        except Exception as exc:
            errors.append({"model": name, "error": f"{type(exc).__name__}: {exc}"})

    if not results:
        raise RuntimeError(f"All {problem_type} experiments failed: {errors}")

    # Persist model-specific row-level outputs for inspection.
    for name, output_frame in output_frames.items():
        if problem_type == "clustering":
            output_frame.to_csv(run_dir / f"{name}_assignments.csv", index=False)
        elif problem_type == "anomaly_detection":
            output_frame.to_csv(run_dir / f"{name}_anomalies.csv", index=False)
        else:
            output_frame.to_csv(run_dir / f"{name}_reduced_features.csv", index=True)

    # Only metrics with a meaningful common interpretation are used to select a
    # default winner. Anomaly detectors have no ground-truth labels, so preserve
    # the first successful candidate and explicitly report that no ranking occurred.
    scored = [r for r in results if r["selection_score"] is not None]
    if scored and problem_type in {"clustering", "dimensionality_reduction"}:
        best = max(scored, key=lambda r: r["selection_score"])
        selection_note = f"Selected by {metric}; interpret unsupervised scores cautiously."
    else:
        best = results[0]
        selection_note = (
            "No reliable ground-truth model ranking is available for this task; "
            "the first successful candidate is reported as the default."
        )

    best_source = Path(best["artifact"])
    best_artifact = run_dir / "best_model.joblib"
    if best_source != best_artifact:
        best_artifact.write_bytes(best_source.read_bytes())

    output = {
        "problem_type": problem_type,
        "target_column": None,
        "target_placeholder_column": target,
        "features": input_columns,
        "selection_metric": metric,
        "best_model": best["model"],
        "selection_note": selection_note,
        "experiments": sorted(results, key=lambda item: item["duration_seconds"]),
        "errors": errors,
        "best_model_path": str(best_artifact),
        "rows_used": int(len(X)),
        "outputs": [str(run_dir / f) for f in sorted(p.name for p in run_dir.glob("*_*.csv"))],
    }
    (run_dir / "metrics.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    _write_training_script(run_dir / "train_model.py", target, features, problem_type, best["model"], metric)
    return output


def train_candidates(frame: pd.DataFrame, target: str, features: list[str], problem_type: str,
                     candidates: list[str], metric: str, run_dir: Path, workers: int = 3) -> dict:
    if problem_type in ADVANCED_MODELS:
        return _train_advanced_candidates(
            frame=frame,
            target=target,
            features=features,
            problem_type=problem_type,
            candidates=candidates,
            metric=metric,
            run_dir=run_dir,
        )

    # Existing supervised path remains unchanged below this point.
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
