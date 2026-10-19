"""Google Colab runtime adapter for supervised and unsupervised scikit-learn training.

The local application performs planning, data preparation and review. This module
sends only the prepared training frame to an already-created Colab runtime,
executes the same allow-listed model workflow remotely, and downloads artifacts.
"""
from __future__ import annotations

import base64
import json
import os
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import requests


COLAB_API = "https://colaboratory.googleapis.com/v1beta"
DEFAULT_RUNTIME = "runtimes/automl-t4-runtime"


class ColabAccessRequired(RuntimeError):
    """Raised when Google Colab API credentials or a runtime are unavailable."""


def _google_credentials():
    try:
        import google.auth
        from google.auth.transport.requests import Request
    except ImportError as exc:
        raise ColabAccessRequired(
            "Google auth dependencies are missing. Run: pip install google-auth requests websocket-client"
        ) from exc

    scopes = [
        "https://www.googleapis.com/auth/colaboratory",
        "https://www.googleapis.com/auth/cloud-platform",
    ]
    try:
        credentials, _ = google.auth.default(scopes=scopes)
        credentials.refresh(Request())
    except Exception as exc:
        raise ColabAccessRequired(
            "Google Application Default Credentials are unavailable. Run the gcloud ADC login command "
            "from your setup instructions, then retry. Details: " + str(exc)
        ) from exc
    return credentials


def _get_runtime_connection() -> tuple[str, str]:
    """Get connection details, creating a replacement runtime if the saved one is gone.

    Runtime names are not credentials. Fresh proxy URL/token values are fetched on
    every run and are never written to disk or printed.
    """
    credentials = _google_credentials()
    project = os.getenv("GOOGLE_CLOUD_PROJECT", "537909573529").strip()
    api_root = COLAB_API.rstrip("/")
    headers = {
        "Authorization": f"Bearer {credentials.token}",
        "Content-Type": "application/json",
    }
    if project:
        headers["x-goog-user-project"] = project

    # Persist only the non-secret runtime resource name so the app can reuse it
    # across PowerShell sessions. Never persist connectionInfo or its token.
    project_root = Path(__file__).resolve().parents[2]
    runtime_config = project_root / "workspace" / "colab_runtime.json"
    runtime_config.parent.mkdir(parents=True, exist_ok=True)

    env_name = os.getenv("COLAB_RUNTIME_NAME", "").strip().strip("/")
    saved_name = ""
    try:
        if runtime_config.exists():
            saved_name = str(json.loads(runtime_config.read_text(encoding="utf-8")).get("runtime_name", "")).strip().strip("/")
    except (OSError, ValueError, TypeError):
        saved_name = ""

    # Explicit environment setting wins; otherwise reuse the saved runtime name.
    runtime_name = env_name or saved_name or DEFAULT_RUNTIME.strip().strip("/")
    if runtime_name.startswith(("https://", "http://")):
        raise ColabAccessRequired("COLAB_RUNTIME_NAME must be a runtime resource name, not a URL.")
    if runtime_name.startswith("projects/"):
        runtime_path = runtime_name
    elif runtime_name.startswith("runtimes/"):
        runtime_path = runtime_name
    else:
        runtime_path = "runtimes/" + runtime_name

    def get_runtime(name: str):
        url = f"{api_root}/{name.lstrip('/')}"
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 404:
            return None
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = response.text[:1200]
            raise ColabAccessRequired(
                f"Could not retrieve Colab runtime {name!r} (HTTP {response.status_code}). "
                f"Project: {project!r}. Details: {detail}"
            ) from exc
        return response.json()

    runtime = get_runtime(runtime_path)

    # If the saved runtime is gone, request a fresh runtime with a unique ID.
    if runtime is None:
        runtime_id = "automl-t4-" + uuid.uuid4().hex[:12]
        create_url = f"{api_root}/runtimes"
        payload = {
            "runtimeSpec": {
                "variant": "VARIANT_GPU",
                "accelerator": "T4",
                "shape": "SHAPE_STANDARD",
            }
        }
        try:
            response = requests.post(
                create_url,
                params={"runtimeId": runtime_id},
                headers=headers,
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            created = response.json()
        except Exception as exc:
            detail = getattr(locals().get("response"), "text", "")[:1200]
            raise ColabAccessRequired(
                "The configured Colab runtime was not found, and creating a replacement T4 "
                f"runtime failed. Check Colab API runtime-creation access. Details: {detail or exc}"
            ) from exc

        if created.get("error"):
            err = created["error"]
            raise ColabAccessRequired(
                "Colab refused to create a replacement T4 runtime: "
                + str(err.get("message", err))
            )

        created_name = (
            (created.get("metadata") or {}).get("runtime")
            or (created.get("response") or {}).get("name")
            or created.get("name")
        )
        if not created_name:
            raise ColabAccessRequired(
                "Colab accepted the create request but did not return a runtime resource name. "
                f"Response keys: {sorted(created.keys())}"
            )
        runtime_path = created_name.strip("/")
        if runtime_path.startswith("https://"):
            raise ColabAccessRequired("Colab returned an unexpected URL instead of a runtime resource name.")

        # Creation may be asynchronous. Poll briefly for the runtime to become readable.
        deadline = time.time() + 180
        runtime = None
        while time.time() < deadline:
            runtime = get_runtime(runtime_path)
            if runtime and (runtime.get("connectionInfo") or runtime.get("connection_info")):
                break
            time.sleep(3)
        if not runtime:
            raise ColabAccessRequired(
                f"Colab created or accepted runtime {runtime_path!r}, but it is not readable yet. "
                "Wait a minute and retry."
            )

    connection = runtime.get("connectionInfo") or runtime.get("connection_info") or {}
    runtime_url = connection.get("url") or connection.get("runtimeUrl") or connection.get("runtime_url")
    token = connection.get("token") or connection.get("proxyToken") or connection.get("proxy_token")
    if not runtime_url or not token:
        state = runtime.get("runtimeState", runtime.get("state", "unknown"))
        raise ColabAccessRequired(
            f"Colab runtime {runtime_path!r} exists but did not provide connection information "
            f"(state={state!r}). It may still be starting. Retry in a minute."
        )

    # Save only the resource name; always fetch a fresh connection token next run.
    try:
        runtime_config.write_text(
            json.dumps({"runtime_name": runtime_path}, indent=2),
            encoding="utf-8",
        )
    except OSError:
        # Runtime still works for this run even if persistence is unavailable.
        pass

    return runtime_url.rstrip("/"), token

def _remote_training_script(csv_b64: str, target: str, features: list[str], problem_type: str,
                            candidates: list[str], metric: str, workers: int) -> str:
    config = {
        "csv_b64": csv_b64, "target": target, "features": features,
        "problem_type": problem_type, "candidates": candidates,
        "metric": metric, "workers": workers,
    }
    return r"""import base64, json, time, zipfile, traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA, TruncatedSVD

CFG = json.loads(__CONFIG_JSON__)
OUT = Path('/content/automl_agent_output')
OUT.mkdir(parents=True, exist_ok=True)

try:
    csv_bytes = base64.b64decode(CFG['csv_b64'])
    input_path = OUT / 'prepared_dataset.csv'
    input_path.write_bytes(csv_bytes)
    frame = pd.read_csv(input_path)
    target, features = CFG['target'], CFG['features']
    problem_type, candidates, metric = CFG['problem_type'], CFG['candidates'], CFG['metric']
    missing = [c for c in list(features) + [target] if c not in frame.columns]
    if missing:
        raise ValueError('Missing required columns: ' + repr(missing))

    if problem_type in {'clustering', 'anomaly_detection', 'dimensionality_reduction'}:
        # Current CLI compatibility: target is included as an ordinary feature, not used as a label.
        input_columns = list(dict.fromkeys(list(features) + [target]))
        X = frame[input_columns].copy()
        if len(X) < 3:
            raise ValueError('At least three rows are required for unsupervised training.')

        def make_preprocessor(data):
            numeric = data.select_dtypes(include=['number', 'bool']).columns.tolist()
            categorical = [c for c in data.columns if c not in numeric]
            transformers = []
            if numeric:
                transformers.append(('numeric', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()),
                ]), numeric))
            if categorical:
                transformers.append(('categorical', Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                ]), categorical))
            if not transformers:
                raise ValueError('No usable input features were provided.')
            return ColumnTransformer(transformers=transformers, remainder='drop')

        results, errors, outputs = [], [], {}
        for name in candidates:
            started = time.time()
            try:
                prep = make_preprocessor(X)
                ready = np.asarray(prep.fit_transform(X), dtype=float)
                if ready.ndim != 2 or ready.shape[1] == 0 or not np.isfinite(ready).all():
                    raise ValueError('Preprocessing produced invalid feature values.')

                if problem_type == 'clustering':
                    count = min(3, max(2, ready.shape[1]))
                    models = {
                        'kmeans': KMeans(n_clusters=count, n_init=10, random_state=42),
                        'dbscan': DBSCAN(eps=0.8, min_samples=5),
                        'agglomerative_clustering': AgglomerativeClustering(n_clusters=count),
                    }
                elif problem_type == 'anomaly_detection':
                    models = {
                        'isolation_forest': IsolationForest(contamination='auto', random_state=42),
                        'local_outlier_factor': LocalOutlierFactor(
                            n_neighbors=min(20, max(2, len(X) - 1)), contamination='auto'
                        ),
                        'one_class_svm': OneClassSVM(kernel='rbf', gamma='scale', nu=0.05),
                    }
                else:
                    components = max(1, min(2, ready.shape[1], len(X) - 1))
                    models = {
                        'pca': PCA(n_components=components, random_state=42),
                        'truncated_svd': TruncatedSVD(n_components=components, random_state=42),
                    }
                if name not in models:
                    raise ValueError('Model ' + repr(name) + ' is not supported for ' + problem_type)
                model = models[name]
                details, selection_score = {}, None

                if problem_type == 'clustering':
                    labels = model.fit_predict(ready)
                    unique_labels = set(labels.tolist())
                    mask = np.asarray(labels) != -1 if name == 'dbscan' else np.ones(len(labels), dtype=bool)
                    scored_labels = np.asarray(labels)[mask]
                    score = None
                    if len(set(scored_labels.tolist())) >= 2 and int(mask.sum()) > len(set(scored_labels.tolist())):
                        score = float(silhouette_score(ready[mask], scored_labels))
                    details = {
                        'cluster_count': int(len(unique_labels - {-1}) if name == 'dbscan' else len(unique_labels)),
                        'noise_points': int(np.sum(np.asarray(labels) == -1)) if name == 'dbscan' else 0,
                        'silhouette_score': score,
                    }
                    outputs[name] = pd.DataFrame({'row_index': frame.index, 'cluster': labels})
                    selection_score = score

                elif problem_type == 'anomaly_detection':
                    if name == 'local_outlier_factor':
                        labels = model.fit_predict(ready)
                        scores = -model.negative_outlier_factor_
                    else:
                        model.fit(ready)
                        labels = model.predict(ready)
                        scores = -model.decision_function(ready) if hasattr(model, 'decision_function') else -model.score_samples(ready)
                    anomaly = np.asarray(labels) == -1
                    details = {
                        'anomaly_count': int(anomaly.sum()),
                        'outlier_fraction': float(anomaly.mean()),
                        'score_direction': 'higher values indicate more anomalous observations',
                    }
                    out_frame = frame.copy()
                    out_frame['anomaly_label'] = np.where(anomaly, 'anomaly', 'normal')
                    out_frame['anomaly_score'] = np.asarray(scores, dtype=float)
                    outputs[name] = out_frame

                else:
                    reduced = np.asarray(model.fit_transform(ready), dtype=float)
                    details = {'n_components': int(reduced.shape[1])}
                    if hasattr(model, 'explained_variance_ratio_'):
                        variance = np.asarray(model.explained_variance_ratio_, dtype=float)
                        details['explained_variance_ratio'] = variance.tolist()
                        details['total_explained_variance'] = float(variance.sum())
                        selection_score = float(variance.sum())
                    outputs[name] = pd.DataFrame(
                        reduced,
                        columns=[f'component_{i + 1}' for i in range(reduced.shape[1])],
                        index=frame.index,
                    )

                artifact = OUT / (name + '.joblib')
                joblib.dump({
                    'task_type': problem_type, 'model_name': name,
                    'input_columns': input_columns, 'preprocessor': prep, 'model': model,
                }, artifact)
                results.append({
                    'model': name, 'task_type': problem_type, 'metrics': details,
                    'selection_score': selection_score,
                    'duration_seconds': round(time.time() - started, 2),
                    'artifact': str(artifact),
                })
            except Exception as exc:
                errors.append({'model': name, 'error': type(exc).__name__ + ': ' + str(exc)})

        if not results:
            raise RuntimeError('All advanced experiments failed: ' + repr(errors))
        for name, out_frame in outputs.items():
            if problem_type == 'clustering':
                out_frame.to_csv(OUT / (name + '_assignments.csv'), index=False)
            elif problem_type == 'anomaly_detection':
                out_frame.to_csv(OUT / (name + '_anomalies.csv'), index=False)
            else:
                out_frame.to_csv(OUT / (name + '_reduced_features.csv'), index=True)

        scored = [item for item in results if item['selection_score'] is not None]
        if scored and problem_type in {'clustering', 'dimensionality_reduction'}:
            best = max(scored, key=lambda item: item['selection_score'])
            selection_note = 'Selected using the available unsupervised score; interpret cautiously.'
        else:
            best = results[0]
            selection_note = 'No reliable ground-truth ranking is available; first successful candidate is the default.'
        best_path = OUT / 'best_model.joblib'
        best_path.write_bytes(Path(best['artifact']).read_bytes())
        metrics = {
            'problem_type': problem_type, 'target_column': None,
            'target_placeholder_column': target, 'features': input_columns,
            'selection_metric': metric, 'best_model': best['model'],
            'selection_note': selection_note,
            'experiments': sorted(results, key=lambda item: item['duration_seconds']),
            'errors': errors, 'best_model_path': str(best_path),
            'rows_used': int(len(X)), 'outputs': [str(p) for p in sorted(OUT.glob('*_*.csv'))],
            'training_backend': 'google_colab',
        }
        (OUT / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')

    else:
        # Existing supervised workflow retained.
        frame = frame[list(features) + [target]].dropna(subset=[target]).copy()
        if len(frame) < 30:
            raise ValueError('Need at least 30 usable rows for training.')
        X, y = frame[features], frame[target]
        stratify = y if problem_type == 'classification' and y.nunique() > 1 and y.value_counts().min() >= 3 else None
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=stratify)
        stratify_temp = y_temp if problem_type == 'classification' and y_temp.nunique() > 1 and y_temp.value_counts().min() >= 2 else None
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=stratify_temp)

        def estimator(name):
            if problem_type == 'classification':
                models = {
                    'logistic_regression': LogisticRegression(max_iter=1000),
                    'random_forest': RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=1),
                    'extra_trees': ExtraTreesClassifier(n_estimators=150, random_state=42, n_jobs=1),
                    'hist_gradient_boosting': HistGradientBoostingClassifier(random_state=42),
                }
            elif problem_type == 'regression':
                models = {
                    'linear_regression': LinearRegression(),
                    'random_forest': RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=1),
                    'extra_trees': ExtraTreesRegressor(n_estimators=150, random_state=42, n_jobs=1),
                    'hist_gradient_boosting': HistGradientBoostingRegressor(random_state=42),
                }
            else:
                raise ValueError('Unsupported problem type: ' + str(problem_type))
            if name not in models:
                raise ValueError('Model ' + repr(name) + ' is not supported for ' + problem_type)
            return models[name]

        def pipeline(X_frame, model):
            numeric = X_frame.select_dtypes(include=['number', 'bool']).columns.tolist()
            categorical = [c for c in X_frame.columns if c not in numeric]
            transformers = []
            if numeric:
                transformers.append(('numeric', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric))
            if categorical:
                transformers.append(('categorical', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical))
            return Pipeline([('preprocess', ColumnTransformer(transformers=transformers, remainder='drop')), ('model', model)])

        def score(y_true, y_pred):
            if problem_type == 'regression':
                if metric == 'mae': return float(mean_absolute_error(y_true, y_pred))
                if metric == 'r2': return float(r2_score(y_true, y_pred))
                return float(np.sqrt(mean_squared_error(y_true, y_pred)))
            if metric == 'accuracy': return float(accuracy_score(y_true, y_pred))
            return float(f1_score(y_true, y_pred, average='weighted', zero_division=0))

        def better(a, b):
            if problem_type == 'regression' and metric in ('rmse', 'mae'): return a < b
            return a > b

        results, errors = [], []
        def train_one(name):
            started = time.time()
            pipe = pipeline(X_train, estimator(name))
            pipe.fit(X_train, y_train)
            value = score(y_val, pipe.predict(X_val))
            joblib.dump(pipe, OUT / (name + '.joblib'))
            return {'model': name, 'validation_metric': metric, 'validation_score': value,
                    'duration_seconds': round(time.time() - started, 2), 'artifact': str(OUT / (name + '.joblib'))}

        with ThreadPoolExecutor(max_workers=max(1, min(int(CFG['workers']), len(candidates)))) as pool:
            futures = {pool.submit(train_one, name): name for name in candidates}
            for future in as_completed(futures):
                try: results.append(future.result())
                except Exception as exc: errors.append({'model': futures[future], 'error': type(exc).__name__ + ': ' + str(exc)})
        if not results: raise RuntimeError('All model experiments failed: ' + repr(errors))
        best = results[0]
        for result in results[1:]:
            if better(result['validation_score'], best['validation_score']): best = result
        X_trainval, y_trainval = pd.concat([X_train, X_val]), pd.concat([y_train, y_val])
        best_pipe = pipeline(X_trainval, estimator(best['model']))
        best_pipe.fit(X_trainval, y_trainval)
        test_score = score(y_test, best_pipe.predict(X_test))
        joblib.dump(best_pipe, OUT / 'best_model.joblib')
        metrics = {
            'problem_type': problem_type, 'target_column': target, 'features': features,
            'selection_metric': metric, 'best_model': best['model'],
            'best_validation_score': best['validation_score'], 'held_out_test_score': test_score,
            'test_metric': metric, 'experiments': sorted(results, key=lambda x: x['duration_seconds']),
            'errors': errors, 'best_model_path': str(OUT / 'best_model.joblib'),
            'rows_used': int(len(frame)), 'training_backend': 'google_colab',
        }
        (OUT / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        script_lines = [
            '# Reproducible training entry point generated by Auto ML Agent.',
            'from pathlib import Path', 'import pandas as pd',
            'from auto_ml_agent.engine.trainer import train_candidates',
            'CSV_PATH = Path("your_dataset.csv")', 'TARGET = ' + repr(target),
            'FEATURES = ' + repr(features), 'PROBLEM_TYPE = ' + repr(problem_type),
            'MODEL = ' + repr(best['model']), 'METRIC = ' + repr(metric),
            'if __name__ == "__main__":',
            '    df = pd.read_csv(CSV_PATH)',
            '    print(train_candidates(df, TARGET, FEATURES, PROBLEM_TYPE, [MODEL], METRIC, Path("model_output"), workers=1))',
        ]
        (OUT / 'train_model.py').write_text('\\n'.join(script_lines) + '\\n', encoding='utf-8')

    archive = OUT / 'colab_artifacts.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as zf:
        for path in OUT.iterdir():
            if path.is_file() and path.name != archive.name:
                zf.write(path, arcname=path.name)
    encoded = base64.b64encode(archive.read_bytes()).decode('ascii')
    print('AUTOML_RESULT_JSON:' + json.dumps(metrics))
    for index in range(0, len(encoded), 30000):
        print('AUTOML_ZIP_CHUNK:' + str(index // 30000) + ':' + encoded[index:index+30000])
except Exception:
    print('AUTOML_REMOTE_ERROR:' + traceback.format_exc())
""".replace("__CONFIG_JSON__", repr(json.dumps(config)))

def _execute_on_runtime(
    code: str,
    runtime_url: str,
    proxy_token: str,
    *,
    log_path: Path | None = None,
) -> str:
    """Execute code and collect output until both execute_reply and IOPub idle arrive."""
    try:
        import websocket
    except ImportError as exc:
        raise ColabAccessRequired("Install websocket-client: pip install websocket-client") from exc

    headers = [f"X-Colab-Runtime-Proxy-Token: {proxy_token}"]
    base = runtime_url.rstrip("/")
    session = str(uuid.uuid4())
    kernel_id = None
    ws = None
    output: list[str] = []
    diagnostic: list[str] = []

    try:
        response = requests.post(
            base + "/api/kernels",
            headers={"X-Colab-Runtime-Proxy-Token": proxy_token},
            json={"name": "python3"},
            timeout=30,
        )
        response.raise_for_status()
        kernel_id = response.json()["id"]
    except Exception as exc:
        raise ColabAccessRequired(
            "Could not create a Python kernel on the Colab runtime. "
            "The runtime may be disconnected. Details: " + str(exc)
        ) from exc

    ws_url = base.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
    ws_url += f"/api/kernels/{kernel_id}/channels?session_id={session}"

    try:
        ws = websocket.create_connection(
            ws_url,
            header=headers,
            timeout=15,
            origin=base,
            suppress_origin=False,
        )
        msg_id = uuid.uuid4().hex
        message = {
            "header": {
                "msg_id": msg_id,
                "username": "automl-agent",
                "session": session,
                "msg_type": "execute_request",
                "version": "5.3",
            },
            "parent_header": {},
            "metadata": {},
            "content": {
                "code": code,
                "silent": False,
                "store_history": False,
                "user_expressions": {},
                "allow_stdin": False,
                "stop_on_error": True,
            },
            "channel": "shell",
            "buffers": [],
        }
        ws.send(json.dumps(message))

        timeout_seconds = max(60, int(os.getenv("COLAB_EXECUTION_TIMEOUT_SECONDS", "3600")))
        deadline = time.time() + timeout_seconds
        reply_received = False
        idle_received = False
        reply_status = None

        while time.time() < deadline:
            # A short receive timeout lets us keep waiting for the other channel's
            # completion message instead of treating a quiet interval as failure.
            ws.settimeout(min(15, max(1, deadline - time.time())))
            try:
                raw = ws.recv()
            except websocket.WebSocketTimeoutException:
                continue
            if not raw:
                continue

            try:
                msg = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                diagnostic.append("[non-JSON websocket message received]")
                continue

            parent = msg.get("parent_header") or {}
            if parent.get("msg_id") != msg_id:
                continue

            msg_type = (msg.get("header") or {}).get("msg_type")
            channel = msg.get("channel")
            content = msg.get("content") or {}

            if msg_type == "stream":
                chunk = content.get("text", "")
                output.append(chunk)
                # Do not duplicate enormous binary artifact chunks in the
                # diagnostic file; preserve normal output and remote errors.
                if "AUTOML_ZIP_CHUNK:" not in chunk:
                    diagnostic.append(chunk)
            elif msg_type == "error":
                error_text = "\n".join(
                    content.get("traceback")
                    or [content.get("ename", "Remote execution error") + ": "
                        + content.get("evalue", "")]
                )
                output.append("\n" + error_text + "\n")
                diagnostic.append("\n[REMOTE EXECUTION ERROR]\n" + error_text + "\n")
            elif msg_type == "execute_reply":
                reply_received = True
                reply_status = content.get("status")
                if reply_status != "ok":
                    # Continue collecting until idle so the traceback is captured.
                    diagnostic.append(
                        f"\n[execute_reply status={reply_status!r}]\n"
                    )
            elif msg_type == "status" and content.get("execution_state") == "idle":
                idle_received = True

            if reply_received and idle_received:
                if reply_status != "ok":
                    raise RuntimeError(
                        "Colab kernel execution failed. "
                        "See colab_runtime_output.log in the run directory."
                    )
                result = "".join(output)
                if log_path:
                    try:
                        log_path.parent.mkdir(parents=True, exist_ok=True)
                        log_path.write_text(
                            "".join(diagnostic)[-500_000:],
                            encoding="utf-8",
                        )
                    except OSError:
                        pass
                return result

        diagnostic.append(
            "\n[TIMEOUT] "
            f"execute_reply_received={reply_received}, "
            f"idle_received={idle_received}, reply_status={reply_status!r}\n"
        )
        if log_path:
            try:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text("".join(diagnostic)[-500_000:], encoding="utf-8")
            except OSError:
                pass
        raise TimeoutError(
            "Timed out waiting for Colab completion "
            f"(execute_reply_received={reply_received}, idle_received={idle_received}). "
            "See colab_runtime_output.log in the run directory."
        )
    except Exception as exc:
        if isinstance(exc, (RuntimeError, TimeoutError)):
            if log_path and not Path(log_path).exists():
                try:
                    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
                    Path(log_path).write_text(
                        "".join(diagnostic)[-500_000:], encoding="utf-8"
                    )
                except OSError:
                    pass
            raise
        raise ColabAccessRequired(
            "Could not communicate with the Colab Jupyter kernel: " + str(exc)
        ) from exc
    finally:
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
        if kernel_id:
            try:
                requests.delete(
                    base + f"/api/kernels/{kernel_id}",
                    headers={"X-Colab-Runtime-Proxy-Token": proxy_token},
                    timeout=10,
                )
            except Exception:
                pass


def run_colab_job(frame: pd.DataFrame, *, target: str, features: list[str],
                  problem_type: str, candidates: list[str], metric: str,
                  run_dir: Path, workers: int = 3) -> dict[str, Any]:
    """Train the configured allow-listed models on a connected Colab runtime."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Colab training requires a non-empty prepared pandas DataFrame.")
    if problem_type not in {
        "classification", "regression", "clustering",
        "anomaly_detection", "dimensionality_reduction",
    }:
        raise ValueError(f"Unsupported problem type: {problem_type!r}")
    if not candidates:
        raise ValueError("At least one candidate model is required.")
    missing = [column for column in list(features) + [target] if column not in frame.columns]
    if missing:
        raise ValueError(f"Prepared dataset is missing planned columns: {missing}")

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    prepared = frame[list(features) + [target]].copy()
    csv_b64 = base64.b64encode(prepared.to_csv(index=False).encode("utf-8")).decode("ascii")
    code = _remote_training_script(csv_b64, target, list(features), problem_type, list(candidates), metric, workers)

    print("Connecting to Google Colab runtime (credentials and proxy token are kept private)...", flush=True)
    runtime_url, proxy_token = _get_runtime_connection()
    print("Connected. Training will run remotely; the prepared dataset is being transferred.", flush=True)
    output = _execute_on_runtime(
        code, runtime_url, proxy_token,
        log_path=run_dir / "colab_runtime_output.log",
    )

    remote_error = next((line[len("AUTOML_REMOTE_ERROR:"):] for line in output.splitlines()
                         if line.startswith("AUTOML_REMOTE_ERROR:")), None)
    if remote_error:
        raise RuntimeError("Colab training failed:\n" + remote_error)
    result_line = next((line[len("AUTOML_RESULT_JSON:"):] for line in output.splitlines()
                        if line.startswith("AUTOML_RESULT_JSON:")), None)
    if not result_line:
        raise RuntimeError(
            "Colab finished without returning metrics. Diagnostic output was saved to "
            + str(run_dir / "colab_runtime_output.log")
            + ". Inspect that file for remote errors or missing output messages."
        )
    result = json.loads(result_line)

    chunks: dict[int, str] = {}
    for line in output.splitlines():
        if line.startswith("AUTOML_ZIP_CHUNK:"):
            _, index, chunk = line.split(":", 2)
            chunks[int(index)] = chunk
    if not chunks:
        raise RuntimeError("Colab returned metrics but no model artifacts.")
    archive_bytes = base64.b64decode("".join(chunks[i] for i in sorted(chunks)))
    archive_path = run_dir / "colab_artifacts.zip"
    archive_path.write_bytes(archive_bytes)
    with zipfile.ZipFile(archive_path, "r") as archive:
        for member in archive.namelist():
            # Keep files flat in the intended local models output directory.
            if Path(member).name != member or member in {"", ".", ".."}:
                raise RuntimeError(f"Unexpected path in returned artifact archive: {member!r}")
        archive.extractall(run_dir)
    archive_path.unlink(missing_ok=True)

    result["best_model_path"] = str(run_dir / "best_model.joblib")
    for experiment in result.get("experiments", []):
        artifact_name = Path(experiment.get("artifact", "")).name
        if artifact_name:
            experiment["artifact"] = str(run_dir / artifact_name)
    result["training_backend"] = "google_colab"
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Colab training finished. Model artifacts and metrics were saved locally.", flush=True)
    return result
