"""Google Colab runtime adapter for tabular scikit-learn training.

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
from ..config import COLLAB_RUNTIME

COLAB_API = COLLAB_RUNTIME
DEFAULT_RUNTIME = "runtimes/r-6f49ecb3-e213-4796-b4d1-6923be00f80e"


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
    credentials = _google_credentials()
    project = os.getenv("GOOGLE_CLOUD_PROJECT", "537909573529").strip()
    runtime_name = os.getenv("COLAB_RUNTIME_NAME", DEFAULT_RUNTIME).strip()
    if not runtime_name:
        raise ColabAccessRequired("Set COLAB_RUNTIME_NAME to an existing Colab runtime resource name.")
    if not runtime_name.startswith("runtimes/") and "/runtimes/" not in runtime_name:
        raise ColabAccessRequired(
            "COLAB_RUNTIME_NAME must look like 'runtimes/r-…' or a full Colab runtime resource name."
        )
    if runtime_name.startswith("projects/"):
        url = f"{COLAB_API}/{runtime_name}"
    else:
        url = f"{COLAB_API}/{runtime_name}"

    headers = {"Authorization": f"Bearer {credentials.token}"}
    if project:
        headers["x-goog-user-project"] = project
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        runtime = response.json()
    except Exception as exc:
        raise ColabAccessRequired(
            f"Could not retrieve Colab runtime {runtime_name!r}. Confirm it exists and that the Colab API "
            f"is enabled for project {project!r}. Details: {exc}"
        ) from exc

    connection = runtime.get("connectionInfo") or runtime.get("connection_info") or {}
    runtime_url = connection.get("url") or connection.get("runtimeUrl") or connection.get("runtime_url")
    token = connection.get("token") or connection.get("proxyToken") or connection.get("proxy_token")
    if not runtime_url or not token:
        state = runtime.get("state", "unknown")
        raise ColabAccessRequired(
            f"Colab runtime did not provide connection information (state={state!r}). "
            "The runtime may still be starting or may have expired. Start/recreate it and retry."
        )
    return runtime_url.rstrip("/"), token


def _remote_training_script(csv_b64: str, target: str, features: list[str], problem_type: str,
                            candidates: list[str], metric: str, workers: int) -> str:
    config = {
        "csv_b64": csv_b64,
        "target": target,
        "features": features,
        "problem_type": problem_type,
        "candidates": candidates,
        "metric": metric,
        "workers": workers,
    }
    return r"""import base64, json, os, time, zipfile, tempfile, traceback
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
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor

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
    missing = [c for c in features + [target] if c not in frame.columns]
    if missing:
        raise ValueError('Missing required columns: ' + repr(missing))
    frame = frame[features + [target]].dropna(subset=[target]).copy()
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
        'errors': errors, 'best_model_path': str(OUT / 'best_model.joblib'), 'rows_used': int(len(frame)),
        'training_backend': 'google_colab'
    }
    (OUT / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    script = '''# Reproducible training entry point generated by Auto ML Agent.\nfrom pathlib import Path\nimport pandas as pd\nfrom auto_ml_agent.engine.trainer import train_candidates\nCSV_PATH = Path("your_dataset.csv")\nTARGET = ''' + repr(target) + '''\nFEATURES = ''' + repr(features) + '''\nPROBLEM_TYPE = ''' + repr(problem_type) + '''\nMODEL = ''' + repr(best['model']) + '''\nMETRIC = ''' + repr(metric) + '''\nif __name__ == "__main__":\n    df = pd.read_csv(CSV_PATH)\n    print(train_candidates(df, TARGET, FEATURES, PROBLEM_TYPE, [MODEL], METRIC, Path("model_output"), workers=1))\n'''
    (OUT / 'train_model.py').write_text(script, encoding='utf-8')
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


def _execute_on_runtime(code: str, runtime_url: str, proxy_token: str) -> str:
    try:
        import websocket
    except ImportError as exc:
        raise ColabAccessRequired("Install websocket-client: pip install websocket-client") from exc

    headers = [f"X-Colab-Runtime-Proxy-Token: {proxy_token}"]
    base = runtime_url.rstrip("/")
    session = str(uuid.uuid4())
    try:
        response = requests.post(
            base + "/api/kernels",
            headers={"X-Colab-Runtime-Proxy-Token": proxy_token},
            json={"name": "python3"}, timeout=30,
        )
        response.raise_for_status()
        kernel_id = response.json()["id"]
    except Exception as exc:
        raise ColabAccessRequired(
            "Could not create a Python kernel on the Colab runtime. The runtime may be disconnected. Details: " + str(exc)
        ) from exc

    ws_url = base.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
    ws_url += f"/api/kernels/{kernel_id}/channels?session_id={session}"
    ws = None
    try:
        ws = websocket.create_connection(
            ws_url, header=headers, timeout=60, origin=base,
            suppress_origin=False,
        )
        msg_id = uuid.uuid4().hex
        message = {
            "header": {"msg_id": msg_id, "username": "automl-agent", "session": session,
                       "msg_type": "execute_request", "version": "5.3"},
            "parent_header": {}, "metadata": {},
            "content": {"code": code, "silent": False, "store_history": False,
                        "user_expressions": {}, "allow_stdin": False, "stop_on_error": True},
            "channel": "shell", "buffers": [],
        }
        ws.send(json.dumps(message))
        output = []
        deadline = time.time() + int(os.getenv("COLAB_EXECUTION_TIMEOUT_SECONDS", "3600"))
        while time.time() < deadline:
            ws.settimeout(min(60, max(1, deadline - time.time())))
            raw = ws.recv()
            if not raw:
                continue
            msg = json.loads(raw)
            parent = msg.get("parent_header") or {}
            if parent.get("msg_id") != msg_id:
                continue
            msg_type = (msg.get("header") or {}).get("msg_type")
            content = msg.get("content") or {}
            if msg_type == "stream":
                output.append(content.get("text", ""))
            elif msg_type == "error":
                output.append("\n".join(content.get("traceback") or [content.get("evalue", "Remote execution error")]))
            elif msg_type == "execute_reply":
                if content.get("status") != "ok":
                    raise RuntimeError("Colab kernel execution failed:\n" + "".join(output))
                return "".join(output)
        raise TimeoutError("Timed out waiting for the Colab training job to finish.")
    except Exception as exc:
        if isinstance(exc, (RuntimeError, TimeoutError)):
            raise
        raise ColabAccessRequired("Could not communicate with the Colab Jupyter kernel: " + str(exc)) from exc
    finally:
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
        try:
            requests.delete(base + f"/api/kernels/{kernel_id}", headers={"X-Colab-Runtime-Proxy-Token": proxy_token}, timeout=10)
        except Exception:
            pass


def run_colab_job(frame: pd.DataFrame, *, target: str, features: list[str],
                  problem_type: str, candidates: list[str], metric: str,
                  run_dir: Path, workers: int = 3) -> dict[str, Any]:
    """Train the configured allow-listed models on a connected Colab runtime."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Colab training requires a non-empty prepared pandas DataFrame.")
    if problem_type not in {"classification", "regression"}:
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
    output = _execute_on_runtime(code, runtime_url, proxy_token)

    remote_error = next((line[len("AUTOML_REMOTE_ERROR:"):] for line in output.splitlines()
                         if line.startswith("AUTOML_REMOTE_ERROR:")), None)
    if remote_error:
        raise RuntimeError("Colab training failed:\n" + remote_error)
    result_line = next((line[len("AUTOML_RESULT_JSON:"):] for line in output.splitlines()
                        if line.startswith("AUTOML_RESULT_JSON:")), None)
    if not result_line:
        raise RuntimeError("Colab finished without returning metrics. Check runtime logs and retry.")
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
