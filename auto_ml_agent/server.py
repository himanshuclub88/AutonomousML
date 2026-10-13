"""FastAPI server for the Autonomous ML Engineering Agent.

Place this file at ``auto_ml_agent/server.py`` (next to ``main.py``).
Start it from the repository root with:

    uvicorn auto_ml_agent.server:app --reload

The API deliberately orchestrates the existing engine modules instead of
calling the interactive CUI entry point. The CLI in main.py remains unchanged.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Query,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, model_validator

from .agent import ALLOWED_MODELS, create_plan, review_plan_with_data_quality
from .backends.colab import run_colab_job
from .backends.local import run_local_job
from .engine.dataset import (
    analyze_features,
    generate_synthetic,
    inspect_csv,
    prepare_dataset,
    validate_generated_data,
    write_data_quality_report,
)
from .engine.llmdatagen import build_dataset_profile, generate_batch
from .schemas import MLExperimentPlan


# ---------------------------------------------------------------------------
# Paths and application settings
# ---------------------------------------------------------------------------

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
WORKSPACE_DIR = PROJECT_ROOT / "workspace"
UPLOADS_DIR = WORKSPACE_DIR / "uploads"
RUNS_DIR = WORKSPACE_DIR / "runs"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_MB = max(1, int(os.getenv("AUTO_ML_MAX_UPLOAD_MB", "100")))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
DEFAULT_APPROVAL_TIMEOUT_MINUTES = 120
ALLOWED_METRICS = {
    "classification": {"accuracy", "f1_weighted"},
    "regression": {"rmse", "mae", "r2"},
}

DEFAULT_CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
CORS_ORIGINS = [
    item.strip()
    for item in os.getenv("AUTO_ML_CORS_ORIGINS", ",".join(DEFAULT_CORS_ORIGINS)).split(",")
    if item.strip()
]

app = FastAPI(
    title="Autonomous ML Engineering Agent API",
    version="1.0.0",
    description=(
        "A GUI-oriented API over the existing ML agent. Supports CSV inspection, "
        "optional LLM row augmentation, plan review and approval checkpoints, "
        "background training jobs, detailed progress events, metrics, and artifacts. "
        "The existing command-line interface remains separate and unchanged."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class ExperimentCreateRequest(BaseModel):
    """Everything the GUI needs to supply before an experiment starts."""

    use_case: str = Field(min_length=3, max_length=2000, description="What should the model predict or classify?")
    backend: Literal["local", "colab"] = Field(default="local")
    dataset_id: str | None = Field(default=None, description="ID returned by POST /api/datasets/inspect")
    target_column: str | None = Field(default=None, description="Optional requested target column")
    problem_type: Literal["classification", "regression"] | None = None
    candidate_models: list[str] | None = Field(default=None, min_length=1, max_length=4)
    feature_columns: list[str] | None = Field(default=None, min_length=1, description="Optional feature selection; defaults to the agent plan")
    evaluation_metric: Literal["accuracy", "f1_weighted", "rmse", "mae", "r2"] | None = None
    synthetic_rows: int = Field(default=1500, ge=200, le=20000)
    additional_rows: int = Field(default=0, ge=0, le=100000, description="Number of extra LLM-generated rows to append to an uploaded CSV")
    batch_size: int = Field(default=200, ge=1, le=2000)
    remove_duplicates: bool = False
    approval_timeout_minutes: int = Field(default=DEFAULT_APPROVAL_TIMEOUT_MINUTES, ge=5, le=1440)
    notes: str | None = Field(default=None, max_length=4000, description="Optional context for the experiment; displayed in the run record")

    @model_validator(mode="after")
    def validate_request_combinations(self):
        if self.additional_rows and not self.dataset_id:
            raise ValueError("LLM row augmentation requires an uploaded CSV dataset.")
        if self.target_column and not self.target_column.strip():
            raise ValueError("target_column cannot be blank when provided.")
        return self


class ApprovalRequest(BaseModel):
    stage: Literal["initial_plan", "revised_plan"]
    action: Literal["approve", "reject", "edit_and_approve"]
    plan: dict[str, Any] | None = Field(default=None, description="Complete MLExperimentPlan JSON when editing a plan")
    message: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def require_plan_for_edit(self):
        if self.action == "edit_and_approve" and self.plan is None:
            raise ValueError("Provide the complete edited plan when action='edit_and_approve'.")
        return self


# ---------------------------------------------------------------------------
# Internal job state and JSON-safe helpers
# ---------------------------------------------------------------------------

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _json_safe(value: Any) -> Any:
    """Convert common pandas/numpy/Pydantic values to JSON-safe values."""
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump())
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and missing:
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(_json_safe(data), indent=2, ensure_ascii=False), encoding="utf-8")
    temp_path.replace(path)


def _enqueue_event(queue: asyncio.Queue, event: dict[str, Any]) -> None:
    """Run on a subscriber's event loop; avoid blocking worker threads."""
    try:
        queue.put_nowait(event)
    except asyncio.QueueFull:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            pass


@dataclass
class ExperimentJob:
    run_id: str
    run_dir: Path
    request_data: dict[str, Any]
    created_at: str = field(default_factory=utc_now)
    started_at: str | None = None
    finished_at: str | None = None
    status: str = "queued"
    stage: str = "queued"
    progress: float = 0.0
    message: str = "Experiment queued."
    initial_plan: dict[str, Any] | None = None
    revised_plan: dict[str, Any] | None = None
    final_plan: dict[str, Any] | None = None
    quality_report: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    pending_approval: str | None = None
    approval_timeout_minutes: int = DEFAULT_APPROVAL_TIMEOUT_MINUTES
    approval_signal: threading.Event = field(default_factory=threading.Event, repr=False)
    approval_response: dict[str, Any] | None = field(default=None, repr=False)
    cancel_requested: bool = False
    events: list[dict[str, Any]] = field(default_factory=list, repr=False)
    next_sequence: int = 1
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    subscribers: list[tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = field(default_factory=list, repr=False)

    def public(self) -> dict[str, Any]:
        with self.lock:
            return _json_safe({
                "run_id": self.run_id,
                "request_data": self.request_data,
                "created_at": self.created_at,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "status": self.status,
                "stage": self.stage,
                "progress": self.progress,
                "message": self.message,
                "pending_approval": self.pending_approval,
                "initial_plan": self.initial_plan,
                "revised_plan": self.revised_plan,
                "final_plan": self.final_plan,
                "quality_report": self.quality_report,
                "result": self.result,
                "error": self.error,
                "run_directory": str(self.run_dir),
                "elapsed_seconds": _elapsed_seconds(self.started_at, self.finished_at),
            })


def _elapsed_seconds(started_at: str | None, finished_at: str | None = None) -> float | None:
    if not started_at:
        return None
    try:
        start = datetime.fromisoformat(started_at)
        end = datetime.fromisoformat(finished_at) if finished_at else datetime.now(timezone.utc)
        return round(max(0.0, (end - start).total_seconds()), 2)
    except (TypeError, ValueError):
        return None


JOBS: dict[str, ExperimentJob] = {}
JOBS_LOCK = threading.RLock()
DATASETS: dict[str, dict[str, Any]] = {}
DATASETS_LOCK = threading.RLock()
JOB_POOL = ThreadPoolExecutor(
    max_workers=max(1, int(os.getenv("AUTO_ML_JOB_WORKERS", "2"))),
    thread_name_prefix="auto-ml-job",
)


def _persist_job(job: ExperimentJob) -> None:
    try:
        _atomic_json(job.run_dir / "job.json", job.public())
    except OSError:
        # A persistence hiccup should not terminate a model run.
        pass


def _emit(
    job: ExperimentJob,
    stage: str,
    message: str,
    *,
    level: str = "info",
    progress: float | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    event: dict[str, Any]
    with job.lock:
        job.stage = stage
        job.message = message
        if progress is not None:
            job.progress = max(0.0, min(100.0, round(float(progress), 2)))
        event = {
            "sequence": job.next_sequence,
            "timestamp": utc_now(),
            "run_id": job.run_id,
            "stage": stage,
            "level": level,
            "message": message,
            "progress": job.progress,
            "details": _json_safe(details or {}),
        }
        job.next_sequence += 1
        job.events.append(event)
        subscribers = list(job.subscribers)
        try:
            with (job.run_dir / "events.jsonl").open("a", encoding="utf-8") as file:
                file.write(json.dumps(event, ensure_ascii=False) + "\n")
        except OSError:
            pass
        for loop, queue in subscribers:
            try:
                loop.call_soon_threadsafe(_enqueue_event, queue, event)
            except RuntimeError:
                # The websocket's event loop has already closed.
                pass
    _persist_job(job)
    return event


def _set_status(job: ExperimentJob, status: str) -> None:
    with job.lock:
        job.status = status
        if status in {"succeeded", "failed", "cancelled"}:
            job.finished_at = utc_now()
    _persist_job(job)


def _restore_jobs_from_disk() -> None:
    """Rebuild the in-memory experiment index from persisted run folders.

    A Uvicorn reload/restart clears JOBS, but should not make existing run IDs
    return 404 when their artifacts are still present on disk. Active jobs cannot
    safely resume in a new process, so mark those as interrupted and preserve logs.
    """
    restored: dict[str, ExperimentJob] = {}
    for run_dir in RUNS_DIR.iterdir():
        if not run_dir.is_dir():
            continue
        job_file = run_dir / "job.json"
        request_file = run_dir / "request.json"
        try:
            saved = json.loads(job_file.read_text(encoding="utf-8"))
            run_id = str(saved.get("run_id") or run_dir.name)
            request_data = saved.get("request_data")
            if not isinstance(request_data, dict):
                request_data = json.loads(request_file.read_text(encoding="utf-8"))
            job = ExperimentJob(
                run_id=run_id,
                run_dir=run_dir,
                request_data=request_data,
                created_at=saved.get("created_at") or utc_now(),
                started_at=saved.get("started_at"),
                finished_at=saved.get("finished_at"),
                status=saved.get("status", "failed"),
                stage=saved.get("stage", "interrupted"),
                progress=float(saved.get("progress") or 0),
                message=saved.get("message") or "Restored from disk.",
                initial_plan=saved.get("initial_plan"),
                revised_plan=saved.get("revised_plan"),
                final_plan=saved.get("final_plan"),
                quality_report=saved.get("quality_report"),
                result=saved.get("result"),
                error=saved.get("error"),
                pending_approval=saved.get("pending_approval"),
                approval_timeout_minutes=int(request_data.get("approval_timeout_minutes") or DEFAULT_APPROVAL_TIMEOUT_MINUTES),
            )
            events_file = run_dir / "events.jsonl"
            if events_file.exists():
                for line in events_file.read_text(encoding="utf-8").splitlines():
                    try:
                        event = json.loads(line)
                        if isinstance(event, dict) and "sequence" in event:
                            job.events.append(event)
                    except json.JSONDecodeError:
                        continue
            job.events.sort(key=lambda item: int(item.get("sequence", 0)))
            job.next_sequence = max((int(e.get("sequence", 0)) for e in job.events), default=0) + 1

            if job.status not in {"succeeded", "failed", "cancelled"}:
                job.status = "failed"
                job.finished_at = job.finished_at or utc_now()
                job.pending_approval = None
                job.error = {
                    "type": "ServerRestarted",
                    "message": "The API process restarted while this experiment was active. Existing files and logs were restored, but the training process cannot be resumed automatically. Start a new experiment to train again.",
                }
                _emit(
                    job, "interrupted", job.error["message"], level="error",
                    details={"restored_from_disk": True, "run_directory": str(run_dir)},
                )
            else:
                _persist_job(job)
            restored[run_id] = job
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            # Ignore incomplete folders; a run is restorable only with its metadata.
            continue
    with JOBS_LOCK:
        JOBS.update(restored)


# Restore previous run IDs after process startup, before serving API requests.
_restore_jobs_from_disk()


def _get_job_or_404(run_id: str) -> ExperimentJob:
    with JOBS_LOCK:
        job = JOBS.get(run_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Experiment {run_id!r} was not found in the active registry or workspace metadata.")
    return job


class ExperimentCancelled(Exception):
    pass


class ApprovalRejected(Exception):
    pass


def _check_cancelled(job: ExperimentJob) -> None:
    with job.lock:
        if job.cancel_requested:
            raise ExperimentCancelled("Cancellation was requested by the user.")


# ---------------------------------------------------------------------------
# Model Studio: generate run-specific, evidence-based charts
# ---------------------------------------------------------------------------
def _generate_model_studio(job: "ExperimentJob", result: dict[str, Any]) -> dict[str, Any]:
    """Create charts from the saved metrics, fitted pipeline, and exact prepared run data.

    Charts are optional outputs: a plotting/introspection failure must not fail a
    successful training run. All files are scoped to this experiment's graphs/ folder.
    """
    graph_dir = job.run_dir / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)
    generated: list[dict[str, str]] = []
    warnings: list[str] = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import joblib
        from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
        from .engine.dataset import split_data
    except Exception as exc:
        return {"generated": [], "warnings": [f"Graph dependencies unavailable: {type(exc).__name__}: {exc}"]}

    def save(fig, filename: str, title: str, description: str) -> None:
        path = graph_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        generated.append({"id": filename, "title": title, "description": description,
                          "url": f"/api/experiments/{job.run_id}/artifacts/graphs/{filename}"})

    # Candidate comparison: values are the trainer's actual validation scores.
    try:
        experiments = result.get("experiments") or []
        experiments = [item for item in experiments if item.get("validation_score") is not None]
        if experiments:
            ordered = sorted(experiments, key=lambda item: float(item["validation_score"]), reverse=(result.get("problem_type") == "classification" or result.get("selection_metric") == "r2"))
            names = [str(item.get("model", "model")) for item in ordered]
            scores = [float(item["validation_score"]) for item in ordered]
            fig, ax = plt.subplots(figsize=(9, max(3.4, 0.55 * len(names))))
            bars = ax.barh(names[::-1], scores[::-1])
            ax.set_title("Candidate model comparison · validation")
            ax.set_xlabel(str(result.get("selection_metric", "validation score")).upper())
            ax.grid(axis="x", alpha=.22)
            for bar, value in zip(bars, scores[::-1]):
                ax.text(bar.get_width(), bar.get_y()+bar.get_height()/2, f" {value:.4g}", va="center", fontsize=9)
            save(fig, "candidate_comparison.png", "Candidate comparison", "Actual validation scores returned by the trainer.")
    except Exception as exc:
        warnings.append(f"Candidate comparison chart could not be generated: {exc}")

    # Model and test diagnostics are only generated from the exact saved prepared data.
    model_path_value = result.get("best_model_path")
    data_path = job.run_dir / "prepared_training_data.csv"
    try:
        if not model_path_value:
            raise FileNotFoundError("The result did not include a best-model path.")
        model_path = Path(model_path_value)
        if not model_path.is_absolute():
            model_path = job.run_dir / "models" / model_path
        if not model_path.is_file():
            model_path = job.run_dir / "models" / "best_model.joblib"
        if not model_path.is_file():
            raise FileNotFoundError("The fitted best_model.joblib artifact was not found.")
        model = joblib.load(model_path)

        # Feature importance / coefficient magnitude from the fitted estimator.
        try:
            estimator = model.named_steps.get("model") if hasattr(model, "named_steps") else model
            preprocess = model.named_steps.get("preprocess") if hasattr(model, "named_steps") else None
            values = getattr(estimator, "feature_importances_", None)
            importance_label = "Feature importance"
            if values is None and getattr(estimator, "coef_", None) is not None:
                coef = np.asarray(estimator.coef_)
                values = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
                importance_label = "Absolute coefficient magnitude"
            if values is not None:
                values = np.asarray(values).ravel()
                try:
                    names = list(preprocess.get_feature_names_out()) if preprocess is not None else []
                except Exception:
                    names = []
                if len(names) != len(values):
                    names = [f"Feature {index+1}" for index in range(len(values))]
                top = sorted(zip(names, values), key=lambda pair: float(pair[1]), reverse=True)[:20]
                if top:
                    labels = [str(pair[0]).replace("numeric__", "").replace("categorical__", "") for pair in top]
                    scores = [float(pair[1]) for pair in top]
                    fig, ax = plt.subplots(figsize=(10, max(4, .34 * len(labels))))
                    bars = ax.barh(labels[::-1], scores[::-1])
                    ax.set_title(f"Top features · {importance_label.lower()}")
                    ax.set_xlabel(importance_label)
                    ax.grid(axis="x", alpha=.22)
                    save(fig, "feature_importance.png", "Feature insights", f"Derived from the fitted {result.get('best_model', 'best')} estimator; encoded features may appear separately.")
            else:
                warnings.append("This model does not expose native feature importance or coefficients; no feature-importance chart was fabricated.")
        except Exception as exc:
            warnings.append(f"Feature importance was unavailable: {exc}")

        # Recreate the deterministic held-out split used by the existing trainer.
        if data_path.is_file():
            frame = pd.read_csv(data_path)
            target = result.get("target_column")
            problem_type = result.get("problem_type")
            if target in frame.columns:
                X_train, X_val, X_test, y_train, y_val, y_test = split_data(frame, target, problem_type)
                predictions = model.predict(X_test)
                diagnostic = pd.DataFrame({"actual": y_test.astype(str).to_numpy(), "predicted": pd.Series(predictions).astype(str).to_numpy()})
                diagnostic.to_csv(graph_dir / "held_out_test_predictions.csv", index=False)
                if problem_type == "classification":
                    labels = list(pd.Series(y_test).astype(str).value_counts().index)
                    labels = sorted(set(labels) | set(pd.Series(predictions).astype(str)))
                    cm = confusion_matrix(pd.Series(y_test).astype(str), pd.Series(predictions).astype(str), labels=labels)
                    fig, ax = plt.subplots(figsize=(max(5, .65*len(labels)), max(4, .6*len(labels))))
                    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
                    display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d", xticks_rotation=35)
                    ax.set_title("Held-out test · confusion matrix")
                    save(fig, "confusion_matrix.png", "Confusion matrix", "Computed from the same deterministic held-out split used for final evaluation.")
                else:
                    actual = pd.to_numeric(pd.Series(y_test), errors="coerce").to_numpy()
                    predicted = pd.to_numeric(pd.Series(predictions), errors="coerce").to_numpy()
                    valid = np.isfinite(actual) & np.isfinite(predicted)
                    if valid.any():
                        fig, ax = plt.subplots(figsize=(7, 6))
                        ax.scatter(actual[valid], predicted[valid], alpha=.65, s=22)
                        low = float(min(actual[valid].min(), predicted[valid].min()))
                        high = float(max(actual[valid].max(), predicted[valid].max()))
                        ax.plot([low, high], [low, high], linestyle="--", linewidth=1.5)
                        ax.set_xlabel("Actual target")
                        ax.set_ylabel("Predicted target")
                        ax.set_title("Held-out test · actual vs predicted")
                        ax.grid(alpha=.22)
                        save(fig, "actual_vs_predicted.png", "Actual vs predicted", "Predictions from the saved model on the held-out test split.")
                        residuals = actual[valid] - predicted[valid]
                        fig, ax = plt.subplots(figsize=(8, 4.5))
                        ax.scatter(predicted[valid], residuals, alpha=.65, s=22)
                        ax.axhline(0, linestyle="--", linewidth=1.5)
                        ax.set_xlabel("Predicted target")
                        ax.set_ylabel("Residual (actual − predicted)")
                        ax.set_title("Held-out test · residual diagnostics")
                        ax.grid(alpha=.22)
                        save(fig, "residuals.png", "Residual diagnostics", "Held-out residuals; patterns can indicate model error structure.")
    except Exception as exc:
        warnings.append(f"Model-based diagnostics could not be generated: {type(exc).__name__}: {exc}")

    # Persist the index for the GUI and for inspection outside the browser.
    payload = {"run_id": job.run_id, "generated": generated, "warnings": warnings}
    _atomic_json(graph_dir / "index.json", payload)
    return payload


# ---------------------------------------------------------------------------
# Dataset summaries
# ---------------------------------------------------------------------------

def _column_summary(frame: pd.DataFrame) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for column in frame.columns:
        series = frame[column]
        non_null = series.dropna()
        item: dict[str, Any] = {
            "name": str(column),
            "dtype": str(series.dtype),
            "missing_count": int(series.isna().sum()),
            "missing_percent": round(float(series.isna().mean() * 100), 2),
            "unique_count": int(series.nunique(dropna=True)),
            "sample_values": [_json_safe(v) for v in non_null.drop_duplicates().head(5).tolist()],
        }
        if pd.api.types.is_numeric_dtype(series) and len(non_null):
            item["statistics"] = {
                "min": _json_safe(non_null.min()),
                "max": _json_safe(non_null.max()),
                "mean": _json_safe(round(float(non_null.mean()), 6)),
                "median": _json_safe(non_null.median()),
                "std": _json_safe(round(float(non_null.std()), 6)) if len(non_null) > 1 else 0.0,
            }
        else:
            counts = series.astype("string").value_counts(dropna=True).head(5)
            item["top_values"] = [
                {"value": str(name), "count": int(count)} for name, count in counts.items()
            ]
        summaries.append(item)
    return summaries


def _dataset_summary(frame: pd.DataFrame, dataset_id: str, original_name: str | None = None) -> dict[str, Any]:
    warnings: list[str] = []
    if len(frame) < 30:
        warnings.append("Fewer than 30 rows are available; the current data preparation module requires at least 30 usable rows.")
    if frame.isna().any().any():
        warnings.append("Missing values are present. Feature missing values may be imputed by the trainer; rows with missing targets are removed during preparation.")
    duplicated = int(frame.duplicated().sum())
    if duplicated:
        warnings.append(f"{duplicated:,} duplicate rows detected. They will be reported and preserved unless duplicate removal is enabled.")
    preview_json = json.loads(frame.head(12).to_json(orient="records", date_format="iso", default_handler=str))
    return _json_safe({
        "dataset_id": dataset_id,
        "original_filename": original_name,
        "rows": int(frame.shape[0]),
        "columns_count": int(frame.shape[1]),
        "memory_usage_bytes": int(frame.memory_usage(deep=True).sum()),
        "duplicate_rows": duplicated,
        "total_missing_cells": int(frame.isna().sum().sum()),
        "columns": _column_summary(frame),
        "preview": preview_json,
        "warnings": warnings,
    })


def _resolve_dataset(dataset_id: str) -> dict[str, Any]:
    # IDs are generated by the server and are used as filenames; validate before path construction.
    if not dataset_id or any(ch not in "0123456789abcdef-" for ch in dataset_id.lower()):
        raise HTTPException(status_code=400, detail="Invalid dataset_id.")
    with DATASETS_LOCK:
        record = DATASETS.get(dataset_id)
    path = UPLOADS_DIR / f"{dataset_id}.csv"
    if record is None and path.is_file():
        record = {"dataset_id": dataset_id, "path": str(path), "original_filename": path.name}
        with DATASETS_LOCK:
            DATASETS[dataset_id] = record
    if record is None or not Path(record["path"]).is_file():
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id!r} was not found. Upload the CSV again.")
    return record


# ---------------------------------------------------------------------------
# Plan validation and approval checkpoints
# ---------------------------------------------------------------------------

def _apply_request_preferences(
    plan: MLExperimentPlan,
    request: ExperimentCreateRequest,
    columns: list[str] | None,
) -> MLExperimentPlan:
    data = plan.model_dump()
    original_problem_type = data["problem_type"]
    if request.problem_type:
        problem_type = request.problem_type
    elif request.evaluation_metric in {"rmse", "mae", "r2"}:
        problem_type = "regression"
    elif request.evaluation_metric in {"accuracy", "f1_weighted"}:
        problem_type = "classification"
    else:
        problem_type = original_problem_type
    allowed = ALLOWED_MODELS[problem_type]
    if request.candidate_models:
        candidates = request.candidate_models
    elif problem_type != original_problem_type:
        candidates = sorted(allowed)[:3]
    else:
        candidates = data["candidate_models"]
    unsupported = [model for model in candidates if model not in allowed]
    if unsupported:
        raise ValueError(
            f"Models {unsupported} are not supported for {problem_type}. Supported models: {sorted(allowed)}."
        )
    metric = request.evaluation_metric or data["evaluation_metric"]
    if metric not in ALLOWED_METRICS[problem_type]:
        if request.evaluation_metric:
            raise ValueError(
                f"Metric {metric!r} is incompatible with {problem_type}. "
                f"Choose one of: {sorted(ALLOWED_METRICS[problem_type])}."
            )
        metric = "f1_weighted" if problem_type == "classification" else "rmse"

    data["problem_type"] = problem_type
    data["candidate_models"] = candidates
    data["evaluation_metric"] = metric
    data["synthetic_rows"] = request.synthetic_rows
    data["generate_synthetic_data"] = columns is None
    if request.target_column:
        data["target_column"] = request.target_column

    if request.feature_columns:
        data["feature_columns"] = list(dict.fromkeys(request.feature_columns))

    if columns is not None:
        target = data["target_column"]
        if target not in columns:
            raise ValueError(f"Target column {target!r} is not present in the uploaded dataset.")
        if request.feature_columns:
            unknown = [column for column in data["feature_columns"] if column not in columns]
            if unknown:
                raise ValueError(f"Selected feature column(s) are not present in the uploaded dataset: {unknown}.")
        else:
            features = [column for column in data["feature_columns"] if column in columns and column != target]
            if not features:
                features = [column for column in columns if column != target]
            data["feature_columns"] = features
        if target in data["feature_columns"]:
            raise ValueError("The target column cannot also be included in feature_columns.")
        if not data["feature_columns"]:
            raise ValueError("At least one feature column is required besides the target.")
    elif data["target_column"] in data["feature_columns"]:
        data["feature_columns"] = [col for col in data["feature_columns"] if col != data["target_column"]]
    if not data["feature_columns"]:
        raise ValueError("At least one feature column is required.")
    return MLExperimentPlan.model_validate(data)


def _validate_plan_for_frame(plan: MLExperimentPlan, frame: pd.DataFrame | None) -> MLExperimentPlan:
    data = plan.model_dump()
    allowed = ALLOWED_MODELS[data["problem_type"]]
    invalid_models = [name for name in data["candidate_models"] if name not in allowed]
    if invalid_models:
        raise ValueError(
            f"Plan contains unsupported model(s) for {data['problem_type']}: {invalid_models}. "
            f"Supported models: {sorted(allowed)}."
        )
    if data["evaluation_metric"] not in ALLOWED_METRICS[data["problem_type"]]:
        raise ValueError(
            f"Metric {data['evaluation_metric']!r} is not supported for {data['problem_type']}. "
            f"Choose one of: {sorted(ALLOWED_METRICS[data['problem_type']])}."
        )
    if frame is not None:
        if data["target_column"] not in frame.columns:
            raise ValueError(f"Plan target {data['target_column']!r} is not present in the dataset.")
        missing = [name for name in data["feature_columns"] if name not in frame.columns]
        if missing:
            raise ValueError(f"Plan features are missing from the dataset: {missing}.")
        if data["target_column"] in data["feature_columns"]:
            raise ValueError("Target column cannot also be a feature column.")
    return MLExperimentPlan.model_validate(data)


def _wait_for_approval(job: ExperimentJob, stage: Literal["initial_plan", "revised_plan"], current_plan: MLExperimentPlan, frame: pd.DataFrame | None) -> MLExperimentPlan:
    with job.lock:
        if job.cancel_requested:
            raise ExperimentCancelled("Cancellation was requested by the user.")
        job.pending_approval = stage
        job.approval_response = None
        job.approval_signal.clear()
        job.status = "awaiting_approval"
    plan_key = "initial_plan" if stage == "initial_plan" else "revised_plan"
    _emit(
        job,
        f"approval_{stage}",
        "Your review is required before the pipeline can continue.",
        level="warning",
        details={"approval_stage": stage, "plan": current_plan.model_dump(), "actions": ["approve", "reject", "edit_and_approve"]},
    )
    timeout_seconds = job.approval_timeout_minutes * 60
    completed = job.approval_signal.wait(timeout=timeout_seconds)
    with job.lock:
        response = job.approval_response
        cancelled = job.cancel_requested
        job.pending_approval = None
        if job.status != "cancelled":
            job.status = "running"
    if cancelled:
        raise ExperimentCancelled("Experiment cancelled while awaiting approval.")
    if not completed or response is None:
        raise TimeoutError(
            f"Approval timed out after {job.approval_timeout_minutes} minutes at {stage}."
        )
    action = response.get("action")
    if action == "reject":
        raise ApprovalRejected(response.get("message") or f"The user rejected the {stage.replace('_', ' ')}.")
    if action == "approve":
        accepted_plan = current_plan
    elif action == "edit_and_approve":
        accepted_plan = MLExperimentPlan.model_validate(response.get("plan") or {})
    else:
        raise ValueError(f"Unsupported approval action: {action!r}")
    accepted_plan = _validate_plan_for_frame(accepted_plan, frame)
    _emit(
        job,
        f"approval_{stage}_complete",
        "Plan approved. Continuing the experiment.",
        progress=36 if stage == "initial_plan" else 72,
        details={"approval_stage": stage, "plan": accepted_plan.model_dump()},
    )
    return accepted_plan


# ---------------------------------------------------------------------------
# Experiment orchestration
# ---------------------------------------------------------------------------

def _augment_frame(job: ExperimentJob, frame: pd.DataFrame, additional_rows: int, batch_size: int) -> pd.DataFrame:
    total_batches = (additional_rows + batch_size - 1) // batch_size
    profile = build_dataset_profile(frame)
    generated_batches: list[pd.DataFrame] = []
    generated_count = 0
    _emit(
        job,
        "data_generation_started",
        f"Starting LLM row augmentation: {additional_rows:,} additional rows in {total_batches} batch(es).",
        progress=10,
        details={"original_rows": len(frame), "additional_rows_requested": additional_rows, "batch_size": batch_size, "total_batches": total_batches},
    )
    for batch_number in range(1, total_batches + 1):
        _check_cancelled(job)
        current_batch_size = min(batch_size, additional_rows - generated_count)
        started = time.monotonic()
        _emit(
            job,
            "data_generation_batch_started",
            f"Generating batch {batch_number} of {total_batches} ({current_batch_size:,} rows requested).",
            progress=10 + (generated_count / additional_rows) * 15,
            details={"batch": batch_number, "total_batches": total_batches, "batch_size": current_batch_size},
        )
        batch = generate_batch(frame, current_batch_size, batch_number=batch_number, dataset_profile=profile)
        if not isinstance(batch, pd.DataFrame):
            batch = pd.DataFrame(batch)
        if list(batch.columns) != list(frame.columns):
            raise ValueError(
                f"LLM batch {batch_number} schema does not match the original dataset columns. "
                f"Expected {list(frame.columns)}, received {list(batch.columns)}."
            )
        generated_batches.append(batch)
        generated_count += len(batch)
        _emit(
            job,
            "data_generation_batch_complete",
            f"Generated {generated_count:,} of {additional_rows:,} requested rows.",
            progress=10 + (generated_count / additional_rows) * 15,
            details={
                "batch": batch_number,
                "total_batches": total_batches,
                "batch_rows": len(batch),
                "generated_rows": generated_count,
                "total_rows_to_generate": additional_rows,
                "batch_duration_seconds": round(time.monotonic() - started, 2),
            },
        )
    generated = pd.concat(generated_batches, ignore_index=True)
    combined = pd.concat([frame.copy(), generated], ignore_index=True)
    combined_path = job.run_dir / "augmented_dataset.csv"
    combined.to_csv(combined_path, index=False)
    _emit(
        job,
        "data_generation_complete",
        f"Dataset augmentation complete: {len(frame):,} original + {len(generated):,} generated = {len(combined):,} rows.",
        progress=27,
        details={"original_rows": len(frame), "generated_rows": len(generated), "combined_rows": len(combined), "artifact": str(combined_path)},
    )
    return combined


def _run_experiment(job: ExperimentJob, request: ExperimentCreateRequest) -> None:
    with job.lock:
        job.status = "running"
        job.started_at = utc_now()
    _persist_job(job)
    frame: pd.DataFrame | None = None
    columns: list[str] | None = None
    source_path: Path | None = None

    try:
        _check_cancelled(job)
        # Both backends share planning, data preparation, quality analysis, and
        # approval checkpoints. The selected backend is used only for training.

        if request.dataset_id:
            dataset_record = _resolve_dataset(request.dataset_id)
            source_path = Path(dataset_record["path"])
            _emit(
                job,
                "dataset_loading",
                f"Loading uploaded dataset: {dataset_record.get('original_filename', source_path.name)}.",
                progress=4,
                details={"dataset_id": request.dataset_id, "stored_path": str(source_path)},
            )
            frame, columns = inspect_csv(str(source_path))
            _emit(
                job,
                "dataset_loaded",
                f"Dataset loaded successfully: {len(frame):,} rows and {len(columns)} columns.",
                progress=8,
                details={"rows": len(frame), "column_count": len(columns), "columns": columns},
            )
            if request.target_column and request.target_column not in columns:
                raise ValueError(
                    f"Target column {request.target_column!r} is not in this CSV. Available columns: {columns}."
                )
            if request.additional_rows:
                frame = _augment_frame(job, frame, request.additional_rows, request.batch_size)
                source_path = job.run_dir / "augmented_dataset.csv"
                columns = list(frame.columns)
            else:
                # Keep a copy with the run so the experiment is reproducible.
                source_copy = job.run_dir / "input_dataset_copy.csv"
                shutil.copy2(source_path, source_copy)
        else:
            _emit(
                job,
                "dataset_configuration",
                "No CSV was supplied. The agent will propose a synthetic demonstration dataset.",
                progress=8,
                details={"synthetic_rows_requested": request.synthetic_rows},
            )

        _check_cancelled(job)
        _emit(job, "planning", "Creating the initial ML experiment plan with the agent.", progress=29,
              details={"use_case": request.use_case, "columns": columns, "requested_target": request.target_column})
        plan = create_plan(
            request.use_case,
            dataset_columns=columns,
            requested_target=request.target_column,
        )
        plan = _apply_request_preferences(plan, request, columns)
        plan = _validate_plan_for_frame(plan, frame)
        job.initial_plan = plan.model_dump()
        _atomic_json(job.run_dir / "initial_plan.json", job.initial_plan)
        _emit(
            job,
            "initial_plan_created",
            "Initial plan created. Review the task type, target, features, candidate models, and evaluation metric.",
            progress=32,
            details={"plan": job.initial_plan},
        )

        plan = _wait_for_approval(job, "initial_plan", plan, frame)
        job.initial_plan = plan.model_dump()
        _atomic_json(job.run_dir / "initial_plan.json", job.initial_plan)
        _check_cancelled(job)

        # Prepare the real or synthetic dataset using the approved plan.
        if frame is None:
            _emit(
                job,
                "synthetic_data_generation",
                f"Generating {plan.synthetic_rows:,} synthetic demonstration rows for {plan.problem_type}.",
                progress=40,
                details={"rows": plan.synthetic_rows, "target_column": plan.target_column, "feature_columns": plan.feature_columns},
            )
            frame = generate_synthetic(
                problem_type=plan.problem_type,
                target=plan.target_column,
                features=plan.feature_columns,
                rows=plan.synthetic_rows,
                use_case=request.use_case,
            )
            validate_generated_data(
                frame,
                target=plan.target_column,
                features=plan.feature_columns,
                expected_rows=plan.synthetic_rows,
            )
            synthetic_path = job.run_dir / "synthetic_dataset.csv"
            frame.to_csv(synthetic_path, index=False)
            _emit(job, "synthetic_data_generated", f"Synthetic dataset generated and validated: {len(frame):,} rows.",
                  progress=45, details={"artifact": str(synthetic_path), "rows": len(frame), "columns": list(frame.columns)})
        else:
            selected_columns = plan.feature_columns + [plan.target_column]
            missing = [name for name in selected_columns if name not in frame.columns]
            if missing:
                raise ValueError(f"Approved plan refers to columns missing from the CSV: {missing}.")
            frame = frame[selected_columns].copy()
            used_path = job.run_dir / "dataset_selected_for_training.csv"
            frame.to_csv(used_path, index=False)
            _emit(job, "dataset_selected", "Selected the approved feature and target columns for training.",
                  progress=45, details={"rows": len(frame), "columns": selected_columns, "artifact": str(used_path)})

        _check_cancelled(job)
        _emit(job, "data_quality", "Running dataset quality checks and feature analysis.", progress=49)
        frame, quality_report = prepare_dataset(
            frame=frame,
            target=plan.target_column,
            features=plan.feature_columns,
            problem_type=plan.problem_type,
            remove_duplicates=request.remove_duplicates,
        )
        feature_report = analyze_features(
            frame=frame,
            target=plan.target_column,
            features=plan.feature_columns,
        )
        quality_report["feature_analysis"] = feature_report
        quality_report["dataset_source"] = "uploaded_csv_with_llm_augmentation" if request.additional_rows else ("uploaded_csv" if request.dataset_id else "synthetic_demo")
        quality_report["run_id"] = job.run_id
        quality_report["original_uploaded_rows"] = int(len(pd.read_csv(source_path))) if request.dataset_id and source_path and source_path.exists() and source_path.name != "augmented_dataset.csv" else None
        job.quality_report = _json_safe(quality_report)
        quality_path = job.run_dir / "data_quality_report.json"
        write_data_quality_report(quality_report, quality_path)
        _emit(
            job,
            "data_quality_complete",
            "Dataset quality checks completed. Review missing values, duplicate counts, constant features, and analysis warnings.",
            progress=56,
            details={"report": job.quality_report, "artifact": str(quality_path), "rows_after_preparation": len(frame)},
        )

        _check_cancelled(job)
        _emit(job, "plan_review", "Asking the agent to review the initial plan against the data-quality report.", progress=60)
        revised_plan = review_plan_with_data_quality(
            plan,
            use_case=request.use_case,
            quality_report=quality_report,
        )
        revised_plan = _validate_plan_for_frame(revised_plan, frame)
        job.revised_plan = revised_plan.model_dump()
        _atomic_json(job.run_dir / "revised_plan.json", job.revised_plan)
        _emit(
            job,
            "revised_plan_created",
            "The data-aware plan review is complete. Compare the initial and revised plans before training.",
            progress=65,
            details={"initial_plan": job.initial_plan, "revised_plan": job.revised_plan, "quality_report": job.quality_report},
        )

        final_plan = _wait_for_approval(job, "revised_plan", revised_plan, frame)
        final_plan = _validate_plan_for_frame(final_plan, frame)
        job.final_plan = final_plan.model_dump()
        _atomic_json(job.run_dir / "plan.json", job.final_plan)
        _check_cancelled(job)

        # The data-aware review may recommend a narrower feature list. Re-run the
        # checks against the approved final plan so the report actually describes
        # the data that will be used for training (without modifying the frozen engine).
        if (
            final_plan.problem_type != plan.problem_type
            or final_plan.target_column != plan.target_column
            or final_plan.feature_columns != plan.feature_columns
        ):
            final_columns = final_plan.feature_columns + [final_plan.target_column]
            missing_final_columns = [name for name in final_columns if name not in frame.columns]
            if missing_final_columns:
                raise ValueError(
                    "The approved final plan refers to columns that are not available in the prepared dataset: "
                    f"{missing_final_columns}. Start a new experiment with those columns selected upfront."
                )
            _emit(
                job,
                "final_dataset_recheck",
                "The reviewed plan changed the task or selected features. Rechecking data quality against the approved final configuration.",
                progress=74,
                details={"problem_type": final_plan.problem_type, "target_column": final_plan.target_column, "feature_columns": final_plan.feature_columns},
            )
            final_input = frame[final_columns].copy()
            frame, final_quality_report = prepare_dataset(
                frame=final_input,
                target=final_plan.target_column,
                features=final_plan.feature_columns,
                problem_type=final_plan.problem_type,
                remove_duplicates=request.remove_duplicates,
            )
            final_quality_report["feature_analysis"] = analyze_features(
                frame=frame,
                target=final_plan.target_column,
                features=final_plan.feature_columns,
            )
            final_quality_report["dataset_source"] = quality_report.get("dataset_source")
            final_quality_report["run_id"] = job.run_id
            job.quality_report = _json_safe(final_quality_report)
            quality_report = final_quality_report
            quality_path = job.run_dir / "data_quality_report.json"
            write_data_quality_report(quality_report, quality_path)
            _emit(
                job,
                "final_data_quality_recheck_complete",
                "Data-quality report refreshed for the exact approved final plan.",
                progress=75,
                details={"report": job.quality_report, "artifact": str(quality_path), "rows_after_preparation": len(frame)},
            )

        # Save the exact prepared rows passed to the trainer. Model Studio uses this
        # snapshot to reproduce the deterministic held-out split for diagnostics.
        prepared_data_path = job.run_dir / "prepared_training_data.csv"
        frame[final_plan.feature_columns + [final_plan.target_column]].to_csv(prepared_data_path, index=False)
        _emit(
            job,
            "training_started",
            f"Training {len(final_plan.candidate_models)} model candidate(s). The prepared dataset snapshot was saved for reproducible diagnostics.",
            progress=76,
            details={
                "models": final_plan.candidate_models,
                "workers": min(4, len(final_plan.candidate_models)),
                "metric": final_plan.evaluation_metric,
                "rows": len(frame),
                "features": final_plan.feature_columns,
                "target": final_plan.target_column,
            },
        )
        training_started = time.monotonic()
        training_args = {
            "target": final_plan.target_column,
            "features": final_plan.feature_columns,
            "problem_type": final_plan.problem_type,
            "candidates": final_plan.candidate_models,
            "metric": final_plan.evaluation_metric,
            "run_dir": job.run_dir / "models",
            "workers": min(4, len(final_plan.candidate_models)),
        }
        if request.backend == "colab":
            _emit(
                job,
                "colab_training",
                "Submitting the prepared dataset to Google Colab for remote training.",
                progress=80,
                details={
                    "backend": "google_colab",
                    "models": final_plan.candidate_models,
                    "rows": len(frame),
                },
            )
            result = run_colab_job(frame, **training_args)
        else:
            result = run_local_job(frame, **training_args)
        result["training_wall_time_seconds"] = round(time.monotonic() - training_started, 2)
        result["run_id"] = job.run_id
        result["run_directory"] = str(job.run_dir)
        try:
            studio = _generate_model_studio(job, result)
            result["model_studio"] = studio
            _emit(job, "model_studio_generated", f"Model Studio generated {len(studio.get('generated', []))} chart(s).", progress=96,
                  details={"graphs": studio.get("generated", []), "warnings": studio.get("warnings", [])})
        except Exception as graph_exc:
            result["model_studio"] = {"generated": [], "warnings": [f"Graph generation failed: {type(graph_exc).__name__}: {graph_exc}"]}
            _emit(job, "model_studio_warning", "Training succeeded, but Model Studio chart generation had a problem.", level="warning",
                  details=result["model_studio"])
        job.result = _json_safe(result)
        _atomic_json(job.run_dir / "result_summary.json", job.result)
        _emit(
            job,
            "training_complete",
            f"Training complete. Best model: {result.get('best_model', 'not reported')}.",
            progress=98,
            details={"result": job.result, "metrics_artifact": str(job.run_dir / "models" / "metrics.json")},
        )
        _set_status(job, "succeeded")
        _emit(job, "experiment_complete", "Experiment completed successfully. Reports, metrics, and model artifacts are ready.", progress=100,
              details={"run_directory": str(job.run_dir), "result": job.result})
    except ApprovalRejected as exc:
        _set_status(job, "cancelled")
        _emit(job, "experiment_rejected", str(exc), level="warning", details={"reason": str(exc)})
    except ExperimentCancelled as exc:
        _set_status(job, "cancelled")
        _emit(job, "experiment_cancelled", str(exc), level="warning")
    except Exception as exc:
        error_data = {
            "type": type(exc).__name__,
            "message": str(exc),
            "stage": job.stage,
            "timestamp": utc_now(),
        }
        job.error = error_data
        try:
            (job.run_dir / "error.txt").write_text(
                f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}", encoding="utf-8"
            )
        except OSError:
            pass
        _set_status(job, "failed")
        _emit(job, "experiment_failed", f"Experiment failed: {type(exc).__name__}: {exc}", level="error", details=error_data)


# ---------------------------------------------------------------------------
# API routes: discovery and configuration
# ---------------------------------------------------------------------------

@app.get("/", tags=["System"])
def root():
    # Serve the bundled GUI when frontend/index.html exists; otherwise show API info.
    index_file = FRONTEND_DIR / "index.html"
    if index_file.is_file():
        return FileResponse(index_file, media_type="text/html", filename=None)
    return {
        "name": "Autonomous ML Engineering Agent API",
        "version": app.version,
        "docs": "/docs",
        "health": "/api/health",
        "message": "API is running. Add frontend/index.html to serve the browser dashboard at this address.",
    }


@app.get("/api/health", tags=["System"])
def health() -> dict[str, Any]:
    with JOBS_LOCK:
        job_list = list(JOBS.values())
    active = sum(1 for job in job_list if job.status in {"queued", "running", "awaiting_approval"})
    return {
        "status": "ok",
        "service": "autonomous-ml-agent",
        "timestamp": utc_now(),
        "active_jobs": active,
        "known_jobs": len(job_list),
        "upload_limit_mb": MAX_UPLOAD_MB,
        "workspace": str(WORKSPACE_DIR),
    }


@app.get("/api/capabilities", tags=["System"])
def capabilities() -> dict[str, Any]:
    return {
        "api_version": app.version,
        "dataset_sources": ["csv_upload", "synthetic_demo"],
        "backends": [
            {"id": "local", "label": "Local Python", "available": True},
            {"id": "colab", "label": "Google Colab", "available": True, "reason": None},
        ],
        "problem_types": ["classification", "regression"],
        "models": {key: sorted(values) for key, values in ALLOWED_MODELS.items()},
        "metrics": {key: sorted(values) for key, values in ALLOWED_METRICS.items()},
        "approval_stages": [
            {"id": "initial_plan", "label": "Review initial plan"},
            {"id": "revised_plan", "label": "Review data-aware plan"},
        ],
        "progress_delivery": ["websocket", "polling"],
        "artifacts": ["input/augmented/synthetic datasets", "plans", "data quality report", "metrics", "trained models", "training script", "event log"],
        "limitations": [
            "The current trainer emits results when candidate training finishes; it does not yet expose true in-training per-model progress callbacks.",
            "Job state is held in memory while this API process is running; run artifacts and event logs are persisted under workspace/runs.",
        ],
    }


# ---------------------------------------------------------------------------
# API routes: datasets
# ---------------------------------------------------------------------------

@app.post("/api/datasets/inspect", tags=["Datasets"])
async def upload_and_inspect_dataset(file: UploadFile = File(...)) -> dict[str, Any]:
    original_name = Path(file.filename or "dataset.csv").name
    if Path(original_name).suffix.lower() != ".csv":
        raise HTTPException(status_code=415, detail="Please upload a .csv file.")
    dataset_id = uuid.uuid4().hex
    final_path = UPLOADS_DIR / f"{dataset_id}.csv"
    temporary_path = UPLOADS_DIR / f"{dataset_id}.uploading"
    bytes_written = 0
    try:
        with temporary_path.open("wb") as destination:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail=f"CSV exceeds the {MAX_UPLOAD_MB} MB upload limit.")
                destination.write(chunk)
        if bytes_written == 0:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")
        temporary_path.replace(final_path)
        frame, _columns = inspect_csv(str(final_path))
        summary = _dataset_summary(frame, dataset_id, original_name)
        record = {
            "dataset_id": dataset_id,
            "path": str(final_path),
            "original_filename": original_name,
            "uploaded_at": utc_now(),
            "bytes": bytes_written,
        }
        with DATASETS_LOCK:
            DATASETS[dataset_id] = record
        summary["upload"] = {"bytes": bytes_written, "size_mb": round(bytes_written / (1024 * 1024), 3), "uploaded_at": record["uploaded_at"]}
        return summary
    except HTTPException:
        temporary_path.unlink(missing_ok=True)
        final_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        temporary_path.unlink(missing_ok=True)
        final_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {type(exc).__name__}: {exc}") from exc
    finally:
        await file.close()


@app.get("/api/datasets/{dataset_id}", tags=["Datasets"])
def get_dataset(dataset_id: str) -> dict[str, Any]:
    record = _resolve_dataset(dataset_id)
    try:
        frame, _columns = inspect_csv(record["path"])
        summary = _dataset_summary(frame, dataset_id, record.get("original_filename"))
        summary["upload"] = {key: record[key] for key in ("uploaded_at", "bytes") if key in record}
        return summary
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not inspect dataset: {type(exc).__name__}: {exc}") from exc


@app.delete("/api/datasets/{dataset_id}", tags=["Datasets"])
def delete_dataset(dataset_id: str) -> dict[str, Any]:
    record = _resolve_dataset(dataset_id)
    path = Path(record["path"]).resolve()
    if UPLOADS_DIR.resolve() not in path.parents:
        raise HTTPException(status_code=400, detail="Refusing to delete a file outside the uploads directory.")
    path.unlink(missing_ok=True)
    with DATASETS_LOCK:
        DATASETS.pop(dataset_id, None)
    return {"deleted": True, "dataset_id": dataset_id}


# ---------------------------------------------------------------------------
# API routes: experiments
# ---------------------------------------------------------------------------

@app.post("/api/experiments", status_code=202, tags=["Experiments"])
def create_experiment(request: ExperimentCreateRequest) -> dict[str, Any]:
    if request.backend == "colab":
        raise HTTPException(
            status_code=409,
            detail="Google Colab is not configured yet. Select Local Python; the existing Colab backend intentionally raises an access-required error.",
        )
    if request.dataset_id:
        record = _resolve_dataset(request.dataset_id)
        try:
            dataset_frame, dataset_columns = inspect_csv(record["path"])
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not load selected dataset: {exc}") from exc
        if request.target_column and request.target_column not in dataset_columns:
            raise HTTPException(status_code=422, detail={
                "message": f"Target column {request.target_column!r} was not found.",
                "available_columns": dataset_columns,
            })
        if request.feature_columns:
            missing_features = [column for column in request.feature_columns if column not in dataset_columns]
            if missing_features:
                raise HTTPException(status_code=422, detail={
                    "message": f"Selected feature column(s) were not found: {missing_features}.",
                    "available_columns": dataset_columns,
                })
            if request.target_column and request.target_column in request.feature_columns:
                raise HTTPException(status_code=422, detail="The target column cannot also be a feature.")
        if request.additional_rows and dataset_frame.empty:
            raise HTTPException(status_code=422, detail="Cannot augment an empty dataset.")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    request_payload = request.model_dump()
    job = ExperimentJob(
        run_id=run_id,
        run_dir=run_dir,
        request_data=request_payload,
        approval_timeout_minutes=request.approval_timeout_minutes,
    )
    with JOBS_LOCK:
        JOBS[run_id] = job
    _atomic_json(run_dir / "request.json", request_payload)
    _emit(
        job,
        "queued",
        "Experiment accepted and queued. Configure review cards will appear at both plan approval stages.",
        progress=0,
        details={"request": request_payload, "run_directory": str(run_dir)},
    )
    JOB_POOL.submit(_run_experiment, job, request)
    return {
        "run_id": run_id,
        "status": job.status,
        "message": "Experiment accepted. Follow status/events and respond when an approval checkpoint appears.",
        "status_url": f"/api/experiments/{run_id}",
        "events_url": f"/api/experiments/{run_id}/events",
        "websocket_url": f"/ws/experiments/{run_id}",
        "results_url": f"/api/experiments/{run_id}/results",
        "approvals_expected": ["initial_plan", "revised_plan"],
    }


@app.get("/api/experiments", tags=["Experiments"])
def list_experiments(limit: int = Query(default=50, ge=1, le=200)) -> dict[str, Any]:
    with JOBS_LOCK:
        jobs = list(JOBS.values())
    jobs.sort(key=lambda item: item.created_at, reverse=True)
    return {"count": len(jobs[:limit]), "experiments": [job.public() for job in jobs[:limit]]}


@app.get("/api/experiments/{run_id}", tags=["Experiments"])
def get_experiment(run_id: str) -> dict[str, Any]:
    return _get_job_or_404(run_id).public()


@app.post("/api/experiments/{run_id}/approval", tags=["Experiments"])
def submit_approval(run_id: str, request: ApprovalRequest) -> dict[str, Any]:
    job = _get_job_or_404(run_id)
    with job.lock:
        if job.status != "awaiting_approval" or not job.pending_approval:
            raise HTTPException(status_code=409, detail="This experiment is not currently waiting for approval.")
        if request.stage != job.pending_approval:
            raise HTTPException(status_code=409, detail=f"The pending approval stage is {job.pending_approval!r}, not {request.stage!r}.")
        if job.approval_response is not None:
            raise HTTPException(status_code=409, detail="An approval response has already been submitted for this checkpoint.")
        if request.action == "edit_and_approve" and not request.plan:
            raise HTTPException(status_code=422, detail="Provide the complete edited plan.")
        if request.action == "edit_and_approve" and request.stage == "revised_plan":
            try:
                edited_plan = MLExperimentPlan.model_validate(request.plan or {})
            except Exception as exc:
                raise HTTPException(status_code=422, detail=f"The edited plan is invalid: {exc}") from exc
            approved_basis = job.revised_plan or {}
            locked_fields = ("problem_type", "target_column", "feature_columns")
            changed = [field for field in locked_fields if getattr(edited_plan, field) != approved_basis.get(field)]
            if changed:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "message": "After the data-quality review, task type, target, and features are locked to the checked dataset. Start a new experiment to change them.",
                        "locked_fields_changed": changed,
                    },
                )
        job.approval_response = request.model_dump()
        _emit(
            job,
            f"approval_response_{request.stage}",
            f"Received '{request.action}' for the {request.stage.replace('_', ' ')}.",
            level="warning" if request.action == "reject" else "info",
            details={"stage": request.stage, "action": request.action, "message": request.message},
        )
        job.approval_signal.set()
    return {"accepted": True, "run_id": run_id, "stage": request.stage, "action": request.action}


@app.post("/api/experiments/{run_id}/cancel", tags=["Experiments"])
def cancel_experiment(run_id: str) -> dict[str, Any]:
    job = _get_job_or_404(run_id)
    with job.lock:
        if job.status in {"succeeded", "failed", "cancelled"}:
            raise HTTPException(status_code=409, detail=f"Cannot cancel an experiment with status {job.status!r}.")
        job.cancel_requested = True
        job.approval_response = {"action": "reject", "message": "Cancellation requested."}
        _emit(
            job,
            "cancellation_requested",
            "Cancellation requested. The current synchronous LLM or model-fit operation may need to return before cancellation takes effect.",
            level="warning",
        )
        job.approval_signal.set()
    return {"accepted": True, "run_id": run_id, "message": "Cancellation will take effect at the next safe stage boundary or after the current operation returns."}


@app.get("/api/experiments/{run_id}/plans", tags=["Results"])
def get_plans(run_id: str) -> dict[str, Any]:
    job = _get_job_or_404(run_id)
    return {"initial_plan": job.initial_plan, "revised_plan": job.revised_plan, "final_plan": job.final_plan}


@app.get("/api/experiments/{run_id}/quality", tags=["Results"])
def get_quality_report(run_id: str) -> dict[str, Any]:
    job = _get_job_or_404(run_id)
    if job.quality_report is not None:
        return job.quality_report
    path = job.run_dir / "data_quality_report.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    raise HTTPException(status_code=409, detail="The data-quality report is not available yet.")


@app.get("/api/experiments/{run_id}/results", tags=["Results"])
def get_results(run_id: str) -> dict[str, Any]:
    job = _get_job_or_404(run_id)
    result = job.result
    if result is None:
        path = job.run_dir / "models" / "metrics.json"
        if path.is_file():
            result = json.loads(path.read_text(encoding="utf-8"))
    return {
        "run_id": run_id,
        "status": job.status,
        "ready": result is not None,
        "result": result,
        "error": job.error,
        "quality_report": job.quality_report,
        "plans": {"initial": job.initial_plan, "revised": job.revised_plan, "final": job.final_plan},
    }


@app.get("/api/experiments/{run_id}/events", tags=["Monitoring"])
def get_events(
    run_id: str,
    after: int = Query(default=0, ge=0, description="Return events with sequence greater than this value"),
    limit: int = Query(default=500, ge=1, le=2000),
    level: str | None = Query(default=None, description="Optional info/warning/error filter"),
) -> dict[str, Any]:
    job = _get_job_or_404(run_id)
    with job.lock:
        events = [event for event in job.events if event["sequence"] > after]
    if level:
        events = [event for event in events if event["level"] == level]
    return {
        "run_id": run_id,
        "events": events[:limit],
        "returned": min(len(events), limit),
        "latest_sequence": job.next_sequence - 1,
        "has_more": len(events) > limit,
    }


@app.get("/api/experiments/{run_id}/model-studio", tags=["Results"])
def get_model_studio(run_id: str) -> dict[str, Any]:
    job = _get_job_or_404(run_id)
    index_path = job.run_dir / "graphs" / "index.json"
    if index_path.is_file():
        return json.loads(index_path.read_text(encoding="utf-8"))
    if job.result and job.result.get("model_studio"):
        return {"run_id": run_id, **job.result["model_studio"]}
    raise HTTPException(status_code=409, detail="Model Studio charts are not available until training completes.")


@app.get("/api/experiments/{run_id}/artifacts", tags=["Results"])
def list_artifacts(run_id: str) -> dict[str, Any]:
    job = _get_job_or_404(run_id)
    artifacts: list[dict[str, Any]] = []
    for path in sorted(job.run_dir.rglob("*")):
        if not path.is_file() or path.name.endswith(".tmp"):
            continue
        relative = path.relative_to(job.run_dir).as_posix()
        try:
            stat = path.stat()
        except OSError:
            continue
        artifacts.append({
            "id": relative,
            "name": path.name,
            "path": relative,
            "extension": path.suffix.lower(),
            "size_bytes": stat.st_size,
            "size_kb": round(stat.st_size / 1024, 2),
            "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(timespec="seconds"),
            "download_url": f"/api/experiments/{run_id}/artifacts/{relative}",
        })
    return {"run_id": run_id, "count": len(artifacts), "artifacts": artifacts}


@app.get("/api/experiments/{run_id}/artifacts/{artifact_path:path}", tags=["Results"])
def download_artifact(run_id: str, artifact_path: str):
    job = _get_job_or_404(run_id)
    root = job.run_dir.resolve()
    candidate = (job.run_dir / artifact_path).resolve()
    if candidate != root and root not in candidate.parents:
        raise HTTPException(status_code=400, detail="Invalid artifact path.")
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found.")
    return FileResponse(candidate, filename=candidate.name)


# ---------------------------------------------------------------------------
# Live event streaming over WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/ws/experiments/{run_id}")
async def experiment_websocket(websocket: WebSocket, run_id: str, after: int = 0):
    try:
        job = _get_job_or_404(run_id)
    except HTTPException:
        await websocket.accept()
        await websocket.send_json({"type": "error", "message": f"Experiment {run_id!r} was not found."})
        await websocket.close(code=1008)
        return

    await websocket.accept()
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
    with job.lock:
        # Register and read history under one lock, so events cannot be lost between these operations.
        job.subscribers.append((loop, queue))
        history = [event for event in job.events if event["sequence"] > after]
        snapshot = job.public()
    try:
        await websocket.send_json({"type": "snapshot", "job": snapshot})
        for event in history:
            await websocket.send_json({"type": "event", "event": event})
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=20)
                await websocket.send_json({"type": "event", "event": event})
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "heartbeat", "timestamp": utc_now(), "run_id": run_id})
    except WebSocketDisconnect:
        pass
    except Exception:
        # Client-side reloads and network drops should not disturb the experiment.
        pass
    finally:
        with job.lock:
            job.subscribers = [(sub_loop, sub_queue) for sub_loop, sub_queue in job.subscribers if sub_queue is not queue]


# ---------------------------------------------------------------------------
# Helpful API error handling
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def unexpected_exception_handler(_request, exc: Exception):
    # Keep server-side stack traces in server logs, not in routine GUI error cards.
    return JSONResponse(
        status_code=500,
        content={"detail": "The API encountered an unexpected error.", "error_type": type(exc).__name__},
    )
