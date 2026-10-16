# Autonomous ML Engineering Agent

An AI-assisted machine-learning workspace that turns a plain-English ML idea into a repeatable tabular ML experiment. The agent helps plan the task, prepare or generate data, compare candidate models, evaluate results, and save experiment artifacts.

> **Project status:** Phase 1 focuses on tabular classification and regression. Feature availability depends on the installed code, configuration, and execution backend. Google Colab GUI execution and model-based diagnostics should be verified in your environment before relying on them.

## What the app can do

### 1. Start from a plain-English use case
Describe the problem you want to solve, for example:

- “Predict house prices.”
- “Classify customers into useful segments.”
- “Predict whether a customer will churn.”

The agent uses the description to help determine the ML task and create an experiment plan. The current scope is **tabular classification and regression**.

### 2. Work with a dataset—or generate example data
- Use a CSV dataset when you have one.
- Generate synthetic data when you do not have a dataset available.
- Prepare data for model training.
- Keep data-generation logic separate from interactive input so it can be integrated with other interfaces later.

**Note:** Synthetic data is useful for testing the pipeline, not for proving that a model will perform well on real-world data.

### 3. Review the plan before training
The intended GUI workflow includes a review step so the user can inspect the data-aware plan and approve or reject it before training starts. This keeps the user involved in important decisions rather than silently launching every experiment.

### 4. Train and compare candidate models
For supported tabular tasks, the agent can evaluate multiple candidate algorithms rather than assuming one model is best for every problem. Depending on the task and configuration, candidates can include:

- Logistic regression
- Random forest
- Histogram-based gradient boosting

The pipeline compares candidates using the selected evaluation metric and identifies the best-performing candidate on the configured validation data.

### 5. Evaluate results
Experiments can record metrics for model comparison and evaluation. The appropriate metric depends on the task—for example, classification metrics such as weighted F1 or regression error metrics.

A high score on synthetic or test data is not a guarantee of production performance. Use representative data, suitable validation, and domain-specific checks before deployment.

### 6. Save experiment artifacts
Runs are saved in a workspace folder so results can be reviewed later. Depending on the run and backend, artifacts can include:

- `metrics.json` — evaluation metrics and comparison results
- `best_model.joblib` — serialized best model, when successfully saved
- `train_model.py` — generated or recorded training script
- Candidate model artifacts
- Diagnostic output and other run files

Typical layout:

```text
workspace/
└── runs/
    └── <run_id>/
        ├── metrics.json
        ├── best_model.joblib
        ├── train_model.py
        └── ...
```

The exact contents vary by experiment.

### 7. Use a local CLI and a web GUI
The project has a command-line entry point and a FastAPI-based web interface.

**CLI**

```powershell
python -m auto_ml_agent.main
```

**Web GUI**

```powershell
uvicorn auto_ml_agent.server:app --host 127.0.0.1 --port 8000
```

Then open <http://127.0.0.1:8000/>. Run these commands from the project root with the project's virtual environment activated.

### 8. Choose where training runs
The design separates experiment planning and data preparation from the training backend:

- **Local Python:** train on the current computer.
- **Google Colab:** run training remotely when Colab API access, credentials, and a compatible runtime are configured.

The local machine is intended to handle the use case, data preparation, planning, and approval flow, while the selected backend performs training and returns results. The Colab path has worked in the CLI in a prior test, but the latest GUI/backend changes have not been independently live-tested end-to-end. Verify a new run before depending on it.

## Extra features and design goals

Treat the following as **roadmap goals unless they are present and working in your current checkout**:

- **Data-quality report:** show missing values, duplicate rows, data types, suspicious columns, and other data issues before training.
- **Smarter experiment planning:** choose task type, metrics, preprocessing, and candidate models based on the use case and dataset.
- **Model comparison dashboard:** compare candidate models, metrics, and training outcomes.
- **Visual analytics:** show performance charts, feature insights, and model comparisons in the GUI.
- **Resilient execution:** capture logs, report failures clearly, and retry safe steps where appropriate.
- **Diagnostics fallback:** if a saved model cannot be loaded, still show metrics and non-model-dependent reports instead of losing the entire results view.
- **API-friendly data generation:** keep data-generation functions separate from interactive prompts so other applications can call them.
- **Experiment history:** organize run artifacts so previous runs can be compared.
- **Future expansion:** add more model families, data types, deployment options, and workflow integrations after tabular Phase 1 is stable.

## Architecture at a glance

```text
User's ML idea
      |
      v
Planner / agent
      |
      v
Dataset selection or synthetic data generation
      |
      v
Data preparation and experiment review
      |
      v
Training backend
  +---+----------------+
  |                    |
  v                    v
Local Python       Google Colab
  |                    |
  +---------+----------+
            |
            v
    Evaluation and comparison
            |
            v
    Run folder and artifacts
            |
            v
       Results in GUI/CLI
```

This diagram describes the intended workflow; exact steps and screens depend on the current implementation.

## Requirements

- Python and the project's configured virtual environment
- Project dependencies installed from the repository's dependency file, if provided
- For LLM-powered planning, the provider configuration and credentials expected by the project
- For Colab training, approved Colab API access, Google authentication/application-default credentials, an appropriate Google Cloud project/quota setup, and a supported runtime

Never commit API keys, access tokens, service-account secrets, or local credential files to the repository.

## Quick start

1. Open PowerShell in the project root.
2. Activate the project's virtual environment.
3. Install dependencies according to the repository's dependency instructions.
4. Configure any required LLM provider credentials.
5. Start the CLI:

   ```powershell
   python -m auto_ml_agent.main
   ```

   Or start the web interface:

   ```powershell
   uvicorn auto_ml_agent.server:app --host 127.0.0.1 --port 8000
   ```

6. Describe the ML use case, select a dataset or synthetic data, review the plan if prompted, and run the experiment.
7. Inspect the generated run folder and `metrics.json`.

If the server command cannot import `auto_ml_agent`, make sure the terminal is in the project root and the intended virtual environment is active.

## Known limitations

- Phase 1 focuses on tabular classification and regression; it is not a universal AutoML platform.
- Synthetic-data results are demonstrations, not evidence of real-world model quality.
- Performance depends on the dataset, available compute, selected metric, and candidate models.
- Serialized models created with different scikit-learn versions may not be compatible. An error involving `sklearn.compose._column_transformer._RemainderColsList` can indicate a serialization/version mismatch.
- Model-based diagnostics may fail even when metrics were saved. The results view should preserve metrics and other available artifacts when this happens.
- Colab GUI execution must be tested in the target environment; a visible backend option alone does not prove remote execution works.
- Production use requires additional checks such as representative data, leakage prevention, reproducible environments, monitoring, and deployment controls.

## Troubleshooting

### Diagnostics error mentioning `_RemainderColsList`
This can happen when a saved scikit-learn model is loaded in an incompatible environment. Check the scikit-learn version used during training against the local version. Prefer a compatible environment or a metrics-only diagnostics fallback; do not assume that retraining or the whole experiment failed based on this warning alone.

### Colab reports that it is not configured
Confirm that the updated backend code is installed, restart Uvicorn, check the backend capabilities endpoint, and verify Google authentication and Colab API configuration. Inspect the response body of a failed experiment request for the specific error.

### No run artifacts appear
Check the server/CLI logs for the run ID and error message, then inspect the configured workspace and `workspace/runs/` directory. The output path and artifact list may vary by configuration.

## Project goals

The goal is to make ML experimentation more accessible and repeatable: explain the problem in plain language, let the agent organize the workflow, keep the user involved at review points, compare candidate models, and preserve useful outputs for inspection.

---

**Disclaimer:** This README describes the current Phase 1 direction and known project workflow. Confirm each feature against the code and configuration in your checkout before treating it as production-ready.
