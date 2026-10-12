# Autonomous ML Engineering Agent — Phase 1

A local-first Python application that accepts a natural-language ML use case and an optional CSV, creates a structured experiment plan using your LLM wrapper, trains several candidate scikit-learn models, compares them, and saves artifacts.

## Current scope

- Interactive terminal application (`python -m auto_ml_agent.main`)
- Per-run choice: local Python or Google Colab
- Local mode: working for tabular regression and classification
- Optional CSV input and target column
- Synthetic dataset generation when no CSV is supplied
- Multiple candidate models trained concurrently
- Metrics, model artifacts, and a generated reproducible training script
- Colab mode: explicitly reports that API access is required; it does not pretend to submit a job

This first version intentionally does not use FastAPI or a frontend.