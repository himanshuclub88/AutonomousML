from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd

from .agent import create_plan, review_plan_with_data_quality
from .backends.colab import run_colab_job
from .backends.local import run_local_job
from .engine.dataset import (
    generate_synthetic,
    inspect_csv,
    validate_generated_data,
    prepare_dataset,
    write_data_quality_report,
    analyze_features,
)
from engine.data_set import augment_dataset


ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "workspace" / "runs"

def ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{prompt}{suffix}: ").strip()
    return value or default

def main() -> None:
    print("\n=== Autonomous ML Engineering Agent (Phase 1) ===")
    print("Local-first prototype: tabular classification and regression.\n")
    use_case = ask("Describe the ML use case")
    if not use_case:
        print("A use case is required.")
        return

    print("\nChoose training environment:")
    print("  1. Local Python (available now)")
    print("  2. Google Colab (requires approved API access)")
    mode = ask("Select 1 or 2", "1")
    backend = "colab" if mode == "2" else "local"

    csv_path = ask("\nCSV dataset path (leave blank to generate synthetic data)")
    frame = None
    columns = None
    target = None
    if csv_path:
        try:
            frame, columns = inspect_csv(csv_path)
        except Exception as exc:
            print(f"Could not load CSV: {type(exc).__name__}: {exc}")
            return
        print(f"Loaded {len(frame):,} rows and {len(columns)} columns.")
        print("Columns:", ", ".join(columns))
        target = ask("Target column name (leave blank to let planner suggest one)")
        target = target or None
        if target and target not in columns:
            print(f"Target {target!r} is not present in the CSV.")
            return

        answer = input("Generate additional data? [y/N]: ").strip().lower()

        if answer in {"y", "yes"}:
            additional_rows = int(input("How many additional rows? "))
            frame = augment_dataset(frame, additional_rows)

    print("\nCreating experiment plan...")
    try:
        plan = create_plan(use_case, dataset_columns=columns, requested_target=target)
    except Exception as exc:
        print(f"Could not create plan: {type(exc).__name__}: {exc}")
        return

    print("\n--- Proposed plan ---")
    print(json.dumps(plan.model_dump(), indent=2))
    if input("\nContinue with this plan? [y/N]: ").strip().lower() != "y":
        print("Cancelled; no experiment was started.")
        return

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plan.json").write_text(json.dumps(plan.model_dump(), indent=2), encoding="utf-8")

    if backend == "colab":
        try:
            run_colab_job(plan=plan, dataset_path=csv_path, run_dir=run_dir)
        except Exception as exc:
            print(f"\nColab job was not started: {exc}")
            print("Choose Local Python when you are ready to run experiments now.")
        return

    try:
        if frame is None:
            frame = generate_synthetic(
                problem_type=plan.problem_type,
                target=plan.target_column,
                features=plan.feature_columns,
                rows=plan.synthetic_rows,
                use_case=use_case,
            )

            validate_generated_data(
                frame,
                target=plan.target_column,
                features=plan.feature_columns,
                expected_rows=plan.synthetic_rows,
            )
            frame.to_csv(run_dir / "synthetic_dataset.csv", index=False)
            print(f"Generated demonstration dataset with {len(frame):,} rows.")
        else:
            # Keep only the features and target specified in the reviewed plan.
            selected = plan.feature_columns + [plan.target_column]
            missing = [column for column in selected if column not in frame.columns]
            if missing:
                raise ValueError(f"Planned columns are missing from CSV: {missing}")
            frame = frame[selected].copy()
            frame.to_csv(run_dir / "input_dataset_copy.csv", index=False)

        
        frame, quality_report = prepare_dataset(
            frame=frame,
            target=plan.target_column,
            features=plan.feature_columns,
            problem_type=plan.problem_type,
            remove_duplicates=False,
        )

        feature_report = analyze_features(
            frame=frame,
            target=plan.target_column,
            features=plan.feature_columns,
        )

        quality_report["feature_analysis"] = feature_report

        write_data_quality_report(
            quality_report,
            run_dir / "data_quality_report.json",
        )

        n=1
        import time
        while(n<11):
            s = n*5*"#"+(10-n)*5*" "
            time.sleep(1)
            print(f"|{s}|", end="\r", flush=True)
            n+=1

        print("\n--- Data quality report ---")
        print(json.dumps(quality_report, indent=2))


        # Phase 3: let the LLM review the initial plan.
        revised_plan = review_plan_with_data_quality(
            plan,
            use_case=use_case,
            quality_report=quality_report,
        )

        print("\n--- Revised experiment plan ---")
        print(json.dumps(revised_plan.model_dump(), indent=2))

        if input("\nContinue with revised plan? [y/N]: ").strip().lower() != "y":
            print("Cancelled; no model training was started.")
            return

        # Use the reviewed plan for subsequent training.
        plan = revised_plan

        # Save the final reviewed plan.
        (run_dir / "plan.json").write_text(
            json.dumps(plan.model_dump(), indent=2),
            encoding="utf-8",
        )



        workers = min(4, len(plan.candidate_models))
        print(f"\nTraining {len(plan.candidate_models)} candidate model(s), up to {workers} concurrently...")
        result = run_local_job(
            frame,
            target=plan.target_column,
            features=plan.feature_columns,
            problem_type=plan.problem_type,
            candidates=plan.candidate_models,
            metric=plan.evaluation_metric,
            run_dir=run_dir / "models",
            workers=workers,
        )
        print("\n=== Experiment complete ===")
        print(json.dumps(result, indent=2))
        print(f"\nAll run artifacts: {run_dir}")
        print("Best model: ", result["best_model_path"])
        print("Metrics:    ", run_dir / "models" / "metrics.json")
        print("Training script:", run_dir / "models" / "train_model.py")

    except Exception as exc:
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "error.txt").write_text(
                f"{type(exc).__name__}: {exc}",
                encoding="utf-8",
            )
        except OSError as log_error:
            print(f"Could not save error log: {log_error}")

        print(f"\nExperiment failed: {type(exc).__name__}: {exc}")
        print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
