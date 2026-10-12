from pathlib import Path
import pandas as pd
from ..engine.trainer import train_candidates

def run_local_job(frame: pd.DataFrame, *, target: str, features: list[str],
                  problem_type: str, candidates: list[str], metric: str,
                  run_dir: Path, workers: int = 3) -> dict:
    return train_candidates(
        frame=frame,
        target=target,
        features=features,
        problem_type=problem_type,
        candidates=candidates,
        metric=metric,
        run_dir=run_dir,
        workers=workers,
    )
