"""
Mass experiments runner leveraging main.py wrappers.

Runs 4 experiment settings (regular/hard x hide/open) over:
- batch sizes B in [1, 10, 20]
- datasets in [
    'MOBO_dataset_rat_myocyte',
    'DBO_dataset_rat_myocyte',
    'df_Human_Hela_regular_mode',
    'df_Human_Hela_timesaving_mode',
    'df_Human_T_Cell_Expanded',
    'df_Human_TF_Cell_Expanded',
]

Special requirements:
- For hide-the-label: hidden_fraction=[0.99]
- For open-race: set R=5 (instead of 10), keep other defaults as in code
- Save results and plots including the term "99pcovered" in filenames
- Write checkpoints to /Users/aliparsaee/Desktop/AmiiResidencyProject/AL_Project/Checkpoints
  and resume from them to avoid re-running completed jobs.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from main import (
    run_regular_hide_the_label,
    run_hard_hide_the_label,
    run_regular_open_race,
    run_hard_open_race,
)


# Configuration
ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = Path("/Users/aliparsaee/Desktop/AmiiResidencyProject/AL_Project/Checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS: List[str] = [
    'MOBO_dataset_rat_myocyte',
    'DBO_dataset_rat_myocyte',
    'df_Human_Hela_regular_mode',
    'df_Human_Hela_timesaving_mode',
    'df_Human_T_Cell_Expanded',
    'df_Human_TF_Cell_Expanded',
]

BATCH_SIZES: List[int] = [1, 10, 20]

# Hide-the-label config
HIDDEN_FRACTIONS = 0.95

# MODES_LIST = ["regular-hide", "hard-hide", "regular-open", "hard-open"]
MODES_LIST = ["hard-open"]

NUM_OF_SYNTHETIC_DATA = 3
NUM_OF_COMPETITIONS = 3


@dataclass(frozen=True)
class Task:
    mode: str  # 'regular-hide' | 'hard-hide' | 'regular-open' | 'hard-open'
    dataset: str
    B: int

    def key(self) -> str:
        return f"{self.mode}__{self.dataset}__B{self.B}"


def _checkpoint_path(task: Task) -> Path:
    return CHECKPOINT_DIR / f"{task.key()}__99pcovered.json"


def _is_completed(task: Task) -> bool:
    return _checkpoint_path(task).exists()


def _write_checkpoint(task: Task, payload: Dict) -> None:
    path = _checkpoint_path(task)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp, path)


def _result_base_name(task: Task) -> str:
    mode_tag = task.mode.replace('-', '_')
    return f"{task.dataset}_{mode_tag}_B{task.B}_95pcovered"


def _run_regular_hide(task: Task) -> Tuple[Dict, Dict]:
    # Use hidden_fraction=[0.99]; leave other defaults (but override batch_size with B)
    # Explicitly set optimizer_names=None to ensure all optimizers are used
    combined_results, all_analyses = run_regular_hide_the_label(
        dataset_name=task.dataset,
        optimizer_names=None,  # Use all optimizers
        hidden_fraction=HIDDEN_FRACTIONS,
        batch_size=task.B,
        save_results=True,
        num_of_synthetic_data=NUM_OF_SYNTHETIC_DATA,
        n_competitions=NUM_OF_COMPETITIONS
        # other defaults from the function are preserved
    )
    return combined_results, all_analyses


def _run_hard_hide(task: Task) -> Tuple[Dict, Dict]:
    # Explicitly set optimizer_names=None to ensure all optimizers are used
    results, analysis = run_hard_hide_the_label(
        dataset_name=task.dataset,
        optimizer_names=None,  # Use all optimizers
        hidden_fraction=HIDDEN_FRACTIONS,
        batch_size=task.B,
        save_results=True,
        num_of_synthetic_data=NUM_OF_SYNTHETIC_DATA,
        n_competitions=NUM_OF_COMPETITIONS
    )
    return results, analysis


def _run_regular_open(task: Task) -> Tuple[Dict, Dict]:
    # Override B and set R=5; keep defaults for others (S=200 is now default)
    # Explicitly set optimizer_names=None to ensure all optimizers are used
    tournament_results, analysis = run_regular_open_race(
        dataset_name=task.dataset,
        optimizer_names=None,  # Use all optimizers
        B=task.B,
        R=5,
        save_results=True,
        n_competitions=NUM_OF_COMPETITIONS
    )
    return tournament_results, analysis


def _run_hard_open(task: Task) -> Tuple[Dict, Dict]:
    # Override B and set R=5; keep defaults for others (S=200 is now default)
    # Explicitly set optimizer_names=None to ensure all optimizers are used
    tournament_results, analysis = run_hard_open_race(
        dataset_name=task.dataset,
        optimizer_names=None,  # Use all optimizers
        B=task.B,
        R=5,
        save_results=True,
        n_competitions=NUM_OF_COMPETITIONS
    )
    return tournament_results, analysis


def _save_with_suffix(base_dir: Path, base_name: str, suffix: str, obj: Dict) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    out_path = base_dir / f"{base_name}_{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    return out_path


def _maybe_save_additional_artifacts(task: Task, results: Dict, analysis: Dict) -> None:
    """Save additional copies with explicit 99pcovered suffixes into Results tree.

    The underlying experiment runners already save results/plots. This function just
    writes an extra copy with a consistent name containing "99pcovered" to make them
    easy to find across all runs.
    """
    base_name = _result_base_name(task)
    if task.mode in ("regular-open", "hard-open"):
        subdir = ROOT / "Results" / ("Regular_Mode" if task.mode == "regular-open" else "Hard_Mode") / "Open_Race"
    else:
        # place hide-the-label extras next to compiled results area
        subdir = ROOT / "Results" / ("Regular_Mode" if task.mode == "regular-hide" else "Hard_Mode") / "Hide_The_Label"
    payload = {"results": results, "analysis": analysis}
    _save_with_suffix(subdir, base_name, "results", payload)


def iter_tasks() -> Iterable[Task]:
    for mode in MODES_LIST:
        for B in BATCH_SIZES:
            for dataset in DATASETS:
                yield Task(mode=mode, dataset=dataset, B=B)


def run_all() -> None:
    for task in iter_tasks():
        if _is_completed(task):
            print(f"[SKIP] {task.key()} already completed (checkpoint exists)")
            continue

        print(f"[RUN ] {task.key()}")
        try:
            if task.mode == "regular-hide":
                results, analysis = _run_regular_hide(task)
            elif task.mode == "hard-hide":
                results, analysis = _run_hard_hide(task)
            elif task.mode == "regular-open":
                results, analysis = _run_regular_open(task)
            elif task.mode == "hard-open":
                results, analysis = _run_hard_open(task)
            else:
                raise ValueError(f"Unknown mode: {task.mode}")

            # Save extra artifacts with 99pcovered in the name
            _maybe_save_additional_artifacts(task, results, analysis)

            # Write checkpoint with summary info
            ckpt_payload = {
                "task": asdict(task),
                "status": "completed",
                "summary": {
                    "optimizer_names": results.get("optimizer_names") or analysis.get("optimizer_names"),
                    "n_competitions": analysis.get("n_competitions") if isinstance(analysis, dict) else None,
                },
            }
            _write_checkpoint(task, ckpt_payload)
            print(f"[DONE] {task.key()} — checkpoint written")

        except Exception as e:
            ckpt_payload = {
                "task": asdict(task),
                "status": "failed",
                "error": str(e),
            }
            _write_checkpoint(task, ckpt_payload)
            print(f"[FAIL] {task.key()} — {e}")


if __name__ == "__main__":
    run_all()


