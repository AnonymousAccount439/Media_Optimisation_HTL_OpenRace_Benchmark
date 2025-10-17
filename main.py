"""
Convenience entrypoints to run experiments and visualizations from the experiments/ folder.

This module provides:
- run_regular_hide_the_label
- run_hard_hide_the_label
- run_regular_open_race
- run_hard_open_race
- visualize_hide_the_label_mean_steps
- visualize_open_race_best_so_far
- run_experiment: a general dispatcher for the four experiment types

Optional CLI usage examples:
  python main.py regular-hide MOBO_dataset_rat_myocyte --n_competitions 5 --hidden_fraction 0.9
  python main.py hard-open df_Human_TF_Cell_Expanded --surrogate_type random_forest --S 50 --R 5 --B 1
  python main.py viz-hide-bar /path/to/results.json --out figures/hide_label_mean_steps.png --title "My Plot"
  python main.py viz-open-best /path/to/results.json --out figures/open_best.png --title "Open Race"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Experiment entrypoints
from experiments.Regular_Mode.Regular_Mode_Hide_The_Label import (
    run_incremental_gp_competition,
)
from experiments.Hard_Mode.Harder_Mode_Hide_The_Label import (
    run_harder_competition,
)
from experiments.Regular_Mode.Regular_Mode_Open_Race import (
    run_open_race_competition,
)
from experiments.Hard_Mode.Harder_Mode_Open_Race import (
    run_harder_open_race_competition,
)

# Visualization helpers
from experiments.Visualization.Hide_The_Label_bar_plots import (
    extract_from_optimizer_stats,
    extract_from_competitions,
    plot_bar,
)
from experiments.Visualization.Open_Race_best_so_far_plots import (
    load_open_race,
    extract_per_optimizer_histories,
    aggregate_histories,
    plot_best_so_far,
)


def run_regular_hide_the_label(
    dataset_name: str,
    optimizer_names: Optional[List[str]] = None,
    n_competitions: int = 10,
    hidden_fraction: Union[float, List[float]] = 0.9,
    pool_size: int = 200,
    batch_size: int = 1,
    n_jobs: int = -1,
    random_state: int = 42,
    save_results: bool = True,
    num_of_synthetic_data: int = 10,
    use_gpu: bool = True,
    cache_dir: str = "model_cache",
    use_cache_model: bool = True,
    show_pca: bool = False,
    compiled_output_dir: str = "results_currentlyfor_hide_and_open",
    compiled_filename: Optional[str] = None,
    save_individual_results: bool = False,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], Any]:
    """Run Regular Mode Hide-the-Label experiment and return results.

    Returns a tuple of (combined_results, all_analyses) as defined by the
    underlying run_incremental_gp_competition implementation.
    """
    return run_incremental_gp_competition(
        dataset_name=dataset_name,
        optimizer_names=optimizer_names,
        n_competitions=n_competitions,
        hidden_fraction=hidden_fraction,
        pool_size=pool_size,
        batch_size=batch_size,
        n_jobs=n_jobs,
        random_state=random_state,
        save_results=save_results,
        num_of_synthetic_data=num_of_synthetic_data,
        use_gpu=use_gpu,
        cache_dir=cache_dir,
        use_cache_model=use_cache_model,
        show_pca=show_pca,
        compiled_output_dir=compiled_output_dir,
        compiled_filename=compiled_filename,
        save_individual_results=save_individual_results,
        verbose=verbose,
    )


def run_hard_hide_the_label(
    dataset_name: str,
    surrogate_type: str = "random_forest",
    difficulty_config: Optional[Dict[str, Any]] = None,
    optimizer_names: Optional[List[str]] = None,
    n_competitions: int = 10,
    hidden_fraction: Union[float, List[float]] = 0.9,
    pool_size: int = 200,
    batch_size: int = 1,
    n_jobs: int = -1,
    random_state: int = 42,
    save_results: bool = True,
    num_of_synthetic_data: int = 10,
    use_gpu: bool = True,
    use_cache_model: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run Hard Mode Hide-the-Label experiment and return results and analysis."""
    return run_harder_competition(
        dataset_name=dataset_name,
        surrogate_type=surrogate_type,
        difficulty_config=difficulty_config,
        optimizer_names=optimizer_names,
        n_competitions=n_competitions,
        hidden_fraction=hidden_fraction,
        pool_size=pool_size,
        batch_size=batch_size,
        n_jobs=n_jobs,
        random_state=random_state,
        save_results=save_results,
        num_of_synthetic_data=num_of_synthetic_data,
        use_gpu=use_gpu,
        use_cache_model=use_cache_model,
    )


def run_regular_open_race(
    dataset_name: str,
    optimizer_names: Optional[List[str]] = None,
    n_competitions: int = 10,
    S: int = 200,
    R: int = 5,
    B: int = 1,
    noise_std: float = 0.0,
    n_jobs: int = -1,
    random_state: int = 42,
    save_results: bool = True,
    use_cache_model: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run Regular Mode Open Race experiment and return results and analysis."""
    return run_open_race_competition(
        dataset_name=dataset_name,
        optimizer_names=optimizer_names,
        n_competitions=n_competitions,
        S=S,
        R=R,
        B=B,
        noise_std=noise_std,
        n_jobs=n_jobs,
        random_state=random_state,
        save_results=save_results,
        use_cache_model=use_cache_model,
    )


def run_hard_open_race(
    dataset_name: str,
    surrogate_type: str = "random_forest",
    difficulty_config: Optional[Dict[str, Any]] = None,
    optimizer_names: Optional[List[str]] = None,
    n_competitions: int = 10,
    S: int = 200,
    R: int = 5,
    B: int = 1,
    n_jobs: int = -1,
    random_state: int = 42,
    save_results: bool = True,
    use_cache_model: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run Hard Mode Open Race experiment and return results and analysis."""
    return run_harder_open_race_competition(
        dataset_name=dataset_name,
        surrogate_type=surrogate_type,
        difficulty_config=difficulty_config,
        optimizer_names=optimizer_names,
        n_competitions=n_competitions,
        S=S,
        R=R,
        B=B,
        n_jobs=n_jobs,
        random_state=random_state,
        save_results=save_results,
        use_cache_model=use_cache_model,
    )


def visualize_hide_the_label_mean_steps(
    results_json: Union[str, Path],
    out_path: Union[str, Path],
    title: str = "Hide-the-Label: Mean Steps per Optimizer",
) -> str:
    """Generate a bar plot of mean steps per optimizer from a results JSON file.

    Returns the output path of the saved figure.
    """
    json_path = Path(results_json).expanduser().resolve()
    with open(json_path, "r") as f:
        data = json.load(f)

    optimizers, means, ses = extract_from_optimizer_stats(data)
    if not optimizers:
        optimizers, means, ses = extract_from_competitions(data)

    # Fallback: compiled wrapper { files: [ { data: {...} }, ... ] }
    if not optimizers and isinstance(data.get("files"), list):
        merged: Dict[str, List[float]] = {}
        for item in data["files"]:
            if not isinstance(item, dict):
                continue
            d = item.get("data")
            if not isinstance(d, dict):
                continue
            local = _collect_steps_by_optimizer(d)
            for opt, steps in local.items():
                merged.setdefault(opt, []).extend(steps)
        if merged:
            optimizers = list(merged.keys())
            means = [float(np.mean(merged[o])) for o in optimizers]
            stds = [float(np.std(merged[o], ddof=1)) if len(merged[o]) > 1 else 0.0 for o in optimizers]
            ns = [len(merged[o]) for o in optimizers]
            ses = [s / (n ** 0.5) if n > 0 else 0.0 for s, n in zip(stds, ns)]

    if not optimizers:
        raise ValueError("Could not find optimizer step data in the provided JSON")

    out_path = str(Path(out_path).expanduser())
    plot_bar(optimizers, means, ses, title, out_path)
    return out_path


def _collect_steps_by_optimizer(data: Dict[str, Any]) -> Dict[str, List[float]]:
    """Helper for visualization fallback aggregation."""
    result: Dict[str, List[float]] = {}
    comps = data.get("competitions")
    if not isinstance(comps, list):
        return result
    for comp in comps:
        hist = comp.get("history") or {}
        for opt, steps in hist.items():
            if isinstance(steps, list):
                result.setdefault(opt, []).extend(steps)
    return result


def visualize_open_race_best_so_far(
    results_json: Union[str, Path],
    out_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
) -> str:
    """Generate a best-so-far line plot for Open Race results JSON.

    Returns the output path of the saved figure.
    """
    json_path = Path(results_json).expanduser().resolve()
    data = load_open_race(json_path)
    per_optimizer_histories, tournament_results = extract_per_optimizer_histories(data)
    aggregated = aggregate_histories(per_optimizer_histories)

    if title is None:
        dataset_name = tournament_results.get("dataset_name", json_path.stem)
        title = f"Best titer so far per step — {dataset_name}"

    if out_path is None:
        out_dir = Path.cwd() / "figures"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{json_path.stem}_best_so_far.png"

    out_path = str(Path(out_path).expanduser())
    plot_best_so_far(aggregated, title, out_path)
    return out_path


def run_experiment(kind: str, **kwargs: Any) -> Tuple[Dict[str, Any], Any]:
    """General dispatcher for experiments 1-4.

    kind may be a descriptive string; simple heuristics route to the right function:
    - contains "regular" and "hide" -> run_regular_hide_the_label
    - contains "hard" and "hide" -> run_hard_hide_the_label
    - contains "regular" and "open" -> run_regular_open_race
    - contains "hard" and "open" -> run_hard_open_race
    """
    k = kind.strip().lower()
    if ("regular" in k) and ("hide" in k):
        return run_regular_hide_the_label(**kwargs)
    if ("hard" in k) and ("hide" in k):
        return run_hard_hide_the_label(**kwargs)
    if ("regular" in k) and ("open" in k):
        return run_regular_open_race(**kwargs)
    if ("hard" in k) and ("open" in k):
        return run_hard_open_race(**kwargs)
    raise ValueError(
        "Unknown kind. Expected one of: 'regular hide', 'hard hide', 'regular open', 'hard open'"
    )


def _parse_optimizer_names(arg: Optional[str]) -> Optional[List[str]]:
    if isinstance(arg, list):
        return arg
    if not arg:
        return None
    parts = [p.strip() for p in arg.split(",")]
    return [p for p in parts if p]


def _parse_hidden_fraction(arg: Optional[str]) -> Union[float, List[float]]:
    if isinstance(arg, float):
        return arg
    if "," in arg:
        return [float(x.strip()) for x in arg.split(",") if x.strip()]
    return float(arg)


def _build_cli() -> argparse.ArgumentParser:
    firsthalf = ['RANDOM','SBO_GP_PV', 'BO_GP_EI', 'SMART_BO', 'SBO_ANN_PV']
    secondhalf = ['DE_DIRECT', 'FULL_FACTORIAL', 'FRACTIONAL_FACTORIAL', 'PLACKETT_BURMAN', 'CENTRAL_COMPOSITE', 'BOX_BEHNKEN', 'LATIN_HYPERCUBE'] # removed 'D_OPTIMAL'
    all_optimisers = firsthalf + secondhalf

    parser = argparse.ArgumentParser(description="Run AL experiments and visualizations")
    subs = parser.add_subparsers(dest="cmd", required=True)

    # regular-hide
    p = subs.add_parser("regular-hide", help="Regular Mode: Hide-the-Label")
    p.add_argument("dataset")
    p.add_argument("--optimizers", default=all_optimisers, help="Comma-separated optimizer names or list of comma-seperated names")
    p.add_argument("--n_competitions", type=int, default=10)
    p.add_argument("--hidden_fraction", default=0.95, help="Float or CSV of floats")
    p.add_argument("--pool_size", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--n_jobs", type=int, default=-1)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--num_of_synthetic_data", type=int, default=10)
    p.add_argument("--use_gpu", action="store_true")
    p.add_argument("--no_save", action="store_true", help="Do not save results to disk")


    # hard-hide
    p = subs.add_parser("hard-hide", help="Hard Mode: Hide-the-Label")
    p.add_argument("dataset")
    p.add_argument("--surrogate_type", default="random_forest")
    p.add_argument("--difficulty_config", default={
        'noise_level': 0.2,
        'heteroscedastic': False,
        'n_modes': 1,
        'n_discontinuities': 0
    }, help="Difficulty configuration")
    p.add_argument("--optimizers", default=all_optimisers, help="Comma-separated optimizer names or list of comma-seperated names")
    p.add_argument("--n_competitions", type=int, default=10)
    p.add_argument("--hidden_fraction", default=0.95, help="Float or CSV of floats")
    p.add_argument("--pool_size", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--n_jobs", type=int, default=-1)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--no_save", action="store_true")
    p.add_argument("--num_of_synthetic_data", type=int, default=10)
    p.add_argument("--use_gpu", action="store_true")



    # regular-open
    p = subs.add_parser("regular-open", help="Regular Mode: Open Race")
    p.add_argument("dataset")
    p.add_argument("--optimizers", default=all_optimisers, help="Comma-separated optimizer names or list of comma-seperated names")
    p.add_argument("--n_competitions", type=int, default=10)
    p.add_argument("--S", type=int, default=200)
    p.add_argument("--R", type=int, default=10)
    p.add_argument("--B", type=int, default=1)
    p.add_argument("--noise_std", type=float, default=0.0)
    p.add_argument("--n_jobs", type=int, default=-1)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--no_save", action="store_true")



    # hard-open
    p = subs.add_parser("hard-open", help="Hard Mode: Open Race")
    p.add_argument("dataset")
    p.add_argument("--surrogate_type", default="random_forest")
    p.add_argument("--difficulty_config", default={
            'noise_level': 0.2,
            'heteroscedastic': True,
            'n_modes': 2,
            'n_discontinuities': 1
            }, help="Difficulty configuration")
    p.add_argument("--optimizers", default=all_optimisers, help="Comma-separated optimizer names or list of comma-seperated names")
    p.add_argument("--n_competitions", type=int, default=10)
    p.add_argument("--S", type=int, default=200)
    p.add_argument("--R", type=int, default=10)
    p.add_argument("--B", type=int, default=1)
    p.add_argument("--n_jobs", type=int, default=-1)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--no_save", action="store_true")

    # viz-hide-bar
    p = subs.add_parser("viz-hide-bar", help="Visualization: Hide-the-Label mean steps bar plot")
    p.add_argument("results_json")
    p.add_argument("--out", required=True)
    p.add_argument("--title", default="Hide-the-Label: Mean Steps per Optimizer")

    # viz-open-best
    p = subs.add_parser("viz-open-best", help="Visualization: Open Race best-so-far plot")
    p.add_argument("results_json")
    p.add_argument("--out", default=None)
    p.add_argument("--title", default=None)

    return parser


def _cli_main(argv: Optional[List[str]] = None) -> None:
    parser = _build_cli()
    args = parser.parse_args(argv)

    if args.cmd == "regular-hide":
        _, _ = run_regular_hide_the_label(
            dataset_name=args.dataset,
            optimizer_names=_parse_optimizer_names(args.optimizers),
            n_competitions=args.n_competitions,
            hidden_fraction=_parse_hidden_fraction(args.hidden_fraction),
            pool_size=args.pool_size,
            batch_size=args.batch_size,
            n_jobs=args.n_jobs,
            random_state=args.random_state,
            num_of_synthetic_data=args.num_of_synthetic_data,
            use_gpu=args.use_gpu,
            cache_dir="model_cache",
            use_cache_model=True,
            show_pca=False,
            verbose=False,
            save_results=not args.no_save,
        )
        return


    if args.cmd == "hard-hide":
        _, _ = run_hard_hide_the_label(
            dataset_name=args.dataset,
            surrogate_type=args.surrogate_type,
            difficulty_config=args.difficulty_config,
            optimizer_names=_parse_optimizer_names(args.optimizers),
            n_competitions=args.n_competitions,
            hidden_fraction=_parse_hidden_fraction(args.hidden_fraction),
            pool_size=args.pool_size,
            batch_size=args.batch_size,
            n_jobs=args.n_jobs,
            random_state=args.random_state,
            save_results=not args.no_save,
            num_of_synthetic_data=args.num_of_synthetic_data,
            use_gpu=args.use_gpu,
            use_cache_model=True
        )
        return


    if args.cmd == "regular-open":
        _, _ = run_regular_open_race(
            dataset_name=args.dataset,
            optimizer_names=_parse_optimizer_names(args.optimizers),
            n_competitions=args.n_competitions,
            S=args.S,
            R=args.R,
            B=args.B,
            noise_std=args.noise_std,
            n_jobs=args.n_jobs,
            random_state=args.random_state,
            save_results=not args.no_save,
            use_cache_model=True
        )
        return

    if args.cmd == "hard-open":
        _, _ = run_hard_open_race(
            dataset_name=args.dataset,
            surrogate_type=args.surrogate_type,
            difficulty_config=args.difficulty_config,
            optimizer_names=_parse_optimizer_names(args.optimizers),
            n_competitions=args.n_competitions,
            S=args.S,
            R=args.R,
            B=args.B,
            n_jobs=args.n_jobs,
            random_state=args.random_state,
            save_results=not args.no_save,
            use_cache_model=True
        )
        return

    if args.cmd == "viz-hide-bar":
        out = visualize_hide_the_label_mean_steps(
            results_json=args.results_json,
            out_path=args.out,
            title=args.title,
        )
        print(f"Saved plot to: {out}")
        return

    if args.cmd == "viz-open-best":
        out = visualize_open_race_best_so_far(
            results_json=args.results_json,
            out_path=args.out,
            title=args.title,
        )
        print(f"Saved plot to: {out}")
        return

    raise SystemExit("Unknown command")


if __name__ == "__main__":
    import time
    start_time = time.time()
    _cli_main()
    total_time = time.time() - start_time
    print(f"Execution time: {total_time:.2f} seconds")


    # # possible datasets: ['MOBO_dataset_rat_myocyte', 'DBO_dataset_rat_myocyte', 'df_Human_Hela_regular_mode', 'df_Human_Hela_timesaving_mode', 'df_Human_T_Cell_Expanded', 'df_Human_TF_Cell_Expanded', 'synthetic_2d', 'synthetic_5d', 'synthetic_10d']

    # firsthalf = ['RANDOM','SBO_GP_PV', 'BO_GP_EI', 'SMART_BO', 'SBO_ANN_PV']
    # secondhalf = ['DE_DIRECT', 'FULL_FACTORIAL', 'FRACTIONAL_FACTORIAL', 'PLACKETT_BURMAN', 'CENTRAL_COMPOSITE', 'BOX_BEHNKEN', 'LATIN_HYPERCUBE'] # removed 'D_OPTIMAL'
    # all_optimisers = firsthalf + secondhalf

# available_modes = ["regular-hide", "hard-hide", "regular-open", "hard-open"]
