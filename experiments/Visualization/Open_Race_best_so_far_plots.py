from __future__ import annotations
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_open_race(json_path: Path) -> dict:
    # Robust load: handle BOM and stray prefix chars before JSON (e.g., 'pl')
    with open(json_path, 'r', encoding='utf-8-sig') as f:
        text = f.read()
    # Trim any leading junk before the first JSON object/array
    first_brace = text.find('{')
    first_bracket = text.find('[')
    candidates = [i for i in (first_brace, first_bracket) if i != -1]
    start = min(candidates) if candidates else 0
    cleaned = text[start:]
    return json.loads(cleaned)


def extract_per_optimizer_histories(data: dict):
    # Support both shapes:
    # 1) Top-level tournament dict with keys including 'competitions'
    # 2) Wrapper dict with key 'tournament_results' containing the tournament dict
    if isinstance(data, dict) and isinstance(data.get('competitions'), list):
        tournament_results = data
    else:
        tournament_results = data.get('tournament_results', {})

    competitions = tournament_results.get('competitions', [])
    if not isinstance(competitions, list) or not competitions:
        return {}, tournament_results

    optimizer_names = (
        tournament_results.get('optimizer_names')
        or data.get('analysis', {}).get('optimizer_names', [])
    )
    if not optimizer_names and 'optimizer_results' in competitions[0]:
        optimizer_names = list(competitions[0]['optimizer_results'].keys())

    per_optimizer_histories: dict[str, list[tuple[list[int], list[float]]]] = {
        name: [] for name in optimizer_names
    }

    for competition in competitions:
        optimizer_results = competition.get('optimizer_results', {})
        for name, result in optimizer_results.items():
            history = result.get('optimization_history') or []
            step_indices: list[int] = []
            best_values: list[float] = []
            for record in history:
                step = record.get('step')
                best_value = record.get('best_value_so_far')
                if best_value is None:
                    best_value = record.get('current_best')
                if step is None or best_value is None:
                    continue
                step_indices.append(int(step))
                best_values.append(float(best_value))

            if step_indices:
                order = np.argsort(step_indices)
                step_indices = [step_indices[i] for i in order]
                best_values = [best_values[i] for i in order]

                for i in range(1, len(best_values)):
                    if best_values[i] < best_values[i - 1]:
                        best_values[i] = best_values[i - 1]

                per_optimizer_histories.setdefault(name, []).append((step_indices, best_values))

    return per_optimizer_histories, tournament_results


def aggregate_histories(per_optimizer_histories: dict[str, list[tuple[list[int], list[float]]]]):
    aggregated: dict[str, tuple[list[int], list[float]]] = {}

    for optimizer_name, runs in per_optimizer_histories.items():
        if not runs:
            continue

        max_step = max(run_steps[-1] if run_steps else -1 for run_steps, _ in runs)
        step_to_values: dict[int, list[float]] = {s: [] for s in range(max_step + 1)}

        for run_steps, run_values in runs:
            for step, value in zip(run_steps, run_values):
                step_to_values[step].append(value)

            if run_steps:
                last_value = run_values[-1]
                for s in range(run_steps[-1] + 1, max_step + 1):
                    step_to_values[s].append(last_value)

        agg_steps = sorted(step_to_values.keys())
        agg_values = [
            float(np.mean(step_to_values[s])) if step_to_values[s] else math.nan
            for s in agg_steps
        ]

        aggregated[optimizer_name] = (agg_steps, agg_values)

    return aggregated


def plot_best_so_far(aggregated: dict[str, tuple[list[int], list[float]]], title: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6), dpi=150)
    any_plotted = False

    for optimizer_name, (steps, values) in aggregated.items():
        if steps:
            plt.plot(steps, values, label=optimizer_name)
            any_plotted = True

    if not any_plotted:
        print(f'No data to plot. Skipping {out_path}')
        plt.close()
        return None

    plt.title(title)
    plt.xlabel('Step')
    plt.ylabel('Best value so far')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


def main():
    # Edit this variable to point to your JSON file
    INPUT_JSON = "/Users/aliparsaee/Desktop/AmiiResidencyProject/AL_Project/harder_competition_results_open_race/harder_MOBO_dataset_rat_myocyte_random_forest_20250908_165630_results.json"


    json_path = Path(INPUT_JSON).expanduser().resolve()
    if not json_path.exists():
        raise SystemExit(f'Input JSON not found: {json_path}')

    data = load_open_race(json_path)
    per_optimizer_histories, tournament_results = extract_per_optimizer_histories(data)
    aggregated = aggregate_histories(per_optimizer_histories)

    dataset_name = tournament_results.get('dataset_name', json_path.stem)
    title = f'Best titer so far per step — {dataset_name}'

    # Always save to repo-level figures directory
    outdir = Path(__file__).resolve().parents[1] / 'figures'
    outdir.mkdir(parents=True, exist_ok=True)

    outname = f'{json_path.stem}_best_so_far.png'
    out_path = outdir / outname

    result = plot_best_so_far(aggregated, title, out_path)
    if result is not None:
        print(f'Saved {result}')


if __name__ == '__main__':
    main()


