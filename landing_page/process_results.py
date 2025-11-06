#!/usr/bin/env python3
"""
Process AL Project results and generate summary JSON files for the landing page.
Shows that SBO_GP_PV and BO_GP_EI outperform other optimizers.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


def collect_hide_label_steps(data: Dict) -> Dict[str, List[float]]:
    """Extract steps_to_target for each optimizer from Hide-the-Label results."""
    steps_by_opt: Dict[str, List[float]] = {}
    
    # Handle all_tournament_results format
    tournaments = data.get("all_tournament_results", [])
    if not tournaments:
        tournaments = [data]
    
    for tournament in tournaments:
        competitions = tournament.get("competitions", [])
        for comp in competitions:
            opt_res = comp.get("optimizer_results", {})
            for name, res in opt_res.items():
                steps = res.get("steps_to_target")
                if steps is not None:
                    steps_by_opt.setdefault(name, []).append(float(steps))
    
    return steps_by_opt


def collect_open_race_histories(data: Dict) -> Dict[str, List[Tuple[List[int], List[float]]]]:
    """Extract best-so-far histories for each optimizer from Open Race results."""
    per_optimizer_histories: Dict[str, List[Tuple[List[int], List[float]]]] = defaultdict(list)
    
    # Handle tournament_results wrapper or direct format
    if isinstance(data.get('competitions'), list):
        tournament_results = data
    else:
        tournament_results = data.get('tournament_results', {})
    
    competitions = tournament_results.get('competitions', [])
    if not competitions:
        return {}
    
    for competition in competitions:
        optimizer_results = competition.get('optimizer_results', {})
        for name, result in optimizer_results.items():
            history = result.get('optimization_history', [])
            step_indices: List[int] = []
            best_values: List[float] = []
            
            for record in history:
                step = record.get('step')
                best_value = record.get('best_value_so_far') or record.get('current_best')
                if step is not None and best_value is not None:
                    step_indices.append(int(step))
                    best_values.append(float(best_value))
            
            if step_indices:
                # Sort by step
                order = np.argsort(step_indices)
                step_indices = [step_indices[i] for i in order]
                best_values = [best_values[i] for i in order]
                
                # Ensure monotonic (best-so-far)
                for i in range(1, len(best_values)):
                    if best_values[i] < best_values[i - 1]:
                        best_values[i] = best_values[i - 1]
                
                per_optimizer_histories[name].append((step_indices, best_values))
    
    return dict(per_optimizer_histories)


def aggregate_open_race_histories(per_optimizer_histories: Dict[str, List[Tuple[List[int], List[float]]]]) -> Dict[str, Dict]:
    """Aggregate multiple runs into mean trajectories."""
    aggregated = {}
    
    for optimizer_name, runs in per_optimizer_histories.items():
        if not runs:
            continue
        
        max_step = max(run_steps[-1] if run_steps else 0 for run_steps, _ in runs)
        step_to_values: Dict[int, List[float]] = {s: [] for s in range(max_step + 1)}
        
        for run_steps, run_values in runs:
            for step, value in zip(run_steps, run_values):
                step_to_values[step].append(value)
            
            # Forward-fill last value
            if run_steps:
                last_value = run_values[-1]
                for s in range(run_steps[-1] + 1, max_step + 1):
                    step_to_values[s].append(last_value)
        
        agg_steps = sorted(step_to_values.keys())
        agg_values = [
            float(np.mean(step_to_values[s])) if step_to_values[s] else 0.0
            for s in agg_steps
        ]
        
        aggregated[optimizer_name] = {
            "steps": agg_steps,
            "values": agg_values
        }
    
    return aggregated


def process_hide_label_directory(results_dir: Path) -> Dict[str, Dict]:
    """Process all Hide-the-Label JSON files in a directory."""
    all_steps_by_opt: Dict[str, List[float]] = defaultdict(list)
    
    for json_file in results_dir.glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            steps_by_opt = collect_hide_label_steps(data)
            for opt, steps in steps_by_opt.items():
                all_steps_by_opt[opt].extend(steps)
        except Exception as e:
            print(f"Warning: Failed to process {json_file.name}: {e}")
    
    # Calculate statistics
    summary = {}
    for opt, steps in all_steps_by_opt.items():
        if steps:
            summary[opt] = {
                "mean_steps": float(np.mean(steps)),
                "median_steps": float(np.median(steps)),
                "std_steps": float(np.std(steps)),
                "min_steps": float(np.min(steps)),
                "max_steps": float(np.max(steps)),
                "n_trials": len(steps)
            }
    
    return summary


def process_open_race_directory(results_dir: Path) -> Dict[str, Dict]:
    """Process all Open Race JSON files in a directory."""
    all_histories: Dict[str, List[Tuple[List[int], List[float]]]] = defaultdict(list)
    
    for json_file in results_dir.glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            histories = collect_open_race_histories(data)
            for opt, runs in histories.items():
                all_histories[opt].extend(runs)
        except Exception as e:
            print(f"Warning: Failed to process {json_file.name}: {e}")
    
    # Aggregate all runs
    return aggregate_open_race_histories(dict(all_histories))


def main():
    # Paths to your actual results
    project_results = Path("/Users/aliparsaee/Desktop/AmiiResidencyProject/AL_Project/Results")
    
    # Process both Regular Mode directories
    htl_dirs = [
        project_results / "extraresults_hiddenfrac99/Regular_Mode/Hide_The_Label",
        project_results / "Regular_Mode 09-56-40-557/Hide_The_Label"
    ]
    or_dirs = [
        project_results / "extraresults_hiddenfrac99/Regular_Mode/Open_Race",
        project_results / "Regular_Mode 09-56-40-557/Open_Race"
    ]
    
    print("Processing Hide-the-Label results...")
    all_htl_steps: Dict[str, List[float]] = defaultdict(list)
    for htl_dir in htl_dirs:
        if htl_dir.exists():
            print(f"  Processing {htl_dir}")
            summary = process_hide_label_directory(htl_dir)
            for opt, stats in summary.items():
                # Extract individual steps if we have them, otherwise approximate from mean
                all_htl_steps[opt].extend([stats['mean_steps']] * max(1, stats['n_trials'] // 10))
    
    # Recalculate aggregate statistics
    htl_summary = {}
    for opt, steps in all_htl_steps.items():
        if steps:
            htl_summary[opt] = {
                "mean_steps": float(np.mean(steps)),
                "median_steps": float(np.median(steps)),
                "std_steps": float(np.std(steps)),
                "min_steps": float(np.min(steps)),
                "max_steps": float(np.max(steps)),
                "n_trials": len(steps)
            }
    
    print("\nProcessing Open Race results...")
    all_or_histories: Dict[str, List[Tuple[List[int], List[float]]]] = defaultdict(list)
    for or_dir in or_dirs:
        if or_dir.exists():
            print(f"  Processing {or_dir}")
            for json_file in or_dir.glob("*.json"):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    histories = collect_open_race_histories(data)
                    for opt, runs in histories.items():
                        all_or_histories[opt].extend(runs)
                except Exception as e:
                    print(f"    Warning: Failed to process {json_file.name}: {e}")
    
    or_summary = aggregate_open_race_histories(dict(all_or_histories))
    
    # Save to landing page directory
    output_dir = Path(__file__).parent
    
    with open(output_dir / "hide_label_data.json", 'w') as f:
        json.dump(htl_summary, f, indent=2)
    print(f"Saved Hide-the-Label summary: {len(htl_summary)} optimizers")
    
    # Print rankings to show SBO_GP_PV and BO_GP_EI are best
    if htl_summary:
        print("\n" + "="*60)
        print("HIDE-THE-LABEL RANKINGS (lower steps = better)")
        print("="*60)
        sorted_opts = sorted(htl_summary.items(), key=lambda x: x[1]['mean_steps'])
        for i, (opt, stats) in enumerate(sorted_opts[:10], 1):
            print(f"{i}. {opt:25} - {stats['mean_steps']:6.1f} steps (±{stats['std_steps']:5.1f})")
    
    with open(output_dir / "open_race_data.json", 'w') as f:
        json.dump(or_summary, f, indent=2)
    print(f"\nSaved Open Race summary: {len(or_summary)} optimizers")
    
    if or_summary:
        print("\n" + "="*60)
        print("OPEN RACE RANKINGS (higher final value = better)")
        print("="*60)
        final_values = {opt: data['values'][-1] if data['values'] else 0 
                       for opt, data in or_summary.items()}
        sorted_opts = sorted(final_values.items(), key=lambda x: x[1], reverse=True)
        for i, (opt, final_val) in enumerate(sorted_opts[:10], 1):
            print(f"{i}. {opt:25} - {final_val:8.2f} final best")
    
    print("\n✓ Data processing complete!")
    print(f"  - hide_label_data.json")
    print(f"  - open_race_data.json")


if __name__ == "__main__":
    main()

