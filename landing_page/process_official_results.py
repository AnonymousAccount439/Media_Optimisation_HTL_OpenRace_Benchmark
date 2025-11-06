#!/usr/bin/env python3
"""
Process Result_Official data and generate playground_data.json for the landing page.
This script processes all result files and organizes them by:
- Hidden fraction (0.95, 0.99)
- Difficulty (Regular, Hard)
- Race type (Hide_The_Label, Open_Race)
- Batch size (1, 10, 20)
- Dataset
- Optimizer
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import re


def extract_file_info(filename: str) -> Dict[str, str]:
    """Extract metadata from filename."""
    # Example: DBO_rat_myocyte_Hard_Hide_The_Label_Notallopt_Batch10_Hidden_Percentage_0.95_20251015_202559.json
    # Or: T_Cell_Easy_Open_Race_Notallopt_Batch1_Hidden_Percentage_0.95_20251002_225033.json
    
    # Use regex to find batch size
    batch_match = re.search(r'Batch(\d+)', filename)
    batch_size = batch_match.group(1) if batch_match else None
    
    # Use regex to find hidden percentage
    hidden_match = re.search(r'Hidden_Percentage_([\d.]+)', filename)
    hidden_frac = hidden_match.group(1) if hidden_match else None
    
    # Determine dataset name
    if filename.startswith('DBO_rat_myocyte'):
        dataset = 'rat_myocyte'
    elif filename.startswith('MOBO_rat_myocyte'):
        dataset = 'rat_myocyte'
    elif filename.startswith('Hela_regular_mode'):
        dataset = 'Hela_regular'
    elif filename.startswith('Hela_timesaving_mode'):
        dataset = 'Hela_timesaving'
    elif filename.startswith('T_Cell'):
        dataset = 'T_Cell'
    elif filename.startswith('TF_Cell'):
        dataset = 'TF_Cell'
    else:
        dataset = 'unknown'
    
    return {
        'batch_size': batch_size,
        'hidden_frac': hidden_frac,
        'dataset': dataset
    }


def collect_hide_label_stats(data: Dict) -> Dict[str, Dict]:
    """Extract steps_to_target for each optimizer from Hide-the-Label results."""
    steps_by_opt: Dict[str, List[float]] = defaultdict(list)
    
    # Check if this is a compiled analysis file (Regular mode 0.95 batch1 format)
    if data.get("type") == "analysis" and "items" in data:
        # This is a compiled analysis file with pre-calculated stats
        items = data.get("items", [])
        for item in items:
            item_data = item.get("data", {})
            optimizer_stats = item_data.get("optimizer_stats", {})
            for opt_name, stats in optimizer_stats.items():
                # Each item has aggregated stats
                mean_steps = stats.get("mean_steps")
                if mean_steps is not None:
                    steps_by_opt[opt_name].append(float(mean_steps))
    
    # Check if this is the format with "all_tournament_results" at top level
    elif "all_tournament_results" in data:
        tournaments = data["all_tournament_results"]
        for tournament in tournaments:
            competitions = tournament.get("competitions", [])
            for comp in competitions:
                opt_res = comp.get("optimizer_results", {})
                for name, res in opt_res.items():
                    steps = res.get("steps_to_target")
                    if steps is not None:
                        steps_by_opt[name].append(float(steps))
    
    # Check if this is the format with "results" > "all_tournament_results"
    elif "results" in data and "all_tournament_results" in data["results"]:
        tournaments = data["results"]["all_tournament_results"]
        for tournament in tournaments:
            competitions = tournament.get("competitions", [])
            for comp in competitions:
                opt_res = comp.get("optimizer_results", {})
                for name, res in opt_res.items():
                    steps = res.get("steps_to_target")
                    if steps is not None:
                        steps_by_opt[name].append(float(steps))
    
    # Check if this is the format with "results" > "tournament_results" (array)
    elif "results" in data and "tournament_results" in data["results"]:
        tournaments = data["results"]["tournament_results"]
        if isinstance(tournaments, list):
            for tournament in tournaments:
                competitions = tournament.get("competitions", [])
                for comp in competitions:
                    opt_res = comp.get("optimizer_results", {})
                    for name, res in opt_res.items():
                        steps = res.get("steps_to_target")
                        if steps is not None:
                            steps_by_opt[name].append(float(steps))
    
    else:
        # Handle standard tournament format (Hard mode format)
        tournaments = data.get("tournament_results", [])
        if not tournaments:
            tournaments = [data]
        
        if not isinstance(tournaments, list):
            tournaments = [tournaments]
        
        for tournament in tournaments:
            competitions = tournament.get("competitions", [])
            for comp in competitions:
                opt_res = comp.get("optimizer_results", {})
                for name, res in opt_res.items():
                    steps = res.get("steps_to_target")
                    if steps is not None:
                        steps_by_opt[name].append(float(steps))
    
    # Calculate summary statistics
    summary = {}
    for opt, steps in steps_by_opt.items():
        if steps:
            summary[opt] = {
                "mean_steps": float(np.mean(steps)),
                "median_steps": float(np.median(steps)),
                "min_steps": float(np.min(steps)),
                "max_steps": float(np.max(steps)),
                "count": len(steps)
            }
    
    return summary


def collect_open_race_histories(data: Dict) -> Dict[str, Tuple[List[int], List[float]]]:
    """Extract and aggregate best-so-far histories for each optimizer from Open Race results."""
    per_optimizer_histories: Dict[str, List[Tuple[List[int], List[float]]]] = defaultdict(list)
    
    # Check if this is the hiddenfrac99 format with "results" wrapper
    if "results" in data and "competitions" in data["results"]:
        tournament_results = data["results"]
    # Handle tournament_results wrapper or direct format
    elif isinstance(data.get('competitions'), list):
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
                
                # Ensure monotonic (best-so-far should not decrease)
                for i in range(1, len(best_values)):
                    if best_values[i] < best_values[i - 1]:
                        best_values[i] = best_values[i - 1]
                
                per_optimizer_histories[name].append((step_indices, best_values))
    
    # Aggregate all runs for each optimizer
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


def process_result_file(json_file: Path, race_type: str) -> Dict[str, any]:
    """Process a single result file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    if race_type == "Hide_The_Label":
        return collect_hide_label_stats(data)
    else:  # Open_Race
        return collect_open_race_histories(data)


def main():
    # Path to Result_Official directory
    result_official = Path("/Users/aliparsaee/Downloads/Active_Learning_for_Cell_Media-organised_code/Result_Official")
    
    if not result_official.exists():
        print(f"Error: {result_official} does not exist!")
        return
    
    # Structure: playground_data[hidden_frac][difficulty][race_type][batch_size][dataset][optimizer]
    playground_data = {
        0.95: {
            "Regular": {"Hide_The_Label": {}, "Open_Race": {}},
            "Hard": {"Hide_The_Label": {}, "Open_Race": {}}
        },
        0.99: {
            "Regular": {"Hide_The_Label": {}, "Open_Race": {}},
            "Hard": {"Hide_The_Label": {}, "Open_Race": {}}
        }
    }
    
    # Initialize batch sizes
    for hidden_frac in [0.95, 0.99]:
        for difficulty in ["Regular", "Hard"]:
            for race_type in ["Hide_The_Label", "Open_Race"]:
                for batch_size in [1, 10, 20]:
                    playground_data[hidden_frac][difficulty][race_type][batch_size] = {}
    
    # Process all files
    print("Processing Result_Official files...\n")
    
    for hidden_frac_dir in result_official.iterdir():
        if not hidden_frac_dir.is_dir():
            continue
        
        # Extract hidden fraction (95 or 99)
        if "hiddenfrac95" in hidden_frac_dir.name:
            hidden_frac = 0.95
        elif "hiddenfrac99" in hidden_frac_dir.name:
            hidden_frac = 0.99
        else:
            continue
        
        print(f"Processing hidden fraction: {hidden_frac}")
        
        for difficulty_dir in hidden_frac_dir.iterdir():
            if not difficulty_dir.is_dir():
                continue
            
            # Extract difficulty (Regular_Mode or Hard_Mode)
            if "Regular_Mode" in difficulty_dir.name:
                difficulty = "Regular"
            elif "Hard_Mode" in difficulty_dir.name:
                difficulty = "Hard"
            else:
                continue
            
            print(f"  Difficulty: {difficulty}")
            
            for race_dir in difficulty_dir.iterdir():
                if not race_dir.is_dir():
                    continue
                
                # Extract race type
                race_type = race_dir.name  # "Hide_The_Label" or "Open_Race"
                
                if race_type not in ["Hide_The_Label", "Open_Race"]:
                    continue
                
                print(f"    Race type: {race_type}")
                
                # Process all JSON files in this directory
                json_files = list(race_dir.glob("*.json"))
                print(f"      Found {len(json_files)} files")
                
                for json_file in json_files:
                    try:
                        file_info = extract_file_info(json_file.name)
                        batch_size = int(file_info['batch_size'])
                        dataset = file_info['dataset']
                        
                        if batch_size not in [1, 10, 20]:
                            print(f"        Skipping {json_file.name}: invalid batch size {batch_size}")
                            continue
                        
                        print(f"        Processing: {json_file.name}")
                        print(f"          Dataset: {dataset}, Batch: {batch_size}")
                        
                        # Process the file
                        results = process_result_file(json_file, race_type)
                        
                        # Store in the playground_data structure
                        if dataset not in playground_data[hidden_frac][difficulty][race_type][batch_size]:
                            playground_data[hidden_frac][difficulty][race_type][batch_size][dataset] = {}
                        
                        # Merge optimizer results
                        for optimizer, stats in results.items():
                            playground_data[hidden_frac][difficulty][race_type][batch_size][dataset][optimizer] = stats
                        
                    except Exception as e:
                        print(f"        Error processing {json_file.name}: {e}")
                        import traceback
                        traceback.print_exc()
    
    # Save the playground data
    output_file = Path(__file__).parent / "playground_data.json"
    with open(output_file, 'w') as f:
        json.dump(playground_data, f, indent=2)
    
    print(f"\n✓ Generated {output_file}")
    print(f"\nSummary of available data:")
    
    for hidden_frac in [0.95, 0.99]:
        print(f"\nHidden Fraction: {hidden_frac}")
        for difficulty in ["Regular", "Hard"]:
            for race_type in ["Hide_The_Label", "Open_Race"]:
                for batch_size in [1, 10, 20]:
                    datasets = playground_data[hidden_frac][difficulty][race_type][batch_size]
                    if datasets:
                        print(f"  {difficulty} - {race_type} - Batch {batch_size}: {len(datasets)} datasets")
                        # Get unique optimizers across all datasets
                        all_opts = set()
                        for dataset_data in datasets.values():
                            all_opts.update(dataset_data.keys())
                        print(f"    Optimizers: {sorted(all_opts)}")


if __name__ == "__main__":
    main()

