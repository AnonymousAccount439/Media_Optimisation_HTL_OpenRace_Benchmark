# Active Learning for Cell Media

This repository evaluates and compares active learning and design-of-experiments strategies for optimizing cell media formulations. It implements two core benchmarking games and their “regular” and “hard” variants, plus utilities to generate plots and run large experiment grids.

- Hide-the-Label (HTL): Active learning on discrete candidate pools with partially hidden labels
- Open Race (OR): Black-box optimization in continuous space, where strategies propose new coordinates
- Regular Mode: Baseline difficulty
- Hard Mode: Increased difficulty via noise, discontinuities, multi-modality, etc.


## Repository Structure

- `experiments/Regular_Mode/Regular_Mode_Hide_The_Label.py`: Regular HTL implementation and entrypoints
- `experiments/Regular_Mode/Regular_Mode_Open_Race.py`: Regular Open Race implementation and entrypoints
- `experiments/Hard_Mode/Harder_Mode_Hide_The_Label.py`: Hard HTL implementation and entrypoints
- `experiments/Hard_Mode/Harder_Mode_Open_Race.py`: Hard Open Race implementation and entrypoints
- `experiments/Visualization/Hide_The_Label_bar_plots.py`: Mean-steps bar plot for HTL
- `experiments/Visualization/Open_Race_best_so_far_plots.py`: “Best-so-far” time-series plot for OR
- `utils/`: Datasets, optimizers, surrogate models
- `main.py`: Unified, programmatic runners and CLI entrypoints
- `mass_experiments.py`: Batch runner over datasets and batch sizes with checkpointing
- `Results/` and `Plotting/`: Standardized outputs from experiments and plotting helpers


## Algorithms

### Hide-the-Label (HTL)
Hide-the-Label simulates an active learning loop over a finite candidate pool. A fraction of labels is “hidden” and the learner must adaptively select points to reveal, subject to a budget. Competing strategies attempt to minimize steps-to-target (or maximize final best under a fixed budget).

- Candidate Pool: Fixed discrete set of candidates (e.g., media recipes)
- Hidden Fraction: Controls how much of the pool is initially unlabeled
- Objective: Reach target quickly or maximize the final best
- Outputs: Per-optimizer steps-to-target, success rates, and distributional stats

Regular Mode uses Gaussian Process (GP) surrogates and stable settings.
Hard Mode uses non-GP surrogates and can introduce noise, heteroscedasticity, discontinuities, and multi-modality for a more challenging landscape.

Key entrypoints:
- Regular: `run_incremental_gp_competition` in `experiments/Regular_Mode/Regular_Mode_Hide_The_Label.py`
- Hard: `run_harder_competition` in `experiments/Hard_Mode/Harder_Mode_Hide_The_Label.py`


### Open Race (OR)
Open Race evaluates optimizers in a continuous search space. Each optimizer receives the same evaluation budget S, with R initial random points and subsequent batches of size B. Strategies propose the next batch to evaluate against a black-box function built from a surrogate model of the dataset.

- Domain: Continuous space bounded by dataset-derived ranges
- Budget: Exactly S evaluations per optimizer; R initial + (S-R) additional
- Batch Size: B (last step adjusts to preserve exactly S)
- Output: Per-optimizer best-so-far trajectories and tournament statistics

Regular Mode builds a GP surrogate with dataset-specific kernel choices.
Hard Mode swaps in harder surrogate types and difficulty configurations.

Key entrypoints:
- Regular: `run_open_race_competition` in `experiments/Regular_Mode/Regular_Mode_Open_Race.py`
- Hard: `run_harder_open_race_competition` in `experiments/Hard_Mode/Harder_Mode_Open_Race.py`


## Regular vs Hard Mode

- Surrogates: Regular uses GP; Hard uses alternative or perturbed surrogates (random forest, neural nets, discontinuities).
- Noise: Hard can add noise (including heteroscedastic), increasing difficulty.
- Landscape: Hard may introduce multiple modes and discontinuities to stress-test optimizers.
- Goals: Interfaces remain similar so comparisons are apples-to-apples under harder conditions.


## Visualizations

- HTL Bar Plot (`experiments/Visualization/Hide_The_Label_bar_plots.py`)
  - Input: A JSON results file with either `optimizer_stats` or a competitions list containing per-optimizer `steps_to_target`.
  - Output: Bar chart of mean steps per optimizer with standard errors.

- OR Best-So-Far (`experiments/Visualization/Open_Race_best_so_far_plots.py`)
  - Input: A standardized Open Race JSON (from the experiment runners)
  - Output: Line plot of best value so far versus step for each optimizer

Both scripts are headless (use Agg backend) and save figures to disk.


## Quick Start

### Environment
- Python 3.10+
- Install dependencies:

```bash
pip install -r requirements.txt
```

### Datasets and Optimizers
- See `utils/datasets.py` for available datasets and loader semantics
- See `utils/optimizers.py` and `utils/open_race_optimizers.py` for optimizers

### Running via main.py
Programmatic API:

```python
from main import (
    run_regular_hide_the_label,
    run_hard_hide_the_label,
    run_regular_open_race,
    run_hard_open_race,
    visualize_hide_the_label_mean_steps,
    visualize_open_race_best_so_far,
    run_experiment,
)

# Regular HTL
combined_results, analyses = run_regular_hide_the_label(
    dataset_name="MOBO_dataset_rat_myocyte",
    hidden_fraction=[0.9],
    batch_size=1,
)

# Hard OR
tournament_results, analysis = run_hard_open_race(
    dataset_name="df_Human_TF_Cell_Expanded",
    S=50, R=5, B=1,
)

# General dispatcher
out = run_experiment("regular open", dataset_name="MOBO_dataset_rat_myocyte", S=50, R=5, B=1)
```

CLI usage:

```bash
# Regular HTL
python main.py regular-hide MOBO_dataset_rat_myocyte --n_competitions 5 --hidden_fraction 0.9

# Hard OR
python main.py hard-open df_Human_TF_Cell_Expanded --surrogate_type random_forest --S 50 --R 5 --B 1

# Visualizations
python main.py viz-hide-bar /path/to/htl_results.json --out figures/hide_steps.png
python main.py viz-open-best /path/to/open_race_results.json --out figures/best_so_far.png
```


### Batch Runs with Checkpointing
Use `mass_experiments.py` to run grids across datasets and batch sizes with resume support.

- Iterates 4 settings: regular/hard × hide/open
- For HTL: forces `hidden_fraction=[0.99]`
- For OR: forces `R=5` and varies only `B` and dataset
- Writes extra results containing `99pcovered` in filenames
- Checkpoints: `/Users/aliparsaee/Desktop/AmiiResidencyProject/AL_Project/Checkpoints`

Run:

```bash
python mass_experiments.py
```

The runner skips any task with an existing checkpoint file and writes a failed checkpoint with error details on exceptions.


## Results and Plots
- Standard outputs are written under `Results/Regular_Mode/...` and `Results/Hard_Mode/...`.
- Additional convenience artifacts with `99pcovered` in filenames are written by `mass_experiments.py`.
- Plotting helpers also save into `Plotting/...` structures when invoked by experiment functions.


## Reproducibility Notes
- Random seeds are configurable; defaults are set in experiment entrypoints.
- Some hard-mode surrogates include stochastic elements; consider setting `random_state` for repeatability.
- Large datasets may trigger subsampling or faster kernel choices in the regular GP surrogate for performance.


## Contributing / Extending
- Add new optimizers under `utils/optimizers.py` or `utils/open_race_optimizers.py` and register them.
- Add new datasets in `utils/datasets.py` with a consistent interface (features `X`, targets `y`, variances `y_var`).
- For new visualizations, place scripts under `experiments/Visualization/` and export a `main()` if you want CLI usage.



