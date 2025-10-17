#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Open Race Competition Framework

This module implements the Open Race Game algorithm where optimization strategies
compete to find the global maximum of a black box function with a limited number
of function evaluations in continuous space.

Algorithm Implementation:
1. Sample R initial points uniformly within the domain of f
2. D_init = {(x_i, f(x_i))}_{i=1}^{R}
3. ObservationList = []
4. For j = 1 to J (For each optimizer):
   5. D = D_init (Reset to initial observation)
   6. For s = 1 to floor((S-R)/B):  # Ensure exactly S total evaluations per optimizer
      7. X_next = O_j(D, B) (Optimizer selects next batch)
      8. Evaluate Y_next = f(X_next)
      9. D = D ∪ {(X_next, Y_next)}
   10. Append max_{(x,y) ∈ D} y to ObservationList
11. Return ObservationList

Note: Each optimizer gets exactly S total evaluations: R initial + (S-R) additional evaluations

Key Features:
- Each optimizer gets the same initial random points
- Each optimizer runs independently for S total function evaluations
- Optimizers can propose ANY coordinate within the search bounds
- No hidden indices or discrete candidate pools
- Uses black box function evaluation
"""

import os
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from pathlib import Path
from datetime import datetime
from joblib import Parallel, delayed
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF, RationalQuadratic
from sklearn.preprocessing import StandardScaler
from scipy.stats import qmc
import warnings
warnings.filterwarnings('ignore')
import logging

from pathlib import Path

# Configure Additional outputs directory and logging
ROOT_DIR = Path(__file__).resolve().parents[2]
ADDITIONAL_OUTPUTS_DIR = ROOT_DIR / 'Additional outputs'
os.makedirs(ADDITIONAL_OUTPUTS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(ADDITIONAL_OUTPUTS_DIR / 'regular_mode_open_race.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Local imports (from utils)
from utils.datasets import load_dataset, get_available_datasets, Dataset, SyntheticDataset
from utils.optimizers import get_all_optimizers, get_optimizer_by_name, OptimizationStrategy
from experiments.Visualization.Open_Race_best_so_far_plots import (
    load_open_race,
    extract_per_optimizer_histories,
    aggregate_histories,
    plot_best_so_far,
)


def sample_initial_points(R: int, bounds: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """
    Sample R initial points uniformly within the domain bounds
    
    Args:
        R: Number of initial points
        bounds: Search space bounds (n_features, 2)
        rng: Random number generator
        
    Returns:
        Array of shape (R, n_features) with initial points
    """
    n_features = bounds.shape[0]
    points = np.zeros((R, n_features))
    for i in range(n_features):
        points[:, i] = rng.uniform(bounds[i, 0], bounds[i, 1], R)
    return points


class BlackBoxFunction:
    """Wrapper for GP model to act as a black box function"""
    
    def __init__(self, gp_model: GaussianProcessRegressor, scaler: StandardScaler, 
                 bounds: np.ndarray, noise_std: float = 0.0):
        """
        Initialize black box function
        
        Args:
            gp_model: Fitted GP model
            scaler: Feature scaler used for GP training
            bounds: Feature bounds (n_features, 2) array with [min, max] for each feature
            noise_std: Standard deviation of observation noise to add
        """
        self.gp_model = gp_model
        self.scaler = scaler
        self.bounds = bounds
        self.noise_std = noise_std
        self.n_features = bounds.shape[0]
        self.evaluation_count = 0
        
        # Track all evaluations
        self.X_evaluated = []
        self.y_evaluated = []
        
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the black box function at point x
        
        Args:
            x: Input point (1D array of length n_features)
            
        Returns:
            Function value at x
        """
        # Ensure x is 2D for sklearn
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        # Clip to bounds
        x_clipped = np.clip(x, self.bounds[:, 0], self.bounds[:, 1])
        
        # Scale features
        x_scaled = self.scaler.transform(x_clipped)
        
        # Get GP prediction
        y_pred = self.gp_model.predict(x_scaled)
        
        # Add noise if specified
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, y_pred.shape)
            y_pred += noise
        
        # Track evaluation
        self.evaluation_count += 1
        self.X_evaluated.append(x_clipped.flatten().copy())
        self.y_evaluated.append(float(y_pred[0]))
        
        return float(y_pred[0])
    
    def get_bounds(self) -> np.ndarray:
        """Get the bounds of the search space"""
        return self.bounds.copy()
    
    def get_evaluation_history(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get all evaluations made so far"""
        if not self.X_evaluated:
            return np.array([]).reshape(0, self.n_features), np.array([])
        return np.array(self.X_evaluated), np.array(self.y_evaluated)
    
    def reset_evaluation_count(self):
        """Reset evaluation counter and history"""
        self.evaluation_count = 0
        self.X_evaluated = []
        self.y_evaluated = []


def open_race_game(f: Callable[[np.ndarray], float], B: int, optimizers: List[OptimizationStrategy], 
                   S: int, R: int, bounds: np.ndarray, rng: np.random.RandomState) -> Tuple[List[float], Dict[str, List[Dict[str, Any]]]]:
    """
    Run the Open Race Game algorithm
    
    Args:
        f: Black box function to optimize
        B: Batch size
        optimizers: List of optimizer instances
        S: Total number of points to sample per optimizer
        R: Number of initial random points
        bounds: Search space bounds (n_features, 2)
        rng: Random number generator
        
    Returns:
        Tuple of (List of best values found by each optimizer, Dictionary of optimization histories)
    """
    # Validate parameters
    if S <= R:
        raise ValueError(f"S ({S}) must be greater than R ({R}) to allow additional evaluations")
    
    if (S - R) % B != 0:
        logger.info(f"Info: (S-R)/B = ({S}-{R})/{B} = {(S-R)/B} is not an integer. "
                   f"Will use variable batch sizes to achieve exactly {S} total evaluations")
    
    # Filter optimizers that support continuous optimization
    continuous_optimizers = [opt for opt in optimizers if opt.supports_continuous()]
    
    if len(continuous_optimizers) != len(optimizers):
        logger.warning(f"Filtered out {len(optimizers) - len(continuous_optimizers)} optimizers that don't support continuous optimization")
        logger.info(f"Using {len(continuous_optimizers)} optimizers: {[opt.name for opt in continuous_optimizers]}")
    
    if not continuous_optimizers:
        raise ValueError("No optimizers support continuous optimization for open race game")
    
    # Sample initial points (Algorithm step 1-2)
    D_init_X = sample_initial_points(R, bounds, rng)
    D_init_y = np.array([f(x) for x in D_init_X])
    
    # Initialize observation list (Algorithm step 3)
    observation_list = []
    optimization_histories = {}
    
    # For each optimizer (Algorithm step 4)
    for j, optimizer in enumerate(continuous_optimizers):
        logger.info(f"Running optimizer {j+1}/{len(continuous_optimizers)}: {optimizer.name}")
        
        # Reset optimizer state
        optimizer.reset()
        
        # Initialize with D_init (Algorithm step 5)
        D_X = D_init_X.copy()
        D_y = D_init_y.copy()
        current_best = np.max(D_y)
        
        # Track optimization history
        optimization_history = []
        
        # Record initial state (after R initial points)
        optimization_history.append({
            'step': 0,
            'evaluations': R,
            'best_value_so_far': float(current_best),
            'current_best': float(current_best)
        })
        
        # Run for (S-R)/B steps to ensure exactly S total evaluations (Algorithm step 6)
        # Each optimizer should get exactly S evaluations: R initial + (S-R) additional
        n_steps = (S - R) // B
        # Adjust batch size for the last step if needed to get exactly S evaluations
        remaining_evaluations = S - R - (n_steps * B)
        if remaining_evaluations > 0:
            n_steps += 1
        
        logger.debug(f"Optimizer {optimizer.name}: Will run {n_steps} steps (remaining_evaluations={remaining_evaluations})")
        
        for s in range(n_steps):
            # Determine batch size for this step
            if s == n_steps - 1 and remaining_evaluations > 0:
                current_batch_size = remaining_evaluations
            else:
                current_batch_size = B
            
            logger.debug(f"Step {s+1}/{n_steps}, batch_size={current_batch_size}")
            
            # Optimizer selects next batch (Algorithm step 7)
            try:
                X_next = optimizer.select_next_batch_open_field(
                    X_observed=D_X,
                    y_observed=D_y,
                    current_best=current_best,
                    batch_size=current_batch_size,
                    bounds=bounds,
                    true_function=f
                )
                
                # Ensure X_next is the right shape
                if X_next.ndim == 1:
                    X_next = X_next.reshape(1, -1)
                
                # Evaluate Y_next (Algorithm step 8)
                Y_next = np.array([f(x) for x in X_next])
                
                # Update D (Algorithm step 9)
                D_X = np.vstack([D_X, X_next])
                D_y = np.append(D_y, Y_next)
                
                # Update current best
                current_best = np.max(D_y)
                
                # Record step information
                optimization_history.append({
                    'step': s + 1,
                    'evaluations': R + (s + 1) * current_batch_size,
                    'best_value_so_far': float(current_best),
                    'current_best': float(current_best),
                    'batch_size': current_batch_size,
                    'new_values': Y_next.tolist()
                })
                
            except Exception as e:
                logger.error(f"Optimizer {optimizer.name} failed at step {s}: {e}")
                logger.error(f"Step {s+1}/{n_steps}, batch_size={current_batch_size}")
                # For open race, we should fail fast rather than use random fallback
                raise RuntimeError(f"Optimizer {optimizer.name} failed during open race game: {e}")
        
        # Record best outcome (Algorithm step 10)
        best_value = np.max(D_y)
        observation_list.append(best_value)
        optimization_histories[optimizer.name] = optimization_history
        logger.info(f"  -> Best value: {best_value:.6f}")
    
    return observation_list, optimization_histories


class OpenRaceCompetition:
    """Main class for organizing open race competitions"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.np_random = np.random.RandomState(random_state)
        self.start_time = time.time()
        self.MODE_TAG = 'Regular_Mode_Open_Race'
        
        # Optimal kernel configurations
        self.optimal_kernels = {
            'MOBO_dataset_rat_myocyte': {
                'kernel_type': 'RationalQuadratic',
                'noise_level': 0.1,
                'normalize_y': True,
                'alpha_mode': 'data_variance'
            },
            'DBO_dataset_rat_myocyte': {
                'kernel_type': 'RBF_plus_Matern_ARD',
                'noise_level': 0.1,
                'normalize_y': False,
                'alpha_mode': 'data_variance'
            },
            'df_Human_Hela_regular_mode': {
                'kernel_type': 'RBF_plus_Matern_iso',
                'noise_level': 0.1,
                'normalize_y': False,
                'alpha_mode': 'data_variance'
            },
            'df_Human_Hela_timesaving_mode': {
                'kernel_type': 'RBF_plus_Matern_iso',
                'noise_level': 0.01,
                'normalize_y': False,
                'alpha_mode': 'data_variance'
            }
        }
        
    def _log_timing(self, step_name: str):
        """Log elapsed time since start"""
        elapsed = time.time() - self.start_time
        logger.info(f"[{elapsed:.2f}s] {step_name}")
    
    def _create_optimal_kernel(self, dataset_name: str, n_features: int) -> Any:
        """Create the optimal kernel for a specific dataset"""
        if dataset_name not in self.optimal_kernels:
            logger.info(f"No optimal kernel found for {dataset_name}, using default")
            if n_features <= 30:
                initial_length_scales = np.ones(n_features) * 1.0
                return (1.0 * RBF(length_scale=initial_length_scales.copy(), length_scale_bounds=(1e-3, 1e2)) + 
                       1.0 * Matern(length_scale=initial_length_scales.copy(), nu=2.5, length_scale_bounds=(1e-3, 1e2)) + 
                       WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-3, 1.0)))
            else:
                return (1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e2)) + 
                       1.0 * Matern(length_scale=1.0, nu=2.5, length_scale_bounds=(1e-3, 1e2)) + 
                       WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-3, 1.0)))
        
        config = self.optimal_kernels[dataset_name]
        kernel_type = config['kernel_type']
        noise_level = config['noise_level']
        
        logger.info(f"Using OPTIMAL kernel for {dataset_name}: {kernel_type}")
        
        if kernel_type == 'RationalQuadratic':
            kernel = (1.0 * RationalQuadratic(length_scale=1.0, alpha=1.0,
                                             length_scale_bounds=(1e-3, 1e2),
                                             alpha_bounds=(1e-2, 1e2)) + 
                     WhiteKernel(noise_level=noise_level, noise_level_bounds=(1e-3, 1.0)))
                     
        elif kernel_type == 'RBF_plus_Matern_ARD':
            kernel = (1.0 * RBF(length_scale=np.ones(n_features), length_scale_bounds=(1e-3, 1e2)) + 
                     1.0 * Matern(length_scale=np.ones(n_features), nu=2.5, length_scale_bounds=(1e-3, 1e2)) + 
                     WhiteKernel(noise_level=noise_level, noise_level_bounds=(1e-3, 1.0)))
                     
        elif kernel_type == 'RBF_plus_Matern_iso':
            kernel = (1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e2)) + 
                     1.0 * Matern(length_scale=1.0, nu=2.5, length_scale_bounds=(1e-3, 1e2)) + 
                     WhiteKernel(noise_level=noise_level, noise_level_bounds=(1e-3, 1.0)))
        else:
            kernel = (1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e2)) + 
                     WhiteKernel(noise_level=noise_level, noise_level_bounds=(1e-3, 1.0)))
        
        return kernel
        
    def fit_surrogate_model(self, X: np.ndarray, y: np.ndarray, y_var: np.ndarray, 
                           dataset_name: str = None) -> Tuple[GaussianProcessRegressor, StandardScaler]:
        """Fit a Gaussian Process surrogate model to the dataset"""
        self._log_timing("Starting GP model fitting for black box function")
        
        n_samples, n_features = X.shape
        logger.info(f"Dataset: {n_samples} samples × {n_features} features")
        
        # For T/TF datasets, reuse cached surrogate via hide-the-label fitter when available
        use_cached_htl = dataset_name in ['df_Human_T_Cell_Expanded', 'df_Human_TF_Cell_Expanded']
        
        if use_cached_htl:
            logger.info(f"Attempting to reuse cached surrogate for {dataset_name} via hide_the_label_competition")
            try:
                from experiments.Regular_Mode.Regular_Mode_Hide_The_Label import HideLabelCompetitionIncrementalGP
                htl = HideLabelCompetitionIncrementalGP(
                    random_state=self.random_state,
                    use_gpu=False,
                    cache_dir=str(ADDITIONAL_OUTPUTS_DIR / "model_cache")
                )
                gp, scaler, r2 = htl.fit_surrogate_model(
                    X, y, y_var, dataset_name=dataset_name, use_cache_model=True
                )
                self._log_timing("Loaded surrogate via hide-the-label fitter (cached if available)")
                logger.info(f"Reused surrogate R^2 on full data: {r2:.4f}")
                return gp, scaler
            except Exception as e:
                logger.warning(f"Failed to load cached surrogate via hide-the-label fitter: {e}. Falling back to local fast_full.")
                # Fallback: fit local fast_full quickly
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(0.1, 10.0)) + WhiteKernel(0.1)
                gp = GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=y_var + 1e-6,
                    normalize_y=True,
                    n_restarts_optimizer=0,
                    random_state=self.random_state
                )
                gp.fit(X_scaled, y)
                self._log_timing("Finished GP model fitting (local fast_full fallback)")
                return gp, scaler
        
        # Otherwise keep existing behavior
        # Subsample if very large dataset
        if n_samples > 1500:
            logger.info("Large dataset detected: Subsampling for GP fitting speed...")
            n_subsample = min(600, n_samples)
            indices = self.np_random.choice(n_samples, n_subsample, replace=False)
            X_sub = X[indices]
            y_sub = y[indices]
            y_var_sub = y_var[indices]
            logger.info(f"Subsampled to: {n_subsample} samples ({n_subsample/n_samples*100:.1f}%)")
        else:
            X_sub, y_sub, y_var_sub = X, y, y_var
            logger.info("Using full dataset")  
        
        # Feature scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_sub)
        logger.info("Features scaled using StandardScaler")
        
        # Use optimal kernel
        if n_samples > 1500:
            kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(0.1, 10.0)) + WhiteKernel(0.1)
            logger.info("Using fast RBF kernel for large dataset")
        else:
            kernel = self._create_optimal_kernel(dataset_name, n_features)
            logger.info("Using optimal kernel configuration")
        
        # GP fitting parameters
        n_restarts = 0 if n_samples > 1500 else 5
        
        if dataset_name and dataset_name in self.optimal_kernels:
            config = self.optimal_kernels[dataset_name]
            normalize_y = config['normalize_y']
            alpha_mode = config['alpha_mode']
            
            if alpha_mode == 'data_variance':
                zero_var_count = np.sum(y_var_sub == 0)
                na_var_count = np.sum(np.isnan(y_var_sub))
                
                if zero_var_count > 0:
                    logger.info(f"Warning: {zero_var_count}/{len(y_var_sub)} target variances are zero")
                if na_var_count > 0:
                    logger.info(f"Warning: {na_var_count}/{len(y_var_sub)} target variances are NaN")
                    y_var_sub = np.nan_to_num(y_var_sub, nan=1e-6)
                
                alpha = y_var_sub + 1e-6
            else:
                alpha = 1e-6
                
            logger.info(f"Using optimal GP settings: normalize_y={normalize_y}, alpha_mode={alpha_mode}")
        else:
            normalize_y = True
            alpha = y_var_sub + 1e-6
            logger.info("Using default GP settings")
        
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            normalize_y=normalize_y,
            n_restarts_optimizer=n_restarts,
            random_state=self.random_state
        )
        
        gp.fit(X_scaled, y_sub)
        self._log_timing("Finished GP model fitting")

        # Evaluate model
        y_pred, sigma = gp.predict(X_scaled, return_std=True)
        mse = np.mean((y_sub - y_pred) ** 2)
        r2 = 1 - (np.sum((y_sub - y_pred) ** 2) / np.sum((y_sub - np.mean(y_sub)) ** 2))
        log_ml = gp.log_marginal_likelihood()
        
        logger.info(f"GP PERFORMANCE:")
        logger.info(f"  Training samples used: {len(y_sub)}/{len(y)}")
        logger.info(f"  Log-Marginal-Likelihood: {log_ml:.3f}")
        logger.info(f"  Mean Squared Error: {mse:.4f}")
        logger.info(f"  R^2 value: {r2:.4f}")

        return gp, scaler
    
    def create_black_box_function(self, gp_model: GaussianProcessRegressor, scaler: StandardScaler,
                                 original_X: np.ndarray, noise_std: float = 0.0) -> BlackBoxFunction:
        """Create a black box function from the fitted GP model"""
        # Calculate bounds based on original data with slight extension
        bounds = np.column_stack([
            original_X.min(axis=0) - 0.1 * (original_X.max(axis=0) - original_X.min(axis=0)),
            original_X.max(axis=0) + 0.1 * (original_X.max(axis=0) - original_X.min(axis=0))
        ])
        
        black_box = BlackBoxFunction(gp_model, scaler, bounds, noise_std)
        logger.info(f"Created black box function with {bounds.shape[0]} dimensions")
        logger.info(f"Search space bounds:")
        for i, (min_val, max_val) in enumerate(bounds):
            logger.info(f"  Feature {i}: [{min_val:.3f}, {max_val:.3f}]")
        
        return black_box
    
    def run_single_competition(self, black_box: BlackBoxFunction, optimizers: List[OptimizationStrategy],
                              S: int = 50, R: int = 5, B: int = 1, seed: int = None) -> Dict[str, Any]:
        """
        Run a single Open Race Game competition
        
        Args:
            black_box: Black box function to optimize
            optimizers: List of optimizer instances
            S: Total number of points to sample per optimizer
            R: Number of initial random points
            B: Batch size
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary containing competition results
        """
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = self.np_random
        
        bounds = black_box.get_bounds()
        
        # Run the Open Race Game
        observation_list, optimization_histories = open_race_game(
            f=black_box.evaluate,
            B=B,
            optimizers=optimizers,
            S=S,
            R=R,
            bounds=bounds,
            rng=rng
        )
        
        competition_results = {
            'S': S,
            'R': R,
            'B': B,
            'search_space_bounds': bounds.tolist(),
            'optimizer_names': [opt.name for opt in optimizers],
            'observation_list': observation_list,
            'optimizer_results': {}
        }
        
        # Store individual results
        for i, (optimizer, best_value) in enumerate(zip(optimizers, observation_list)):
            competition_results['optimizer_results'][optimizer.name] = {
                'best_value': float(best_value),
                'rank': i + 1,
                'optimization_history': optimization_histories[optimizer.name]
            }
        
        return competition_results
    
    def run_competition_tournament(self, dataset_name: str, optimizer_names: List[str] = None,
                                  n_competitions: int = 10, S: int = 50, R: int = 5, B: int = 1,
                                  noise_std: float = 0.0, n_jobs: int = -1) -> Dict[str, Any]:
        """
        Run a tournament of Open Race Game competitions
        
        Args:
            dataset_name: Name of the dataset to use
            optimizer_names: List of optimizer names to include
            n_competitions: Number of competitions to run
            S: Total number of points to sample per optimizer
            R: Number of initial random points
            B: Batch size
            noise_std: Standard deviation of observation noise
            n_jobs: Number of parallel jobs
            
        Returns:
            Dictionary containing tournament results
        """
        # Load dataset
        X, y, y_var, dataset = load_dataset(dataset_name)
        
        # Get optimizers
        if optimizer_names is None:
            # Use open race optimizers by default
            from utils.open_race_optimizers import get_open_race_optimizers
            open_race_optimizers = get_open_race_optimizers(random_state=self.random_state)
            optimizers = list(open_race_optimizers.values())
        else:
            # Use open race optimizers for the specified names
            from utils.open_race_optimizers import get_open_race_optimizers
            open_race_optimizers = get_open_race_optimizers(random_state=self.random_state)
            optimizers = []
            for name in optimizer_names:
                if name in open_race_optimizers:
                    optimizers.append(open_race_optimizers[name])
                else:
                    logger.warning(f"Optimizer {name} not found in open race optimizers")
        
        # Filter for continuous optimization support
        continuous_optimizers = [opt for opt in optimizers if opt.supports_continuous()]
        
        if len(continuous_optimizers) != len(optimizers):
            logger.warning(f"Filtered out {len(optimizers) - len(continuous_optimizers)} optimizers that don't support continuous optimization")
            filtered_names = [opt.name for opt in optimizers if not opt.supports_continuous()]
            logger.warning(f"Filtered optimizers: {filtered_names}")
        
        optimizers = continuous_optimizers
        
        if not optimizers:
            raise ValueError("No optimizers support continuous optimization for open race game")
        
        logger.info(f"Selected {len(optimizers)} optimizers: {[opt.name for opt in optimizers]}")
        
        # Fit GP model once
        gp_model, scaler = self.fit_surrogate_model(X, y, y_var, dataset_name)
        
        # Create black box function
        black_box = self.create_black_box_function(gp_model, scaler, X, noise_std)
        
        # Run competitions
        start_time = time.time()
        
        def run_competition_with_seed(seed, comp_idx=None, total_comps=None):
            local_competition = OpenRaceCompetition(random_state=seed)
            local_black_box = BlackBoxFunction(gp_model, scaler, black_box.get_bounds(), noise_std)
            
            if comp_idx is not None and total_comps is not None:
                logger.info(f"Starting Competition {comp_idx+1}/{total_comps}")
            
            result = local_competition.run_single_competition(
                black_box=local_black_box,
                optimizers=optimizers,
                S=S,
                R=R,
                B=B,
                seed=seed
            )
            
            if comp_idx is not None and total_comps is not None:
                logger.info(f"Finished Competition {comp_idx+1}/{total_comps}")
            
            return result
        
        if n_jobs == 1:
            competition_results = [
                run_competition_with_seed(self.random_state + i, i, n_competitions)
                for i in range(n_competitions)
            ]
        else:
            competition_results = Parallel(n_jobs=n_jobs)(
                delayed(run_competition_with_seed)(self.random_state + i)
                for i in range(n_competitions)
            )
        
        elapsed_time = time.time() - start_time
        
        tournament_results = {
            'dataset_name': dataset_name,
            'dataset_info': dataset.get_info(),
            'competition_settings': {
                'n_competitions': n_competitions,
                'S': S,
                'R': R,
                'B': B,
                'noise_std': noise_std,
                'random_state': self.random_state
            },
            'optimizer_names': [opt.name for opt in optimizers],
            'competitions': competition_results,
            'timestamp': datetime.now().isoformat(),
            'elapsed_time': elapsed_time
        }
        
        return tournament_results
    
    def analyze_tournament_results(self, tournament_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze tournament results and compute statistics"""
        optimizer_names = tournament_results['optimizer_names']
        competitions = tournament_results['competitions']
        n_competitions = len(competitions)
        
        analysis = {
            'optimizer_names': optimizer_names,
            'n_competitions': n_competitions,
            'optimizer_stats': {}
        }
        
        # Analyze each optimizer
        for optimizer_name in optimizer_names:
            best_values = []
            
            for comp in competitions:
                best_value = comp['optimizer_results'][optimizer_name]['best_value']
                best_values.append(best_value)
            
            # Compute statistics
            mean_best = np.mean(best_values)
            median_best = np.median(best_values)
            std_best = np.std(best_values)
            min_best = np.min(best_values)
            max_best = np.max(best_values)
            
            analysis['optimizer_stats'][optimizer_name] = {
                'mean_best_value': mean_best,
                'median_best_value': median_best,
                'std_best_value': std_best,
                'min_best_value': min_best,
                'max_best_value': max_best,
                'all_best_values': best_values
            }
        
        return analysis
    
    def print_tournament_summary(self, analysis: Dict[str, Any]):
        """Print a summary of tournament results"""
        print("\n" + "="*80)
        print("OPEN RACE COMPETITION RESULTS")
        print("="*80)
        
        print(f"Number of competitions: {analysis['n_competitions']}")
        print(f"Optimizers tested: {len(analysis['optimizer_names'])}")
        
        # Sort optimizers by mean best value (descending)
        sorted_optimizers = sorted(
            analysis['optimizer_names'], 
            key=lambda name: analysis['optimizer_stats'][name]['mean_best_value'],
            reverse=True
        )
        
        print("\nRanking (by mean best value found):")
        print("-" * 80)
        print(f"{'Rank':<4} {'Optimizer':<20} {'Mean Best':<12} {'Median Best':<13} {'Max Best':<10}")
        print("-" * 80)
        
        for i, optimizer_name in enumerate(sorted_optimizers, 1):
            stats = analysis['optimizer_stats'][optimizer_name]
            mean_best = f"{stats['mean_best_value']:.4f}"
            median_best = f"{stats['median_best_value']:.4f}"
            max_best = f"{stats['max_best_value']:.4f}"
            
            print(f"{i:<4} {optimizer_name:<20} {mean_best:<12} {median_best:<13} {max_best:<10}")
        
        print("-" * 80)
        
        # Best performer
        best_optimizer = sorted_optimizers[0]
        best_stats = analysis['optimizer_stats'][best_optimizer]
        print(f"\nBest performer: {best_optimizer}")
        print(f"  Mean best value: {best_stats['mean_best_value']:.4f}")
        print(f"  Median best value: {best_stats['median_best_value']:.4f}")
        print(f"  Max best value: {best_stats['max_best_value']:.4f}")


def run_open_race_competition(dataset_name: str, 
                             optimizer_names: List[str] = None,
                             n_competitions: int = 10,
                             S: int = 200,
                             R: int = 5,
                             B: int = 1,
                             noise_std: float = 0.0,
                             n_jobs: int = -1,
                             random_state: int = 42,
                             save_results: bool = True) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Run a complete Open Race Game competition tournament
    
    Args:
        dataset_name: Name of the dataset to use
        optimizer_names: List of optimizer names to include
        n_competitions: Number of competitions to run
        S: Total number of points to sample per optimizer
        R: Number of initial random points
        B: Batch size
        noise_std: Standard deviation of observation noise
        n_jobs: Number of parallel jobs
        random_state: Random seed
        save_results: Whether to save results to file
        
    Returns:
        Tuple of (tournament_results, analysis)
    """
    start_time = time.time()
    logger.info(f"Starting Open Race Game competition with {n_competitions} runs")
    logger.info(f"Each optimizer gets S={S} total function evaluations with B={B} batch size")
    
    # Create competition instance
    competition = OpenRaceCompetition(random_state=random_state)
    
    # Run tournament
    tournament_results = competition.run_competition_tournament(
        dataset_name=dataset_name,
        optimizer_names=optimizer_names,
        n_competitions=n_competitions,
        S=S,
        R=R,
        B=B,
        noise_std=noise_std,
        n_jobs=n_jobs
    )
    
    # Analyze results
    analysis = competition.analyze_tournament_results(tournament_results)
    
    # Print summary
    competition.print_tournament_summary(analysis)
    
    # Save standardized results JSON and generate plot to new structure
    if save_results:
        try:
            root_dir = Path(__file__).resolve().parents[2]
            results_dir = root_dir / 'Results' / 'Regular_Mode' / 'Open_Race'
            results_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_name = f"{dataset_name}_regular_mode_open_race_{timestamp}"
            results_path = results_dir / f"{base_name}.json"

            combined_results = {
                'tournament_results': tournament_results,
                'analysis': analysis
            }
            with open(results_path, 'w') as f:
                json.dump(combined_results, f, indent=2, default=str)
            logger.info(f"Results saved to: {results_path}")

            # Create plot
            plotting_dir = root_dir / 'Plotting' / 'Regular_Mode' / 'Open_Race'
            plotting_dir.mkdir(parents=True, exist_ok=True)
            plot_path = plotting_dir / f"{base_name}.png"

            try:
                # Use helper to plot best-so-far curves
                data = combined_results['tournament_results']
                per_opt_histories, tr = extract_per_optimizer_histories(data)
                aggregated = aggregate_histories(per_opt_histories)
                title = f"Open Race: Best-so-far — {dataset_name}"
                plot_best_so_far(aggregated, title, plot_path)
                logger.info(f"Plot saved to: {plot_path}")
            except Exception as e:
                logger.warning(f"Failed to generate Open Race plot: {e}")
        except Exception as e:
            logger.warning(f"Failed to save standardized Open Race results/plots: {e}")
    
    total_time = time.time() - start_time
    logger.info(f"Total execution time: {total_time:.2f} seconds")
    
    return tournament_results, analysis


if __name__ == "__main__":
    # Example usage
    print("Available datasets:")
    from utils.datasets import print_dataset_info
    print_dataset_info()
    
    print("\nAvailable optimizers:")
    from optimizers import print_optimizer_info
    print_optimizer_info()

    percentage_list = [0.95]  # List of hidden fractions to test (match AL_that_works)
    # working optimizers: ['RANDOM', 'BO_GP_EI', 'SMART_BO', 'SBO_GP_PV', 'SBO_ANN_PV', 'SBO_POLY_PV', 'SBO_GP_EI_TRUNCDE', 'DE_DIRECT', 'FULL_FACTORIAL', 'FRACTIONAL_FACTORIAL', 'PLACKETT_BURMAN', 'CENTRAL_COMPOSITE', 'BOX_BEHNKEN', 'LATIN_HYPERCUBE', 'D_OPTIMAL']
    # unavailable optimizers: ['GA_DIRECT', 'PSO_DIRECT'] (require DEAP and pyswarms libraries)
    # possible datasets: ['MOBO_dataset_rat_myocyte', 'DBO_dataset_rat_myocyte', 'df_Human_Hela_regular_mode', 'df_Human_Hela_timesaving_mode', 'df_Human_T_Cell_Expanded', 'df_Human_TF_Cell_Expanded', 'synthetic_2d', 'synthetic_5d', 'synthetic_10d']


    firsthalf = ['RANDOM','SBO_GP_PV', 'BO_GP_EI', 'SMART_BO', 'SBO_ANN_PV']
    secondhalf = ['DE_DIRECT', 'FULL_FACTORIAL', 'FRACTIONAL_FACTORIAL', 'PLACKETT_BURMAN', 'CENTRAL_COMPOSITE', 'BOX_BEHNKEN', 'LATIN_HYPERCUBE'] # removed 'D_OPTIMAL'
    all_optimisers = firsthalf + secondhalf

    print(f"\nRunning Open Race Game competition...")
    results, analysis = run_open_race_competition(
        dataset_name='df_Human_TF_Cell_Expanded',
        optimizer_names=all_optimisers,
        n_competitions=10,
        S=200,  # Total function evaluations per optimizer
        R=10,   # Initial random points
        B=20,   # Batch size
        noise_std=0,
        n_jobs=-1,
        random_state=42
    ) 