#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Harder Hide-the-Label Competition Framework

This module implements a harder version of the hide-the-label competition
that makes it more difficult for Bayesian optimizers by using non-GP models
and adding various difficulty factors.

Methods implemented:
1. Non-GP Type 1 Models (Random Forest, Neural Networks)
2. Noisy Label Revelation
3. Multi-modal Functions
4. Heteroscedastic Noise
"""

import os
import json
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
from datetime import datetime
from joblib import Parallel, delayed, dump, load
import hashlib
from sklearn.preprocessing import StandardScaler
from scipy.stats import qmc
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import existing competition framework
import sys

# Ensure both the project root and the experiments directory are importable,
# so this file can be executed from any working directory.
_CURRENT_DIR = os.path.dirname(__file__)
_EXPERIMENTS_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, '..'))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, '..', '..'))
for _p in (_PROJECT_ROOT, _EXPERIMENTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.Regular_Mode.Regular_Mode_Hide_The_Label import HideLabelCompetitionIncrementalGP
from utils.datasets import load_dataset, get_available_datasets, Dataset, SyntheticDataset
from utils.optimizers import get_all_optimizers, get_optimizer_by_name, OptimizationStrategy

# Import harder surrogate models (prefer project utils, then fallbacks)
try:
    from utils.harder_surrogates import (
        HarderSurrogateFactory,
        HarderRandomForestSurrogate,
        HarderNeuralNetworkSurrogate,
        DiscontinuousSurrogate,
    )
except Exception:
    try:
        from experiments.make_it_harder_attempts.harder_surrogates import (
            HarderSurrogateFactory,
            HarderRandomForestSurrogate,
            HarderNeuralNetworkSurrogate,
            DiscontinuousSurrogate,
        )
    except Exception:
        try:
            from .harder_surrogates import (
                HarderSurrogateFactory,
                HarderRandomForestSurrogate,
                HarderNeuralNetworkSurrogate,
                DiscontinuousSurrogate,
            )  # type: ignore
        except Exception:
            # Fallback to direct import from current directory when executed as a script
            _CURRENT_DIR = os.path.dirname(__file__)
            if _CURRENT_DIR not in sys.path:
                sys.path.insert(0, _CURRENT_DIR)
            from harder_surrogates import (
                HarderSurrogateFactory,
                HarderRandomForestSurrogate,
                HarderNeuralNetworkSurrogate,
                DiscontinuousSurrogate,
            )  # noqa: E402

from pathlib import Path

# Configure Additional outputs directory and logging
ROOT_DIR = Path(__file__).resolve().parents[2]
ADDITIONAL_OUTPUTS_DIR = ROOT_DIR / 'Additional outputs'
os.makedirs(ADDITIONAL_OUTPUTS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(ADDITIONAL_OUTPUTS_DIR / 'harder_competition.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class HarderHideLabelCompetition(HideLabelCompetitionIncrementalGP):
    """Harder version of hide-the-label competition using non-GP models"""
    
    def __init__(self, 
                 surrogate_type: str = 'random_forest',
                 difficulty_config: Dict[str, Any] = None,
                 random_state: int = 42, 
                 use_gpu: bool = True,
                 harder_cache_dir: str = "harder_model_cache"):
        
        # Initialize parent class
        super().__init__(random_state=random_state, use_gpu=use_gpu)
        
        # Set surrogate type and difficulty configuration
        self.surrogate_type = surrogate_type
        self.difficulty_config = difficulty_config or {}
        
        # Dedicated cache for harder competition models (separate from hide-the-label)
        # Place harder cache under Additional outputs unless absolute
        hc = Path(harder_cache_dir)
        if not hc.is_absolute():
            hc = ADDITIONAL_OUTPUTS_DIR / hc
        self.harder_cache_dir = str(hc)
        os.makedirs(self.harder_cache_dir, exist_ok=True)
        
        # Default difficulty configuration
        self.default_difficulty_config = {
            'noise_level': 0.1,
            'heteroscedastic': False,
            'n_modes': 1,  # 1 = no multi-modality
            'mode_separation': 2.0,
            'n_discontinuities': 0,  # 0 = no discontinuities
            'discontinuity_strength': 2.0
        }
        
        # Update with provided config
        self.default_difficulty_config.update(self.difficulty_config)
        self.difficulty_config = self.default_difficulty_config
        
        logger.info(f"Initialized Harder Competition with {surrogate_type} surrogate")
        logger.info(f"Difficulty config: {self.difficulty_config}")
        logger.info(f"Harder model cache dir: {self.harder_cache_dir}")

    def _generate_harder_model_cache_key(self, X: np.ndarray, y: np.ndarray, dataset_name: Optional[str]) -> str:
        """Create a unique cache key for harder surrogate based on data and config."""
        cfg_str = json.dumps(self.difficulty_config, sort_keys=True)
        h = hashlib.md5()
        h.update(f"{dataset_name}_{self.surrogate_type}_{self.random_state}".encode())
        h.update(cfg_str.encode())
        # Include limited data fingerprint for stability
        h.update(str(X.shape).encode()); h.update(str(y.shape).encode())
        if X.size > 0:
            h.update(X.astype(np.float64).ravel()[:200].tobytes())
        if y.size > 0:
            h.update(y.astype(np.float64).ravel()[:200].tobytes())
        return f"harder_{h.hexdigest()[:16]}"

    def _get_harder_cached_model_path(self, cache_key: str) -> str:
        return os.path.join(self.harder_cache_dir, f"{cache_key}.joblib")

    def _load_harder_cached_model(self, cache_key: str) -> Optional[Tuple[Any, StandardScaler, float]]:
        path = self._get_harder_cached_model_path(cache_key)
        if os.path.exists(path):
            try:
                logger.info(f"Loading harder cached model from: {path}")
                surrogate, scaler, r2 = load(path)
                logger.info(f"Loaded harder cached model with R²: {r2:.4f}")
                return surrogate, scaler, r2
            except Exception as e:
                logger.warning(f"Failed to load harder cached model: {e}")
        return None

    def _save_harder_cached_model(self, cache_key: str, surrogate: Any, scaler: StandardScaler, r2: float) -> None:
        path = self._get_harder_cached_model_path(cache_key)
        try:
            dump((surrogate, scaler, r2), path)
            logger.info(f"Saved harder model to cache: {path}")
        except Exception as e:
            logger.warning(f"Failed to save harder cached model: {e}")
        
    def fit_surrogate_model(self, X: np.ndarray, y: np.ndarray, y_var: np.ndarray, dataset_name: str = None, use_cache_model: bool = True) -> Tuple[Any, StandardScaler]:
        """
        Fit a harder surrogate model to the dataset
        
        Args:
            X: Input features
            y: Target values
            y_var: Target variances
            dataset_name: Name of the dataset (for logging)
            
        Returns:
            Tuple of (fitted harder surrogate model, feature scaler)
        """
        self._log_timing("Starting harder surrogate model fitting")
        
        n_samples, n_features = X.shape
        logger.info(f"Dataset: {n_samples} samples × {n_features} features")
        logger.info(f"Surrogate type: {self.surrogate_type}")
        logger.info(f"Difficulty level: {self.difficulty_config}")

        # Try cache first
        cache_key = self._generate_harder_model_cache_key(X, y, dataset_name or "unknown")
        logger.info(f"Harder model cache key: {cache_key}")
        if use_cache_model:
            cached = self._load_harder_cached_model(cache_key)
            if cached is not None:
                surrogate, scaler, _r2 = cached
                self._log_timing("Loaded harder cached surrogate model")
                return surrogate, scaler
        
        # Subsample very large datasets
        if n_samples > 1500:
            logger.info("Large dataset detected: subsampling for speed")
            n_subsample = min(600, n_samples)
            indices = self.np_random.choice(n_samples, n_subsample, replace=False)
            X_sub = X[indices]
            y_sub = y[indices]
            y_var_sub = y_var[indices]
            logger.info(f"Subsampled to: {n_subsample} samples ({n_subsample/n_samples*100:.1f}%)")
        else:
            X_sub, y_sub, y_var_sub = X, y, y_var
            logger.debug("Using full dataset")
        
        # Feature scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_sub)
        logger.debug("Features scaled using StandardScaler")
        
        # Create harder surrogate model
        if self.surrogate_type == 'random_forest':
            surrogate = HarderRandomForestSurrogate(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                bootstrap=True,
                random_state=self.random_state,
                n_jobs=1,
                noise_level=self.difficulty_config['noise_level'],
                heteroscedastic=self.difficulty_config['heteroscedastic'],
                n_modes=self.difficulty_config['n_modes'],
                mode_separation=self.difficulty_config['mode_separation']
            )
        elif self.surrogate_type == 'neural_network':
            surrogate = HarderNeuralNetworkSurrogate(
                hidden_sizes=[100, 50],
                activation='relu',
                learning_rate=0.001,
                batch_size=32,
                max_epochs=100,
                early_stopping_patience=10,
                dropout_rate=0.1,
                random_state=self.random_state,
                use_gpu=self.use_gpu,
                noise_level=self.difficulty_config['noise_level'],
                heteroscedastic=self.difficulty_config['heteroscedastic'],
                n_modes=self.difficulty_config['n_modes'],
                mode_separation=self.difficulty_config['mode_separation']
            )
        elif self.surrogate_type == 'discontinuous':
            # Create base surrogate first
            base_surrogate = HarderRandomForestSurrogate(
                random_state=self.random_state,
                noise_level=self.difficulty_config['noise_level'],
                heteroscedastic=self.difficulty_config['heteroscedastic']
            )
            surrogate = DiscontinuousSurrogate(
                base_surrogate=base_surrogate,
                n_discontinuities=self.difficulty_config['n_discontinuities'],
                discontinuity_strength=self.difficulty_config['discontinuity_strength'],
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown surrogate type: {self.surrogate_type}")
        
        # Fit the surrogate model
        surrogate.fit(X_sub, y_sub, y_var_sub)
        
        self._log_timing("Finished harder surrogate model fitting")
        
        # Evaluate the fitted model
        y_pred = surrogate.predict(X_sub)
        if isinstance(y_pred, tuple):
            y_pred = y_pred[0]  # Extract mean prediction
            
        mse = np.mean((y_sub - y_pred) ** 2)
        r2 = 1 - (np.sum((y_sub - y_pred) ** 2) / np.sum((y_sub - np.mean(y_sub)) ** 2))
        
        logger.info(f"Harder {self.surrogate_type} Performance - R²: {r2:.4f}, MSE: {mse:.4f}")
        logger.debug(f"Training samples: {len(y_sub)}/{len(y)}")
        
        # Save to harder cache using R² on training subset as metric
        try:
            self._save_harder_cached_model(cache_key, surrogate, scaler, float(r2))
        except Exception:
            pass
        
        return surrogate, scaler
    
    def generate_candidate_pool(self, surrogate_model: Any, scaler: StandardScaler,
                               original_X: np.ndarray, n_points: int = 200, synthetic_seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate candidate pool using the harder surrogate model
        
        Args:
            surrogate_model: Fitted harder surrogate model
            scaler: Feature scaler used for model training
            original_X: Original dataset features (for dimensionality)
            n_points: Number of points to generate
            synthetic_seed: Random seed for this specific synthetic dataset
            
        Returns:
            Tuple of (X_pool, y_pool) where y_pool are the function values
        """
        self._log_timing("Starting harder candidate pool generation")
        
        # Use provided synthetic_seed or default to self.random_state
        if synthetic_seed is None:
            seed_to_use = self.random_state
        else:
            seed_to_use = synthetic_seed
        
        # Generate Latin Hypercube design
        sampler = qmc.LatinHypercube(d=original_X.shape[1], seed=seed_to_use)
        X_pool = sampler.random(n_points)
        
        # Scale to match original data range
        X_min = np.min(original_X, axis=0)
        X_max = np.max(original_X, axis=0)
        X_pool = X_min + (X_max - X_min) * X_pool
        
        # Sample function values from harder surrogate model (surrogate scales internally)
        y_pool = surrogate_model.sample_y(X_pool, 1, random_state=seed_to_use).flatten()
        
        self._log_timing("Finished harder candidate pool generation")
        return X_pool, y_pool
    
    def get_difficulty_description(self) -> str:
        """Get a human-readable description of the difficulty configuration"""
        desc = f"Surrogate: {self.surrogate_type}"
        
        if self.difficulty_config['noise_level'] > 0:
            noise_type = "heteroscedastic" if self.difficulty_config['heteroscedastic'] else "homoscedastic"
            desc += f", {noise_type} noise ({self.difficulty_config['noise_level']:.2f})"
        
        if self.difficulty_config['n_modes'] > 1:
            desc += f", {self.difficulty_config['n_modes']} modes"
        
        if self.difficulty_config['n_discontinuities'] > 0:
            desc += f", {self.difficulty_config['n_discontinuities']} discontinuities"
        
        return desc


def run_harder_competition(dataset_name: str, 
                          surrogate_type: str = 'random_forest',
                          difficulty_config: Dict[str, Any] = None,
                          optimizer_names: List[str] = None,
                          n_competitions: int = 10,
                          hidden_fraction: Union[float, List[float]] = 0.9,
                          pool_size: int = 200,
                          batch_size: int = 1,
                          n_jobs: int = -1,
                          random_state: int = 42,
                          save_results: bool = True,
                          num_of_synthetic_data: int = 10,
                          use_gpu: bool = True) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Run harder hide-the-label competition using non-GP surrogate models
    
    Args:
        dataset_name: Name of the dataset to use
        surrogate_type: Type of harder surrogate ('random_forest', 'neural_network', 'discontinuous')
        difficulty_config: Configuration for difficulty factors
        optimizer_names: List of optimizer names to include
        n_competitions: Number of competitions to run
        hidden_fraction: Fraction of data to hide (can be list for multiple values)
        pool_size: Size of candidate pool
        n_jobs: Number of parallel jobs
        random_state: Random seed for reproducibility
        save_results: Whether to save results to files
        num_of_synthetic_data: Number of synthetic datasets to generate
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        Tuple of (tournament_results, analysis)
    """
    
    print("HARDER HIDE-THE-LABEL COMPETITION")
    print("="*80)
    print(f"Dataset: {dataset_name}")
    print(f"Surrogate: {surrogate_type}")
    print(f"Competitions: {n_competitions}")
    print(f"Hidden fraction: {hidden_fraction}")
    print(f"Pool size: {pool_size}")
    print(f"Parallel jobs: {n_jobs}")
    print(f"Random state: {random_state}")
    print(f"GPU acceleration: {use_gpu}")
    print(f"Synthetic datasets: {num_of_synthetic_data}")
    print("="*80)
    
    # Create harder competition instance
    competition = HarderHideLabelCompetition(
        surrogate_type=surrogate_type,
        difficulty_config=difficulty_config,
        random_state=random_state,
        use_gpu=use_gpu
    )
    
    # Print difficulty description
    difficulty_desc = competition.get_difficulty_description()
    print(f"Difficulty: {difficulty_desc}")
    print("="*80)
    
    logger.info(f"Starting harder competition with {dataset_name}")
    logger.info(f"Difficulty: {difficulty_desc}")
    
    # Convert hidden_fraction to list if it's a single float
    if isinstance(hidden_fraction, float):
        hidden_fraction = [hidden_fraction]
    
    # Initialize combined results
    all_tournament_results = []
    all_analyses = []
    
    # Load dataset and train harder surrogate model once
    print("\n=== Training Harder Surrogate Model ===")
    X, y, y_var, dataset = load_dataset(dataset_name)
    surrogate_model, scaler = competition.fit_surrogate_model(X, y, y_var, dataset_name)
    # Report R^2 on full dataset
    try:
        X_scaled_full = scaler.transform(X)
        y_pred_full = surrogate_model.predict(X_scaled_full)
        if isinstance(y_pred_full, tuple):
            y_pred_full = y_pred_full[0]
        r2_full = 1 - (np.sum((y - y_pred_full) ** 2) / np.sum((y - np.mean(y)) ** 2))
        print(f"Full-data surrogate R^2: {r2_full:.4f}")
    except Exception as e:
        print(f"Failed to compute full-data R^2: {e}")
    
    # Generate all synthetic datasets at once
    print(f"\n=== Generating {num_of_synthetic_data} Synthetic Datasets ===")
    synthetic_datasets = []
    for synth_idx in tqdm(range(num_of_synthetic_data), desc="Generating datasets"):
        # Use unique seed for each synthetic dataset
        synthetic_seed = random_state + synth_idx * 1000
        X_pool, y_pool = competition.generate_candidate_pool(
            surrogate_model, scaler, X, n_points=pool_size, synthetic_seed=synthetic_seed
        )
        synthetic_datasets.append((X_pool, y_pool))
        
        # Log basic stats to verify datasets are different
        logger.debug(f"Dataset {synth_idx + 1}: X_pool mean={np.mean(X_pool):.3f}, y_pool mean={np.mean(y_pool):.3f}, y_pool max={np.max(y_pool):.3f}")
    
    logger.info(f"Generated {len(synthetic_datasets)} unique synthetic datasets")
    
    # Run tournaments for each synthetic dataset
    for synth_idx, (X_pool, y_pool) in enumerate(synthetic_datasets):
        print(f"\n=== Running Tournament {synth_idx + 1}/{num_of_synthetic_data} ===")
        
        # Run tournament
        tournament_results = competition.run_competition_tournament(
            dataset_name=dataset_name,
            optimizer_names=optimizer_names,
            n_competitions=n_competitions,
            batch_size=batch_size,
            hidden_fraction=hidden_fraction,
            pool_size=pool_size,
            n_jobs=n_jobs,
            X_pool=X_pool,
            y_pool=y_pool
        )
        
        # Analyze results
        analysis = competition.analyze_tournament_results(tournament_results)
        
        # Print summary
        competition.print_tournament_summary(analysis)
        
        # Store results
        all_tournament_results.append(tournament_results)
        all_analyses.append(analysis)
    
    # Combine results from all synthetic datasets
    if num_of_synthetic_data > 1:
        print(f"\n=== COMBINED RESULTS FROM {num_of_synthetic_data} SYNTHETIC DATASETS ===")
        
        # Combine analyses
        combined_analysis = {
            'dataset_name': dataset_name,
            'surrogate_type': surrogate_type,
            'difficulty_config': difficulty_config,
            'optimizer_names': all_analyses[0]['optimizer_names'],
            'n_competitions': all_analyses[0]['n_competitions'] * num_of_synthetic_data,
            # Align with analyze_tournament_results key naming
            'hidden_fraction': all_analyses[0].get('hidden_fraction', hidden_fraction[0] if isinstance(hidden_fraction, list) else hidden_fraction),
            'pool_size': pool_size,
            'tournament_time': sum(analysis.get('tournament_time', 0) for analysis in all_analyses),
            'num_synthetic_datasets': num_of_synthetic_data,
            'optimizer_stats': {},
            'hidden_fraction_stats': {}
        }
        
        # Copy hidden_fraction_stats from the first analysis if present
        if 'hidden_fraction_stats' in all_analyses[0]:
            combined_analysis['hidden_fraction_stats'] = all_analyses[0]['hidden_fraction_stats']
        
        # Aggregate statistics across all synthetic datasets
        for optimizer_name in combined_analysis['optimizer_names']:
            all_steps = []
            all_success = []
            all_final_best = []
            
            for analysis in all_analyses:
                stats = analysis['optimizer_stats'][optimizer_name]
                # Simplified aggregation
                all_steps.extend([stats['mean_steps']] * analysis['n_competitions'])
                all_success.extend([stats['success_rate']] * analysis['n_competitions'])
                all_final_best.extend([stats['mean_final_best']] * analysis['n_competitions'])
            
            combined_analysis['optimizer_stats'][optimizer_name] = {
                'mean_steps': np.mean(all_steps),
                'std_steps': np.std(all_steps),
                'median_steps': np.median(all_steps),
                'min_steps': np.min(all_steps),
                'max_steps': np.max(all_steps),
                'success_rate': np.mean(all_success),
                'mean_final_best': np.mean(all_final_best),
                'std_final_best': np.std(all_final_best)
            }
        
        # Print combined summary
        competition.print_tournament_summary(combined_analysis)
        
        # Save combined results in standardized location and generate plot
        if save_results:
            try:
                root_dir = Path(__file__).resolve().parents[2]
                results_dir = root_dir / 'Results' / 'Hard_Mode' / 'Hide_The_Label'
                results_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                base_name = f"{dataset_name}_hard_mode_hide_the_label_{timestamp}"
                results_path = results_dir / f"{base_name}.json"

                combined_results = {
                    'dataset_name': dataset_name,
                    'surrogate_type': surrogate_type,
                    'difficulty_config': difficulty_config,
                    'num_synthetic_datasets': num_of_synthetic_data,
                    'tournament_results': all_tournament_results,
                    'analyses': all_analyses,
                    'combined_analysis': combined_analysis
                }
                with open(results_path, 'w') as f:
                    json.dump(combined_results, f, indent=2, default=str)
                logger.info(f"Results saved to: {results_path}")

                # Generate bar plot
                from experiments.Visualization.Hide_The_Label_bar_plots import (
                    extract_from_optimizer_stats, extract_from_competitions, plot_bar,
                )
                plotting_dir = root_dir / 'Plotting' / 'Hard_Mode' / 'Hide_The_Label'
                plotting_dir.mkdir(parents=True, exist_ok=True)
                plot_path = plotting_dir / f"{base_name}.png"

                try:
                    optimizers, means, ses = extract_from_optimizer_stats(combined_analysis)
                    if not optimizers:
                        optimizers, means, ses = extract_from_competitions({
                            'all_tournament_results': all_tournament_results
                        })
                    title = f"Hide-the-Label (Hard): Mean Steps — {dataset_name}"
                    plot_bar(optimizers, means, ses, title, str(plot_path))
                    logger.info(f"Plot saved to: {plot_path}")
                except Exception as e:
                    logger.warning(f"Failed to generate plot: {e}")
            except Exception as e:
                logger.warning(f"Failed to save standardized results/plots: {e}")
        
        return combined_results, combined_analysis
    
    else:
        # Single synthetic dataset
        if save_results:
            try:
                root_dir = Path(__file__).resolve().parents[2]
                results_dir = root_dir / 'Results' / 'Hard_Mode' / 'Hide_The_Label'
                results_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                base_name = f"{dataset_name}_hard_mode_hide_the_label_{timestamp}"
                results_path = results_dir / f"{base_name}.json"

                payload = {
                    'tournament_results': all_tournament_results[0],
                    'analysis': all_analyses[0],
                    'surrogate_type': surrogate_type,
                    'difficulty_config': difficulty_config,
                }
                with open(results_path, 'w') as f:
                    json.dump(payload, f, indent=2, default=str)
                logger.info(f"Results saved to: {results_path}")

                from experiments.Visualization.Hide_The_Label_bar_plots import (
                    extract_from_optimizer_stats, extract_from_competitions, plot_bar,
                )
                plotting_dir = root_dir / 'Plotting' / 'Hard_Mode' / 'Hide_The_Label'
                plotting_dir.mkdir(parents=True, exist_ok=True)
                plot_path = plotting_dir / f"{base_name}.png"

                try:
                    optimizers, means, ses = extract_from_optimizer_stats(all_analyses[0])
                    if not optimizers:
                        optimizers, means, ses = extract_from_competitions(all_tournament_results[0])
                    title = f"Hide-the-Label (Hard): Mean Steps — {dataset_name}"
                    plot_bar(optimizers, means, ses, title, str(plot_path))
                    logger.info(f"Plot saved to: {plot_path}")
                except Exception as e:
                    logger.warning(f"Failed to generate plot: {e}")
            except Exception as e:
                logger.warning(f"Failed to save standardized results/plots: {e}")
        
        return all_tournament_results[0], all_analyses[0]


if __name__ == "__main__":
    # Example usage with different difficulty configurations
    
    # Configuration 1: Noisy Random Forest
    print("Testing noisy Random Forest...")
    noisy_config = {
        'noise_level': 0.2,
        'heteroscedastic': False,
        'n_modes': 1,
        'n_discontinuities': 0
    }
    
    percentage_list = [0.95]  # List of hidden fractions to test (match AL_that_works)
    # working optimizers: ['RANDOM', 'BO_GP_EI', 'SMART_BO', 'SBO_GP_PV', 'SBO_ANN_PV', 'SBO_POLY_PV', 'SBO_GP_EI_TRUNCDE', 'DE_DIRECT', 'FULL_FACTORIAL', 'FRACTIONAL_FACTORIAL', 'PLACKETT_BURMAN', 'CENTRAL_COMPOSITE', 'BOX_BEHNKEN', 'LATIN_HYPERCUBE', 'D_OPTIMAL']
    # unavailable optimizers: ['GA_DIRECT', 'PSO_DIRECT'] (require DEAP and pyswarms libraries)
    # possible datasets: ['MOBO_dataset_rat_myocyte', 'DBO_dataset_rat_myocyte', 'df_Human_Hela_regular_mode', 'df_Human_Hela_timesaving_mode', 'df_Human_T_Cell_Expanded', 'df_Human_TF_Cell_Expanded', 'synthetic_2d', 'synthetic_5d', 'synthetic_10d']


    firsthalf = ['RANDOM', 'BO_GP_EI', 'SMART_BO', 'SBO_GP_PV', 'SBO_ANN_PV']
    secondhalf = ['DE_DIRECT', 'FULL_FACTORIAL', 'FRACTIONAL_FACTORIAL', 'PLACKETT_BURMAN', 'CENTRAL_COMPOSITE', 'BOX_BEHNKEN', 'LATIN_HYPERCUBE']  #, 'D_OPTIMAL'
    all_optimisers = firsthalf + secondhalf


    results1, analysis1 = run_harder_competition(
        dataset_name='df_Human_TF_Cell_Expanded',
        surrogate_type='random_forest',
        difficulty_config=noisy_config,
        optimizer_names= all_optimisers,
        n_competitions=10,
        hidden_fraction=0.95,  # Use float, not list
        pool_size=200,
        batch_size=20,
        n_jobs=-1,
        random_state=42,
        save_results = True,
        num_of_synthetic_data=10,
        use_gpu=False
    )


    
    # # Configuration 2: Multi-modal Neural Network
    # print("\nTesting multi-modal Neural Network...")
    # multimodal_config = {
    #     'noise_level': 0.1,
    #     'heteroscedastic': True,
    #     'n_modes': 3,
    #     'mode_separation': 2.0,
    #     'n_discontinuities': 0
    # }
    
    # results2, analysis2 = run_harder_competition(
    #     dataset_name='synthetic_2d',
    #     surrogate_type='neural_network',
    #     difficulty_config=multimodal_config,
    #     optimizer_names=['RANDOM', 'BO_GP_EI', 'SMART_BO', 'SBO_GP_PV', 'SBO_ANN_PV', 'SBO_POLY_PV', 'SBO_GP_EI_TRUNCDE', 'DE_DIRECT', 'FULL_FACTORIAL', 'FRACTIONAL_FACTORIAL', 'PLACKETT_BURMAN', 'CENTRAL_COMPOSITE', 'BOX_BEHNKEN', 'LATIN_HYPERCUBE', 'D_OPTIMAL'],
    #     n_competitions=3,
    #     hidden_fraction=0.95,  # Use float, not list
    #     pool_size=50,
    #     n_jobs=1,
    #     random_state=42,
    #     num_of_synthetic_data=1,
    #     use_gpu=False
    # ) 
