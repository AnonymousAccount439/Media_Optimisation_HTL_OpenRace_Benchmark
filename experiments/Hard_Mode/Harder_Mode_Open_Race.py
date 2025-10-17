#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Harder Open Race Competition Framework

This module implements a harder version of the open race competition
that makes it more difficult for Bayesian optimizers by using non-GP models
and adding various difficulty factors.

Methods implemented:
1. Non-GP Type 1 Models (Random Forest, Neural Networks)
2. Noisy Function Evaluation
3. Multi-modal Functions
4. Heteroscedastic Noise
5. Discontinuous Functions
"""

import os
import sys
import json
import time
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from joblib import Parallel, delayed, dump, load
import hashlib
from sklearn.preprocessing import StandardScaler
import warnings

# Import existing open race framework with robust path fallback
try:
    from experiments.Regular_Mode.Regular_Mode_Open_Race import OpenRaceCompetition, BlackBoxFunction
    from utils.datasets import load_dataset
except Exception:
    _CURRENT_DIR = os.path.dirname(__file__)
    _EXPERIMENTS_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, '..'))
    _PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, '..', '..'))
    for _p in (_PROJECT_ROOT, _EXPERIMENTS_DIR):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    from experiments.Regular_Mode.Regular_Mode_Open_Race import OpenRaceCompetition, BlackBoxFunction  # noqa: E402
    from utils.datasets import load_dataset  # noqa: E402

# Import harder surrogate models with robust fallback
try:
    from utils.harder_surrogates import HarderSurrogateFactory
except Exception:
    _CURRENT_DIR = os.path.dirname(__file__)
    if _CURRENT_DIR not in sys.path:
        sys.path.insert(0, _CURRENT_DIR)
    from harder_surrogates import HarderSurrogateFactory  # noqa: E402

from pathlib import Path

# Configure Additional outputs directory and logging
ROOT_DIR = Path(__file__).resolve().parents[2]
ADDITIONAL_OUTPUTS_DIR = ROOT_DIR / 'Additional outputs'
os.makedirs(ADDITIONAL_OUTPUTS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(ADDITIONAL_OUTPUTS_DIR / 'harder_open_race_competition.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Silence noisy warnings after imports are complete
warnings.filterwarnings('ignore')


class HarderBlackBoxFunction(BlackBoxFunction):
    """Harder version of black box function with additional difficulty factors"""
    
    def __init__(self, surrogate_model: Any, scaler: StandardScaler, 
                 bounds: np.ndarray, difficulty_config: Dict[str, Any] = None,
                 random_state: Optional[int] = None):
        """
        Initialize harder black box function
        
        Args:
            surrogate_model: Fitted surrogate model (non-GP)
            scaler: Feature scaler used for model training
            bounds: Feature bounds (n_features, 2) array with [min, max] for each feature
            difficulty_config: Configuration for difficulty factors
        """
        # Initialize parent class with dummy GP model
        from sklearn.gaussian_process import GaussianProcessRegressor
        dummy_gp = GaussianProcessRegressor()
        super().__init__(dummy_gp, scaler, bounds, noise_std=0.0)
        
        # Replace with actual surrogate model
        self.surrogate_model = surrogate_model
        
        # Set difficulty configuration
        self.difficulty_config = difficulty_config or {}
        self.default_difficulty_config = {
            'noise_level': 0.1,
            'heteroscedastic': False,
            'n_modes': 1,  # 1 = no multi-modality
            'mode_separation': 2.0,
            'n_discontinuities': 0,  # 0 = no discontinuities
            'discontinuity_strength': 2.0
        }
        self.default_difficulty_config.update(self.difficulty_config)
        self.difficulty_config = self.default_difficulty_config
        
        # Local RNG for reproducible noise within a competition
        self.rng = np.random.RandomState(random_state) if random_state is not None else np.random.RandomState()
        
        logger.info(f"Initialized Harder Black Box Function with difficulty: {self.difficulty_config}")
        
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the harder black box function at point x
        
        Args:
            x: Input point (n_features,)
            
        Returns:
            Function value with difficulty factors applied
        """
        # Ensure x is 2D for prediction
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        # Clip to bounds then scale input (aligns with baseline behavior)
        x_clipped = np.clip(x, self.bounds[:, 0], self.bounds[:, 1])
        x_scaled = self.scaler.transform(x_clipped)
        
        # Get base prediction from surrogate model
        if hasattr(self.surrogate_model, 'predict'):
            base_value = self.surrogate_model.predict(x_scaled)
            if isinstance(base_value, tuple):
                base_value = base_value[0]
            base_value = base_value.flatten()[0]
        else:
            # Fallback for models without predict method
            base_value = 0.0
        
        # Apply difficulty factors
        final_value = self._apply_difficulty_factors(x_scaled[0], base_value)
        
        # Track evaluation
        self.evaluation_count += 1
        self.X_evaluated.append(x_clipped.flatten())
        # Track evaluated value for parity with baseline wrapper
        if hasattr(self, 'y_evaluated'):
            self.y_evaluated.append(float(final_value))
        
        return final_value
    
    def _apply_difficulty_factors(self, x_scaled: np.ndarray, base_value: float) -> float:
        """Apply various difficulty factors to the function value"""
        value = base_value
        
        # 1. Add noise
        noise_level = self.difficulty_config.get('noise_level', 0.1)
        if self.difficulty_config.get('heteroscedastic', False):
            # Heteroscedastic noise: more noise in certain regions
            noise_scale = 1.0 + np.sum(x_scaled**2) * 0.5
            noise = self.rng.normal(0, noise_level * noise_scale)
        else:
            # Homoscedastic noise
            noise = self.rng.normal(0, noise_level)
        
        value += noise
        
        # 2. Add multi-modality (if enabled)
        n_modes = self.difficulty_config.get('n_modes', 1)
        if n_modes > 1:
            mode_separation = self.difficulty_config.get('mode_separation', 2.0)
            # Add additional modes at different locations
            for i in range(1, n_modes):
                mode_offset = i * mode_separation
                mode_contribution = np.exp(-np.sum((x_scaled - mode_offset)**2))
                value += mode_contribution * 0.5
        
        # 3. Add discontinuities (if enabled)
        n_discontinuities = self.difficulty_config.get('n_discontinuities', 0)
        if n_discontinuities > 0:
            discontinuity_strength = self.difficulty_config.get('discontinuity_strength', 2.0)
            for i in range(n_discontinuities):
                # Create discontinuities at specific boundaries
                boundary = (i + 1) / (n_discontinuities + 1)
                if np.any(x_scaled > boundary):
                    value += discontinuity_strength
        
        return value


class HarderOpenRaceCompetition(OpenRaceCompetition):
    """Harder version of open race competition using non-GP models"""
    
    def __init__(self, 
                 surrogate_type: str = 'random_forest',
                 difficulty_config: Dict[str, Any] = None,
                 random_state: int = 42,
                 harder_or_cache_dir: str = "harder_open_race_model_cache"):
        
        # Initialize parent class
        super().__init__(random_state=random_state)
        
        # Set surrogate type and difficulty configuration
        self.surrogate_type = surrogate_type
        self.difficulty_config = difficulty_config or {}
        
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
        
        # Dedicated cache for harder open race models (one file per dataset)
        # Place harder open race cache under Additional outputs unless absolute
        oc = Path(harder_or_cache_dir)
        if not oc.is_absolute():
            oc = ADDITIONAL_OUTPUTS_DIR / oc
        self.harder_or_cache_dir = str(oc)
        os.makedirs(self.harder_or_cache_dir, exist_ok=True)
        
        logger.info(f"Initialized Harder Open Race Competition with {surrogate_type} surrogate")
        logger.info(f"Difficulty config: {self.difficulty_config}")
        logger.info(f"Harder open race cache dir: {self.harder_or_cache_dir}")

    def _generate_or_model_cache_key(self, X: np.ndarray, y: np.ndarray, dataset_name: Optional[str]) -> str:
        cfg_str = json.dumps(self.difficulty_config, sort_keys=True)
        h = hashlib.md5()
        h.update(f"{dataset_name}_{self.surrogate_type}_{self.random_state}".encode())
        h.update(cfg_str.encode())
        h.update(str(X.shape).encode()); h.update(str(y.shape).encode())
        if X.size:
            h.update(X.astype(np.float64).ravel()[:200].tobytes())
        if y.size:
            h.update(y.astype(np.float64).ravel()[:200].tobytes())
        return f"harder_or_{h.hexdigest()[:16]}"

    def _get_or_cached_model_path(self, filename: str) -> str:
        return os.path.join(self.harder_or_cache_dir, filename)

    def _load_or_cached_model(self, cache_key: str) -> Optional[Tuple[Any, StandardScaler, float, Optional[Dict[str, Any]]]]:
        candidates = []
        try:
            for f in os.listdir(self.harder_or_cache_dir):
                if f.endswith('.joblib') and (f == f"{cache_key}.joblib" or f.startswith(f"{cache_key}__")):
                    p = self._get_or_cached_model_path(f)
                    try:
                        m = os.path.getmtime(p)
                    except Exception:
                        m = 0
                    candidates.append((m, p))
        except Exception:
            pass
        if not candidates:
            return None
        candidates.sort(key=lambda t: t[0], reverse=True)
        _, chosen = candidates[0]
        try:
            payload = load(chosen)
            meta = None
            if isinstance(payload, tuple) and len(payload) >= 3:
                model, scaler, r2 = payload[:3]
                if len(payload) >= 4 and isinstance(payload[3], dict):
                    meta = payload[3]
                return model, scaler, float(r2), meta
        except Exception:
            return None
        return None

    def _save_or_cached_model(self, cache_key: str, model: Any, scaler: StandardScaler, r2: float, metadata: Optional[Dict[str, Any]] = None) -> None:
        dataset_part = (metadata or {}).get('dataset_name') or 'unknown'
        created_part = (metadata or {}).get('created_at') or datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_dataset = str(dataset_part).replace(' ', '_')
        # Include mode tag in cache filename
        mode_tag = 'Hard_Mode_Open_Race'
        filename = f"{cache_key}__{mode_tag}__{safe_dataset}__{created_part}.joblib"
        path = self._get_or_cached_model_path(filename)
        try:
            dump((model, scaler, float(r2), metadata or {}), path)
            logger.info(f"Saved harder open race model to cache: {path}")
        except Exception as e:
            logger.warning(f"Failed to save harder open race cached model: {e}")
        
    def fit_surrogate_model(self, X: np.ndarray, y: np.ndarray, y_var: np.ndarray, dataset_name: str = None, use_cache_model: bool = True) -> Tuple[Any, StandardScaler]:
        """
        Fit a harder surrogate model to the dataset
        
        Args:
            X: Input features
            y: Target values
            y_var: Target variances
            dataset_name: Name of the dataset
            
        Returns:
            Tuple of (fitted surrogate model, feature scaler)
        """
        self._log_timing("Starting harder surrogate model fitting")
        
        n_samples, n_features = X.shape
        logger.info(f"Dataset: {n_samples} samples × {n_features} features")
        logger.info(f"Using {self.surrogate_type} surrogate model")
        
        # Try cache first
        cache_key = self._generate_or_model_cache_key(X, y, dataset_name or "unknown")
        logger.info(f"Harder open race cache key: {cache_key}")
        if use_cache_model:
            cached = self._load_or_cached_model(cache_key)
            if cached is not None:
                model, scaler, r2, meta = cached
                self._log_timing("Loaded harder open race cached surrogate model")
                logger.info(f"Cached model meta: {meta}")
                return model, scaler

        # Fast path for large datasets (including T/TF): subsample to cap for speed
        n_samples, n_features = X.shape
        if n_samples > 1500 or (dataset_name in ['df_Human_T_Cell_Expanded', 'df_Human_TF_Cell_Expanded']):
            logger.info("Large dataset detected for harder open race: subsampling for speed")
            n_subsample = min(600, n_samples)
            idx = np.random.RandomState(self.random_state).choice(n_samples, n_subsample, replace=False)
            X_fit, y_fit, yv_fit = X[idx], y[idx], y_var[idx]
        else:
            X_fit, y_fit, yv_fit = X, y, y_var

        # Feature scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_fit)
        logger.info("Features scaled using StandardScaler")
        
        # Create and fit harder surrogate model using static factory method
        if self.surrogate_type == 'random_forest':
            # Filter out parameters that don't apply to random forest
            rf_config = {k: v for k, v in self.difficulty_config.items() 
                        if k not in ['n_discontinuities', 'discontinuity_strength']}
            surrogate_model = HarderSurrogateFactory.create_harder_surrogate(
                surrogate_type='random_forest',
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                **rf_config
            )
        elif self.surrogate_type == 'neural_network':
            surrogate_model = HarderSurrogateFactory.create_harder_surrogate(
                surrogate_type='neural_network',
                hidden_sizes=[100, 50],
                activation='relu',
                learning_rate=0.001,
                max_epochs=500,
                random_state=self.random_state,
                **self.difficulty_config
            )
        elif self.surrogate_type == 'discontinuous':
            surrogate_model = HarderSurrogateFactory.create_harder_surrogate(
                surrogate_type='discontinuous',
                base_type='random_forest',
                n_discontinuities=self.difficulty_config.get('n_discontinuities', 3),
                discontinuity_strength=self.difficulty_config.get('discontinuity_strength', 2.0),
                random_state=self.random_state,
                **self.difficulty_config
            )
        else:
            raise ValueError(f"Unknown surrogate type: {self.surrogate_type}")
        
        # Fit the model
        surrogate_model.fit(X_scaled, y_fit, yv_fit)
        
        self._log_timing("Finished harder surrogate model fitting")
        
        # Evaluate the fitted model
        y_pred = surrogate_model.predict(X_scaled)
        if isinstance(y_pred, tuple):
            y_pred = y_pred[0]
            
        # Compute metrics on the training subset actually used for fitting
        mse = np.mean((y_fit - y_pred) ** 2)
        r2 = 1 - (np.sum((y_fit - y_pred) ** 2) / np.sum((y_fit - np.mean(y_fit)) ** 2))
        
        logger.info(f"Harder Surrogate Performance - R²: {r2:.4f}, MSE: {mse:.4f}")
        
        # Save cache with metadata
        meta = {
            'dataset_name': dataset_name or 'unknown',
            'created_at': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'n_samples_fit': int(len(y_fit))
        }
        self._save_or_cached_model(cache_key, surrogate_model, scaler, float(r2), metadata=meta)
        
        return surrogate_model, scaler
    
    def create_black_box_function(self, surrogate_model: Any, scaler: StandardScaler,
                                 original_X: np.ndarray, noise_std: float = 0.0) -> HarderBlackBoxFunction:
        """
        Create a harder black box function from the surrogate model
        
        Args:
            surrogate_model: Fitted surrogate model
            scaler: Feature scaler used for model training
            original_X: Original dataset features (for bounds calculation)
            noise_std: Additional noise standard deviation (overridden by difficulty config)
            
        Returns:
            HarderBlackBoxFunction instance
        """
        # Calculate bounds from original data
        bounds = np.column_stack([
            np.min(original_X, axis=0),
            np.max(original_X, axis=0)
        ])
        
        # Create harder black box function
        black_box = HarderBlackBoxFunction(
            surrogate_model=surrogate_model,
            scaler=scaler,
            bounds=bounds,
            difficulty_config=self.difficulty_config
        )
        
        logger.info(f"Created harder black box function with bounds: {bounds}")
        return black_box
    
    def get_difficulty_description(self) -> str:
        """Get a description of the current difficulty configuration"""
        config = self.difficulty_config
        
        description = f"Surrogate: {self.surrogate_type}"
        description += f", Noise: {config.get('noise_level', 0.1)}"
        description += f", Heteroscedastic: {config.get('heteroscedastic', False)}"
        description += f", Modes: {config.get('n_modes', 1)}"
        description += f", Discontinuities: {config.get('n_discontinuities', 0)}"
        
        return description
    
    def run_harder_competition_tournament(self, black_box: HarderBlackBoxFunction,
                                        optimizer_names: List[str] = None,
                                        n_competitions: int = 10,
                                        S: int = 50, R: int = 5, B: int = 1,
                                        n_jobs: int = -1) -> Dict[str, Any]:
        """
        Run a tournament of harder open race competitions
        
        Args:
            black_box: Harder black box function to optimize
            optimizer_names: List of optimizer names to include
            n_competitions: Number of competitions to run
            S: Total number of points to sample per optimizer
            R: Number of initial random points
            B: Batch size
            n_jobs: Number of parallel jobs
            
        Returns:
            Dictionary containing tournament results
        """
        # Get optimizers (use robust variants for harder open race)
        try:
            from utils.optimisers_for_harder_open_race import (
                get_open_race_optimizers as get_harder_optimizers
            )
        except Exception:
            try:
                from utils.optimisers_for_harder_open_race import (
                    get_open_race_optimizers as get_harder_optimizers
                )  # type: ignore
            except Exception:
                _CURRENT_DIR = os.path.dirname(__file__)
                if _CURRENT_DIR not in sys.path:
                    sys.path.insert(0, _CURRENT_DIR)
                from optimisers_for_harder_open_race import (
                    get_open_race_optimizers as get_harder_optimizers
                )  # noqa: E402

        open_race_optimizers = get_harder_optimizers(random_state=self.random_state)
        if optimizer_names is None:
            optimizers = list(open_race_optimizers.values())
        else:
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
        
        # Run competitions
        start_time = time.time()
        
        def run_competition_with_seed(seed, comp_idx=None, total_comps=None):
            if comp_idx is not None and total_comps is not None:
                logger.info(f"Starting Competition {comp_idx+1}/{total_comps}")
            
            # Create a fresh black box per competition for isolation and seeded noise
            local_black_box = HarderBlackBoxFunction(
                surrogate_model=black_box.surrogate_model,
                scaler=black_box.scaler,
                bounds=black_box.get_bounds(),
                difficulty_config=self.difficulty_config,
                random_state=seed
            )
            
            result = self.run_single_competition(
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
            # Sequential execution
            competitions = []
            for i in range(n_competitions):
                seed = self.random_state + i if self.random_state is not None else None
                result = run_competition_with_seed(seed, i, n_competitions)
                competitions.append(result)
        else:
            # Parallel execution
            seeds = [self.random_state + i if self.random_state is not None else None 
                    for i in range(n_competitions)]
            
            competitions = Parallel(n_jobs=n_jobs, verbose=0)(
                delayed(run_competition_with_seed)(seed, i, n_competitions)
                for i, seed in enumerate(seeds)
            )
        
        tournament_time = time.time() - start_time
        
        # Compile tournament results
        tournament_results = {
            'dataset_name': 'harder_open_race',
            'surrogate_type': self.surrogate_type,
            'difficulty_config': self.difficulty_config,
            'n_competitions': n_competitions,
            'S': S,
            'R': R,
            'B': B,
            'tournament_time': tournament_time,
            'competitions': competitions,
            'optimizer_names': [opt.name for opt in optimizers]
        }
        
        return tournament_results
    
    def save_results(self, tournament_results: Dict[str, Any], analysis: Dict[str, Any], 
                    dataset_name: str, surrogate_type: str) -> Tuple[str, str]:
        """
        Save tournament results and analysis to files
        
        Args:
            tournament_results: Tournament results dictionary
            analysis: Analysis results dictionary
            dataset_name: Name of the dataset
            surrogate_type: Type of surrogate model
            
        Returns:
            Tuple of (results_filepath, analysis_filepath)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory
        output_dir = "harder_competition_results_open_race"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results
        results_filename = f"harder_{dataset_name}_{surrogate_type}_{timestamp}_results.json"
        results_filepath = os.path.join(output_dir, results_filename)
        
        with open(results_filepath, 'w') as f:
            json.dump(tournament_results, f, indent=2, default=str)
        
        # Save analysis
        analysis_filename = f"harder_{dataset_name}_{surrogate_type}_{timestamp}_analysis.json"
        analysis_filepath = os.path.join(output_dir, analysis_filename)
        
        with open(analysis_filepath, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        logger.info(f"Results saved to: {results_filepath}")
        logger.info(f"Analysis saved to: {analysis_filepath}")
        
        return results_filepath, analysis_filepath


def run_harder_open_race_competition(dataset_name: str, 
                                   surrogate_type: str = 'random_forest',
                                   difficulty_config: Dict[str, Any] = None,
                                   optimizer_names: List[str] = None,
                                   n_competitions: int = 10,
                                   S: int = 200,
                                   R: int = 5,
                                   B: int = 1,
                                   n_jobs: int = -1,
                                   random_state: int = 42,
                                   save_results: bool = True) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Run harder open race competition using non-GP surrogate models
    
    Args:
        dataset_name: Name of the dataset to use
        surrogate_type: Type of surrogate model ('random_forest', 'neural_network', 'discontinuous')
        difficulty_config: Configuration for difficulty factors
        optimizer_names: List of optimizer names to include
        n_competitions: Number of competitions to run
        S: Total number of function evaluations per optimizer
        R: Number of initial random points
        B: Batch size for optimization
        n_jobs: Number of parallel jobs
        random_state: Random seed for reproducibility
        save_results: Whether to save results to files
        
    Returns:
        Tuple of (tournament_results, analysis)
    """
    
    print("HARDER OPEN RACE COMPETITION")
    print("="*80)
    print(f"Dataset: {dataset_name}")
    print(f"Surrogate type: {surrogate_type}")
    print(f"Competitions: {n_competitions}")
    print(f"Total evaluations (S): {S}")
    print(f"Initial points (R): {R}")
    print(f"Batch size (B): {B}")
    print(f"Parallel jobs: {n_jobs}")
    print(f"Random state: {random_state}")
    print("="*80)
    
    logger.info(f"Starting harder open race competition with {dataset_name}")
    
    # Create competition instance
    competition = HarderOpenRaceCompetition(
        surrogate_type=surrogate_type,
        difficulty_config=difficulty_config,
        random_state=random_state
    )
    
    # Load dataset and train harder surrogate model
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
    
    # Create black box function
    print("\n=== Creating Harder Black Box Function ===")
    black_box = competition.create_black_box_function(surrogate_model, scaler, X)
    print(f"Difficulty: {competition.get_difficulty_description()}")
    
    # Run tournament with harder black box function
    print(f"\n=== Running {n_competitions} Competitions ===")
    tournament_results = competition.run_harder_competition_tournament(
        black_box=black_box,
        optimizer_names=optimizer_names,
        n_competitions=n_competitions,
        S=S,
        R=R,
        B=B,
        n_jobs=n_jobs
    )
    
    # Analyze results
    print("\n=== Analyzing Results ===")
    analysis = competition.analyze_tournament_results(tournament_results)
    
    # Print summary
    competition.print_tournament_summary(analysis)
    
    # Save standardized results JSON and generate plot
    if save_results:
        try:
            root_dir = Path(__file__).resolve().parents[2]
            results_dir = root_dir / 'Results' / 'Hard_Mode' / 'Open_Race'
            results_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_name = f"{dataset_name}_hard_mode_open_race_{timestamp}"
            results_path = results_dir / f"{base_name}.json"

            payload = {
                'tournament_results': tournament_results,
                'analysis': analysis,
                'surrogate_type': surrogate_type,
                'difficulty_config': difficulty_config,
            }
            with open(results_path, 'w') as f:
                json.dump(payload, f, indent=2, default=str)
            logger.info(f"Results saved to: {results_path}")

            # Generate plot using helper
            plotting_dir = root_dir / 'Plotting' / 'Hard_Mode' / 'Open_Race'
            plotting_dir.mkdir(parents=True, exist_ok=True)
            plot_path = plotting_dir / f"{base_name}.png"

            from experiments.Visualization.Open_Race_best_so_far_plots import (
                extract_per_optimizer_histories, aggregate_histories, plot_best_so_far,
            )
            per_opt_histories, tr = extract_per_optimizer_histories(tournament_results)
            aggregated = aggregate_histories(per_opt_histories)
            title = f"Open Race (Hard): Best-so-far — {dataset_name}"
            plot_best_so_far(aggregated, title, plot_path)
            logger.info(f"Plot saved to: {plot_path}")
        except Exception as e:
            logger.warning(f"Failed to save standardized Hard Mode Open Race results/plots: {e}")
    
    return tournament_results, analysis


if __name__ == "__main__":
    # Example usage
    
    percentage_list = [0.95]  # List of hidden fractions to test (match AL_that_works)
    # working optimizers: ['RANDOM', 'BO_GP_EI', 'SMART_BO', 'SBO_GP_PV', 'SBO_ANN_PV', 'SBO_POLY_PV', 'SBO_GP_EI_TRUNCDE', 'DE_DIRECT', 'FULL_FACTORIAL', 'FRACTIONAL_FACTORIAL', 'PLACKETT_BURMAN', 'CENTRAL_COMPOSITE', 'BOX_BEHNKEN', 'LATIN_HYPERCUBE', 'D_OPTIMAL']
    # unavailable optimizers: ['GA_DIRECT', 'PSO_DIRECT'] (require DEAP and pyswarms libraries)
    # possible datasets: ['MOBO_dataset_rat_myocyte', 'DBO_dataset_rat_myocyte', 'df_Human_Hela_regular_mode', 'df_Human_Hela_timesaving_mode', 'df_Human_T_Cell_Expanded', 'df_Human_TF_Cell_Expanded', 'synthetic_2d', 'synthetic_5d', 'synthetic_10d']


    firsthalf = ['RANDOM','SBO_GP_PV', 'BO_GP_EI', 'SMART_BO', 'SBO_ANN_PV']
    secondhalf = ['DE_DIRECT', 'FULL_FACTORIAL', 'FRACTIONAL_FACTORIAL', 'PLACKETT_BURMAN', 'CENTRAL_COMPOSITE', 'BOX_BEHNKEN', 'LATIN_HYPERCUBE'] # removed 'D_OPTIMAL'
    all_optimisers = firsthalf + secondhalf


        # S: Total number of function evaluations per optimizer
        # R: Number of initial random points
        # B: Batch size for optimization


    # Test with harder open race competition
    results, analysis = run_harder_open_race_competition(
        dataset_name='df_Human_TF_Cell_Expanded',
        surrogate_type='random_forest',
        difficulty_config={
            'noise_level': 0.2,
            'heteroscedastic': True,
            'n_modes': 2,
            'n_discontinuities': 1
        },
        optimizer_names=all_optimisers,
        n_competitions=10,
        S=200,
        R=20,
        B=20,
        n_jobs=-1,
        random_state=42
    ) 
