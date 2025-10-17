#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hide-the-Label Competition Framework with Optimized GP for Better BO_GP_EI Performance

This module implements a competition framework where optimization strategies
compete to find the target (maximum value) in a dataset with minimal queries.

Uses optimized Gaussian Process surrogate modeling:
- RBF + Matern ARD kernel combination for all datasets
- Automatic Relevance Determination (ARD) for feature importance
- Enhanced hyperparameter optimization (5 restarts)
- Full dataset usage (no subsampling)
- Optimized for BO_GP_EI performance to match AL_that_works results
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
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF
from scipy.stats import qmc
from tqdm import tqdm
import warnings
import hashlib
import pickle
warnings.filterwarnings('ignore')

# Configure Additional outputs directory and logging
ROOT_DIR = Path(__file__).resolve().parents[2]
ADDITIONAL_OUTPUTS_DIR = ROOT_DIR / 'Additional outputs'
os.makedirs(ADDITIONAL_OUTPUTS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(ADDITIONAL_OUTPUTS_DIR / 'hide_the_label_competition_incremental_gp.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Mode tag for standardized cache naming
MODE_TAG = 'Regular_Mode_Hide_The_Label'

# Local imports
from utils.datasets import load_dataset, get_available_datasets, Dataset, SyntheticDataset
from utils.optimizers import get_all_optimizers, get_optimizer_by_name, OptimizationStrategy
from experiments.Visualization.Hide_The_Label_bar_plots import (
    extract_from_optimizer_stats,
    extract_from_competitions,
    plot_bar,
)

# Import Incremental GP
try:
    # Temporarily disable to avoid NumPy compatibility issues
    # from speed_up_attempts.fast_surrogates import IncrementalGPSurrogate
    INCREMENTAL_GP_AVAILABLE = False
    logger.warning("Incremental GP temporarily disabled due to NumPy compatibility")
    # Add dummy class to avoid import errors
    class IncrementalGPSurrogate:
        pass
except ImportError:
    logger.warning("Incremental GP not available, falling back to standard GP")
    INCREMENTAL_GP_AVAILABLE = False
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF
    # Add dummy class to avoid import errors
    class IncrementalGPSurrogate:
        pass

# Import Sparse GP Approximations
try:
    from utils.sparse_gp_approximations import (
        SparseGPFactory,
        NystromGPSurrogate,
        FITCGPSurrogate,
        VFEGPSurrogate,
    )
    SPARSE_GP_AVAILABLE = True
    logger.info("Sparse GP approximations available")
except ImportError:
    logger.warning("Sparse GP approximations not available, falling back to standard methods")
    SPARSE_GP_AVAILABLE = False


class HideLabelCompetitionIncrementalGP:
    """
    Main class for organizing hide-the-label competitions with optimized GP for better BO_GP_EI performance.
    
    This class implements a competition framework where different optimization strategies compete to find
    the maximum value (target) in a dataset with minimal queries. It uses optimized Gaussian Process
    surrogate modeling to generate synthetic datasets and evaluate optimizer performance.
    """
    
    def __init__(self, random_state: int = 42, use_gpu: bool = True, cache_dir: str = "model_cache"):
        """
        Initialize the competition framework.
        
        Args:
            random_state: Random seed for reproducibility
            use_gpu: Whether to use GPU acceleration for computations
            cache_dir: Directory to store cached models
        """
        self.random_state = random_state
        self.np_random = np.random.RandomState(random_state)
        self.start_time = time.time()
        self.use_gpu = use_gpu
        # Place cache under Additional outputs unless an absolute path is provided
        cache_path = Path(cache_dir)
        if not cache_path.is_absolute():
            cache_path = ADDITIONAL_OUTPUTS_DIR / cache_path
        self.cache_dir = str(cache_path)
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Store diagnostics from last fast-surrogate selection (label, r2, fit_time, eval_time, total_time)
        self.last_fast_surrogate_trials: List[Dict[str, Any]] = []
        
        # Optimal kernel configurations from GP optimization results
        self.optimal_kernels = {
            'MOBO_dataset_rat_myocyte': {
                'kernel_type': 'rbf',
                'noise_level': 0.1,
                'normalize_y': True,
                'alpha_mode': 'data_variance'
            },
            'DBO_dataset_rat_myocyte': {
                'kernel_type': 'rbf',
                'noise_level': 0.1,
                'normalize_y': False,
                'alpha_mode': 'data_variance'
            },
            'df_Human_Hela_regular_mode': {
                'kernel_type': 'rbf',
                'noise_level': 0.1,
                'normalize_y': False,
                'alpha_mode': 'data_variance'
            },
            'df_Human_Hela_timesaving_mode': {
                'kernel_type': 'rbf',
                'noise_level': 0.01,
                'normalize_y': False,
                'alpha_mode': 'data_variance'
            },
            'df_Human_T_Cell_Expanded': {
                'kernel_type': 'rbf',
                'noise_level': 0.05,
                'normalize_y': True,
                'alpha_mode': 'data_variance'
            },
            'df_Human_TF_Cell_Expanded': {
                'kernel_type': 'rbf',
                'noise_level': 0.05,
                'normalize_y': True,
                'alpha_mode': 'data_variance'
            }
        }
        
    def _log_timing(self, step_name: str, level: str = "info"):
        """
        Log elapsed time since the start of the competition.
        
        Args:
            step_name: Name of the current step being logged
            level: Logging level (debug, info, warning)
        """
        elapsed = time.time() - self.start_time
        message = f"[{elapsed:.2f}s] {step_name}"
        if level == "debug":
            logger.debug(message)
        elif level == "warning":
            logger.warning(message)
        else:
            logger.info(message)
    
    def _generate_model_cache_key(self, X: np.ndarray, y: np.ndarray, dataset_name: str) -> str:
        """
        Generate a unique cache key for the model based on data and dataset name.
        
        Args:
            X: Feature matrix
            y: Target values
            dataset_name: Name of the dataset
            
        Returns:
            Unique cache key string
        """
        # Create a hash of the data shape and first few values for uniqueness
        data_hash = hashlib.md5()
        data_hash.update(f"{X.shape}_{y.shape}_{dataset_name}_{self.random_state}".encode())
        data_hash.update(X.flatten()[:100].tobytes())  # First 100 values
        data_hash.update(y.flatten()[:100].tobytes())  # First 100 values
        
        return f"gp_model_{data_hash.hexdigest()[:16]}"
    
    def _get_cached_model_path(self, cache_key: str) -> str:
        """
        Get the file path for a cached model.
        
        Args:
            cache_key: Unique cache key for the model
            
        Returns:
            File path for the cached model
        """
        return os.path.join(self.cache_dir, f"{cache_key}.joblib")
    
    def _load_cached_model(self, cache_key: str) -> Optional[Tuple[Any, StandardScaler, float, Optional[Dict[str, Any]]]]:
        """
        Load a cached model if it exists. Supports legacy filenames and new metadata-named files.
        
        Returns:
            Tuple of (gp_model, scaler, r2_score, metadata_dict|None) if found, None otherwise
        """
        # Collect candidate files: exact legacy name or any file starting with cache_key__
        candidates = []
        try:
            for file in os.listdir(self.cache_dir):
                if not file.endswith('.joblib'):
                    continue
                if file == f"{cache_key}.joblib" or file.startswith(f"{cache_key}__"):
                    path = os.path.join(self.cache_dir, file)
                    try:
                        mtime = os.path.getmtime(path)
                    except Exception:
                        mtime = 0
                    candidates.append((mtime, path))
        except Exception as e:
            logger.warning(f"Failed to list cache directory: {e}")
            candidates = []
        
        # Fallback to exact legacy path if directory listing failed
        if not candidates:
            legacy_path = self._get_cached_model_path(cache_key)
            if os.path.exists(legacy_path):
                candidates = [(os.path.getmtime(legacy_path), legacy_path)]
        
        if not candidates:
            return None
        
        # Load the most recent candidate
        candidates.sort(key=lambda t: t[0], reverse=True)
        _, chosen_path = candidates[0]
        try:
            logger.info(f"Loading cached model from: {chosen_path}")
            cached_data = load(chosen_path)
            metadata = None
            # Support both 3-tuple and 4-tuple payloads
            if isinstance(cached_data, tuple) and len(cached_data) >= 3:
                gp_model = cached_data[0]
                scaler = cached_data[1]
                r2_score = cached_data[2]
                if len(cached_data) >= 4 and isinstance(cached_data[3], dict):
                    metadata = cached_data[3]
            else:
                logger.warning("Cached model format unrecognized; skipping load")
                return None
            logger.info(f"Successfully loaded cached model with R²: {float(r2_score):.4f}")
            return gp_model, scaler, float(r2_score), metadata
        except Exception as e:
            logger.warning(f"Failed to load cached model: {e}")
            return None
    
    def _save_cached_model(self, cache_key: str, gp_model: Any, scaler: StandardScaler, r2_score: float, metadata: Optional[Dict[str, Any]] = None):
        """
        Save a fitted model to cache. Writes metadata into filename and payload when provided.
        """
        # Build filename with optional dataset and timestamp for discoverability
        dataset_part = (metadata or {}).get('dataset_name') or 'unknown'
        created_part = (metadata or {}).get('created_at') or datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_dataset = str(dataset_part).replace(' ', '_')
        # Include mode tag in filename for clarity
        filename = f"{cache_key}__{MODE_TAG}__{safe_dataset}__{created_part}.joblib"
        cache_path = os.path.join(self.cache_dir, filename)
        
        try:
            cached_data = (gp_model, scaler, float(r2_score), metadata or {})
            dump(cached_data, cache_path)
            logger.info(f"Saved model to cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save model to cache: {e}")
    
    def clear_model_cache(self):
        """
        Clear all cached models from the cache directory.
        
        This method removes all cached model files, forcing new model fitting on next run.
        """
        if os.path.exists(self.cache_dir):
            for file in os.listdir(self.cache_dir):
                if file.endswith('.joblib'):
                    file_path = os.path.join(self.cache_dir, file)
                    try:
                        os.remove(file_path)
                        logger.info(f"Removed cached model: {file}")
                    except Exception as e:
                        logger.warning(f"Failed to remove cached model {file}: {e}")
            logger.info(f"Cleared model cache directory: {self.cache_dir}")
        else:
            logger.info("Cache directory does not exist, nothing to clear")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about the model cache.
        
        Returns:
            Dictionary containing cache information
        """
        cache_info = {
            'cache_dir': self.cache_dir,
            'exists': os.path.exists(self.cache_dir),
            'cached_models': []
        }
        
        if cache_info['exists']:
            for file in os.listdir(self.cache_dir):
                if file.endswith('.joblib'):
                    file_path = os.path.join(self.cache_dir, file)
                    file_size = os.path.getsize(file_path)
                    # Attempt to parse dataset and timestamp from filename
                    dataset = None
                    created_at = None
                    try:
                        base = file[:-7]  # strip .joblib
                        parts = base.split('__')
                        if len(parts) >= 3:
                            # pattern: cache_key__dataset__YYYYmmdd_HHMMSS
                            dataset = parts[1]
                            created_at = parts[2]
                    except Exception:
                        pass
                    cache_info['cached_models'].append({
                        'filename': file,
                        'size_mb': file_size / (1024 * 1024),
                        'path': file_path,
                        'dataset': dataset,
                        'created_at': created_at
                    })
        
        return cache_info
    
    def _create_optimal_kernel(self, dataset_name: str, n_features: int):
        """
        Create optimal kernel configuration for the dataset using RBF + Matern ARD combination.
        
        This method creates a sophisticated kernel that combines RBF and Matern kernels with
        Automatic Relevance Determination (ARD) to allow the GP to learn which features are
        most important for prediction.
        
        Args:
            dataset_name: Name of the dataset (used for potential dataset-specific tuning)
            n_features: Number of features in the dataset
            
        Returns:
            Combined kernel object for Gaussian Process regression
        """
        # Use the same sophisticated kernel as AL_that_works version for better BO_GP_EI performance
        # Initialize with short length scales to allow optimization to find relevant dimensions
        initial_length_scales = np.ones(n_features) * 0.1
        
        # Create combination kernel: RBF + Matern with different length scales per dimension (ARD)
        rbf_kernel = 1.0 * RBF(
            length_scale=initial_length_scales.copy(), 
            length_scale_bounds=(1e-2, 1e2)
        )
        
        matern_kernel = 1.0 * Matern(
            length_scale=initial_length_scales.copy(), 
            nu=2.5, 
            length_scale_bounds=(1e-2, 1e2)
        )
        
        # Additive combination + WhiteKernel for noise
        kernel = (rbf_kernel + matern_kernel + 
                 WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-3, 1.0)))
        
        logger.info(f"Using optimized RBF + Matern ARD kernel with {n_features} dimensions")
        return kernel
        
    def fit_surrogate_model(self, X: np.ndarray, y: np.ndarray, y_var: np.ndarray, dataset_name: str = None, use_cache_model: bool = True) -> Tuple[Any, StandardScaler, float]:
        """
        Fit a surrogate model to the dataset using optimized GP for better BO_GP_EI performance.
        
        This method trains a Gaussian Process surrogate model on the provided dataset using
        optimized hyperparameters and kernel configurations. The model is designed to provide
        high-quality predictions for Bayesian Optimization algorithms.
        
        The method can optionally use cached models or always fit a new model based on the use_cache_model parameter.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
            y_var: Variance estimates for each target value (n_samples,)
            dataset_name: Name of the dataset (optional, used for dataset-specific tuning)
            use_cache_model: Whether to use cached models if available (default: True)
            
        Returns:
            Tuple of (fitted surrogate model, feature scaler, R^2 score on training data)
        """
        self._log_timing("Starting optimized surrogate model fitting")
        
        n_samples, n_features = X.shape
        logger.info(f"Dataset: {n_samples} samples × {n_features} features")
        
        # Generate cache key for this dataset
        cache_key = self._generate_model_cache_key(X, y, dataset_name or "unknown")
        logger.info(f"Model cache key: {cache_key}")
        
        # Try to load cached model first (only if use_cache_model is True)
        if use_cache_model:
            cached_result = self._load_cached_model(cache_key)
            if cached_result is not None:
                gp_model, scaler, r2, meta = cached_result
                self._log_timing("Loaded cached surrogate model")
                return gp_model, scaler, r2
            else:
                logger.info("No cached model found, fitting new model...")
        else:
            logger.info("Cache disabled, fitting new model...")
        
        # If no cached model found, fit a new one
        logger.info("No cached model found, fitting new model...")
        self._log_timing("Starting new model fitting")
        
        # FAST PATH: For large biological datasets (T and TF), use a fast GP path
        # Leave behavior unchanged for other datasets
        if dataset_name in ['df_Human_T_Cell_Expanded', 'df_Human_TF_Cell_Expanded']:
            logger.info(f"Using fast_full surrogate for dataset: {dataset_name}")
            # Reset diagnostics and record timing for transparency
            self.last_fast_surrogate_trials = []
            start_fit = time.time()
            gp_model, scaler = self._fit_fast_full_gp_model(X, y, y_var, dataset_name)
            fit_time = time.time() - start_fit
            self.last_fast_surrogate_trials.append({'label': 'fast_full', 'fit_time': fit_time})
        
        else:
            # Default behavior (unchanged): optimized full GP on all data
            # Ensures BO_GP_EI gets the best possible surrogate model
            gp_model, scaler = self._fit_standard_gp_model(X, y, y_var, dataset_name)
        
        # Evaluate R^2 on training data (use scaled features since model was trained on scaled data)
        try:
            if hasattr(gp_model, 'predict'):
                X_scaled = scaler.transform(X)  # Scale the features for prediction
                y_pred = gp_model.predict(X_scaled)
                if isinstance(y_pred, tuple):
                    y_pred = y_pred[0]
                r2 = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))
            else:
                r2 = float('nan')
        except Exception as e:
            logger.warning(f"Failed to compute R^2 for full-data model: {e}")
            r2 = float('nan')
        
        logger.info(f"Full-data surrogate model R^2: {r2:.4f}")
        
        # Cache the fitted model for future use
        # Attach metadata (dataset and creation time) when saving
        meta = {
            'dataset_name': dataset_name or 'unknown',
            'created_at': datetime.now().strftime('%Y%m%d_%H%M%S')
        }
        self._save_cached_model(cache_key, gp_model, scaler, r2, metadata=meta)
        
        self._log_timing("Finished fitting and caching surrogate model")
        return gp_model, scaler, r2
    
    def _fit_sparse_gp_model(self, X: np.ndarray, y: np.ndarray, y_var: np.ndarray, dataset_name: str = None, n_inducing_points_override: Optional[int] = None) -> Tuple[Any, StandardScaler]:
        """
        Fit a sparse GP model for large datasets.
        
        This method uses sparse Gaussian Process approximations (Nystrom, FITC, or VFE) to handle
        large datasets efficiently by using a subset of inducing points instead of the full dataset.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
            y_var: Variance estimates for each target value (n_samples,)
            dataset_name: Name of the dataset (used to select optimal sparse GP type)
            
        Returns:
            Tuple of (fitted sparse GP model, feature scaler)
        """
        self._log_timing("Starting sparse GP model fitting")
        
        n_samples, n_features = X.shape
        logger.info(f"Fitting sparse GP for {n_samples} samples × {n_features} features")
        
        # Determine optimal sparse GP type and parameters based on dataset
        if dataset_name == 'df_Human_T_Cell_Expanded':
            sparse_type = 'fitc'  # FITC works well for biological data
            n_inducing_points = min(200, n_samples // 10)  # 10% of data as inducing points
        elif dataset_name == 'df_Human_TF_Cell_Expanded':
            sparse_type = 'vfe'   # VFE for better uncertainty estimates
            n_inducing_points = min(200, n_samples // 10)
        else:
            sparse_type = 'nystrom'  # Default for other large datasets
            n_inducing_points = min(150, n_samples // 15)

        # Allow override of inducing point count for model selection sweeps
        if n_inducing_points_override is not None:
            n_inducing_points = int(max(5, min(n_inducing_points_override, n_samples)))
        
        # Get optimal settings for this dataset
        if dataset_name and dataset_name in self.optimal_kernels:
            config = self.optimal_kernels[dataset_name]
            kernel_type = config.get('kernel_type', 'rbf')
            noise_level = config.get('noise_level', 0.1)
        else:
            kernel_type = 'rbf'
            noise_level = 0.1
        
        logger.info(f"Using {sparse_type.upper()} sparse GP with {n_inducing_points} inducing points")
        logger.info(f"Settings: kernel_type={kernel_type}, noise_level={noise_level}")
        
        # Create and fit sparse GP model
        if sparse_type == 'nystrom':
            gp_model = NystromGPSurrogate(
                n_inducing_points=n_inducing_points,
                kernel_type=kernel_type,
                noise_level=noise_level,
                random_state=self.random_state,
                use_gpu=self.use_gpu
            )
        elif sparse_type == 'fitc':
            gp_model = FITCGPSurrogate(
                n_inducing_points=n_inducing_points,
                kernel_type=kernel_type,
                noise_level=noise_level,
                random_state=self.random_state,
                use_gpu=self.use_gpu
            )
        elif sparse_type == 'vfe':
            gp_model = VFEGPSurrogate(
                n_inducing_points=n_inducing_points,
                kernel_type=kernel_type,
                noise_level=noise_level,
                random_state=self.random_state,
                use_gpu=self.use_gpu
            )
        else:
            raise ValueError(f"Unknown sparse GP type: {sparse_type}")
        
        # Fit the model
        gp_model.fit(X, y, y_var)
        
        self._log_timing("Finished sparse GP model fitting")
        
        # Evaluate the fitted model
        y_pred = gp_model.predict(X)
        if isinstance(y_pred, tuple):
            y_pred = y_pred[0]  # Extract mean prediction
            
        mse = np.mean((y - y_pred) ** 2)
        r2 = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))
        
        logger.info(f"Sparse GP Performance - R²: {r2:.4f}, MSE: {mse:.4f}")
        logger.info(f"Using {n_inducing_points} inducing points from {n_samples} total samples")
        
        return gp_model, gp_model.scaler
    
    def _fit_subsampled_gp_model(self, X: np.ndarray, y: np.ndarray, y_var: np.ndarray, dataset_name: str = None, n_subsample_override: Optional[int] = None) -> Tuple[Any, StandardScaler]:
        """
        Fit a subsampled GP model for large datasets (fallback method).
        
        This method is used as a fallback when sparse GP methods are not available. It subsamples
        the dataset to a manageable size and fits a standard GP model on the subsampled data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
            y_var: Variance estimates for each target value (n_samples,)
            dataset_name: Name of the dataset (used for optimal settings)
            
        Returns:
            Tuple of (fitted GP model, feature scaler)
        """
        self._log_timing("Starting subsampled GP model fitting")
        
        n_samples, n_features = X.shape
        logger.info(f"Subsampling large dataset: {n_samples} samples × {n_features} features")
        
        # Use stratified sampling to maintain data distribution
        if n_subsample_override is not None:
            n_subsample = int(max(50, min(n_subsample_override, n_samples)))
        else:
            n_subsample = min(600, n_samples)  # Default cap at 600 samples
        
        # Smart subsampling: keep diverse points
        indices = self.np_random.choice(n_samples, n_subsample, replace=False)
        X_sub = X[indices]
        y_sub = y[indices]
        y_var_sub = y_var[indices]
        
        logger.info(f"Subsampled to: {n_subsample} samples ({n_subsample/n_samples*100:.1f}%)")
        
        # Feature scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_sub)
        logger.debug("Features scaled using StandardScaler")
        
        # Get optimal settings for this dataset
        if dataset_name and dataset_name in self.optimal_kernels:
            config = self.optimal_kernels[dataset_name]
            kernel_type = config.get('kernel_type', 'rbf')
            noise_level = config.get('noise_level', 0.1)
            normalize_y = config.get('normalize_y', True)
        else:
            kernel_type = 'rbf'
            noise_level = 0.1
            normalize_y = True
            
        logger.debug(f"Using subsampled GP settings: kernel_type={kernel_type}, noise_level={noise_level}, normalize_y={normalize_y}")
        
        if INCREMENTAL_GP_AVAILABLE:
            # Use Incremental GP
            gp_model = IncrementalGPSurrogate(
                kernel_type=kernel_type,
                noise_level=noise_level,
                normalize_y=normalize_y,
                random_state=self.random_state,
                use_gpu=self.use_gpu,
                update_frequency=10
            )
            
            # Fit the model
            gp_model.fit(X_sub, y_sub, y_var_sub)
            
        else:
            # Fallback to standard GP
            logger.warning("Using standard GP as fallback")
            kernel = self._create_optimal_kernel(dataset_name, n_features)
            
            gp_model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=y_var_sub + 1e-6,
                normalize_y=normalize_y,
                n_restarts_optimizer=3,
                random_state=self.random_state
            )
            
            gp_model.fit(X_scaled, y_sub)
        
        self._log_timing("Finished subsampled GP model fitting")

        # Evaluate the fitted model
        if INCREMENTAL_GP_AVAILABLE:
            y_pred = gp_model.predict(X_sub)
            if isinstance(y_pred, tuple):
                y_pred = y_pred[0]  # Extract mean prediction
        else:
            y_pred, _ = gp_model.predict(X_scaled, return_std=True)
            
        mse = np.mean((y_sub - y_pred) ** 2)
        r2 = 1 - (np.sum((y_sub - y_pred) ** 2) / np.sum((y_sub - np.mean(y_sub)) ** 2))
        
        logger.info(f"Subsampled GP Performance - R²: {r2:.4f}, MSE: {mse:.4f}")
        logger.debug(f"Training samples: {len(y_sub)}/{len(y)}")
        
        return gp_model, scaler
    
    def _fit_standard_gp_model(self, X: np.ndarray, y: np.ndarray, y_var: np.ndarray, dataset_name: str = None) -> Tuple[Any, StandardScaler]:
        """
        Fit a standard GP model using optimized configuration for better BO_GP_EI performance.
        
        This method fits a full Gaussian Process model using the complete dataset with optimized
        hyperparameters and kernel configurations. It's the primary method used for surrogate
        modeling in the competition framework.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
            y_var: Variance estimates for each target value (n_samples,)
            dataset_name: Name of the dataset (used for optimal settings)
            
        Returns:
            Tuple of (fitted GP model, feature scaler)
        """
        self._log_timing("Starting optimized GP model fitting")
        
        n_samples, n_features = X.shape
        logger.info(f"Using full dataset: {n_samples} samples × {n_features} features")
        
        # Feature scaling (critical for ARD kernels)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        logger.info("Features scaled using StandardScaler")
        
        # Use the same optimized kernel configuration as AL_that_works version
        kernel = self._create_optimal_kernel(dataset_name, n_features)
        
        # Enhanced GP configuration for better performance
        gp_model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=y_var + 1e-6,  # Keep using data variance as noise (this worked well)
            normalize_y=True,    # This was optimal in our tests
            n_restarts_optimizer=5,  # Increased for better hyperparameter optimization
            random_state=self.random_state
        )
        
        # Fit on scaled features
        gp_model.fit(X_scaled, y)
        self._log_timing("Finished optimized GP model fitting")

        # Evaluate the fitted model
        y_pred, sigma = gp_model.predict(X_scaled, return_std=True)
        mse = np.mean((y - y_pred) ** 2)
        r2 = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))
        log_ml = gp_model.log_marginal_likelihood()
        
        logger.info(f"OPTIMIZED GP PERFORMANCE:")
        logger.info(f"  Log-Marginal-Likelihood: {log_ml:.3f}")
        logger.info(f"  Mean Squared Error: {mse:.4f}")
        logger.info(f"  R^2 value: {r2:.4f}")
        logger.info(f"  Fitted kernel: {gp_model.kernel_}")
        
        # Analyze learned length scales for feature importance
        if hasattr(gp_model.kernel_, 'k1') and hasattr(gp_model.kernel_.k1, 'length_scale'):
            rbf_scales = gp_model.kernel_.k1.length_scale
            matern_scales = gp_model.kernel_.k2.length_scale
            
            # Identify important features (small length scales = important)
            rbf_important = rbf_scales < 10.0
            matern_important = matern_scales < 10.0
            combined_important = rbf_important | matern_important
            
            n_important = np.sum(combined_important)
            logger.info(f"  Identified {n_important}/{n_features} important features")
            if n_important > 0 and n_important <= 10:  # Only show if reasonable number
                important_indices = np.where(combined_important)[0]
                logger.info(f"  Important feature indices: {important_indices.tolist()}")
        
        return gp_model, scaler
    
    def _fit_fast_full_gp_model(self, X: np.ndarray, y: np.ndarray, y_var: np.ndarray, dataset_name: str = None) -> Tuple[Any, StandardScaler]:
        """
        Fit a simplified full GP quickly (RBF + White, 0 restarts).
        Intended as a fast alternative when subsampled/sparse do not achieve target R^2.
        """
        self._log_timing("Starting fast full GP model fitting")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(0.1, 10.0)) + WhiteKernel(0.1)
        gp_model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=y_var + 1e-6,
            normalize_y=True,
            n_restarts_optimizer=0,
            random_state=self.random_state
        )
        gp_model.fit(X_scaled, y)
        self._log_timing("Finished fast full GP model fitting")
        return gp_model, scaler

    def _compute_full_data_r2(self, gp_model: Any, scaler: Optional[StandardScaler], X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute R^2 of a surrogate model on the full dataset.
        Handles both sklearn GP (expects scaled X) and custom sparse GP (does internal scaling).
        """
        try:
            y_pred = None
            # First, try prediction assuming sklearn GP that expects scaled features
            if scaler is not None and hasattr(scaler, 'transform'):
                try:
                    X_scaled = scaler.transform(X)
                    y_pred = gp_model.predict(X_scaled)
                except Exception:
                    y_pred = None
            # Fallback: predict on raw X (sparse surrogates do internal scaling)
            if y_pred is None:
                y_pred = gp_model.predict(X)
            if isinstance(y_pred, tuple):
                y_pred = y_pred[0]
            r2 = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))
            return float(r2)
        except Exception as e:
            logger.warning(f"Failed to compute full-data R^2 for model: {e}")
            return float('nan')

    def generate_candidate_pool(self, gp_model: Any, scaler: StandardScaler,
                               original_X: np.ndarray, n_points: int = 200, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate candidate pool of points using the fitted surrogate model.
        
        This method creates a synthetic dataset by sampling from the fitted Gaussian Process model.
        It uses Latin Hypercube sampling to ensure good coverage of the input space and then
        evaluates the GP model at these points to create the synthetic dataset.
        
        Args:
            gp_model: Fitted surrogate model (optimized GP)
            scaler: Feature scaler used for model training
            original_X: Original dataset features (for dimensionality and bounds)
            n_points: Number of points to generate in the candidate pool
            seed: Random seed for reproducibility (optional)
            
        Returns:
            Tuple of (X_pool, y_pool) where X_pool are the candidate points and y_pool are the function values
        """
        self._log_timing("Starting candidate pool generation")
        # Use provided seed or default to self.random_state
        if seed is None:
            seed = self.random_state
        # Generate Latin Hypercube design using provided random state
        sampler = qmc.LatinHypercube(d=original_X.shape[1], seed=seed)
        X_pool = sampler.random(n_points)
        # Scale to match original data range
        X_min = np.min(original_X, axis=0)
        X_max = np.max(original_X, axis=0)
        X_pool = X_min + (X_max - X_min) * X_pool
        # IMPORTANT: Scale features using the same scaler as model training
        X_pool_scaled = scaler.transform(X_pool)
        # Sample function values from optimized GP model using provided random state
        y_pool = gp_model.sample_y(X_pool_scaled, 1, random_state=seed).flatten()
        self._log_timing("Finished candidate pool generation")
        return X_pool, y_pool
    
    def run_single_competition(self, X_pool: np.ndarray, y_pool: np.ndarray,
                              optimizers: Dict[str, OptimizationStrategy],
                              hidden_fraction: float = 0.9,
                              batch_size: int = 1,
                              seed: int = None,
                              verbose: bool = True) -> Dict[str, Any]:
        """
        Run a single competition between optimizers.
        
        This method implements the core competition logic where different optimization strategies
        compete to find the target (maximum value) in the dataset. A fraction of the data is
        hidden initially, and optimizers take turns selecting points to reveal until the target
        is found.
        
        Args:
            X_pool: Pool of candidate points (synthetic dataset features)
            y_pool: Function values for candidate points (synthetic dataset targets)
            optimizers: Dictionary of optimizer instances to compete
            hidden_fraction: Fraction of data to hide initially (e.g., 0.9 means 90% hidden)
            batch_size: Number of points to select per optimization step
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary containing detailed competition results for all optimizers
        """
        # Ensure hidden_fraction is a float
        if isinstance(hidden_fraction, list):
            hidden_fraction = float(hidden_fraction[0])
        # FIX: Use local RNG instead of overwriting object-level RNG
        if seed is not None:
            local_rng = np.random.RandomState(seed)
        else:
            local_rng = self.np_random
        
        # Find target (point with highest value)
        target_idx = np.argmax(y_pool)
        target_value = y_pool[target_idx]
        
        # Determine which points to hide
        n_points = len(y_pool)
        n_hidden = int(n_points * hidden_fraction)
        
        # CRITICAL FIX: Ensure target is always hidden
        target_idx = np.argmax(y_pool)
        
        # Force target to be in hidden set
        other_indices = list(range(n_points))
        other_indices.remove(target_idx)
        
        # Select n_hidden-1 other points to hide (excluding target)
        if n_hidden > 1:
            other_hidden = local_rng.choice(other_indices, n_hidden-1, replace=False)
            hidden_indices = np.concatenate([[target_idx], other_hidden])
        else:
            hidden_indices = np.array([target_idx])
        
        # Revealed indices are the complement
        revealed_indices = np.array([i for i in range(n_points) if i not in hidden_indices])
        
        # Convert hidden_indices to list for easier manipulation
        hidden_indices = hidden_indices.tolist()
        
        if verbose:
            logger.info(f"=== COMPETITION SETUP DETAILS ===")
            logger.info(f"  Total points: {n_points}")
            logger.info(f"  Hidden points: {len(hidden_indices)}")
            logger.info(f"  Revealed points: {len(revealed_indices)}")
            logger.info(f"  Target index: {target_idx}")
            logger.info(f"  Target value: {target_value:.4f}")
            logger.info(f"  Hidden indices: {sorted(hidden_indices)}")
            logger.info(f"  Revealed indices: {sorted(revealed_indices.tolist())}")
            logger.info(f"  Target in hidden: {target_idx in hidden_indices}")
        
        # ALWAYS log target information for debugging (but only if verbose)
        if verbose:
            logger.info(f"[Competition] Target setup: target_idx={target_idx}, target_value={target_value:.4f}, target_in_hidden={target_idx in hidden_indices}")
        
        # Initialize competition state
        competition_state = {
            'X_pool': X_pool,
            'y_pool': y_pool,
            'hidden_indices': hidden_indices,
            'revealed_indices': revealed_indices.tolist(),
            'target_idx': int(target_idx),
            'target_value': float(target_value),
            'current_best': float(np.max(y_pool[revealed_indices])) if len(revealed_indices) > 0 else float('-inf'),
            'n_initial_revealed': len(revealed_indices)
        }
        
        # Run each optimizer
        optimizer_results = {}
        
        for optimizer_name, optimizer in optimizers.items():
            logger.debug(f"Running optimizer: {optimizer_name}")
            
            # Reset optimizer state
            optimizer.reset()
            
            # FIXED: Clear global GP cache between competitions to prevent state persistence
            from utils.optimizers import _GP_CACHE
            _GP_CACHE.clear()
            
            # CRITICAL FIX: Each optimizer gets a fresh copy of hidden indices
            hidden_indices = competition_state['hidden_indices'].copy()
            revealed_indices = competition_state['revealed_indices'].copy()
            
            # Initialize revealed data
            X_observed = X_pool[revealed_indices]
            y_observed = y_pool[revealed_indices]
            
            # Evaluate surrogate model R^2 at each step (for BO_GP_EI)
            surrogate_r2_history = []
            steps_to_target = 0
            found_target = False
            optimization_history = []
            
            # Run optimization until target is found (SUPPORTING BATCH PROCESSING)
            while not found_target and len(hidden_indices) > 0:
                # Calculate current_best locally for THIS optimizer only
                current_best_local = float(np.max(y_observed)) if len(y_observed) > 0 else float('-inf')
                
                # Determine actual batch size for this step
                actual_batch_size = min(batch_size, len(hidden_indices))
                
                # Select next batch of points
                try:
                    if batch_size == 1:
                        # Single point selection (original behavior)
                        next_indices = [optimizer.select_next_point(
                            X_pool=X_pool,
                            hidden_indices=hidden_indices,
                            X_observed=X_observed,
                            y_observed=y_observed,
                            current_best=current_best_local
                        )]
                    else:
                        # Batch selection
                        next_indices = optimizer.select_next_batch(
                            X_pool=X_pool,
                            hidden_indices=hidden_indices,
                            X_observed=X_observed,
                            y_observed=y_observed,
                            current_best=current_best_local,
                            batch_size=actual_batch_size
                        )
                    
                    if verbose:
                        logger.info(f"[Competition] Optimizer '{optimizer_name}' selected next indices: {next_indices}")
                        print(f"[Competition] Optimizer '{optimizer_name}' selected next indices: {next_indices}")
                
                except Exception as e:
                    logger.warning(f"[Competition] Exception in optimizer '{optimizer_name}' at step {steps_to_target}: {e}", exc_info=True)
                    # Fallback to random selection
                    next_indices = local_rng.choice(hidden_indices, size=actual_batch_size, replace=False).tolist()
                
                # Validate and sanitize selections: ensure unique, unseen, and correct batch size
                if next_indices is None:
                    next_indices = []
                # Coerce to list of ints
                if not isinstance(next_indices, (list, tuple, np.ndarray)):
                    next_indices = [int(next_indices)]
                next_indices = [int(i) for i in list(next_indices)]

                # Remove duplicates while preserving order
                seen = set()
                deduped = []
                for idx in next_indices:
                    if idx not in seen:
                        deduped.append(idx)
                        seen.add(idx)

                # Keep only indices that are still hidden
                proposed = [idx for idx in deduped if idx in hidden_indices]

                # Top-up with random unseen points to reach the required batch size
                needed = actual_batch_size - len(proposed)
                if needed > 0:
                    remaining_choices = [idx for idx in hidden_indices if idx not in proposed]
                    if len(remaining_choices) >= needed:
                        rand_fill = local_rng.choice(remaining_choices, size=needed, replace=False).tolist()
                        proposed.extend(rand_fill)
                    else:
                        # Fallback: if optimizer returned too many invalids and we somehow can't fill, sample fresh
                        proposed = local_rng.choice(hidden_indices, size=actual_batch_size, replace=False).tolist()

                valid_indices = proposed[:actual_batch_size]
                if len(valid_indices) < actual_batch_size:
                    # Final guard
                    fill_needed = actual_batch_size - len(valid_indices)
                    extras = local_rng.choice([idx for idx in hidden_indices if idx not in valid_indices], size=fill_needed, replace=False).tolist()
                    valid_indices.extend(extras)
                
                # Record batch of points
                batch_history = []
                for i, next_idx in enumerate(valid_indices):
                    batch_history.append({
                        'step': steps_to_target,
                        'batch_position': i,
                        'selected_idx': int(next_idx),
                        'selected_value': float(y_pool[next_idx]),
                        'current_best': float(current_best_local),
                        'hidden_remaining': len(hidden_indices),
                        'found_target': found_target
                    })
                
                optimization_history.extend(batch_history)
                
                # Reveal batch of points
                for next_idx in valid_indices:
                    hidden_indices.remove(next_idx)
                    revealed_indices = np.append(revealed_indices, next_idx)
                    X_observed = np.vstack([X_observed, X_pool[next_idx:next_idx+1]])
                    y_observed = np.append(y_observed, y_pool[next_idx])
                    
                    # Check if target found
                    if next_idx == target_idx:
                        found_target = True
                        if verbose:
                            logger.info(f"[Competition] Target FOUND by {optimizer_name} at step {steps_to_target + 1}! Target {target_idx} was selected")
                
                # Increment step counter (one step per batch)
                steps_to_target += 1
                
                # Log if we're getting close to exhausting hidden indices
                if len(hidden_indices) <= 5 and verbose:
                    logger.info(f"[Competition] {optimizer_name} step {steps_to_target}: Only {len(hidden_indices)} hidden indices left: {hidden_indices}, target_in_remaining={target_idx in hidden_indices}")
                
                if steps_to_target % 10 == 0:
                    logger.debug(f"  Step {steps_to_target}: selected point {next_idx}, best {current_best_local:.4f}, found_target={found_target}")
                
                if optimizer_name == 'BO_GP_EI' and len(y_observed) > 1:
                    try:
                        from sklearn.gaussian_process import GaussianProcessRegressor
                        from sklearn.gaussian_process.kernels import RBF, WhiteKernel
                        from sklearn.preprocessing import StandardScaler
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X_observed)
                        kernel = RBF() + WhiteKernel()
                        gp = GaussianProcessRegressor(kernel=kernel, random_state=0)
                        gp.fit(X_scaled, y_observed)
                        y_pred = gp.predict(X_scaled)
                        r2 = 1 - (np.sum((y_observed - y_pred) ** 2) / np.sum((y_observed - np.mean(y_observed)) ** 2))
                    except Exception as e:
                        r2 = float('nan')
                    surrogate_r2_history.append(r2)
                    if verbose:
                        logger.info(f"[BO_GP_EI] Step {steps_to_target+1}: Surrogate R^2 on observed data: {r2:.4f}")
            
            # Store results
            optimizer_results[optimizer_name] = {
                'steps_to_target': steps_to_target,
                'found_target': found_target,
                'final_best': float(np.max(y_observed)) if len(y_observed) > 0 else float('-inf'),  # Use local final best
                'optimization_history': optimization_history,
                'surrogate_r2_history': surrogate_r2_history if optimizer_name == 'BO_GP_EI' else None
            }
            
            if verbose:
                logger.debug(f"  {optimizer_name} completed: {steps_to_target} steps, found_target={found_target}")
                
                # Log completion status with detailed information
                logger.info(f"=== {optimizer_name} COMPETITION RESULTS ===")
                logger.info(f"  Steps taken: {steps_to_target}")
                logger.info(f"  Found target: {found_target}")
                logger.info(f"  Remaining hidden: {len(hidden_indices)}")
                logger.info(f"  Final best value: {float(np.max(y_observed)) if len(y_observed) > 0 else float('-inf'):.4f}")
                logger.info(f"  Target value: {target_value:.4f}")
                logger.info(f"  Optimization history:")
                for step_info in optimization_history:
                    logger.info(f"    Step {step_info['step']}: Selected point {step_info['selected_idx']} (value: {step_info['selected_value']:.4f})")
                logger.info(f"[Competition] {optimizer_name} completed: {steps_to_target} steps, found_target={found_target}, remaining_hidden={len(hidden_indices)}")
        
        # Compile competition results
        competition_result = {
            'target_idx': int(target_idx),
            'target_value': float(target_value),
            'n_initial_revealed': len(revealed_indices),
            'n_total_points': n_points,
            'hidden_fraction': hidden_fraction,
            'optimizer_results': optimizer_results,
            'competition_state': competition_state
        }
        
        return competition_result
    
    def run_competition_tournament(self, dataset_name: str, 
                                  optimizer_names: List[str] = None,
                                  n_competitions: int = 10,
                                  hidden_fraction: float = 0.9,
                                  pool_size: int = 200,
                                  batch_size: int = 1,
                                  n_jobs: int = -1,
                                  X_pool: np.ndarray = None,
                                  y_pool: np.ndarray = None,
                                  use_cache_model: bool = True,
                                  synthetic_dataset_index: int = 0,
                                  verbose: bool = True) -> Dict[str, Any]:
        """
        Run a tournament of competitions between optimizers.
        
        This method runs multiple competitions between optimizers to get statistically
        meaningful results. It can run competitions in parallel and supports both
        pre-generated synthetic datasets and on-the-fly generation.
        
        Args:
            dataset_name: Name of the dataset to use for surrogate model training
            optimizer_names: List of optimizer names to include in the tournament
            n_competitions: Number of competitions to run for statistical significance
            hidden_fraction: Fraction of data to hide in each competition
            pool_size: Size of candidate pool to generate
            batch_size: Number of points to select per optimization step
            n_jobs: Number of parallel jobs (-1 for all available cores)
            X_pool: Pre-generated candidate pool (optional, for reusing synthetic data)
            y_pool: Pre-generated candidate values (optional, for reusing synthetic data)
            
        Returns:
            Dictionary containing comprehensive tournament results
        """
        # Ensure hidden_fraction is a float
        if isinstance(hidden_fraction, list):
            hidden_fraction = float(hidden_fraction[0])
        if verbose:
            self._log_timing("Starting competition tournament")
        
        # Load dataset
        X, y, y_var, dataset = load_dataset(dataset_name)
        
        # Use provided synthetic data or generate new pool
        if X_pool is None or y_pool is None:
            # Fit optimized GP surrogate model
            gp_model, scaler = self.fit_surrogate_model(X, y, y_var, dataset_name)
            # Generate candidate pool
            X_pool, y_pool = self.generate_candidate_pool(gp_model, scaler, X, n_points=pool_size)
        
        # Store synthetic dataset information if applicable
        synthetic_info = None
        if isinstance(dataset, SyntheticDataset):
            synthetic_info = {
                'type': 'synthetic',
                'generation_method': 'Optimized Gaussian Process',
                'n_features': dataset.n_features,
                'n_samples': dataset.n_samples,
                'noise_level': dataset.noise_level,
                'random_state': dataset.random_state,
                'X_pool_shape': X_pool.shape,
                'y_pool_shape': y_pool.shape,
                'y_pool_stats': {
                    'mean': float(np.mean(y_pool)),
                    'std': float(np.std(y_pool)),
                    'min': float(np.min(y_pool)),
                    'max': float(np.max(y_pool))
                },
                'synth_idx': synthetic_dataset_index
            }
        
        # Run competitions in parallel
        start_time = time.time()
        
        # Get optimizers
        if optimizer_names is None:
            from utils.optimizers import get_all_optimizers
            optimizer_names = list(get_all_optimizers().keys())
        
        # CRITICAL: Store optimizer names only - actual optimizers will be created with unique seeds in each competition
        
        if verbose:
            logger.info(f"Running tournament with {len(optimizer_names)} optimizers: {optimizer_names}")
        
        # Run competitions in parallel
        def run_competition_with_seed(seed, comp_idx=None, total_comps=None, verbose=verbose):
            # Create a new competition instance with proper seed (reverted - state persistence was causing issues)
            local_competition = HideLabelCompetitionIncrementalGP(
                random_state=seed,
                use_gpu=self.use_gpu,
                cache_dir=self.cache_dir
            )
            
            # Create optimizers with the unique seed for this competition
            from utils.optimizers import get_all_optimizers
            optimizers = get_all_optimizers(random_state=seed)
            selected_optimizers = {name: optimizers[name] for name in optimizer_names if name in optimizers}
            
            if len(selected_optimizers) == 0:
                raise ValueError(f"No valid optimizers found. Available: {list(optimizers.keys())}")
            
            # Run single competition with unique optimizer instances
            result = local_competition.run_single_competition(
                X_pool=X_pool,
                y_pool=y_pool,
                optimizers=selected_optimizers,
                hidden_fraction=hidden_fraction,
                batch_size=batch_size,
                seed=seed,
                verbose=verbose
            )
            

            if comp_idx is not None and total_comps is not None:
                logger.debug(f"Completed competition {comp_idx}/{total_comps}")
            
            return result
        
        # Run competitions with unique seeds (use stable hashing)
        if n_jobs == 1:
            # Sequential execution
            competitions = []
            for i in range(n_competitions):
                # Create unique seed by combining tournament seed with competition index AND synthetic dataset info
                # Use deterministic MD5-based hashes for reproducibility across runs
                synth_info_hash = int(hashlib.md5(f"synth_{synthetic_dataset_index}".encode()).hexdigest()[:8], 16) % 100000
                comp_hash = int(hashlib.md5(f"comp_{i}".encode()).hexdigest()[:8], 16) % 10000
                seed = self.random_state + synth_info_hash + i * 10000 + comp_hash
                result = run_competition_with_seed(seed, i+1, n_competitions)
                competitions.append(result)
        else:
            # Parallel execution
            synth_info_hash = int(hashlib.md5(f"synth_{synthetic_dataset_index}".encode()).hexdigest()[:8], 16) % 100000
            seeds = [
                self.random_state + synth_info_hash + i * 10000 + int(hashlib.md5(f"comp_{i}".encode()).hexdigest()[:8], 16) % 10000
                for i in range(n_competitions)
            ]
            competitions = Parallel(n_jobs=n_jobs, verbose=0)(
                delayed(run_competition_with_seed)(seed, i+1, n_competitions)
                for i, seed in enumerate(seeds)
            )
        
        tournament_time = time.time() - start_time
        
        # Compile tournament results
        tournament_results = {
            'dataset_name': dataset_name,
            'optimizer_names': optimizer_names,
            'n_competitions': n_competitions,
            'hidden_fraction': hidden_fraction,
            'pool_size': pool_size,
            'tournament_time': tournament_time,
            'synthetic_info': synthetic_info,
            'competitions': competitions
        }
        
        if verbose:
            self._log_timing("Finished competition tournament")
        
        return tournament_results
    
    def _create_competition_pca_visualization(self, X_pool: np.ndarray, y_pool: np.ndarray, 
                                            competition_result: Dict[str, Any], 
                                            dataset_name: str, synth_idx: int, comp_idx: int):
        """
        Create PCA visualization showing optimizer's chosen points and target point.
        
        Args:
            X_pool: Pool of candidate points
            y_pool: Function values for candidate points
            competition_result: Results from a single competition
            dataset_name: Name of the dataset
            synth_idx: Synthetic dataset index
            comp_idx: Competition index
        """
        try:
            import matplotlib.pyplot as plt
            from sklearn.decomposition import PCA
            
            # Get competition state and optimizer results
            comp_state = competition_result['competition_state']
            optimizer_results = competition_result['optimizer_results']
            
            # Get target information
            target_idx = comp_state['target_idx']
            target_value = comp_state['target_value']
            
            # Get optimizer's chosen points
            optimizer_name = list(optimizer_results.keys())[0]  # Get first optimizer
            opt_result = optimizer_results[optimizer_name]
            
            # Extract chosen indices from optimization history
            chosen_indices = [step_info['selected_idx'] for step_info in opt_result['optimization_history']]
            
            # Perform PCA on the feature space
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_pool)
            
            # Create the plot
            plt.figure(figsize=(12, 8))
            
            # Plot all points in light gray
            plt.scatter(X_pca[:, 0], X_pca[:, 1], c='lightgray', alpha=0.3, s=20, label='All points')
            
            # Plot optimizer's chosen points in red
            chosen_pca = X_pca[chosen_indices]
            plt.scatter(chosen_pca[:, 0], chosen_pca[:, 1], c='red', s=50, alpha=0.8, 
                       label=f'{optimizer_name} chosen points', edgecolors='darkred', linewidth=1)
            
            # Plot target point in yellow
            target_pca = X_pca[target_idx].reshape(1, -1)
            plt.scatter(target_pca[:, 0], target_pca[:, 1], c='yellow', s=100, alpha=1.0,
                       label=f'Target (value: {target_value:.3f})', edgecolors='orange', linewidth=2)
            
            # Add step numbers to chosen points
            for i, idx in enumerate(chosen_indices):
                plt.annotate(f'{i+1}', (X_pca[idx, 0], X_pca[idx, 1]), 
                           xytext=(5, 5), textcoords='offset points', 
                           fontsize=8, color='darkred', weight='bold')
            
            # Highlight target point with step number if it was found
            if target_idx in chosen_indices:
                target_step = chosen_indices.index(target_idx) + 1
                plt.annotate(f'TARGET (step {target_step})', (X_pca[target_idx, 0], X_pca[target_idx, 1]), 
                           xytext=(10, 10), textcoords='offset points', 
                           fontsize=10, color='orange', weight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
            
            plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
            plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
            plt.title(f'PCA Visualization: {optimizer_name} Competition\n'
                     f'Dataset: {dataset_name}, Synthetic {synth_idx}, Competition {comp_idx}\n'
                     f'Steps: {len(chosen_indices)}, Target Found: {target_idx in chosen_indices}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add performance summary
            steps_taken = len(chosen_indices)
            found_target = target_idx in chosen_indices
            final_best = opt_result['final_best']
            
            summary_text = f'Steps: {steps_taken}\nTarget Found: {found_target}\nFinal Best: {final_best:.3f}'
            plt.text(0.02, 0.98, summary_text, transform=plt.gca().transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            logger.warning(f"Failed to create PCA visualization: {e}")
    
    def analyze_tournament_results(self, tournament_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze tournament results and compute statistics.
        
        This method processes the raw tournament results to compute performance statistics
        for each optimizer, including mean steps, success rates, and final best values.
        
        Args:
            tournament_results: Dictionary containing raw tournament results
            
        Returns:
            Dictionary containing analyzed statistics for each optimizer
        """
        
        analysis = {
            'dataset_name': tournament_results['dataset_name'],
            'optimizer_names': tournament_results['optimizer_names'],
            'n_competitions': tournament_results['n_competitions'],
            'hidden_fraction': tournament_results['hidden_fraction'],
            'pool_size': tournament_results['pool_size'],
            'tournament_time': tournament_results['tournament_time'],
            'optimizer_stats': {}
        }
        
        # Compile statistics for each optimizer
        competitions = tournament_results['competitions']
        
        for optimizer_name in tournament_results['optimizer_names']:
            steps_list = []
            found_target_list = []
            final_best_list = []
            
            for comp in competitions:
                opt_result = comp['optimizer_results'][optimizer_name]
                steps_list.append(opt_result['steps_to_target'])
                found_target_list.append(opt_result['found_target'])
                final_best_list.append(opt_result['final_best'])
            
            # Compute statistics
            analysis['optimizer_stats'][optimizer_name] = {
                'mean_steps': np.mean(steps_list),
                'std_steps': np.std(steps_list),
                'median_steps': np.median(steps_list),
                'min_steps': np.min(steps_list),
                'max_steps': np.max(steps_list),
                'success_rate': np.mean(found_target_list),
                'mean_final_best': np.mean(final_best_list),
                'std_final_best': np.std(final_best_list)
            }
        
        return analysis
    
    def print_tournament_summary(self, analysis: Dict[str, Any], verbose: bool = True):
        """
        Print a summary of tournament results.
        
        This method displays a formatted summary of the tournament results, showing
        performance statistics for each optimizer in a readable table format.
        
        Args:
            analysis: Dictionary containing analyzed tournament statistics
            verbose: Whether to print detailed information (default: True)
        """
        
        print("\n" + "="*80)
        print("OPTIMIZED GP HIDE-THE-LABEL COMPETITION RESULTS")
        print("="*80)
        
        if verbose:
            print(f"Dataset: {analysis['dataset_name']}")
            print(f"Optimizers: {', '.join(analysis['optimizer_names'])}")
            print(f"Competitions: {analysis['n_competitions']}")
            print(f"Hidden fraction: {analysis['hidden_fraction']}")
            print(f"Pool size: {analysis['pool_size']}")
            print(f"Tournament time: {analysis['tournament_time']:.2f}s")
        
        print("\nOVERALL OPTIMIZER PERFORMANCE:")
        print("-" * 60)
        
        # Sort optimizers by mean steps (lower is better)
        optimizer_stats = analysis['optimizer_stats']
        sorted_optimizers = sorted(optimizer_stats.items(), key=lambda x: float('inf') if x[1]['mean_steps'] is None else x[1]['mean_steps'])
        
        print(f"{'Optimizer':<15} {'Mean Steps':<12} {'Std Steps':<12} {'Success Rate':<12} {'Mean Final Best':<15}")
        print("-" * 80)
        
        for optimizer_name, stats in sorted_optimizers:
            mean_steps_str = 'N/A' if stats['mean_steps'] is None else f"{stats['mean_steps']:.2f}"
            std_steps_str = 'N/A' if stats['std_steps'] is None else f"{stats['std_steps']:.2f}"
            print(f"{optimizer_name:<15} {mean_steps_str:<12} {std_steps_str:<12} "
                  f"{stats['success_rate']:<12.2%} {stats['mean_final_best']:<15.4f}")
        
        print("\n" + "="*80)
    
    def save_results(self, tournament_results: Dict[str, Any], 
                    analysis: Dict[str, Any], 
                    output_dir: str = "results",
                    filename: Optional[str] = None) -> str:
        """
        Save tournament results and analysis to files.
        
        This method saves both the raw tournament results and the analyzed statistics
        to JSON files for later analysis or visualization.
        
        Args:
            tournament_results: Raw tournament results to save
            analysis: Analyzed statistics to save
            output_dir: Directory to save the files in
            filename: Base filename (optional, will generate timestamped name if not provided)
            
        Returns:
            Path to the output directory
        """
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tournament_{tournament_results['dataset_name']}_{timestamp}"
        
        # Save tournament results
        results_file = os.path.join(output_dir, f"{filename}_results.json")
        with open(results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = json.loads(json.dumps(tournament_results, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x))
            json.dump(json_results, f, indent=2)
        
        # Save analysis
        analysis_file = os.path.join(output_dir, f"{filename}_analysis.json")
        with open(analysis_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_analysis = json.loads(json.dumps(analysis, default=lambda x: x.tolist() if hasattr(x, 'tolist') else float(x) if hasattr(x, 'item') else x))
            json.dump(json_analysis, f, indent=2)
        
        logger.info(f"Results saved to: {results_file}")
        logger.info(f"Analysis saved to: {analysis_file}")
        
        return output_dir


def run_incremental_gp_competition(dataset_name: str, 
                                 optimizer_names: List[str] = None,
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
                                 # New saving controls
                                 compiled_output_dir: str = "results_currentlyfor_hide_and_open",
                                 compiled_filename: Optional[str] = None,
                                 save_individual_results: bool = False,
                                 verbose: bool = True) -> Tuple[Dict[str, Any], Dict[str,Any]]:
    """
    Run hide-the-label competition using optimized GP for better BO_GP_EI performance.
    
    This is the main entry point function that orchestrates the entire competition process.
    It loads a dataset, trains an optimized GP surrogate model, generates synthetic datasets,
    and runs competitions between different optimization strategies.
    
    Args:
        dataset_name: Name of the dataset to use for surrogate model training
        optimizer_names: List of optimizer names to include in the competition
        n_competitions: Number of competitions to run for statistical significance
        hidden_fraction: Fraction of data to hide (can be list for multiple values)
        pool_size: Size of candidate pool to generate
        batch_size: Number of points to select per optimization step
        n_jobs: Number of parallel jobs (-1 for all available cores)
        random_state: Random seed for reproducibility
        save_results: Whether to save results to files
        num_of_synthetic_data: Number of synthetic datasets to generate
        use_gpu: Whether to use GPU acceleration
        cache_dir: Directory to store cached models
        
    Returns:
        Tuple of (combined_results, all_analyses) containing all tournament results and analyses
    """
    
    if verbose:
        print("OPTIMIZED GP HIDE-THE-LABEL COMPETITION")
        print("="*80)
        print(f"Dataset: {dataset_name}")
        print(f"Competitions: {n_competitions}")
        print(f"Hidden fraction: {hidden_fraction}")
        print(f"Pool size: {pool_size}")
        print(f"Parallel jobs: {n_jobs}")
        print(f"Random state: {random_state}")
        print(f"GPU acceleration: {use_gpu}")
        print(f"Model cache directory: {cache_dir}")
        print("="*80)
    
    logger.info(f"Starting optimized GP competition with {dataset_name}")
    
    # Convert hidden_fraction to list if it's a single float
    if isinstance(hidden_fraction, float):
        hidden_fraction = [hidden_fraction]
    
    # Initialize combined results
    all_tournament_results = []
    all_analyses = []
    
    # Create competition instance
    competition = HideLabelCompetitionIncrementalGP(random_state=random_state, use_gpu=use_gpu, cache_dir=cache_dir)
    
    # Show cache information
    cache_info = competition.get_cache_info()
    if verbose:
        print(f"\n=== Model Cache Information ===")
        print(f"Cache directory: {cache_info['cache_dir']}")
        print(f"Cache exists: {cache_info['exists']}")
        print(f"Number of cached models: {len(cache_info['cached_models'])}")
        if cache_info['cached_models']:
            total_size = sum(model['size_mb'] for model in cache_info['cached_models'])
            print(f"Total cache size: {total_size:.2f} MB")
            for model in cache_info['cached_models']:
                print(f"  - {model['filename']}: {model['size_mb']:.2f} MB")
    
    # Load dataset and train optimized GP model once
    if verbose:
        print("\n=== Training Optimized GP Model ===")
    X, y, y_var, dataset = load_dataset(dataset_name)
    gp_model, scaler, full_data_r2 = competition.fit_surrogate_model(X, y, y_var, dataset_name, use_cache_model=use_cache_model)
    if verbose:
        print(f"Full-data surrogate model R^2: {full_data_r2:.4f}")
    
    # Generate all synthetic datasets at once
    if verbose:
        print(f"\n=== Generating {num_of_synthetic_data} Synthetic Datasets ===")
    synthetic_datasets = []
    for synth_idx in range(num_of_synthetic_data):
        if verbose:
            print(f"Generating dataset {synth_idx + 1}/{num_of_synthetic_data}")
        # Create unique seed for each synthetic dataset (deterministic)
        synth_seed = (
            random_state
            + synth_idx * 100000
            + int(hashlib.md5(f"synth_{synth_idx}".encode()).hexdigest()[:8], 16) % 100000
        )
        X_pool, y_pool = competition.generate_candidate_pool(gp_model, scaler, X, n_points=pool_size, seed=synth_seed)
        synthetic_datasets.append((X_pool, y_pool))
        # Print first 5 y_pool values for check
        if verbose:
            print(f"  y_pool[:5] for synthetic dataset {synth_idx + 1}: {y_pool[:5]}")
    
    # Set logging level based on verbose
    if not verbose:
        import logging
        logging.getLogger().setLevel(logging.WARNING)
    
    

    # Run competitions for each synthetic dataset and hidden fraction
    for synth_idx, (X_pool, y_pool) in enumerate(synthetic_datasets):
        if verbose:
            print(f"\n=== Running Competitions for Synthetic Dataset {synth_idx + 1}/{num_of_synthetic_data} ===")
        # Run tournament for each hidden fraction

        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        
        # Combine X_pool and y_pool for PCA
        combined_data = np.hstack((X_pool, y_pool.reshape(-1, 1)))
        
        if show_pca:
            # Perform PCA
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(combined_data)
            
            # Plot PCA results
            plt.figure(figsize=(10, 6))
            plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
            plt.title('PCA of Candidate Pool')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.grid()
            plt.show()



        for frac in hidden_fraction:
            if verbose:
                print(f"\n--- Running competitions with {frac*100}% hidden data ---")
            # Run tournament with the current synthetic dataset
            tournament_results = competition.run_competition_tournament(
                dataset_name=dataset_name,
                optimizer_names=optimizer_names,
                n_competitions=n_competitions,
                hidden_fraction=frac,
                pool_size=pool_size,
                batch_size=batch_size,
                n_jobs=n_jobs,
                X_pool=X_pool,
                y_pool=y_pool,
                use_cache_model=use_cache_model,
                synthetic_dataset_index=synth_idx,  # Pass synthetic dataset index
                verbose=verbose  # Pass verbose parameter
            )
            # Print initial hidden indices for each competition
            if verbose:
                print(f"Initial hidden indices for each competition in synthetic dataset {synth_idx + 1}:")
                for comp_idx, comp in enumerate(tournament_results['competitions']):
                    print(f"  Competition {comp_idx + 1}: {comp['competition_state']['hidden_indices'][:10]} ... (total {len(comp['competition_state']['hidden_indices'])})")
                
                # Create PCA visualization for this competition (only if show_pca is True)
                if show_pca:
                    competition._create_competition_pca_visualization(
                        X_pool, y_pool, comp, 
                        dataset_name, synth_idx + 1, comp_idx + 1
                    )
            # Add synthetic dataset index to results
            tournament_results['synthetic_dataset_index'] = synth_idx
            # Analyze results
            analysis = competition.analyze_tournament_results(tournament_results)
            # Always print summary, but control verbosity
            competition.print_tournament_summary(analysis, verbose=verbose)
            # Save per-synthetic results only if explicitly requested
            if save_results and save_individual_results:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"tournament_{dataset_name}_synth{synth_idx}_hidden{int(frac*100)}_{timestamp}.json"
                competition.save_results(tournament_results, analysis, filename=filename)
            all_tournament_results.append(tournament_results)
            all_analyses.append(analysis)
    
    # Combine all results
    combined_results = {
        'all_tournament_results': all_tournament_results,
        'all_analyses': all_analyses,
        'num_synthetic_datasets': num_of_synthetic_data,
        'hidden_fractions': hidden_fraction
    }
    
    # Calculate aggregate optimizer performance across all synthetic datasets
    # Always print the tables regardless of verbose setting
    print("\n" + "="*80)
    print("AGGREGATE OPTIMIZER PERFORMANCE ACROSS ALL SYNTHETIC DATASETS")
    print("="*80)
    
    # Collect all step counts for each optimizer
    optimizer_step_counts = {}
    optimizer_success_rates = {}
    optimizer_final_bests = {}
    
    for analysis in all_analyses:
        for optimizer_name, stats in analysis['optimizer_stats'].items():
            if optimizer_name not in optimizer_step_counts:
                optimizer_step_counts[optimizer_name] = []
                optimizer_success_rates[optimizer_name] = []
                optimizer_final_bests[optimizer_name] = []
            
            optimizer_step_counts[optimizer_name].append(stats['mean_steps'])
            optimizer_success_rates[optimizer_name].append(stats['success_rate'])
            optimizer_final_bests[optimizer_name].append(stats['mean_final_best'])
    
    # Calculate aggregate statistics
    print(f"{'Optimizer':<15} {'Avg Steps':<12} {'Std Steps':<12} {'Avg Success':<12} {'Avg Final Best':<15}")
    print("-" * 80)
    
    for optimizer_name in sorted(optimizer_step_counts.keys()):
        steps = optimizer_step_counts[optimizer_name]
        success_rates = optimizer_success_rates[optimizer_name]
        final_bests = optimizer_final_bests[optimizer_name]
        
        avg_steps = np.mean(steps)
        std_steps = np.std(steps)
        avg_success = np.mean(success_rates)
        avg_final_best = np.mean(final_bests)
        
        print(f"{optimizer_name:<15} {avg_steps:<12.2f} {std_steps:<12.2f} {avg_success:<12.1%} {avg_final_best:<15.4f}")
    
    print("="*80)
    print(f"Total synthetic datasets: {num_of_synthetic_data}")
    print(f"Hidden fractions tested: {hidden_fraction}")
    print("="*80)
    
    # Save a single standardized results JSON and generate plot
    if save_results:
        try:
            root_dir = Path(__file__).resolve().parents[2]
            results_dir = root_dir / 'Results' / 'Regular_Mode' / 'Hide_The_Label'
            results_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_name = f"{dataset_name}_regular_mode_hide_the_label_{timestamp}"
            results_path = results_dir / f"{base_name}.json"

            # Ensure JSON serializable (handle numpy types)
            serializable_combined = json.loads(json.dumps(
                combined_results,
                default=lambda x: x.tolist() if hasattr(x, 'tolist') else float(x) if hasattr(x, 'item') else x
            ))

            with open(results_path, 'w') as f:
                json.dump(serializable_combined, f, indent=2)
            logger.info(f"Results saved to: {results_path}")

            # Generate plot into Plotting folder with the same naming structure (png)
            plotting_dir = root_dir / 'Plotting' / 'Regular_Mode' / 'Hide_The_Label'
            plotting_dir.mkdir(parents=True, exist_ok=True)
            plot_path = plotting_dir / f"{base_name}.png"

            # Use visualization helpers to render the bar plot
            try:
                optimizers, means, ses = extract_from_optimizer_stats(serializable_combined)
                if not optimizers:
                    optimizers, means, ses = extract_from_competitions(serializable_combined)
                title = f"Hide-the-Label: Mean Steps — {dataset_name}"
                plot_bar(optimizers, means, ses, title, str(plot_path))
                logger.info(f"Plot saved to: {plot_path}")
            except Exception as e:
                logger.warning(f"Failed to generate plot: {e}")
        except Exception as e:
            logger.warning(f"Failed to save standardized results/plots: {e}")

    return combined_results, all_analyses


if __name__ == "__main__":
    # Example usage

    percentage_list = [0.95]  # List of hidden fractions to test (match AL_that_works)
    # working optimizers: ['RANDOM', 'BO_GP_EI', 'SMART_BO', 'SBO_GP_PV', 'SBO_ANN_PV', 'SBO_POLY_PV', 'SBO_GP_EI_TRUNCDE', 'DE_DIRECT', 'FULL_FACTORIAL', 'FRACTIONAL_FACTORIAL', 'PLACKETT_BURMAN', 'CENTRAL_COMPOSITE', 'BOX_BEHNKEN', 'LATIN_HYPERCUBE', 'D_OPTIMAL']
    # unavailable optimizers: ['GA_DIRECT', 'PSO_DIRECT'] (require DEAP and pyswarms libraries)
    # possible datasets: ['MOBO_dataset_rat_myocyte', 'DBO_dataset_rat_myocyte', 'df_Human_Hela_regular_mode', 'df_Human_Hela_timesaving_mode', 'df_Human_T_Cell_Expanded', 'df_Human_TF_Cell_Expanded', 'synthetic_2d', 'synthetic_5d', 'synthetic_10d']


    firsthalf = ['RANDOM','SBO_GP_PV', 'BO_GP_EI', 'SMART_BO', 'SBO_ANN_PV']
    secondhalf = ['DE_DIRECT', 'FULL_FACTORIAL', 'FRACTIONAL_FACTORIAL', 'PLACKETT_BURMAN', 'CENTRAL_COMPOSITE', 'BOX_BEHNKEN', 'LATIN_HYPERCUBE', 'D_OPTIMAL']
    all_optimisers = firsthalf + secondhalf


    # Test with optimized GP for better BO_GP_EI performance
    # Example usage with model caching
    verbose = False  # Set this to control output verbosity
     
    if verbose:
        print("\n" + "="*80)
        print("EXAMPLE: Running competition with model caching")
        print("="*80)
        print("\n--- First run (will fit and cache model) ---")
    
    
    # Track total runtime for this script execution
    _script_start_time = time.time()

    results, analysis = run_incremental_gp_competition(
        dataset_name='df_Human_TF_Cell_Expanded',
        optimizer_names= all_optimisers,    #all_optimisers,  #,'SBO_GP_PV'],
        n_competitions=10,  # 3 competitions per synthetic dataset
        hidden_fraction=[0.95],
        pool_size=200,
        batch_size=20,
        n_jobs=-1,
        random_state=42,
        num_of_synthetic_data=10,  # 3 synthetic datasets
        use_gpu=False,
        cache_dir="model_cache",  # Specify cache directory
        use_cache_model=True,  # Use cached model if available
        show_pca=False,  # Disable PCA for cleaner output
        verbose=False  # Enable verbose to see all competitions
    )
    _script_elapsed = time.time() - _script_start_time
    print(f"Hide-the-Label competition finished in {_script_elapsed:.2f}s")
                                 