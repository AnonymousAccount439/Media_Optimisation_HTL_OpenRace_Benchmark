#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Harder Surrogate Models for Making Competition More Difficult

This module implements surrogate models that make the hide-the-label competition
harder for Bayesian optimizers by using non-GP models and adding noise.

Methods implemented:
1. Non-GP Type 1 Models (Random Forest, Neural Networks)
2. Noisy Label Revelation
3. Multi-modal Functions
4. Heteroscedastic Noise
"""

import os
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import existing surrogate models
import sys
# Try local import first; then fallback to Extras; finally provide minimal in-file fallbacks
try:
    from speed_up_attempts.fast_surrogates import (
        RandomForestSurrogate, 
        NeuralNetworkSurrogate, 
        IncrementalGPSurrogate,
        FastSurrogateBase,
    )
except Exception:
    try:
        _ROOT_DIR = Path(__file__).resolve().parents[1]
        _EXTRAS_SPEED = os.path.join(
            str(_ROOT_DIR), 'Extras', 'experiments_and_making_harder', 'speed_up_attempts'
        )
        if os.path.isdir(_EXTRAS_SPEED) and _EXTRAS_SPEED not in sys.path:
            sys.path.insert(0, _EXTRAS_SPEED)
        from fast_surrogates import (
            RandomForestSurrogate,
            NeuralNetworkSurrogate,
            IncrementalGPSurrogate,
            FastSurrogateBase,
        )  # type: ignore
    except Exception:
        # Minimal fallbacks using scikit-learn to keep Hard Mode runnable without Extras
        class FastSurrogateBase:  # type: ignore
            def __init__(self, random_state: int = 42):
                self.random_state = int(random_state)
                self.np_random = np.random.RandomState(self.random_state)
                self.is_fitted = False

            def fit(self, X: np.ndarray, y: np.ndarray, y_var: Optional[np.ndarray] = None):
                self.is_fitted = True
                return self

            def predict(self, X: np.ndarray, return_std: bool = False):
                raise NotImplementedError

            def sample_y(self, X: np.ndarray, n_samples: int = 1, random_state: Optional[int] = None) -> np.ndarray:
                mu = self.predict(X)
                rng = np.random.RandomState(random_state) if random_state is not None else self.np_random
                mu = mu if mu.ndim == 1 else mu.ravel()
                return mu[:, None] + rng.normal(0, 1e-6, size=(mu.shape[0], n_samples))

        from sklearn.ensemble import RandomForestRegressor  # type: ignore
        class RandomForestSurrogate(FastSurrogateBase):  # type: ignore
            def __init__(
                self,
                n_estimators: int = 100,
                max_depth: Optional[int] = None,
                min_samples_split: int = 2,
                min_samples_leaf: int = 1,
                max_features: Union[str, int, float] = 'sqrt',
                bootstrap: bool = True,
                random_state: int = 42,
                n_jobs: int = -1,
            ):
                super().__init__(random_state)
                self.model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    bootstrap=bootstrap,
                    random_state=random_state,
                    n_jobs=n_jobs,
                )

            def fit(self, X: np.ndarray, y: np.ndarray, y_var: Optional[np.ndarray] = None) -> 'RandomForestSurrogate':
                self.model.fit(X, y)
                self.is_fitted = True
                return self

            def predict(self, X: np.ndarray, return_std: bool = False):
                y_pred = self.model.predict(X)
                if return_std:
                    if hasattr(self.model, 'estimators_') and len(self.model.estimators_) > 1:
                        preds = np.stack([t.predict(X) for t in self.model.estimators_], axis=0)
                        y_std = preds.std(axis=0)
                    else:
                        y_std = np.zeros_like(y_pred)
                    return y_pred, y_std
                return y_pred

            def sample_y(self, X: np.ndarray, n_samples: int = 1, random_state: Optional[int] = None) -> np.ndarray:
                mu = self.predict(X)
                rng = np.random.RandomState(random_state) if random_state is not None else self.np_random
                return mu[:, None] + rng.normal(0, 1e-3, size=(mu.shape[0], n_samples))

        from sklearn.neural_network import MLPRegressor  # type: ignore
        class NeuralNetworkSurrogate(FastSurrogateBase):  # type: ignore
            def __init__(
                self,
                hidden_sizes: List[int] = [100, 50],
                activation: str = 'relu',
                learning_rate: float = 0.001,
                batch_size: int = 32,
                max_epochs: int = 200,
                early_stopping_patience: int = 10,
                dropout_rate: float = 0.0,
                random_state: int = 42,
                use_gpu: bool = True,
            ):
                super().__init__(random_state)
                self.model = MLPRegressor(
                    hidden_layer_sizes=tuple(hidden_sizes),
                    activation=activation,
                    learning_rate_init=learning_rate,
                    max_iter=max_epochs,
                    random_state=random_state,
                )

            def fit(self, X: np.ndarray, y: np.ndarray, y_var: Optional[np.ndarray] = None) -> 'NeuralNetworkSurrogate':
                self.model.fit(X, y)
                self.is_fitted = True
                return self

            def predict(self, X: np.ndarray, return_std: bool = False):
                y_pred = self.model.predict(X)
                if return_std:
                    y_std = np.zeros_like(y_pred)
                    return y_pred, y_std
                return y_pred

            def sample_y(self, X: np.ndarray, n_samples: int = 1, random_state: Optional[int] = None) -> np.ndarray:
                mu = self.predict(X)
                rng = np.random.RandomState(random_state) if random_state is not None else self.np_random
                return mu[:, None] + rng.normal(0, 1e-3, size=(mu.shape[0], n_samples))

        class IncrementalGPSurrogate(FastSurrogateBase):  # type: ignore
            pass

# Standard ML libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mean_squared_error, r2_score

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NoisySurrogateMixin:
    """Mixin to add noise to surrogate models"""
    
    def __init__(self, noise_level: float = 0.1, heteroscedastic: bool = False):
        self.noise_level = noise_level
        self.heteroscedastic = heteroscedastic
        
    def add_noise(self, y: np.ndarray, X: Optional[np.ndarray] = None, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        """Add noise to predictions using a reproducible RNG."""
        rng = rng or getattr(self, 'np_random', np.random.RandomState(0))
        if self.heteroscedastic and X is not None:
            # Heteroscedastic noise: noise level varies with input
            # Higher noise in regions with higher function values
            noise_scale = np.abs(y) * self.noise_level + 0.01
            noise = rng.normal(0, noise_scale)
        else:
            # Homoscedastic noise
            noise = rng.normal(0, self.noise_level, size=y.shape)
        return y + noise


class MultiModalSurrogateMixin:
    """Mixin to create multi-modal functions"""
    
    def __init__(self, n_modes: int = 3, mode_separation: float = 2.0):
        self.n_modes = n_modes
        self.mode_separation = mode_separation
        self.mode_centers = None
        self.mode_weights = None
        
    def create_multimodal_function(self, X: np.ndarray, base_function: Callable) -> np.ndarray:
        """Create a multi-modal function from a base function"""
        n_samples = X.shape[0]
        
        # Create multiple modes
        if self.mode_centers is None:
            # Generate random mode centers using model RNG for reproducibility
            low = np.min(X, axis=0)
            high = np.max(X, axis=0)
            centers_unit = getattr(self, 'np_random', np.random.RandomState(0)).rand(self.n_modes, X.shape[1])
            self.mode_centers = low + centers_unit * (high - low)
            self.mode_weights = getattr(self, 'np_random', np.random.RandomState(0)).dirichlet(np.ones(self.n_modes))
        
        # Evaluate base function at each mode
        y_multimodal = np.zeros(n_samples)
        
        for i, (center, weight) in enumerate(zip(self.mode_centers, self.mode_weights)):
            # Shift input to center around this mode
            X_shifted = X - center
            y_mode = base_function(X_shifted)
            
            # Add Gaussian kernel to create mode
            distances = np.linalg.norm(X - center, axis=1)
            mode_kernel = np.exp(-distances**2 / (2 * self.mode_separation**2))
            
            y_multimodal += weight * y_mode * mode_kernel
        
        return y_multimodal


class HarderRandomForestSurrogate(RandomForestSurrogate, NoisySurrogateMixin, MultiModalSurrogateMixin):
    """Random Forest surrogate that makes competition harder"""
    
    def __init__(self, 
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: Union[str, int, float] = 'sqrt',
                 bootstrap: bool = True,
                 random_state: int = 42,
                 n_jobs: int = -1,
                 noise_level: float = 0.1,
                 heteroscedastic: bool = False,
                 n_modes: int = 3,
                 mode_separation: float = 2.0):
        
        # Initialize parent classes
        RandomForestSurrogate.__init__(
            self, n_estimators, max_depth, min_samples_split, 
            min_samples_leaf, max_features, bootstrap, random_state, n_jobs
        )
        NoisySurrogateMixin.__init__(self, noise_level, heteroscedastic)
        MultiModalSurrogateMixin.__init__(self, n_modes, mode_separation)
        
    def fit(self, X: np.ndarray, y: np.ndarray, y_var: Optional[np.ndarray] = None) -> 'HarderRandomForestSurrogate':
        """Fit the harder Random Forest model"""
        start_time = time.time()
        
        # Create multi-modal function if requested
        if self.n_modes > 1:
            # Use a simple base function (e.g., sum of features)
            def base_function(X_base):
                return np.sum(X_base, axis=1)
            
            y_multimodal = self.create_multimodal_function(X, base_function)
            y = y_multimodal
        
        # Add noise to training data (reproducible)
        y_noisy = self.add_noise(y, X, rng=self.np_random)
        
        # Call parent fit method
        super().fit(X, y_noisy, y_var)
        
        elapsed = time.time() - start_time
        logger.info(f"Harder Random Forest fitted in {elapsed:.3f}s (noise: {self.noise_level}, modes: {self.n_modes})")
        
        return self
        
    def predict(self, X: np.ndarray, return_std: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Make predictions with added noise"""
        # Get base predictions
        if return_std:
            y_pred, y_std = super().predict(X, return_std=True)
        else:
            y_pred = super().predict(X, return_std=False)
            y_std = None
        
        # Add noise to predictions
        if self.noise_level > 0:
            y_pred = self.add_noise(y_pred, X, rng=self.np_random)
        
        if return_std:
            return y_pred, y_std
        else:
            return y_pred
        
    def sample_y(self, X: np.ndarray, n_samples: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        """Sample from the harder Random Forest with additional noise"""
        # Get base samples
        samples = super().sample_y(X, n_samples, random_state)
        
        # Add additional noise to samples (use provided seed for reproducibility)
        local_rng = np.random.RandomState(random_state) if random_state is not None else self.np_random
        if self.noise_level > 0:
            samples = self.add_noise(samples, X, rng=local_rng)
            
        return samples


class HarderNeuralNetworkSurrogate(NeuralNetworkSurrogate, NoisySurrogateMixin, MultiModalSurrogateMixin):
    """Neural Network surrogate that makes competition harder"""
    
    def __init__(self, 
                 hidden_sizes: List[int] = [100, 50],
                 activation: str = 'relu',
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 max_epochs: int = 100,
                 early_stopping_patience: int = 10,
                 dropout_rate: float = 0.1,
                 random_state: int = 42,
                 use_gpu: bool = True,
                 noise_level: float = 0.1,
                 heteroscedastic: bool = False,
                 n_modes: int = 3,
                 mode_separation: float = 2.0):
        
        # Initialize parent classes
        NeuralNetworkSurrogate.__init__(
            self, hidden_sizes, activation, learning_rate, batch_size,
            max_epochs, early_stopping_patience, dropout_rate, random_state, use_gpu
        )
        NoisySurrogateMixin.__init__(self, noise_level, heteroscedastic)
        MultiModalSurrogateMixin.__init__(self, n_modes, mode_separation)
        
    def fit(self, X: np.ndarray, y: np.ndarray, y_var: Optional[np.ndarray] = None) -> 'HarderNeuralNetworkSurrogate':
        """Fit the harder Neural Network model"""
        start_time = time.time()
        
        # Create multi-modal function if requested
        if self.n_modes > 1:
            # Use a simple base function (e.g., sum of features)
            def base_function(X_base):
                return np.sum(X_base, axis=1)
            
            y_multimodal = self.create_multimodal_function(X, base_function)
            y = y_multimodal
        
        # Add noise to training data (reproducible)
        y_noisy = self.add_noise(y, X, rng=self.np_random)
        
        # Call parent fit method
        super().fit(X, y_noisy, y_var)
        
        elapsed = time.time() - start_time
        logger.info(f"Harder Neural Network fitted in {elapsed:.3f}s (noise: {self.noise_level}, modes: {self.n_modes})")
        
        return self
        
    def predict(self, X: np.ndarray, return_std: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Make predictions with added noise"""
        # Get base predictions
        if return_std:
            y_pred, y_std = super().predict(X, return_std=True)
        else:
            y_pred = super().predict(X, return_std=False)
            y_std = None
        
        # Add noise to predictions
        if self.noise_level > 0:
            y_pred = self.add_noise(y_pred, X, rng=self.np_random)
        
        if return_std:
            return y_pred, y_std
        else:
            return y_pred
        
    def sample_y(self, X: np.ndarray, n_samples: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        """Sample from the harder Neural Network with additional noise"""
        # Get base samples
        samples = super().sample_y(X, n_samples, random_state)
        
        # Add additional noise to samples (use provided seed for reproducibility)
        local_rng = np.random.RandomState(random_state) if random_state is not None else self.np_random
        if self.noise_level > 0:
            samples = self.add_noise(samples, X, rng=local_rng)
            
        return samples


class DiscontinuousSurrogate(FastSurrogateBase):
    """Surrogate model that creates discontinuous functions"""
    
    def __init__(self, 
                 base_surrogate: FastSurrogateBase,
                 n_discontinuities: int = 3,
                 discontinuity_strength: float = 2.0,
                 random_state: int = 42):
        super().__init__(random_state)
        
        self.base_surrogate = base_surrogate
        self.n_discontinuities = n_discontinuities
        self.discontinuity_strength = discontinuity_strength
        self.discontinuity_planes = None
        
    def _create_discontinuity_planes(self, X: np.ndarray):
        """Create random discontinuity planes"""
        n_features = X.shape[1]
        self.discontinuity_planes = []
        
        for _ in range(self.n_discontinuities):
            # Random normal vector
            normal = self.np_random.randn(n_features)
            normal = normal / np.linalg.norm(normal)
            
            # Random point on the plane
            point = self.np_random.uniform(np.min(X, axis=0), np.max(X, axis=0))
            
            self.discontinuity_planes.append((normal, point))
            
    def _apply_discontinuities(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Apply discontinuities to the function"""
        if self.discontinuity_planes is None:
            return y
            
        y_discontinuous = y.copy()
        
        for normal, point in self.discontinuity_planes:
            # Calculate distance to plane
            distances = np.dot(X - point, normal)
            
            # Add discontinuity: positive side gets a boost
            discontinuity = np.where(distances > 0, self.discontinuity_strength, 0)
            y_discontinuous += discontinuity
            
        return y_discontinuous
        
    def fit(self, X: np.ndarray, y: np.ndarray, y_var: Optional[np.ndarray] = None) -> 'DiscontinuousSurrogate':
        """Fit the discontinuous surrogate model"""
        start_time = time.time()
        
        # Create discontinuity planes
        self._create_discontinuity_planes(X)
        
        # Apply discontinuities to training data
        y_discontinuous = self._apply_discontinuities(X, y)
        
        # Fit base surrogate
        self.base_surrogate.fit(X, y_discontinuous, y_var)
        self.is_fitted = True
        
        elapsed = time.time() - start_time
        logger.info(f"Discontinuous surrogate fitted in {elapsed:.3f}s ({self.n_discontinuities} discontinuities)")
        
        return self
        
    def predict(self, X: np.ndarray, return_std: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        return self.base_surrogate.predict(X, return_std)
        
    def sample_y(self, X: np.ndarray, n_samples: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        """Sample from the discontinuous surrogate"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before sampling")
            
        return self.base_surrogate.sample_y(X, n_samples, random_state)


class HarderSurrogateFactory:
    """Factory for creating harder surrogate models"""
    
    @staticmethod
    def create_harder_surrogate(surrogate_type: str, **kwargs) -> FastSurrogateBase:
        """Create a harder surrogate model"""
        
        if surrogate_type == 'random_forest':
            return HarderRandomForestSurrogate(**kwargs)
        elif surrogate_type == 'neural_network':
            return HarderNeuralNetworkSurrogate(**kwargs)
        elif surrogate_type == 'discontinuous':
            # Create base surrogate first
            base_type = kwargs.pop('base_type', 'random_forest')
            base_surrogate = HarderSurrogateFactory.create_harder_surrogate(base_type, **kwargs)
            return DiscontinuousSurrogate(base_surrogate, **kwargs)
        else:
            raise ValueError(f"Unknown harder surrogate type: {surrogate_type}")
    
    @staticmethod
    def get_available_harder_surrogates() -> List[str]:
        """Get list of available harder surrogate types"""
        return ['random_forest', 'neural_network', 'discontinuous']


def benchmark_harder_surrogates(X: np.ndarray, y: np.ndarray, y_var: Optional[np.ndarray] = None) -> Dict[str, Dict[str, float]]:
    """Benchmark harder surrogate models"""
    
    results = {}
    
    # Test different harder surrogate configurations
    configs = [
        ('rf_noisy', {'surrogate_type': 'random_forest', 'noise_level': 0.2}),
        ('rf_multimodal', {'surrogate_type': 'random_forest', 'n_modes': 3}),
        ('rf_heteroscedastic', {'surrogate_type': 'random_forest', 'noise_level': 0.1, 'heteroscedastic': True}),
        ('nn_noisy', {'surrogate_type': 'neural_network', 'noise_level': 0.2}),
        ('nn_multimodal', {'surrogate_type': 'neural_network', 'n_modes': 3}),
        ('discontinuous', {'surrogate_type': 'discontinuous', 'base_type': 'random_forest', 'n_discontinuities': 3}),
    ]
    
    for name, config in configs:
        try:
            start_time = time.time()
            
            surrogate = HarderSurrogateFactory.create_harder_surrogate(**config)
            surrogate.fit(X, y, y_var)
            
            y_pred = surrogate.predict(X)
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            
            elapsed = time.time() - start_time
            
            results[name] = {
                'mse': mse,
                'r2': r2,
                'time': elapsed
            }
            
            logger.info(f"{name}: MSE={mse:.4f}, R²={r2:.4f}, Time={elapsed:.3f}s")
            
        except Exception as e:
            logger.error(f"Failed to benchmark {name}: {e}")
            results[name] = {'error': str(e)}
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Testing harder surrogate models...")
    
    # Create synthetic data
    np.random.seed(42)
    X = np.random.rand(100, 2)
    y = np.sin(X[:, 0]) + np.cos(X[:, 1]) + 0.1 * np.random.randn(100)
    
    # Benchmark harder surrogates
    results = benchmark_harder_surrogates(X, y)
    
    print("\nBenchmark Results:")
    for name, metrics in results.items():
        if 'error' not in metrics:
            print(f"{name}: MSE={metrics['mse']:.4f}, R²={metrics['r2']:.4f}, Time={metrics['time']:.3f}s")
        else:
            print(f"{name}: ERROR - {metrics['error']}") 