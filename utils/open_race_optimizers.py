#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Open Race Optimizers Compatibility Layer

This module wraps our existing optimizers to make them compatible with the open race competition.
The key differences are:
- Open race uses continuous function evaluation, not hidden points
- Optimizers need to propose points within continuous bounds
- No candidate pools or hidden indices
"""

import numpy as np
from typing import List, Dict, Any, Callable
from utils.optimizers import (
    RandomStrategy, BOGPEIStrategy, DEDirectStrategy, SmartBOStrategy,
    SBOGPPVStrategy, SBOANNPVStrategy, SBOPolyPVStrategy, SBOGPEITruncDEStrategy,
    FullFactorialDesignStrategy, FractionalFactorialDesignStrategy, 
    PlackettBurmanDesignStrategy, CentralCompositeDesignStrategy, 
    BoxBehnkenDesignStrategy, LatinHypercubeStrategy, DOptimalDesignStrategy,
    get_all_optimizers as get_original_optimizers
)


class OpenRaceOptimizerWrapper:
    """Wrapper to make hide-the-label optimizers work with open race competition."""
    
    def __init__(self, base_optimizer, random_state: int = None,
                 candidates_per_dim: int = 500,
                 min_candidates: int = 1000):
        """
        Initialize wrapper around a base optimizer.
        
        Args:
            base_optimizer: The original optimizer from optimizers.py
            random_state: Random state for reproducibility
        """
        self.base_optimizer = base_optimizer
        self.name = base_optimizer.name
        self.random_state = random_state or base_optimizer.random_state
        
        # Create a random number generator for continuous sampling
        self.rng = np.random.RandomState(self.random_state)
        
        # Track evaluation history for surrogate-based optimizers
        self.X_history = []
        self.y_history = []
        self.current_best = float('-inf')
        # Candidate generation controls (bigger sets help BO acquisitions)
        self.candidates_per_dim = int(max(50, candidates_per_dim))
        self.min_candidates = int(max(200, min_candidates))
    
    def supports_continuous(self) -> bool:
        """Indicate that this optimizer supports continuous optimization."""
        return True
    
    def select_next_batch_open_field(self, X_observed: np.ndarray, y_observed: np.ndarray, 
                                   current_best: float, batch_size: int, 
                                   bounds: np.ndarray, true_function: Callable) -> np.ndarray:
        """
        Select next batch of points for open race competition.
        
        Args:
            X_observed: Previously evaluated points (n_observed, n_features)
            y_observed: Function values at observed points (n_observed,)
            current_best: Current best function value
            batch_size: Number of points to select
            bounds: Search space bounds (n_features, 2)
            true_function: The black box function to evaluate
            
        Returns:
            Array of shape (batch_size, n_features) with new points to evaluate
        """
        n_features = bounds.shape[0]
        
        # Update internal history
        if len(X_observed) > len(self.X_history):
            # New observations available
            new_X = X_observed[len(self.X_history):]
            new_y = y_observed[len(self.X_history):]
            self.X_history.extend(new_X)
            self.y_history.extend(new_y)
            self.current_best = current_best
        
        # Generate candidate pool for the optimizer
        # For open race, we generate points within the continuous bounds
        # Use a larger, dimension-aware candidate set so acquisitions can work
        n_candidates = max(self.min_candidates,
                           self.candidates_per_dim * n_features,
                           batch_size * 50)
        
        # Use Latin Hypercube sampling for good coverage
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=n_features, seed=self.random_state)
        candidates = sampler.random(n_candidates)
        
        # Scale to bounds
        for i in range(n_features):
            candidates[:, i] = bounds[i, 0] + (bounds[i, 1] - bounds[i, 0]) * candidates[:, i]
        
        # Prevent duplicates across steps for this optimizer by filtering out
        # candidates that are essentially identical to previously evaluated points
        candidates = self._filter_duplicate_candidates(candidates, bounds)
        
        # Create "hidden indices" for the optimizer (all candidates are available)
        hidden_indices = list(range(len(candidates)))
        
        # Let the base optimizer select from candidates
        selected_points = []
        for _ in range(batch_size):
            if len(hidden_indices) == 0:
                # If no candidates left, generate more
                more_candidates = self.rng.uniform(
                    bounds[:, 0], bounds[:, 1], (n_candidates, n_features)
                )
                candidates = np.vstack([candidates, more_candidates])
                hidden_indices = list(range(len(candidates) - n_candidates, len(candidates)))
            
            try:
                selected_idx = self.base_optimizer.select_next_point(
                    X_pool=candidates,
                    hidden_indices=hidden_indices,
                    X_observed=np.array(self.X_history),
                    y_observed=np.array(self.y_history),
                    current_best=self.current_best
                )
                
                # Remove selected index from hidden indices
                hidden_indices.remove(selected_idx)
                
                # Add selected point
                selected_points.append(candidates[selected_idx])
                
            except Exception as e:
                # Fallback to random selection
                selected_idx = self.rng.choice(hidden_indices)
                hidden_indices.remove(selected_idx)
                selected_points.append(candidates[selected_idx])
        
        return np.array(selected_points)

    def _filter_duplicate_candidates(self, candidates: np.ndarray, bounds: np.ndarray,
                                     rounding_decimals: int = 12) -> np.ndarray:
        """Remove candidate points that duplicate previously evaluated points.

        Duplicates are detected in normalized [0,1] space and rounded to a
        high precision to avoid floating-point artifacts. This ensures an
        optimizer cannot re-select the exact same location it already tried,
        while keeping fairness (no cross-optimizer information sharing).
        """
        if len(self.X_history) == 0 or candidates.shape[0] == 0:
            return candidates
        
        # Normalize to [0,1] per feature using bounds
        ranges = bounds[:, 1] - bounds[:, 0]
        safe_ranges = np.where(ranges > 0, ranges, 1.0)
        def _normalize(X: np.ndarray) -> np.ndarray:
            return (X - bounds[:, 0]) / safe_ranges
        
        X_hist = np.asarray(self.X_history)
        hist_norm = _normalize(X_hist)
        cand_norm = _normalize(candidates)
        
        # Build a hash set of history keys at high precision
        hist_keys = set(map(tuple, np.round(hist_norm, rounding_decimals)))
        
        # Keep only candidates whose rounded key is not in history
        keep_mask = []
        for i in range(cand_norm.shape[0]):
            key = tuple(np.round(cand_norm[i], rounding_decimals))
            keep_mask.append(key not in hist_keys)
        keep_mask = np.array(keep_mask, dtype=bool)
        
        filtered = candidates[keep_mask]
        
        # If we filtered too aggressively and don't have enough candidates
        # for the current batch, top up by random sampling (and re-filter)
        attempts = 0
        while filtered.shape[0] < 1 and attempts < 3:
            attempts += 1
            extra = self.rng.uniform(bounds[:, 0], bounds[:, 1],
                                     size=(max(256, candidates.shape[0] // 2), candidates.shape[1]))
            extra_norm = _normalize(extra)
            extra_keep = []
            for i in range(extra_norm.shape[0]):
                key = tuple(np.round(extra_norm[i], rounding_decimals))
                extra_keep.append(key not in hist_keys)
            extra_keep = np.array(extra_keep, dtype=bool)
            extra_filtered = extra[extra_keep]
            if extra_filtered.shape[0] > 0:
                filtered = np.vstack([filtered, extra_filtered]) if filtered.shape[0] > 0 else extra_filtered
        
        return filtered
    
    def reset(self):
        """Reset optimizer state."""
        self.base_optimizer.reset()
        self.X_history = []
        self.y_history = []
        self.current_best = float('-inf')


def get_open_race_optimizers(random_state: int = 42) -> Dict[str, OpenRaceOptimizerWrapper]:
    """
    Get optimizers wrapped for open race competition.
    
    Args:
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary of optimizer name -> OpenRaceOptimizerWrapper
    """
    original_optimizers = get_original_optimizers(random_state=random_state)
    
    open_race_optimizers = {}
    for name, optimizer in original_optimizers.items():
        # Include ALL optimizers - experimental design methods can work in continuous space too
        open_race_optimizers[name] = OpenRaceOptimizerWrapper(optimizer, random_state)
    
    return open_race_optimizers


def get_open_race_optimizer_by_name(name: str, random_state: int = 42) -> OpenRaceOptimizerWrapper:
    """
    Get a specific optimizer by name for open race competition.
    
    Args:
        name: Optimizer name
        random_state: Random state for reproducibility
        
    Returns:
        OpenRaceOptimizerWrapper instance
        
    Raises:
        ValueError: If optimizer name is not found
    """
    optimizers = get_open_race_optimizers(random_state=random_state)
    
    if name not in optimizers:
        available = list(optimizers.keys())
        raise ValueError(f"Optimizer '{name}' not found. Available: {available}")
    
    return optimizers[name]


# Create simple continuous optimizers for testing
class SimpleRandomOptimizer:
    """Simple random optimizer for open race competition."""
    
    def __init__(self, random_state: int = 42):
        self.name = "SIMPLE_RANDOM"
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
    
    def supports_continuous(self) -> bool:
        return True
    
    def select_next_batch_open_field(self, X_observed: np.ndarray, y_observed: np.ndarray, 
                                   current_best: float, batch_size: int, 
                                   bounds: np.ndarray, true_function: Callable) -> np.ndarray:
        """Select random points within bounds."""
        n_features = bounds.shape[0]
        points = np.zeros((batch_size, n_features))
        
        for i in range(n_features):
            points[:, i] = self.rng.uniform(bounds[i, 0], bounds[i, 1], batch_size)
        
        return points
    
    def reset(self):
        """Reset optimizer state."""
        pass


class SimpleGPOptimizer:
    """Simple GP-based optimizer for open race competition."""
    
    def __init__(self, random_state: int = 42):
        self.name = "SIMPLE_GP"
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        self.gp_model = None
    
    def supports_continuous(self) -> bool:
        return True
    
    def select_next_batch_open_field(self, X_observed: np.ndarray, y_observed: np.ndarray, 
                                   current_best: float, batch_size: int, 
                                   bounds: np.ndarray, true_function: Callable) -> np.ndarray:
        """Select points using GP-based acquisition function."""
        n_features = bounds.shape[0]
        
        if len(X_observed) < 2:
            # Not enough data for GP, use random
            points = np.zeros((batch_size, n_features))
            for i in range(n_features):
                points[:, i] = self.rng.uniform(bounds[i, 0], bounds[i, 1], batch_size)
            return points
        
        # Fit GP model
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, WhiteKernel
            
            kernel = RBF() + WhiteKernel()
            self.gp_model = GaussianProcessRegressor(kernel=kernel, random_state=self.random_state)
            self.gp_model.fit(X_observed, y_observed)
            
            # Generate candidate points
            n_candidates = 100
            candidates = np.zeros((n_candidates, n_features))
            for i in range(n_features):
                candidates[:, i] = self.rng.uniform(bounds[i, 0], bounds[i, 1], n_candidates)
            
            # Predict with GP
            y_pred, y_std = self.gp_model.predict(candidates, return_std=True)
            
            # Simple acquisition function: maximize predicted value + exploration
            acquisition = y_pred + 0.1 * y_std
            
            # Select top batch_size points
            top_indices = np.argsort(acquisition)[-batch_size:]
            selected_points = candidates[top_indices]
            
            return selected_points
            
        except Exception as e:
            # Fallback to random
            points = np.zeros((batch_size, n_features))
            for i in range(n_features):
                points[:, i] = self.rng.uniform(bounds[i, 0], bounds[i, 1], batch_size)
            return points
    
    def reset(self):
        """Reset optimizer state."""
        self.gp_model = None


def get_simple_open_race_optimizers(random_state: int = 42) -> Dict[str, Any]:
    """Get simple optimizers designed specifically for open race competition."""
    return {
        "SIMPLE_RANDOM": SimpleRandomOptimizer(random_state),
        "SIMPLE_GP": SimpleGPOptimizer(random_state)
    } 