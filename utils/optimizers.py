#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimization Strategies for Hide-the-Label Competition

This module implements various optimization strategies including:
- Random selection
- Bayesian Optimization with Gaussian Processes
- Evolutionary algorithms (GA, DE, PSO)
- Surrogate-based optimization methods
- Experimental Design Methods (Factorial, Fractional Factorial, etc.)
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from scipy.stats import norm, qmc
from scipy.optimize import differential_evolution
from itertools import product
import warnings
import hashlib
import time
import threading
from sklearn.neighbors import NearestNeighbors
warnings.filterwarnings('ignore')

# Performance optimization globals
_GP_CACHE = {}  # Cache for fitted GP models
_DESIGN_CACHE = {}  # Cache for pre-computed design matrices
_SCALER_CACHE = {}  # Cache for fitted scalers
_CACHE_LOCK = threading.Lock()  # Thread safety for caching

# Performance settings
FAST_GP_RESTARTS = 1  # Reduced from 5 for speed
MEDIUM_GP_RESTARTS = 2  # For medium complexity
FULL_GP_RESTARTS = 3  # For critical computations

# Use float32 for faster computations where precision allows
FAST_DTYPE = np.float32

def _get_data_fingerprint(X: np.ndarray, y: np.ndarray) -> str:
    """Generate fingerprint for caching based on data characteristics"""
    # Use data statistics for fingerprint (faster than hashing all data)
    stats = np.array([
        X.shape[0], X.shape[1],
        np.mean(X), np.std(X), np.min(X), np.max(X),
        np.mean(y), np.std(y), np.min(y), np.max(y)
    ])
    return hashlib.md5(stats.tobytes()).hexdigest()

def _get_X_fingerprint(X: np.ndarray) -> str:
    """Generate a lightweight fingerprint for X-only data to scope design caches.
    Uses basic statistics so identical or very similar pools won't collide accidentally.
    """
    try:
        stats = np.array([
            X.shape[0], X.shape[1],
            float(np.mean(X)), float(np.std(X)), float(np.min(X)), float(np.max(X))
        ], dtype=np.float64)
    except Exception:
        # Fallback to shape-only if stats fail for any reason
        stats = np.array([X.shape[0], X.shape[1], 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return hashlib.md5(stats.tobytes()).hexdigest()

def _assign_unique_nearest_indices(
    X_hidden: np.ndarray,
    candidate_coords: np.ndarray,
    hidden_indices: List[int],
    max_neighbors: int = 10,
) -> List[int]:
    """Map each candidate coordinate to a unique nearest hidden index.

    Uses KNN with multiple neighbors and greedily assigns the first available
    hidden index per candidate to avoid duplicates and reduce collapse.
    """
    if len(hidden_indices) == 0 or len(candidate_coords) == 0:
        return []

    k = min(max_neighbors, len(hidden_indices))
    nn = NearestNeighbors(n_neighbors=k, algorithm='kd_tree')
    nn.fit(X_hidden)
    distances, indices = nn.kneighbors(candidate_coords)

    used = set()
    mapped: List[int] = []
    for neighbor_row in indices:
        chosen = None
        for local_idx in neighbor_row:
            global_idx = hidden_indices[local_idx]
            if global_idx not in used:
                chosen = global_idx
                used.add(global_idx)
                break
        if chosen is None:
            # Fallback to the first neighbor even if duplicate; caller can de-dup later
            chosen = hidden_indices[neighbor_row[0]]
        mapped.append(chosen)
    return mapped

def _get_cached_gp(fingerprint: str) -> Optional[Tuple[GaussianProcessRegressor, StandardScaler]]:
    """Retrieve cached GP model and scaler"""
    with _CACHE_LOCK:
        return _GP_CACHE.get(fingerprint)

def _cache_gp(fingerprint: str, gp: GaussianProcessRegressor, scaler: StandardScaler):
    """Cache GP model and scaler"""
    with _CACHE_LOCK:
        # Limit cache size to prevent memory issues
        if len(_GP_CACHE) > 10:
            # Remove oldest entry
            oldest_key = next(iter(_GP_CACHE))
            del _GP_CACHE[oldest_key]
        _GP_CACHE[fingerprint] = (gp, scaler)

def _create_fast_kernel(n_features: int, complexity: str = "fast") -> Any:
    """Create optimized kernels based on complexity requirement"""
    if complexity == "fast":
        # Simple RBF kernel for speed
        return 1.0 * RBF(length_scale=1.0, length_scale_bounds=(0.1, 10.0)) + WhiteKernel(0.1)
    elif complexity == "medium":
        # Matern kernel with limited ARD
        if n_features <= 10:
            length_scales = np.ones(n_features)
            return 1.0 * Matern(length_scale=length_scales, nu=1.5, length_scale_bounds=(0.1, 5.0)) + WhiteKernel(0.1)
        else:
            # Single length scale for high dimensions
            return 1.0 * Matern(length_scale=1.0, nu=1.5, length_scale_bounds=(0.1, 5.0)) + WhiteKernel(0.1)
    else:  # "full"
        # Original complex kernel for critical computations
        initial_length_scales = np.ones(min(n_features, 20)) * 0.5  # Limit ARD dimensions
        if n_features > 20:
            # Use single length scale for very high dimensions
            return 1.0 * RBF(length_scale=1.0, length_scale_bounds=(0.1, 10.0)) + WhiteKernel(0.1)
        else:
            rbf_kernel = 1.0 * RBF(length_scale=initial_length_scales.copy(), length_scale_bounds=(0.1, 5.0))
            matern_kernel = 1.0 * Matern(length_scale=initial_length_scales.copy(), nu=2.5, length_scale_bounds=(0.1, 5.0))
            return rbf_kernel + matern_kernel + WhiteKernel(0.1)

def _fit_fast_gp(X: np.ndarray, y: np.ndarray, complexity: str = "fast", cache_key: str = None) -> Tuple[GaussianProcessRegressor, StandardScaler]:
    """Fit GP with performance optimizations"""
    
    # Check cache first
    if cache_key:
        cached = _get_cached_gp(cache_key)
        if cached is not None:
            return cached
    
    # Optimize data types for speed
    X = X.astype(FAST_DTYPE)
    y = y.astype(FAST_DTYPE)
    
    # Feature scaling with caching
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create appropriate kernel
    kernel = _create_fast_kernel(X.shape[1], complexity)
    
    # Set restarts based on complexity
    if complexity == "fast":
        n_restarts = FAST_GP_RESTARTS
    elif complexity == "medium":
        n_restarts = MEDIUM_GP_RESTARTS
    else:
        n_restarts = FULL_GP_RESTARTS
    
    # Fit GP with optimized settings
    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=n_restarts,
        alpha=1e-6
    )
    
    gp.fit(X_scaled, y)
    
    # Cache if requested
    if cache_key:
        _cache_gp(cache_key, gp, scaler)
    
    return gp, scaler

# Optional dependencies
try:
    import pyswarms as ps
    HAS_PYSWARMS = True
except ImportError:
    HAS_PYSWARMS = False
    print("Warning: pyswarms not available. PSO_DIRECT strategy will be disabled.")

try:
    from deap import algorithms, base, creator, tools
    import random
    HAS_DEAP = True
except ImportError:
    HAS_DEAP = False
    print("Warning: DEAP not available. GA_DIRECT strategy will be disabled.")


class OptimizationStrategy(ABC):
    """Base class for optimization strategies"""
    
    def __init__(self, name: str, random_state: int = 0):
        self.name = name
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        self.iteration_count = 0
    
    @abstractmethod
    def select_next_point(self, X_pool: np.ndarray, hidden_indices: List[int], 
                         X_observed: np.ndarray, y_observed: np.ndarray,
                         current_best: float, **kwargs) -> int:
        """
        Select the next point to evaluate
        
        Args:
            X_pool: Complete pool of candidate points
            hidden_indices: Indices of points not yet evaluated
            X_observed: Points already observed
            y_observed: Observed function values
            current_best: Current best function value found
            **kwargs: Additional strategy-specific parameters
            
        Returns:
            Index of the next point to evaluate
        """
        pass
    
    def select_next_batch(self, X_pool: np.ndarray, hidden_indices: List[int], 
                         X_observed: np.ndarray, y_observed: np.ndarray,
                         current_best: float, batch_size: int = 1, **kwargs) -> List[int]:
        """
        Select the next batch of points to evaluate
        
        Args:
            X_pool: Complete pool of candidate points
            hidden_indices: Indices of points not yet evaluated
            X_observed: Points already observed
            y_observed: Observed function values
            current_best: Current best function value found
            batch_size: Number of points to select
            **kwargs: Additional strategy-specific parameters
            
        Returns:
            List of indices of the next points to evaluate
        """
        selected_indices = []
        remaining_hidden = hidden_indices.copy()
        
        for _ in range(min(batch_size, len(hidden_indices))):
            if len(remaining_hidden) == 0:
                break
                
            # Select next point using the single-point method
            next_idx = self.select_next_point(
                X_pool=X_pool,
                hidden_indices=remaining_hidden,
                X_observed=X_observed,
                y_observed=y_observed,
                current_best=current_best,
                **kwargs
            )
            
            # Remove selected point from remaining hidden indices
            if next_idx in remaining_hidden:
                remaining_hidden.remove(next_idx)
                selected_indices.append(next_idx)
        
        return selected_indices
    
    def reset(self):
        """Reset the strategy for a new experiment"""
        self.iteration_count = 0
    
    def supports_continuous(self) -> bool:
        """Indicate whether this optimizer supports continuous optimization.
        Default implementation returns False for hide-the-label optimizers."""
        return False


class RandomStrategy(OptimizationStrategy):
    """Uniform random selection strategy"""
    
    def __init__(self, random_state: int = 0):
        super().__init__("RANDOM", random_state)
    
    def select_next_point(self, X_pool: np.ndarray, hidden_indices: List[int], 
                         X_observed: np.ndarray, y_observed: np.ndarray,
                         current_best: float, **kwargs) -> int:
        """Select a random point from hidden indices"""
        self.iteration_count += 1
        return self.rng.choice(hidden_indices)


class BOGPEIStrategy(OptimizationStrategy):
    """Bayesian Optimization with Gaussian Process and Expected Improvement - Optimized"""
    
    def __init__(self, random_state: int = 0, debug: bool = False, fast_mode: bool = True):
        super().__init__("BO_GP_EI", random_state)
        self.debug = debug
        self.fast_mode = fast_mode
        self.gp_cache_key = None
        self.last_data_size = 0
        
    def select_next_point(self, X_pool: np.ndarray, hidden_indices: List[int], 
                         X_observed: np.ndarray, y_observed: np.ndarray,
                         current_best: float, **kwargs) -> int:
        """Select point with highest Expected Improvement - Optimized version"""
        self.iteration_count += 1
        
        start_time = time.time()
        
        # Early termination: if we have very few points left, just pick randomly
        if len(hidden_indices) <= 2:
            return self.rng.choice(hidden_indices)
        
        # Generate cache key for GP model
        data_fingerprint = _get_data_fingerprint(X_observed, y_observed)
        
        # Determine GP complexity based on data size and fast_mode
        if self.fast_mode:
            complexity = "fast" if len(y_observed) < 50 else "medium"
        else:
            complexity = "medium" if len(y_observed) < 100 else "full"
        
        # Use fast GP fitting with caching
        try:
            gp, scaler = _fit_fast_gp(
                X_observed, y_observed, 
                complexity=complexity, 
                cache_key=data_fingerprint
            )
            
            if self.debug:
                print(f"GP fitted in {time.time() - start_time:.3f}s (complexity: {complexity})")
                if data_fingerprint in _GP_CACHE:
                    print("Used cached GP model")
            
        except Exception as e:
            if self.debug:
                print(f"GP fitting failed: {e}")
            return self.rng.choice(hidden_indices)
        
        # Optimize data types and vectorize operations
        X_pool = X_pool.astype(FAST_DTYPE)
        X_hidden = X_pool[hidden_indices]
        X_hidden_scaled = scaler.transform(X_hidden)
        # Denoised best-so-far: use GP posterior mean at observed points instead of raw noisy max
        try:
            X_observed_scaled = scaler.transform(X_observed.astype(FAST_DTYPE)) if len(X_observed) > 0 else None
            if X_observed_scaled is not None and len(X_observed_scaled) > 0:
                mean_obs = gp.predict(X_observed_scaled)
                f_best_denoised = float(np.max(mean_obs))
            else:
                f_best_denoised = float(current_best)
        except Exception:
            f_best_denoised = float(current_best)
        
        # Vectorized EI calculation
        try:
            # Batch prediction for efficiency
            mean, std = gp.predict(X_hidden_scaled, return_std=True)
            
            # Vectorized EI computation with denoised target
            std = np.maximum(std, 1e-6)
            ei = self._expected_improvement_vectorized(mean, std, f_best_denoised)
            
            # Fast argmax
            best_local_idx = np.argmax(ei)
            selected_idx = hidden_indices[best_local_idx]
            
            if self.debug:
                elapsed = time.time() - start_time
                print(f"BO_GP_EI iteration {self.iteration_count} completed in {elapsed:.3f}s")
                print(f"  Selected point {selected_idx} with EI={ei[best_local_idx]:.6f}")
                print(f"  EI range: [{np.min(ei):.6f}, {np.max(ei):.6f}]")
            
            return selected_idx
            
        except Exception as e:
            if self.debug:
                print(f"EI calculation failed: {e}")
            return self.rng.choice(hidden_indices)
    
    def _expected_improvement_vectorized(self, mean: np.ndarray, std: np.ndarray, f_best: float) -> np.ndarray:
        """Vectorized Expected Improvement calculation"""
        z = (mean - f_best) / std
        ei = (mean - f_best) * norm.cdf(z) + std * norm.pdf(z)
        return np.maximum(ei, 0.0)  # Ensure non-negative


class GADirectStrategy(OptimizationStrategy):
    """Genetic Algorithm on true objective (direct optimization)"""
    
    def __init__(self, random_state: int = 0):
        super().__init__("GA_DIRECT", random_state)
        if not HAS_DEAP:
            raise ImportError("DEAP library required for GA_DIRECT strategy")
    
    def select_next_point(self, X_pool: np.ndarray, hidden_indices: List[int], 
                         X_observed: np.ndarray, y_observed: np.ndarray,
                         current_best: float, **kwargs) -> int:
        """Select point using Genetic Algorithm on the true function"""
        self.iteration_count += 1
        
        if 'true_function' not in kwargs:
            # Fallback to random if true function not available
            return self.rng.choice(hidden_indices)
        
        true_function = kwargs['true_function']
        
        # Setup DEAP for GA
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        
        # Restrict search to hidden points only
        X_hidden = X_pool[hidden_indices]
        
        # Create individual as index into hidden points
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                        lambda: self.rng.randint(0, len(hidden_indices)), n=1)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        def evaluate(individual):
            idx = individual[0] % len(hidden_indices)
            return (true_function[hidden_indices[idx]],)
        
        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutUniformInt, low=0, up=len(hidden_indices)-1, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Run GA for a few generations
        random.seed(self.random_state)
        pop = toolbox.population(n=min(20, len(hidden_indices)))
        
        # Evaluate population
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        
        # Evolve for a few generations
        for gen in range(5):  # Limited generations for speed
            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))
            
            # Crossover and mutation
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.5:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            for mutant in offspring:
                if random.random() < 0.1:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # Evaluate invalid individuals
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Replace population
            pop[:] = offspring
        
        # Return best individual
        best = tools.selBest(pop, 1)[0]
        return hidden_indices[best[0] % len(hidden_indices)]


class DEDirectStrategy(OptimizationStrategy):
    """Differential Evolution on true objective (direct optimization)"""
    
    def __init__(self, random_state: int = 0):
        super().__init__("DE_DIRECT", random_state)
    
    def select_next_point(self, X_pool: np.ndarray, hidden_indices: List[int], 
                         X_observed: np.ndarray, y_observed: np.ndarray,
                         current_best: float, **kwargs) -> int:
        """Select point using Differential Evolution on the true function"""
        self.iteration_count += 1
        
        if 'true_function' not in kwargs:
            return self.rng.choice(hidden_indices)
        
        true_function = kwargs['true_function']
        
        def objective(x):
            # Find closest point in the pool
            distances = np.sum((X_pool - x)**2, axis=1)
            closest_idx = np.argmin(distances)
            
            # Only consider hidden indices
            if closest_idx not in hidden_indices:
                # Find closest among hidden indices
                hidden_distances = [distances[i] for i in hidden_indices]
                closest_hidden_idx = np.argmin(hidden_distances)
                closest_idx = hidden_indices[closest_hidden_idx]
            
            return -true_function[closest_idx]  # Negative for minimization
        
        # Run DE
        bounds = [(0, 1)] * X_pool.shape[1]  # Assume normalized inputs
        try:
            result = differential_evolution(
                objective, bounds, maxiter=10, popsize=5, 
                seed=self.random_state, atol=1e-6, tol=1e-6
            )
            x_best = result.x
            
            # Find closest point in hidden indices
            distances = [np.sum((X_pool[i] - x_best)**2) for i in hidden_indices]
            closest_idx = hidden_indices[np.argmin(distances)]
            return closest_idx
        except:
            return self.rng.choice(hidden_indices)


class PSODirectStrategy(OptimizationStrategy):
    """Particle Swarm Optimization on true objective (direct optimization)"""
    
    def __init__(self, random_state: int = 0):
        super().__init__("PSO_DIRECT", random_state)
        if not HAS_PYSWARMS:
            raise ImportError("pyswarms library required for PSO_DIRECT strategy")
    
    def select_next_point(self, X_pool: np.ndarray, hidden_indices: List[int], 
                         X_observed: np.ndarray, y_observed: np.ndarray,
                         current_best: float, **kwargs) -> int:
        """Select point using Particle Swarm Optimization on the true function"""
        self.iteration_count += 1
        
        if 'true_function' not in kwargs:
            return self.rng.choice(hidden_indices)
        
        true_function = kwargs['true_function']
        
        def objective(x):
            # x is a matrix where each row is a particle position
            costs = []
            for particle in x:
                # Find closest point in the pool
                distances = np.sum((X_pool - particle)**2, axis=1)
                closest_idx = np.argmin(distances)
                
                # Only consider hidden indices
                if closest_idx not in hidden_indices:
                    # Find closest among hidden indices
                    hidden_distances = [distances[i] for i in hidden_indices]
                    closest_hidden_idx = np.argmin(hidden_distances)
                    closest_idx = hidden_indices[closest_hidden_idx]
                
                costs.append(-true_function[closest_idx])  # Negative for minimization
            
            return np.array(costs)
        
        # Run PSO
        bounds = (np.zeros(X_pool.shape[1]), np.ones(X_pool.shape[1]))
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        
        try:
            optimizer = ps.single.GlobalBestPSO(
                n_particles=10, dimensions=X_pool.shape[1], 
                options=options, bounds=bounds
            )
            cost, x_best = optimizer.optimize(objective, iters=5)
            
            # Find closest point in hidden indices
            distances = [np.sum((X_pool[i] - x_best)**2) for i in hidden_indices]
            closest_idx = hidden_indices[np.argmin(distances)]
            return closest_idx
        except:
            return self.rng.choice(hidden_indices)


class SBOGPPVStrategy(OptimizationStrategy):
    """Surrogate-based optimization with GP predicting values - Optimized"""
    
    def __init__(self, random_state: int = 0, debug: bool = False, fast_mode: bool = True,
                 ucb_kappa: float = 1.0):
        super().__init__("SBO_GP_PV", random_state)
        self.debug = debug
        self.fast_mode = fast_mode
        self.ucb_kappa = ucb_kappa
    
    def select_next_point(self, X_pool: np.ndarray, hidden_indices: List[int], 
                         X_observed: np.ndarray, y_observed: np.ndarray,
                         current_best: float, **kwargs) -> int:
        """Select point with highest predicted value using optimized GP surrogate"""
        self.iteration_count += 1
        
        start_time = time.time()
        
        # Early termination: if we have very few points left, just pick randomly
        if len(hidden_indices) <= 2:
            return self.rng.choice(hidden_indices)
        
        # Generate cache key for GP model
        data_fingerprint = _get_data_fingerprint(X_observed, y_observed)
        
        # Determine GP complexity based on data size and fast_mode
        if self.fast_mode:
            complexity = "fast" if len(y_observed) < 50 else "medium"
        else:
            complexity = "medium" if len(y_observed) < 100 else "full"
        
        # Use fast GP fitting with caching
        try:
            gp, scaler = _fit_fast_gp(
                X_observed, y_observed, 
                complexity=complexity, 
                cache_key=data_fingerprint
            )
            
            if self.debug:
                print(f"SBO_GP_PV GP fitted in {time.time() - start_time:.3f}s (complexity: {complexity})")
                if data_fingerprint in _GP_CACHE:
                    print("Used cached GP model")
            
        except Exception as e:
            if self.debug:
                print(f"SBO_GP_PV GP fitting failed: {e}")
            return self.rng.choice(hidden_indices)
        
        # Optimize data types and vectorize operations
        X_pool = X_pool.astype(FAST_DTYPE)
        X_hidden = X_pool[hidden_indices]
        X_hidden_scaled = scaler.transform(X_hidden)
        
        # Vectorized prediction with optional exploration via UCB
        try:
            # Batch prediction for efficiency
            mean, std = gp.predict(X_hidden_scaled, return_std=True)
            if self.ucb_kappa is not None and self.ucb_kappa > 0:
                acquisition = mean + self.ucb_kappa * np.maximum(std, 1e-6)
            else:
                acquisition = mean
            
            # Select point with highest acquisition
            best_local_idx = np.argmax(acquisition)
            selected_idx = hidden_indices[best_local_idx]
            
            if self.debug:
                elapsed = time.time() - start_time
                print(f"SBO_GP_PV iteration {self.iteration_count} completed in {elapsed:.3f}s")
                print(f"  Selected point {selected_idx} with acquisition={acquisition[best_local_idx]:.6f}")
                print(f"  Mean range: [{np.min(mean):.6f}, {np.max(mean):.6f}], Std range: [{np.min(std):.6f}, {np.max(std):.6f}]")
            
            return selected_idx
            
        except Exception as e:
            if self.debug:
                print(f"SBO_GP_PV prediction failed: {e}")
            return self.rng.choice(hidden_indices)


class SBOANNPVStrategy(OptimizationStrategy):
    """Surrogate-based optimization with ANN predicting values - Optimized"""
    
    def __init__(self, random_state: int = 0, debug: bool = False):
        super().__init__("SBO_ANN_PV", random_state)
        self.debug = debug
    
    def select_next_point(self, X_pool: np.ndarray, hidden_indices: List[int], 
                         X_observed: np.ndarray, y_observed: np.ndarray,
                         current_best: float, **kwargs) -> int:
        """Select point with highest predicted value using optimized ANN surrogate"""
        self.iteration_count += 1
        
        start_time = time.time()
        
        # Early termination: if we have very few points left, just pick randomly
        if len(hidden_indices) <= 2:
            return self.rng.choice(hidden_indices)
        
        # Optimize data types for speed
        X_observed = X_observed.astype(FAST_DTYPE)
        y_observed = y_observed.astype(FAST_DTYPE)
        X_pool = X_pool.astype(FAST_DTYPE)
        
        # Fit ANN surrogate with optimized settings
        ann = MLPRegressor(
            hidden_layer_sizes=(50, 20),
            random_state=self.random_state,
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10
        )
        
        try:
            ann.fit(X_observed, y_observed)
            
            # Predict on hidden points
            X_hidden = X_pool[hidden_indices]
            predictions = ann.predict(X_hidden)
            
            # Select point with highest predicted value
            best_local_idx = np.argmax(predictions)
            selected_idx = hidden_indices[best_local_idx]
            
            if self.debug:
                elapsed = time.time() - start_time
                print(f"SBO_ANN_PV iteration {self.iteration_count} completed in {elapsed:.3f}s")
                print(f"  Selected point {selected_idx} with predicted value={predictions[best_local_idx]:.6f}")
                print(f"  Prediction range: [{np.min(predictions):.6f}, {np.max(predictions):.6f}]")
            
            return selected_idx
        except Exception as e:
            if self.debug:
                print(f"SBO_ANN_PV failed: {e}, falling back to random")
            # Fallback to random if ANN fails
            return self.rng.choice(hidden_indices)


class SBOPolyPVStrategy(OptimizationStrategy):
    """Surrogate-based optimization with polynomial regression predicting values - Optimized"""
    
    def __init__(self, random_state: int = 0, debug: bool = False):
        super().__init__("SBO_POLY_PV", random_state)
        self.debug = debug
    
    def select_next_point(self, X_pool: np.ndarray, hidden_indices: List[int], 
                         X_observed: np.ndarray, y_observed: np.ndarray,
                         current_best: float, **kwargs) -> int:
        """Select point with highest predicted value using optimized polynomial surrogate"""
        self.iteration_count += 1
        
        start_time = time.time()
        
        # Early termination: if we have very few points left, just pick randomly
        if len(hidden_indices) <= 2:
            return self.rng.choice(hidden_indices)
        
        # Optimize data types for speed
        X_observed = X_observed.astype(FAST_DTYPE)
        y_observed = y_observed.astype(FAST_DTYPE)
        X_pool = X_pool.astype(FAST_DTYPE)
        
        # Fit polynomial surrogate with optimized settings
        poly_features = PolynomialFeatures(degree=2, include_bias=False)
        poly_reg = Pipeline([
            ('poly', poly_features),
            ('linear', LinearRegression())
        ])
        
        try:
            poly_reg.fit(X_observed, y_observed)
            
            # Predict on hidden points
            X_hidden = X_pool[hidden_indices]
            predictions = poly_reg.predict(X_hidden)
            
            # Select point with highest predicted value
            best_local_idx = np.argmax(predictions)
            selected_idx = hidden_indices[best_local_idx]
            
            if self.debug:
                elapsed = time.time() - start_time
                print(f"SBO_POLY_PV iteration {self.iteration_count} completed in {elapsed:.3f}s")
                print(f"  Selected point {selected_idx} with predicted value={predictions[best_local_idx]:.6f}")
                print(f"  Prediction range: [{np.min(predictions):.6f}, {np.max(predictions):.6f}]")
            
            return selected_idx
        except Exception as e:
            if self.debug:
                print(f"SBO_POLY_PV failed: {e}, falling back to random")
            # Fallback to random if polynomial regression fails
            return self.rng.choice(hidden_indices)


class SBOGPEITruncDEStrategy(OptimizationStrategy):
    """GP surrogate with EI acquisition optimized using truncated DE - Optimized"""
    
    def __init__(self, random_state: int = 0, debug: bool = False, fast_mode: bool = True):
        super().__init__("SBO_GP_EI_TRUNCDE", random_state)
        self.debug = debug
        self.fast_mode = fast_mode
    
    def select_next_point(self, X_pool: np.ndarray, hidden_indices: List[int], 
                         X_observed: np.ndarray, y_observed: np.ndarray,
                         current_best: float, **kwargs) -> int:
        """Select point by optimizing EI with truncated DE using optimized GP"""
        self.iteration_count += 1
        
        start_time = time.time()
        
        # Early termination: if we have very few points left, just pick randomly
        if len(hidden_indices) <= 2:
            return self.rng.choice(hidden_indices)
        
        # Generate cache key for GP model
        data_fingerprint = _get_data_fingerprint(X_observed, y_observed)
        
        # Determine GP complexity based on data size and fast_mode
        if self.fast_mode:
            complexity = "fast" if len(y_observed) < 50 else "medium"
        else:
            complexity = "medium" if len(y_observed) < 100 else "full"
        
        # Use fast GP fitting with caching
        try:
            gp, scaler = _fit_fast_gp(
                X_observed, y_observed, 
                complexity=complexity, 
                cache_key=data_fingerprint
            )
            
            if self.debug:
                print(f"SBO_GP_EI_TRUNCDE GP fitted in {time.time() - start_time:.3f}s (complexity: {complexity})")
                if data_fingerprint in _GP_CACHE:
                    print("Used cached GP model")
            
        except Exception as e:
            if self.debug:
                print(f"SBO_GP_EI_TRUNCDE GP fitting failed: {e}")
            return self.rng.choice(hidden_indices)
        
        # Optimize data types and vectorize operations
        X_pool = X_pool.astype(FAST_DTYPE)
        X_hidden = X_pool[hidden_indices]
        X_hidden_scaled = scaler.transform(X_hidden)
        
        # Vectorized EI calculation on all hidden points (like BO_GP_EI)
        try:
            # Batch prediction for efficiency
            mean, std = gp.predict(X_hidden_scaled, return_std=True)
            
            # Vectorized EI computation
            std = np.maximum(std, 1e-6)
            ei = self._expected_improvement_vectorized(mean, std, current_best)
            
            if self.debug:
                print(f"  EI range: [{np.min(ei):.6f}, {np.max(ei):.6f}]")
                print(f"  EI std: {np.std(ei):.6f}")
            
            # Find the best candidate (like BO_GP_EI)
            best_local_idx = np.argmax(ei)
            best_ei = ei[best_local_idx]
            best_candidate = X_hidden[best_local_idx]
            
            # Run a small DE optimization around the best candidate
            bounds = []
            for i in range(len(best_candidate)):
                # Create bounds around the best candidate point
                margin = 0.2 * (X_pool[:, i].max() - X_pool[:, i].min())
                bounds.append((best_candidate[i] - margin, best_candidate[i] + margin))
            
            def local_ei_objective(x):
                # Scale the input using the same scaler
                x_scaled = scaler.transform(x.reshape(1, -1))
                mean, std = gp.predict(x_scaled, return_std=True)
                ei = self._expected_improvement_vectorized(mean, std, current_best)
                return -ei[0]  # Negative for minimization
            
            try:
                result = differential_evolution(
                    local_ei_objective, bounds, maxiter=2, popsize=2,
                    seed=self.random_state, atol=1e-6, tol=1e-6
                )
                
                if result.success:
                    x_optimized = result.x
                    # Find closest point in hidden indices
                    distances = [np.sum((X_hidden[i] - x_optimized)**2) for i in range(len(hidden_indices))]
                    closest_local_idx = np.argmin(distances)
                    closest_global_idx = hidden_indices[closest_local_idx]
                    
                    # Use the optimized point if it's better, otherwise use the original best
                    x_optimized_scaled = scaler.transform(x_optimized.reshape(1, -1))
                    opt_mean, opt_std = gp.predict(x_optimized_scaled, return_std=True)
                    opt_ei = self._expected_improvement_vectorized(opt_mean, opt_std, current_best)[0]
                    
                    if opt_ei > best_ei:
                        selected_idx = closest_global_idx
                        if self.debug:
                            print(f"  DE optimization improved EI from {best_ei:.6f} to {opt_ei:.6f}")
                    else:
                        selected_idx = hidden_indices[best_local_idx]
                        if self.debug:
                            print(f"  Using original best candidate (EI={best_ei:.6f})")
                else:
                    selected_idx = hidden_indices[best_local_idx]
                    if self.debug:
                        print(f"  DE optimization failed, using original best candidate")
                        
            except Exception as e:
                if self.debug:
                    print(f"  DE optimization failed: {e}, using original best candidate")
                selected_idx = hidden_indices[best_local_idx]
            
            if self.debug:
                elapsed = time.time() - start_time
                print(f"SBO_GP_EI_TRUNCDE iteration {self.iteration_count} completed in {elapsed:.3f}s")
                print(f"  Selected point {selected_idx}")
            
            return selected_idx
            
        except Exception as e:
            if self.debug:
                print(f"SBO_GP_EI_TRUNCDE failed: {e}, falling back to random")
            return self.rng.choice(hidden_indices)
    
    def _expected_improvement(self, mean: np.ndarray, std: np.ndarray, f_best: float) -> np.ndarray:
        """Calculate Expected Improvement"""
        std = std.copy()
        std[std < 1e-9] = 1e-9
        z = (mean - f_best) / std
        ei = (mean - f_best) * norm.cdf(z) + std * norm.pdf(z)
        return np.maximum(ei, 0.0)  # Ensure non-negative EI like BO_GP_EI
    
    def _expected_improvement_vectorized(self, mean: np.ndarray, std: np.ndarray, f_best: float) -> np.ndarray:
        """Vectorized Expected Improvement calculation"""
        z = (mean - f_best) / std
        ei = (mean - f_best) * norm.cdf(z) + std * norm.pdf(z)
        return np.maximum(ei, 0.0)  # Ensure non-negative


class SmartBOStrategy(OptimizationStrategy):
    """
    Smart Bayesian Optimization for High-Dimensional Problems
    
    Combines multiple techniques:
    - Adaptive feature selection based on correlation/importance
    - Random embedding to lower dimensions
    - Trust region search around promising areas
    - Ensemble of GP models for robust uncertainty
    """
    
    def __init__(self, random_state: int = 0, debug: bool = False):
        super().__init__("SMART_BO", random_state)
        self.debug = debug
        self.scaler = None
        self.feature_selector = None
        self.embedding_matrix = None
        self.embedding_dim = None
        self.trust_region_center = None
        self.trust_region_radius = 0.3
        self.ensemble_gps = []
        self.feature_importance_scores = None
        
    def select_next_point(self, X_pool: np.ndarray, hidden_indices: List[int], 
                         X_observed: np.ndarray, y_observed: np.ndarray,
                         current_best: float, **kwargs) -> int:
        """Select point using smart high-dimensional BO techniques"""
        self.iteration_count += 1
        
        if self.debug:
            print(f"\n--- SMART_BO Debug - Iteration {self.iteration_count} ---")
            print(f"X_observed shape: {X_observed.shape}")
            print(f"Hidden points remaining: {len(hidden_indices)}")
        
        # Step 1: Initialize or update feature scaling
        if self.scaler is None:
            self.scaler = StandardScaler()
            X_observed_scaled = self.scaler.fit_transform(X_observed)
        else:
            X_observed_scaled = self.scaler.transform(X_observed)
            
        X_pool_scaled = self.scaler.transform(X_pool)
        X_hidden_scaled = X_pool_scaled[hidden_indices]
        
        # Step 2: Adaptive feature selection (every few iterations)
        if self.iteration_count <= 3 or self.iteration_count % 5 == 0:
            self._update_feature_importance(X_observed_scaled, y_observed)
            
        # Step 3: Determine embedding dimension based on data size
        n_features = X_pool.shape[1]
        if self.embedding_dim is None:
            # Start with smaller embedding for high-dim problems
            self.embedding_dim = min(max(3, len(X_observed) // 3), n_features // 2, 8)
            if self.debug:
                print(f"Using embedding dimension: {self.embedding_dim}")
        
        # Step 4: Create/update random embedding matrix
        if self.embedding_matrix is None or self.iteration_count % 10 == 0:
            self._create_embedding_matrix(n_features)
            
        # Step 5: Project to lower dimensional space using feature selection + embedding
        X_observed_embedded = self._embed_features(X_observed_scaled)
        X_hidden_embedded = self._embed_features(X_hidden_scaled)
        
        if self.debug:
            print(f"Embedded to {X_observed_embedded.shape[1]}D (from {n_features}D)")
        
        # Step 6: Update trust region around current best
        self._update_trust_region(X_observed_embedded, y_observed)
        
        # Step 7: Filter candidates within trust region
        trust_candidates = self._filter_trust_region_candidates(
            X_hidden_embedded, hidden_indices
        )
        
        if len(trust_candidates) == 0:
            # Expand trust region if no candidates
            self.trust_region_radius *= 1.5
            trust_candidates = self._filter_trust_region_candidates(
                X_hidden_embedded, hidden_indices
            )
            if self.debug:
                print(f"Expanded trust region to radius {self.trust_region_radius:.3f}")
        
        if len(trust_candidates) == 0:
            # Fallback to all candidates if trust region is still empty
            trust_candidates = list(range(len(hidden_indices)))
            if self.debug:
                print("Using all candidates as fallback")
        
        # Step 8: Train ensemble of GP models
        try:
            self._train_ensemble_gps(X_observed_embedded, y_observed)
            
            # Step 9: Calculate ensemble predictions and uncertainty
            ensemble_means, ensemble_stds = self._ensemble_predict(
                X_hidden_embedded[trust_candidates]
            )
            
            # Step 10: Calculate Expected Improvement with uncertainty bonus
            ei_values = self._calculate_smart_ei(
                ensemble_means, ensemble_stds, current_best
            )
            
            if self.debug:
                print(f"Trust region candidates: {len(trust_candidates)}")
                print(f"EI range: [{np.min(ei_values):.6f}, {np.max(ei_values):.6f}]")
                print(f"EI std: {np.std(ei_values):.6f}")
                
                # Show top candidates
                top_3 = np.argsort(ei_values)[-3:][::-1]
                print("Top 3 candidates:")
                for i, idx in enumerate(top_3):
                    global_idx = hidden_indices[trust_candidates[idx]]
                    print(f"  {i+1}. Index {global_idx}: EI={ei_values[idx]:.6f}")
            
            # Select best candidate within trust region
            best_trust_idx = np.argmax(ei_values)
            best_hidden_idx = trust_candidates[best_trust_idx]
            selected_global_idx = hidden_indices[best_hidden_idx]
            
            # Adaptive trust region adjustment
            if ei_values[best_trust_idx] > np.mean(ei_values) + np.std(ei_values):
                # Good candidate found, shrink trust region for exploitation
                self.trust_region_radius *= 0.9
            else:
                # No great candidates, expand for exploration
                self.trust_region_radius *= 1.1
                
            self.trust_region_radius = np.clip(self.trust_region_radius, 0.1, 1.0)
            
            if self.debug:
                print(f"Selected: global index {selected_global_idx}")
                print(f"Adjusted trust region radius: {self.trust_region_radius:.3f}")
            
            return selected_global_idx
            
        except Exception as e:
            if self.debug:
                print(f"Smart BO failed: {e}, falling back to random")
            return self.rng.choice(hidden_indices)
    
    def _update_feature_importance(self, X_observed: np.ndarray, y_observed: np.ndarray):
        """Update feature importance scores using correlation and variance"""
        # Calculate correlation-based importance
        correlations = []
        for i in range(X_observed.shape[1]):
            corr = np.abs(np.corrcoef(X_observed[:, i], y_observed)[0, 1])
            if np.isnan(corr):
                corr = 0.0
            correlations.append(corr)
        
        # Calculate variance-based importance (features with low variance are less useful)
        variances = np.var(X_observed, axis=0)
        variance_scores = variances / (np.max(variances) + 1e-6)
        
        # Combine correlation and variance scores
        correlation_scores = np.array(correlations)
        self.feature_importance_scores = 0.7 * correlation_scores + 0.3 * variance_scores
        
        if self.debug:
            print(f"Feature importance: max={np.max(self.feature_importance_scores):.3f}, "
                  f"mean={np.mean(self.feature_importance_scores):.3f}")
    
    def _create_embedding_matrix(self, n_features: int):
        """Create random embedding matrix weighted by feature importance"""
        if self.feature_importance_scores is not None:
            # Weight random projection by feature importance
            importance_weights = self.feature_importance_scores / np.sum(self.feature_importance_scores)
            
            # Create weighted random matrix
            self.embedding_matrix = self.rng.normal(0, 1, (n_features, self.embedding_dim))
            
            # Scale each row by importance
            for i in range(n_features):
                self.embedding_matrix[i, :] *= np.sqrt(importance_weights[i] + 0.1)  # +0.1 to avoid zero
        else:
            # Standard random projection
            self.embedding_matrix = self.rng.normal(0, 1, (n_features, self.embedding_dim))
            
        # Normalize columns
        for j in range(self.embedding_dim):
            norm = np.linalg.norm(self.embedding_matrix[:, j])
            if norm > 0:
                self.embedding_matrix[:, j] /= norm
    
    def _embed_features(self, X: np.ndarray) -> np.ndarray:
        """Project features to lower dimensional embedding"""
        return X @ self.embedding_matrix
    
    def _update_trust_region(self, X_observed_embedded: np.ndarray, y_observed: np.ndarray):
        """Update trust region center around current best point"""
        best_idx = np.argmax(y_observed)
        self.trust_region_center = X_observed_embedded[best_idx].copy()
    
    def _filter_trust_region_candidates(self, X_hidden_embedded: np.ndarray, 
                                       hidden_indices: List[int]) -> List[int]:
        """Filter candidates within trust region"""
        if self.trust_region_center is None:
            return list(range(len(hidden_indices)))
        
        # Calculate distances to trust region center
        distances = np.linalg.norm(
            X_hidden_embedded - self.trust_region_center.reshape(1, -1), axis=1
        )
        
        # Return indices within trust region
        within_region = distances <= self.trust_region_radius
        return np.where(within_region)[0].tolist()
    
    def _train_ensemble_gps(self, X_observed_embedded: np.ndarray, y_observed: np.ndarray):
        """Train ensemble of GP models with different kernels - Optimized"""
        self.ensemble_gps = []
        
        # Use fast GP fitting for ensemble
        complexities = ["fast", "fast", "medium"]  # Reduced complexity for speed
        
        for i, complexity in enumerate(complexities):
            try:
                # Generate unique cache key for each ensemble member
                data_fingerprint = _get_data_fingerprint(X_observed_embedded, y_observed)
                cache_key = f"{data_fingerprint}_ensemble_{i}"
                
                gp, scaler = _fit_fast_gp(
                    X_observed_embedded, y_observed,
                    complexity=complexity,
                    cache_key=cache_key
                )
                
                # Store both GP and scaler for ensemble
                self.ensemble_gps.append((gp, scaler))
                
            except Exception as e:
                if self.debug:
                    print(f"GP {i} training failed: {e}")
                continue
    
    def _ensemble_predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get ensemble predictions with uncertainty"""
        if len(self.ensemble_gps) == 0:
            raise ValueError("No trained GPs in ensemble")
        
        all_means = []
        all_stds = []
        
        for gp, scaler in self.ensemble_gps:  # Unpack both GP and scaler
            try:
                # Scale the test data using this ensemble member's scaler
                X_test_scaled = scaler.transform(X_test)
                mean, std = gp.predict(X_test_scaled, return_std=True)
                all_means.append(mean)
                all_stds.append(std)
            except Exception as e:
                if self.debug:
                    print(f"Ensemble GP prediction failed: {e}")
                continue
        
        if len(all_means) == 0:
            raise ValueError("All ensemble GPs failed to predict")
        
        # Ensemble mean is average of individual means
        ensemble_mean = np.mean(all_means, axis=0)
        
        # Ensemble uncertainty combines aleatoric and epistemic uncertainty
        mean_of_vars = np.mean([std**2 for std in all_stds], axis=0)  # Aleatoric
        var_of_means = np.var(all_means, axis=0)  # Epistemic
        ensemble_std = np.sqrt(mean_of_vars + var_of_means)
        
        return ensemble_mean, ensemble_std
    
    def _calculate_smart_ei(self, mean: np.ndarray, std: np.ndarray, 
                           current_best: float) -> np.ndarray:
        """Calculate Expected Improvement with exploration bonus"""
        # Standard EI
        std = np.maximum(std, 1e-6)
        z = (mean - current_best) / std
        ei = (mean - current_best) * norm.cdf(z) + std * norm.pdf(z)
        
        # Add exploration bonus for high uncertainty
        exploration_bonus = 0.1 * std  # Encourage exploration in uncertain regions
        
        return ei + exploration_bonus


class FullFactorialDesignStrategy(OptimizationStrategy):
    """Full Factorial Design - Tests every combination of levels for all factors - Optimized"""
    
    def __init__(self, random_state: int = 0, n_levels: int = 3):
        super().__init__("FULL_FACTORIAL", random_state)
        self.n_levels = n_levels
        self.design_order = None
        self.design_initialized = False
        self.nn_model = None  # For fast nearest neighbor search
    
    def select_next_point(self, X_pool: np.ndarray, hidden_indices: List[int], 
                         X_observed: np.ndarray, y_observed: np.ndarray,
                         current_best: float, **kwargs) -> int:
        """Select point according to full factorial design order - Optimized"""
        self.iteration_count += 1
        
        # Early termination for small datasets
        if len(hidden_indices) <= self.n_levels:
            return self.rng.choice(hidden_indices)
        
        if not self.design_initialized:
            self._initialize_design_fast(X_pool, hidden_indices)
        
        # Select next valid point from design order (skip invalid without random fallback)
        while self.design_order:
            idx = self.design_order.pop(0)
            if idx in hidden_indices:
                return idx
        
        # If design exhausted, return random as last resort
        else:
            # Fallback to random if design is exhausted
            return self.rng.choice(hidden_indices)
    
    def _initialize_design_fast(self, X_pool: np.ndarray, hidden_indices: List[int]):
        """Fast initialization using vectorized operations and caching"""
        X_hidden = X_pool[hidden_indices].astype(FAST_DTYPE)
        n_features = min(X_hidden.shape[1], 10)  # Limit dimensions for factorial explosion
        
        # Build a cache key that is specific to the actual hidden pool content
        x_fp = _get_X_fingerprint(X_hidden[:, :n_features])
        hidden_fp = hashlib.md5(np.array(hidden_indices, dtype=np.int64).tobytes()).hexdigest()
        cache_key = f"factorial_{self.n_levels}_{n_features}_{len(hidden_indices)}_{x_fp}_{hidden_fp}"
        if cache_key in _DESIGN_CACHE:
            self.design_order = _DESIGN_CACHE[cache_key].copy()
            self.design_initialized = True
            return
        
        # Create factorial design points vectorized
        level_indices = np.arange(self.n_levels)
        factorial_grid = np.array(np.meshgrid(*[level_indices] * n_features)).T.reshape(-1, n_features)
        
        # Scale to data range
        X_min, X_max = X_hidden[:, :n_features].min(axis=0), X_hidden[:, :n_features].max(axis=0)
        factorial_coords = X_min + (factorial_grid / (self.n_levels - 1)) * (X_max - X_min)
        
        # Use KNN for fast nearest neighbor search
        if self.nn_model is None:
            self.nn_model = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')
            
        self.nn_model.fit(X_hidden[:, :n_features])
        
        # Unique nearest neighbor assignment to reduce collapse
        design_points = _assign_unique_nearest_indices(
            X_hidden[:, :n_features], factorial_coords, hidden_indices
        )
        
        # Remove duplicates while preserving order
        seen = set()
        unique_design_points = []
        for point in design_points:
            if point not in seen:
                seen.add(point)
                unique_design_points.append(point)
        
        # Randomize order
        self.rng.shuffle(unique_design_points)
        self.design_order = unique_design_points
        
        # Cache the result scoped to this exact hidden pool
        with _CACHE_LOCK:
            _DESIGN_CACHE[cache_key] = self.design_order.copy()
        
        self.design_initialized = True


class FractionalFactorialDesignStrategy(OptimizationStrategy):
    """Fractional Factorial Design - Uses subset of factorial combinations"""
    
    def __init__(self, random_state: int = 0, n_levels: int = 3, fraction: float = 0.5):
        super().__init__("FRACTIONAL_FACTORIAL", random_state)
        self.n_levels = n_levels
        self.fraction = fraction
        self.design_order = None
        self.factor_bins = None
        self.design_initialized = False
    
    def select_next_point(self, X_pool: np.ndarray, hidden_indices: List[int], 
                         X_observed: np.ndarray, y_observed: np.ndarray,
                         current_best: float, **kwargs) -> int:
        """Select point according to fractional factorial design order"""
        self.iteration_count += 1
        
        if not self.design_initialized:
            self._initialize_design(X_pool, hidden_indices)
        
        # Select next valid point from design order
        while self.design_order:
            idx = self.design_order.pop(0)
            if idx in hidden_indices:
                return idx
        else:
            # Fallback to random if design is exhausted
            return self.rng.choice(hidden_indices)
    
    def _initialize_design(self, X_pool: np.ndarray, hidden_indices: List[int]):
        """Initialize fractional factorial design"""
        X_hidden = X_pool[hidden_indices]
        # Cap the number of features used for factorial construction to avoid combinatorial explosion
        n_features_total = X_hidden.shape[1]
        n_features = min(n_features_total, 10)
        X_hidden_eff = X_hidden[:, :n_features]
        
        # Create bins for each feature
        self.factor_bins = []
        for i in range(n_features):
            feature_values = X_hidden_eff[:, i]
            bins = np.linspace(feature_values.min(), feature_values.max(), self.n_levels + 1)
            self.factor_bins.append(bins)
        
        # Avoid materializing the full factorial grid (explodes with dimensions)
        # Estimate target sample count and cap for safety
        estimated_total = self.n_levels ** n_features
        n_selected = max(1, int(estimated_total * self.fraction))
        n_selected = min(n_selected, 2000)  # hard cap for performance
        
        # Randomly sample level combinations
        self.rng.seed(self.random_state)
        selected_combinations = [
            tuple(self.rng.randint(0, self.n_levels, size=n_features))
            for _ in range(n_selected)
        ]
        
        # Map to continuous coordinates at bin centers
        target_coords = []
        for combo in selected_combinations:
            coord = []
            for i, level in enumerate(combo):
                bins = self.factor_bins[i]
                coord.append((bins[level] + bins[level + 1]) / 2)
            target_coords.append(coord)
        target_coords = np.array(target_coords)

        # Assign unique nearest indices to reduce duplicates
        design_points = _assign_unique_nearest_indices(
            X_hidden_eff, target_coords, hidden_indices
        )
        
        self.rng.shuffle(design_points)
        self.design_order = design_points
        self.design_initialized = True
    
    def _find_closest_point(self, X_hidden: np.ndarray, factor_combo: Tuple, 
                           hidden_indices: List[int]) -> Optional[int]:
        """Find the hidden point closest to a factorial combination"""
        target_coords = []
        for i, level in enumerate(factor_combo):
            bins = self.factor_bins[i]
            target_coord = (bins[level] + bins[level + 1]) / 2
            target_coords.append(target_coord)
        
        target_coords = np.array(target_coords)
        distances = np.linalg.norm(X_hidden - target_coords, axis=1)
        closest_idx = np.argmin(distances)
        
        return hidden_indices[closest_idx]


class PlackettBurmanDesignStrategy(OptimizationStrategy):
    """Plackett-Burman Design - Efficient screening design for main effects"""
    
    def __init__(self, random_state: int = 0):
        super().__init__("PLACKETT_BURMAN", random_state)
        self.design_order = None
        self.design_initialized = False
        self.pb_matrix = None
    
    def select_next_point(self, X_pool: np.ndarray, hidden_indices: List[int], 
                         X_observed: np.ndarray, y_observed: np.ndarray,
                         current_best: float, **kwargs) -> int:
        """Select point according to Plackett-Burman design"""
        self.iteration_count += 1
        
        if not self.design_initialized:
            self._initialize_design(X_pool, hidden_indices)
        
        # Select next valid point from design order
        while self.design_order:
            idx = self.design_order.pop(0)
            if idx in hidden_indices:
                return idx
        else:
            return self.rng.choice(hidden_indices)
    
    def _initialize_design(self, X_pool: np.ndarray, hidden_indices: List[int]):
        """Initialize Plackett-Burman design"""
        X_hidden = X_pool[hidden_indices]
        n_features = X_hidden.shape[1]
        
        # Generate Plackett-Burman matrix
        self.pb_matrix = self._generate_pb_matrix(n_features)
        
        # Normalize hidden data to [-1, 1] range for matching
        X_normalized = self._normalize_to_pm1(X_hidden)
        
        # Assign unique nearest indices for PB rows
        design_points = _assign_unique_nearest_indices(
            X_normalized, self.pb_matrix, hidden_indices
        )
        
        self.rng.shuffle(design_points)
        self.design_order = design_points
        self.design_initialized = True
    
    def _generate_pb_matrix(self, n_factors: int) -> np.ndarray:
        """Generate Plackett-Burman design matrix"""
        # Find appropriate PB design size (multiple of 4, >= n_factors)
        n_runs = 4
        while n_runs < n_factors:
            n_runs += 4
        
        # Use Hadamard-like construction for PB design
        if n_runs == 4:
            base_row = np.array([1, 1, -1])
        elif n_runs == 8:
            base_row = np.array([1, 1, 1, -1, 1, -1, -1])
        elif n_runs == 12:
            base_row = np.array([1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1])
        else:
            # Fallback: use random +1/-1 pattern
            base_row = self.rng.choice([-1, 1], size=n_runs-1)
        
        # Generate PB matrix by cyclic permutation
        pb_matrix = []
        for i in range(n_runs-1):
            row = np.roll(base_row, i)[:n_factors]  # Take only needed factors
            pb_matrix.append(row)
        
        # Add all -1 row
        pb_matrix.append(np.full(n_factors, -1))
        
        return np.array(pb_matrix)
    
    def _normalize_to_pm1(self, X: np.ndarray) -> np.ndarray:
        """Normalize features to [-1, 1] range"""
        X_norm = np.zeros_like(X)
        for i in range(X.shape[1]):
            x_min, x_max = X[:, i].min(), X[:, i].max()
            if x_max > x_min:
                X_norm[:, i] = 2 * (X[:, i] - x_min) / (x_max - x_min) - 1
            else:
                X_norm[:, i] = 0
        return X_norm
    
    def _find_closest_normalized_point(self, X_normalized: np.ndarray, 
                                     pb_row: np.ndarray, 
                                     hidden_indices: List[int]) -> Optional[int]:
        """Find point closest to PB design point"""
        distances = np.linalg.norm(X_normalized - pb_row, axis=1)
        closest_idx = np.argmin(distances)
        return hidden_indices[closest_idx]


class CentralCompositeDesignStrategy(OptimizationStrategy):
    """Central Composite Design - Response surface methodology design"""
    
    def __init__(self, random_state: int = 0, alpha: float = 1.414):
        super().__init__("CENTRAL_COMPOSITE", random_state)
        self.alpha = alpha  # Distance for star points
        self.design_order = None
        self.design_initialized = False
        self.surrogate_model = None
    
    def select_next_point(self, X_pool: np.ndarray, hidden_indices: List[int], 
                         X_observed: np.ndarray, y_observed: np.ndarray,
                         current_best: float, **kwargs) -> int:
        """Select point according to Central Composite Design"""
        self.iteration_count += 1
        
        if not self.design_initialized:
            self._initialize_design(X_pool, hidden_indices, X_observed, y_observed)
        
        # Select next valid point from design order
        while self.design_order:
            idx = self.design_order.pop(0)
            if idx in hidden_indices:
                return idx
        else:
            return self.rng.choice(hidden_indices)
    
    def _initialize_design(self, X_pool: np.ndarray, hidden_indices: List[int],
                          X_observed: np.ndarray, y_observed: np.ndarray):
        """Initialize Central Composite Design"""
        X_hidden = X_pool[hidden_indices]
        # Cap the number of features used to build CCD to control size
        n_features_total = X_hidden.shape[1]
        n_features = min(n_features_total, 10)
        X_hidden_eff = X_hidden[:, :n_features]
        
        # Fit surrogate model to guide design
        self._fit_surrogate(X_observed, y_observed)
        
        # Generate CCD design points
        ccd_points = self._generate_ccd_points(n_features)
        
        # Normalize to data range
        X_min, X_max = X_hidden_eff.min(axis=0), X_hidden_eff.max(axis=0)
        ccd_points_scaled = []
        for point in ccd_points:
            scaled_point = X_min + (point + 1) / 2 * (X_max - X_min)
            ccd_points_scaled.append(scaled_point)
        
        # Assign unique nearest indices for CCD points
        design_points = _assign_unique_nearest_indices(
            X_hidden_eff, np.array(ccd_points_scaled), hidden_indices
        )
        
        self.rng.shuffle(design_points)
        self.design_order = design_points
        self.design_initialized = True
    
    def _generate_ccd_points(self, n_factors: int) -> List[np.ndarray]:
        """Generate Central Composite Design points"""
        ccd_points = []
        
        # 1. Factorial points (2^k corners)
        factorial_points = list(product([-1, 1], repeat=n_factors))
        ccd_points.extend([np.array(point) for point in factorial_points])
        
        # 2. Star points (axial points)
        for i in range(n_factors):
            # Positive star point
            star_pos = np.zeros(n_factors)
            star_pos[i] = self.alpha
            ccd_points.append(star_pos)
            
            # Negative star point
            star_neg = np.zeros(n_factors)
            star_neg[i] = -self.alpha
            ccd_points.append(star_neg)
        
        # 3. Center points
        center_point = np.zeros(n_factors)
        ccd_points.append(center_point)
        
        return ccd_points
    
    def _fit_surrogate(self, X_observed: np.ndarray, y_observed: np.ndarray):
        """Fit quadratic surrogate model"""
        # Use polynomial features for quadratic model
        poly_features = PolynomialFeatures(degree=2, include_bias=False)
        scaler = StandardScaler()
        
        self.surrogate_model = Pipeline([
            ('scaler', scaler),
            ('poly', poly_features),
            ('regressor', LinearRegression())
        ])
        
        try:
            self.surrogate_model.fit(X_observed, y_observed)
        except:
            # Fallback to simple linear model
            self.surrogate_model = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', LinearRegression())
            ])
            self.surrogate_model.fit(X_observed, y_observed)
    
    def _find_closest_ccd_point(self, X_hidden: np.ndarray, 
                               ccd_point: np.ndarray, 
                               hidden_indices: List[int]) -> Optional[int]:
        """Find hidden point closest to CCD design point"""
        distances = np.linalg.norm(X_hidden - ccd_point, axis=1)
        closest_idx = np.argmin(distances)
        return hidden_indices[closest_idx]


class BoxBehnkenDesignStrategy(OptimizationStrategy):
    """Box-Behnken Design - Response surface design avoiding extreme combinations"""
    
    def __init__(self, random_state: int = 0):
        super().__init__("BOX_BEHNKEN", random_state)
        self.design_order = None
        self.design_initialized = False
        self.surrogate_model = None
    
    def select_next_point(self, X_pool: np.ndarray, hidden_indices: List[int], 
                         X_observed: np.ndarray, y_observed: np.ndarray,
                         current_best: float, **kwargs) -> int:
        """Select point according to Box-Behnken design"""
        self.iteration_count += 1
        
        if not self.design_initialized:
            self._initialize_design(X_pool, hidden_indices, X_observed, y_observed)
        
        # Select next valid point from design order
        while self.design_order:
            idx = self.design_order.pop(0)
            if idx in hidden_indices:
                return idx
        else:
            return self.rng.choice(hidden_indices)
    
    def _initialize_design(self, X_pool: np.ndarray, hidden_indices: List[int],
                          X_observed: np.ndarray, y_observed: np.ndarray):
        """Initialize Box-Behnken design"""
        X_hidden = X_pool[hidden_indices]
        n_features = X_hidden.shape[1]
        
        # Fit surrogate model
        self._fit_surrogate(X_observed, y_observed)
        
        # Generate Box-Behnken design points
        bb_points = self._generate_bb_points(n_features)
        
        # Scale to data range
        X_min, X_max = X_hidden.min(axis=0), X_hidden.max(axis=0)
        bb_points_scaled = []
        for point in bb_points:
            scaled_point = X_min + (point + 1) / 2 * (X_max - X_min)
            bb_points_scaled.append(scaled_point)
        
        # Assign unique nearest indices for BB points
        design_points = _assign_unique_nearest_indices(
            X_hidden, np.array(bb_points_scaled), hidden_indices
        )
        
        self.rng.shuffle(design_points)
        self.design_order = design_points
        self.design_initialized = True
    
    def _generate_bb_points(self, n_factors: int) -> List[np.ndarray]:
        """Generate Box-Behnken design points"""
        bb_points = []
        
        # Box-Behnken uses midpoints of edges (avoiding corners)
        # For each pair of factors, create edge midpoints
        if n_factors >= 2:
            for i in range(n_factors):
                for j in range(i + 1, n_factors):
                    # Create 4 points for each factor pair
                    for val_i in [-1, 1]:
                        for val_j in [-1, 1]:
                            point = np.zeros(n_factors)
                            point[i] = val_i
                            point[j] = val_j
                            # Other factors stay at center (0)
                            bb_points.append(point)
        
        # Add center points
        center_point = np.zeros(n_factors)
        bb_points.append(center_point)
        
        return bb_points
    
    def _fit_surrogate(self, X_observed: np.ndarray, y_observed: np.ndarray):
        """Fit quadratic surrogate model"""
        poly_features = PolynomialFeatures(degree=2, include_bias=False)
        scaler = StandardScaler()
        
        self.surrogate_model = Pipeline([
            ('scaler', scaler),
            ('poly', poly_features),
            ('regressor', LinearRegression())
        ])
        
        try:
            self.surrogate_model.fit(X_observed, y_observed)
        except:
            # Fallback to linear model
            self.surrogate_model = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', LinearRegression())
            ])
            self.surrogate_model.fit(X_observed, y_observed)
    
    def _find_closest_bb_point(self, X_hidden: np.ndarray, 
                              bb_point: np.ndarray, 
                              hidden_indices: List[int]) -> Optional[int]:
        """Find hidden point closest to Box-Behnken design point"""
        distances = np.linalg.norm(X_hidden - bb_point, axis=1)
        closest_idx = np.argmin(distances)
        return hidden_indices[closest_idx]


class LatinHypercubeStrategy(OptimizationStrategy):
    """Latin Hypercube Sampling - Stratified sampling for uniform coverage - Optimized"""
    
    def __init__(self, random_state: int = 0, n_samples: int = 50):
        super().__init__("LATIN_HYPERCUBE", random_state)
        self.n_samples = n_samples
        self.point_order = None
        self.order_index = 0
        self.design_initialized = False
        self.nn_model = None
    
    def select_next_point(self, X_pool: np.ndarray, hidden_indices: List[int], 
                         X_observed: np.ndarray, y_observed: np.ndarray,
                         current_best: float, **kwargs) -> int:
        """Select point closest to next LHS coordinate - Optimized"""
        self.iteration_count += 1
        
        if not self.design_initialized:
            self._initialize_design_fast(X_pool, hidden_indices)
        
        if self.order_index < len(self.point_order):
            selected_point = self.point_order[self.order_index]
            self.order_index += 1
            return selected_point
        else:
            # Fallback to random when LHS coordinates are exhausted
            return self.rng.choice(hidden_indices)
    
    def _initialize_design_fast(self, X_pool: np.ndarray, hidden_indices: List[int]):
        """Fast LHS initialization with KNN and caching"""
        X_hidden = X_pool[hidden_indices].astype(FAST_DTYPE)
        n_features = X_hidden.shape[1]
        
        # Limit sample size based on available points
        effective_samples = min(self.n_samples, len(hidden_indices) // 2)
        
        # Build a cache key that is specific to the actual hidden pool content
        x_fp = _get_X_fingerprint(X_hidden)
        hidden_fp = hashlib.md5(np.array(hidden_indices, dtype=np.int64).tobytes()).hexdigest()
        cache_key = f"lhs_{effective_samples}_{n_features}_{len(hidden_indices)}_{x_fp}_{hidden_fp}"
        if cache_key in _DESIGN_CACHE:
            self.point_order = _DESIGN_CACHE[cache_key].copy()
            self.design_initialized = True
            self.order_index = 0
            return
        
        # Generate LHS samples
        sampler = qmc.LatinHypercube(d=n_features, seed=self.random_state)
        lhs_unit = sampler.random(n=effective_samples)
        
        # Scale to data range
        X_min, X_max = X_hidden.min(axis=0), X_hidden.max(axis=0)
        lhs_coordinates = X_min + lhs_unit * (X_max - X_min)
        
        # Use KNN for fast nearest neighbor matching
        self.nn_model = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')
        self.nn_model.fit(X_hidden)
        
        # Vectorized nearest neighbor search
        distances, indices = self.nn_model.kneighbors(lhs_coordinates)
        self.point_order = [hidden_indices[idx[0]] for idx in indices]
        
        # Remove duplicates
        seen = set()
        unique_points = []
        for point in self.point_order:
            if point not in seen:
                seen.add(point)
                unique_points.append(point)
        
        self.point_order = unique_points
        
        # Cache the result scoped to this exact hidden pool
        with _CACHE_LOCK:
            _DESIGN_CACHE[cache_key] = self.point_order.copy()
        
        self.design_initialized = True
        self.order_index = 0


class DOptimalDesignStrategy(OptimizationStrategy):
    """D-optimal Design - Maximizes determinant of information matrix"""
    
    def __init__(self, random_state: int = 0, model_order: int = 1):
        super().__init__("D_OPTIMAL", random_state)
        self.model_order = model_order
        self.information_matrix = None
        self.design_matrix = None
    
    def select_next_point(self, X_pool: np.ndarray, hidden_indices: List[int], 
                         X_observed: np.ndarray, y_observed: np.ndarray,
                         current_best: float, **kwargs) -> int:
        """Select point that maximizes D-optimal criterion"""
        self.iteration_count += 1
        
        # Update design matrix with current observations
        self._update_design_matrix(X_observed)
        
        # Evaluate D-optimal criterion for each hidden point
        X_hidden = X_pool[hidden_indices]
        d_values = []
        
        for i, hidden_idx in enumerate(hidden_indices):
            x_candidate = X_pool[hidden_idx:hidden_idx+1]  # Keep 2D shape
            d_value = self._calculate_d_criterion(x_candidate)
            d_values.append(d_value)
        
        # Select point with maximum D-criterion
        best_local_idx = np.argmax(d_values)
        return hidden_indices[best_local_idx]
    
    def _update_design_matrix(self, X_observed: np.ndarray):
        """Update design matrix with polynomial features"""
        if self.model_order == 1:
            # Linear model: add intercept
            self.design_matrix = np.column_stack([
                np.ones(X_observed.shape[0]), X_observed
            ])
        elif self.model_order == 2:
            # Quadratic model
            poly_features = PolynomialFeatures(degree=2, include_bias=True)
            self.design_matrix = poly_features.fit_transform(X_observed)
        else:
            # Higher order polynomial
            poly_features = PolynomialFeatures(degree=self.model_order, include_bias=True)
            self.design_matrix = poly_features.fit_transform(X_observed)
        
        # Calculate current information matrix
        self.information_matrix = self.design_matrix.T @ self.design_matrix
    
    def _calculate_d_criterion(self, x_candidate: np.ndarray) -> float:
        """Calculate D-optimal criterion for candidate point"""
        # Create design vector for candidate
        if self.model_order == 1:
            design_vector = np.column_stack([
                np.ones(x_candidate.shape[0]), x_candidate
            ])
        elif self.model_order == 2:
            poly_features = PolynomialFeatures(degree=2, include_bias=True)
            design_vector = poly_features.fit_transform(x_candidate)
        else:
            poly_features = PolynomialFeatures(degree=self.model_order, include_bias=True)
            design_vector = poly_features.fit_transform(x_candidate)
        
        # Calculate updated information matrix
        updated_info_matrix = (self.information_matrix + 
                              design_vector.T @ design_vector)
        
        # Calculate determinant (D-criterion)
        try:
            d_criterion = np.linalg.det(updated_info_matrix)
            if d_criterion <= 0:
                return 1e-10  # Small positive value for numerical stability
            return d_criterion
        except np.linalg.LinAlgError:
            return 1e-10  # Handle singular matrices


def get_all_optimizers(random_state: int = 0, debug: bool = False) -> Dict[str, OptimizationStrategy]:
    """Get all available optimization strategies"""
    optimizers = {
        "RANDOM": RandomStrategy(random_state),
        "BO_GP_EI": BOGPEIStrategy(random_state, debug=debug),
        "SMART_BO": SmartBOStrategy(random_state, debug=debug),
        "SBO_GP_PV": SBOGPPVStrategy(random_state, debug=debug),
        "SBO_ANN_PV": SBOANNPVStrategy(random_state, debug=debug),
        "SBO_POLY_PV": SBOPolyPVStrategy(random_state, debug=debug),
        "SBO_GP_EI_TRUNCDE": SBOGPEITruncDEStrategy(random_state, debug=debug),
        "DE_DIRECT": DEDirectStrategy(random_state),
        # Experimental Design Methods
        "FULL_FACTORIAL": FullFactorialDesignStrategy(random_state),
        "FRACTIONAL_FACTORIAL": FractionalFactorialDesignStrategy(random_state),
        "PLACKETT_BURMAN": PlackettBurmanDesignStrategy(random_state),
        "CENTRAL_COMPOSITE": CentralCompositeDesignStrategy(random_state),
        "BOX_BEHNKEN": BoxBehnkenDesignStrategy(random_state),
        "LATIN_HYPERCUBE": LatinHypercubeStrategy(random_state),
        "D_OPTIMAL": DOptimalDesignStrategy(random_state),
    }
    
    # Add optional optimizers if dependencies are available
    if HAS_DEAP:
        optimizers["GA_DIRECT"] = GADirectStrategy(random_state)
    
    if HAS_PYSWARMS:
        optimizers["PSO_DIRECT"] = PSODirectStrategy(random_state)
    
    return optimizers


def get_optimizer_by_name(name: str, random_state: int = 0) -> OptimizationStrategy:
    """Get a specific optimization strategy by name"""
    optimizers = get_all_optimizers(random_state)
    
    if name not in optimizers:
        available = list(optimizers.keys())
        raise ValueError(f"Optimizer '{name}' not found. Available optimizers: {available}")
    
    return optimizers[name]


def print_optimizer_info():
    """Print information about available optimizers"""
    optimizers = get_all_optimizers()
    
    print("Available optimization strategies:")
    print("=" * 50)
    
    categories = {
        "Random Methods": ["RANDOM"],
        "Bayesian Optimization": ["BO_GP_EI", "SMART_BO"],
        "Surrogate-Based": ["SBO_GP_PV", "SBO_ANN_PV", "SBO_POLY_PV", "SBO_GP_EI_TRUNCDE"],
        "Direct Methods": ["GA_DIRECT", "DE_DIRECT", "PSO_DIRECT"],
        "Experimental Design": [
            "FULL_FACTORIAL", "FRACTIONAL_FACTORIAL", "PLACKETT_BURMAN",
            "CENTRAL_COMPOSITE", "BOX_BEHNKEN", "LATIN_HYPERCUBE", "D_OPTIMAL"
        ]
    }
    
    for category, methods in categories.items():
        print(f"\n{category}:")
        for method in methods:
            if method in optimizers:
                print(f"  - {method}")
            else:
                print(f"  - {method} (unavailable)")


if __name__ == "__main__":
    print_optimizer_info() 