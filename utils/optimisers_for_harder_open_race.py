#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimizers for Harder Open Race

This module provides the same optimizers used in the Open Race setting, but
with more robust versions for the BO methods to better handle:
- Higher noise levels
- Heteroscedastic noise (spatially varying)
- Non-smoothness/non-stationarity (e.g., discontinuities)

Key ideas applied to BO variants (names preserved):
- BO_GP_EI: Use a more robust kernel (RationalQuadratic + Matern 1.5),
  robust per-observation noise estimates via KNN variance, and noise-aware EI
  using an effective standard deviation s_eff = sqrt(s^2 + sigma_n(x)^2).
- SBO_GP_EI_TRUNCDE: Same robust GP and noise-aware EI before any local search.

All other optimizers are identical to the originals.
"""

import os
import sys
import numpy as np
from typing import Dict, List, Any

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, RationalQuadratic
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.stats import norm

# Robust imports for project paths
try:
    from utils.optimizers import (
        OptimizationStrategy,
        get_all_optimizers,
    )
except Exception:
    _CURRENT_DIR = os.path.dirname(__file__)
    _PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, '..', '..'))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)
    from utils.optimizers import (  # type: ignore
        OptimizationStrategy,
        get_all_optimizers,
    )

# Use the same open-race wrapper so behavior is identical at the interface level
try:
    from utils.open_race_optimizers import OpenRaceOptimizerWrapper
except Exception:
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)
    from utils.open_race_optimizers import OpenRaceOptimizerWrapper  # type: ignore


def _mad_std(y: np.ndarray) -> float:
    """Robust global noise scale via MAD -> sigma."""
    if y.size == 0:
        return 1e-6
    med = np.median(y)
    mad = np.median(np.abs(y - med))
    return float(1.4826 * mad + 1e-9)


def _estimate_local_noise_per_observation(X: np.ndarray, y: np.ndarray, k_neighbors: int = 10) -> np.ndarray:
    """Estimate per-observation noise variance using KNN variance of y.

    Returns alpha_i (variance) for each observation i. Robust to heteroscedasticity.
    """
    n = len(y)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([max(_mad_std(y) ** 2, 1e-9)])

    k = int(max(1, min(k_neighbors, n - 1)))
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    local_vars = np.zeros(n, dtype=float)
    # Compute variance of neighboring y (including self)
    for i in range(n):
        neighbor_idx = indices[i]
        y_neigh = y[neighbor_idx]
        var = float(np.var(y_neigh))
        local_vars[i] = var

    # Floor with robust global noise
    global_sigma2 = _mad_std(y) ** 2
    local_vars = np.maximum(local_vars, 0.1 * global_sigma2)
    return local_vars + 1e-9


def _estimate_local_noise_for_candidates(X_obs: np.ndarray, y: np.ndarray, X_cand: np.ndarray, k_neighbors: int = 10) -> np.ndarray:
    """Estimate local noise variance at candidate locations via KNN of observations."""
    n = len(y)
    if n == 0 or X_cand.shape[0] == 0:
        return np.zeros(X_cand.shape[0], dtype=float)

    k = int(max(1, min(k_neighbors, n)))
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X_obs)
    distances, indices = nn.kneighbors(X_cand)

    sigma2 = np.zeros(X_cand.shape[0], dtype=float)
    global_sigma2 = _mad_std(y) ** 2
    for i in range(X_cand.shape[0]):
        y_neigh = y[indices[i]]
        var = float(np.var(y_neigh))
        sigma2[i] = max(var, 0.1 * global_sigma2)
    return sigma2 + 1e-9


class RobustBOGPEIStrategy(OptimizationStrategy):
    """Bayesian Optimization with GP and noise-aware EI (robust)"""

    def __init__(self, random_state: int = 0, debug: bool = False):
        super().__init__("BO_GP_EI", random_state)
        self.debug = debug
        self.scaler: StandardScaler = None  # type: ignore

    def _fit_gp(self, X: np.ndarray, y: np.ndarray) -> GaussianProcessRegressor:
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Robust per-observation alphas via local KNN variance
        alphas = _estimate_local_noise_per_observation(X_scaled, y)

        # Robust kernel: RationalQuadratic + Matern(1.5) + small White noise
        n_features = X.shape[1]
        kernel = (1.0 * RationalQuadratic(length_scale=1.0, alpha=1.0, length_scale_bounds=(1e-3, 1e2), alpha_bounds=(1e-2, 1e2)) +
                  1.0 * Matern(length_scale=1.0, nu=1.5, length_scale_bounds=(1e-3, 1e2)) +
                  WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1)))

        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alphas,
            normalize_y=True,
            n_restarts_optimizer=1,
            random_state=self.random_state
        )
        gp.fit(X_scaled, y)
        return gp

    def select_next_point(self, X_pool: np.ndarray, hidden_indices: List[int],
                          X_observed: np.ndarray, y_observed: np.ndarray,
                          current_best: float, **kwargs) -> int:
        self.iteration_count += 1

        # Fallbacks
        if len(hidden_indices) <= 2 or X_observed.shape[0] < 3:
            return self.rng.choice(hidden_indices)

        try:
            gp = self._fit_gp(X_observed, y_observed)
        except Exception:
            return self.rng.choice(hidden_indices)

        # Prepare candidate set
        X_hidden = X_pool[hidden_indices]
        X_hidden_scaled = self.scaler.transform(X_hidden)

        # Noise-aware EI: s_eff = sqrt(s^2 + sigma_n(x)^2)
        try:
            mean_obs, _ = gp.predict(self.scaler.transform(X_observed), return_std=True)
            f_best_denoised = float(np.max(mean_obs)) if mean_obs.size > 0 else float(current_best)

            mean_cand, std_cand = gp.predict(X_hidden_scaled, return_std=True)
            std_cand = np.maximum(std_cand, 1e-8)

            # Local noise at candidate points via KNN of observations (in scaled space)
            sigma2_loc = _estimate_local_noise_for_candidates(
                self.scaler.transform(X_observed), y_observed, X_hidden_scaled, k_neighbors=10
            )
            s_eff = np.sqrt(std_cand**2 + sigma2_loc)

            z = (mean_cand - f_best_denoised) / s_eff
            ei = (mean_cand - f_best_denoised) * norm.cdf(z) + s_eff * norm.pdf(z)

            # Small exploration bonus to combat poor local modeling in rough regions
            ei = ei + 0.01 * std_cand

            best_local_idx = int(np.argmax(ei))
            return hidden_indices[best_local_idx]
        except Exception:
            return self.rng.choice(hidden_indices)


class RobustSBOGPEITruncDEStrategy(OptimizationStrategy):
    """GP surrogate with noise-aware EI (robust) and optional local search."""

    def __init__(self, random_state: int = 0, debug: bool = False):
        super().__init__("SBO_GP_EI_TRUNCDE", random_state)
        self.debug = debug
        self.scaler: StandardScaler = None  # type: ignore

    def _fit_gp(self, X: np.ndarray, y: np.ndarray) -> GaussianProcessRegressor:
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        alphas = _estimate_local_noise_per_observation(X_scaled, y)
        kernel = (1.0 * RationalQuadratic(length_scale=1.0, alpha=1.0, length_scale_bounds=(1e-3, 1e2), alpha_bounds=(1e-2, 1e2)) +
                  1.0 * Matern(length_scale=1.0, nu=1.5, length_scale_bounds=(1e-3, 1e2)) +
                  WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1)))

        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alphas,
            normalize_y=True,
            n_restarts_optimizer=1,
            random_state=self.random_state
        )
        gp.fit(X_scaled, y)
        return gp

    def select_next_point(self, X_pool: np.ndarray, hidden_indices: List[int],
                          X_observed: np.ndarray, y_observed: np.ndarray,
                          current_best: float, **kwargs) -> int:
        self.iteration_count += 1

        if len(hidden_indices) <= 2 or X_observed.shape[0] < 3:
            return self.rng.choice(hidden_indices)

        try:
            gp = self._fit_gp(X_observed, y_observed)
        except Exception:
            return self.rng.choice(hidden_indices)

        X_hidden = X_pool[hidden_indices]
        X_hidden_scaled = self.scaler.transform(X_hidden)

        try:
            mean_obs, _ = gp.predict(self.scaler.transform(X_observed), return_std=True)
            f_best_denoised = float(np.max(mean_obs)) if mean_obs.size > 0 else float(current_best)

            mean_cand, std_cand = gp.predict(X_hidden_scaled, return_std=True)
            std_cand = np.maximum(std_cand, 1e-8)

            sigma2_loc = _estimate_local_noise_for_candidates(
                self.scaler.transform(X_observed), y_observed, X_hidden_scaled, k_neighbors=10
            )
            s_eff = np.sqrt(std_cand**2 + sigma2_loc)

            z = (mean_cand - f_best_denoised) / s_eff
            ei = (mean_cand - f_best_denoised) * norm.cdf(z) + s_eff * norm.pdf(z)
            ei = ei + 0.01 * std_cand

            best_local_idx = int(np.argmax(ei))
            return hidden_indices[best_local_idx]
        except Exception:
            return self.rng.choice(hidden_indices)


def get_open_race_optimizers(random_state: int = 42) -> Dict[str, OpenRaceOptimizerWrapper]:
    """Return open-race-compatible optimizers with robust BO variants.

    Names are preserved. Only BO strategies are swapped for robust versions.
    """
    base_opts = get_all_optimizers(random_state=random_state)

    # Swap BO variants for robust versions
    if "BO_GP_EI" in base_opts:
        base_opts["BO_GP_EI"] = RobustBOGPEIStrategy(random_state)
    if "SBO_GP_EI_TRUNCDE" in base_opts:
        base_opts["SBO_GP_EI_TRUNCDE"] = RobustSBOGPEITruncDEStrategy(random_state)

    # Wrap for open race continuous setting
    wrapped: Dict[str, OpenRaceOptimizerWrapper] = {}
    for name, opt in base_opts.items():
        wrapped[name] = OpenRaceOptimizerWrapper(opt, random_state)

    return wrapped




