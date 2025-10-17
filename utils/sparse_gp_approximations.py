#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sparse GP Approximations for Speed-Up Comparison

This module implements various sparse GP approximation methods:
- Nyström approximation
- FITC (Fully Independent Training Conditional)
- VFE (Variational Free Energy)

These methods provide significant speedups for large datasets while maintaining good accuracy.
"""

import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# GPU acceleration
try:
    import cupy as cp
    import cupyx.scipy.linalg as cp_linalg
    GPU_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("CuPy detected - GPU acceleration available for sparse GPs")
except ImportError:
    GPU_AVAILABLE = False
    cp = None
    cp_linalg = None
    logger = logging.getLogger(__name__)
    logger.warning("CuPy not available - using CPU only for sparse GPs")


class SparseGPBase:
    """Base class for sparse GP approximations"""
    
    def __init__(self, n_inducing_points: int = 50, random_state: int = 42):
        self.n_inducing_points = n_inducing_points
        self.random_state = random_state
        self.np_random = np.random.RandomState(random_state)
        self.is_fitted = False
        self.scaler = StandardScaler()
        
    def _select_inducing_points(self, X: np.ndarray) -> np.ndarray:
        """Select inducing points using K-means clustering"""
        if len(X) <= self.n_inducing_points:
            return X.copy()
        
        kmeans = KMeans(
            n_clusters=self.n_inducing_points, 
            random_state=self.random_state,
            n_init=1  # Single run for speed
        )
        kmeans.fit(X)
        return kmeans.cluster_centers_
        
    def fit(self, X: np.ndarray, y: np.ndarray, y_var: Optional[np.ndarray] = None) -> 'SparseGPBase':
        """Fit the sparse GP model"""
        raise NotImplementedError
        
    def predict(self, X: np.ndarray, return_std: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Make predictions"""
        raise NotImplementedError
        
    def sample_y(self, X: np.ndarray, n_samples: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        """Sample from the sparse GP model"""
        raise NotImplementedError


class NystromGPSurrogate(SparseGPBase):
    """Nyström approximation for GP regression"""
    
    def __init__(self, 
                 n_inducing_points: int = 50,
                 kernel_type: str = 'rbf',
                 noise_level: float = 0.1,
                 random_state: int = 42,
                 use_gpu: bool = True):
        super().__init__(n_inducing_points, random_state)
        self.kernel_type = kernel_type
        self.noise_level = noise_level
        self.use_gpu = use_gpu and GPU_AVAILABLE
        
        # Initialize kernel
        self.kernel = self._create_kernel()
        
        # Storage for Nyström approximation
        self.X_inducing = None
        self.K_mm = None  # Kernel matrix of inducing points
        self.K_mm_inv = None  # Inverse of K_mm
        self.alpha = None  # Solution vector
        
    def _create_kernel(self):
        """Create the kernel based on type"""
        if self.kernel_type == 'rbf':
            return ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=self.noise_level)
        elif self.kernel_type == 'matern':
            return ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=self.noise_level)
        else:
            return ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=self.noise_level)
            
    def _compute_kernel_matrix(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute kernel matrix between two sets of points"""
        if self.use_gpu and GPU_AVAILABLE:
            return self._compute_kernel_matrix_gpu(X1, X2)
        else:
            return self._compute_kernel_matrix_cpu(X1, X2)
            
    def _compute_kernel_matrix_cpu(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """CPU kernel matrix computation"""
        # Simplified RBF kernel computation
        sqdist = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
        return np.exp(-0.5 * sqdist)
        
    def _compute_kernel_matrix_gpu(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """GPU kernel matrix computation"""
        X1_gpu = cp.asarray(X1)
        X2_gpu = cp.asarray(X2)
        sqdist = cp.sum(X1_gpu**2, axis=1).reshape(-1, 1) + cp.sum(X2_gpu**2, axis=1) - 2 * cp.dot(X1_gpu, X2_gpu.T)
        K = cp.exp(-0.5 * sqdist)
        return cp.asnumpy(K)
        
    def fit(self, X: np.ndarray, y: np.ndarray, y_var: Optional[np.ndarray] = None) -> 'NystromGPSurrogate':
        """Fit the Nyström GP approximation"""
        start_time = time.time()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Select inducing points
        self.X_inducing = self._select_inducing_points(X_scaled)
        
        # Compute kernel matrices
        self.K_mm = self._compute_kernel_matrix(self.X_inducing, self.X_inducing)
        K_nm = self._compute_kernel_matrix(X_scaled, self.X_inducing)
        
        # Add noise to diagonal of K_mm
        self.K_mm += np.eye(self.K_mm.shape[0]) * self.noise_level
        
        # Compute inverse of K_mm
        try:
            self.K_mm_inv = np.linalg.inv(self.K_mm)
        except np.linalg.LinAlgError:
            # Add jitter if singular
            self.K_mm_inv = np.linalg.inv(self.K_mm + np.eye(self.K_mm.shape[0]) * 1e-6)
        
        # Solve for alpha using Nyström approximation
        # K_nm^T @ K_mm_inv @ K_nm approximates the full kernel matrix
        Q = K_nm @ self.K_mm_inv @ K_nm.T
        Q += np.eye(Q.shape[0]) * self.noise_level
        
        # Solve linear system
        try:
            self.alpha = np.linalg.solve(Q, y)
        except np.linalg.LinAlgError:
            # Add jitter if singular
            Q += np.eye(Q.shape[0]) * 1e-6
            self.alpha = np.linalg.solve(Q, y)
        
        self.is_fitted = True
        
        # Evaluate performance
        y_pred = self.predict(X_scaled)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        elapsed = time.time() - start_time
        logger.info(f"Nyström GP fitted in {elapsed:.3f}s - R²: {r2:.4f}, MSE: {mse:.4f}")
        
        return self
        
    def predict(self, X: np.ndarray, return_std: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Make predictions using Nyström approximation"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        X_scaled = self.scaler.transform(X)
        
        # Compute kernel between test points and inducing points
        K_star = self._compute_kernel_matrix(X_scaled, self.X_inducing)
        
        # Predict mean
        mean_pred = K_star @ self.K_mm_inv @ K_star.T @ self.alpha
        
        if return_std:
            # Approximate variance (simplified)
            var_pred = np.ones(len(X_scaled)) * self.noise_level
            std_pred = np.sqrt(var_pred)
            return mean_pred, std_pred
        else:
            return mean_pred
            
    def sample_y(self, X: np.ndarray, n_samples: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        """Sample from the Nyström GP model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before sampling")
            
        X_scaled = self.scaler.transform(X)
        
        # Get predictions
        mean_pred, std_pred = self.predict(X_scaled, return_std=True)
        
        # Sample from normal distribution
        if random_state is not None:
            local_rng = np.random.RandomState(random_state)
        else:
            local_rng = self.np_random
            
        samples = []
        for _ in range(n_samples):
            sample = local_rng.normal(mean_pred, std_pred)
            samples.append(sample)
            
        return np.array(samples).T


class FITCGPSurrogate(SparseGPBase):
    """Fully Independent Training Conditional (FITC) GP approximation"""
    
    def __init__(self, 
                 n_inducing_points: int = 50,
                 kernel_type: str = 'rbf',
                 noise_level: float = 0.1,
                 random_state: int = 42,
                 use_gpu: bool = True):
        super().__init__(n_inducing_points, random_state)
        self.kernel_type = kernel_type
        self.noise_level = noise_level
        self.use_gpu = use_gpu and GPU_AVAILABLE
        
        # Initialize kernel
        self.kernel = self._create_kernel()
        
        # Storage for FITC approximation
        self.X_inducing = None
        self.K_mm = None
        self.K_mm_inv = None
        self.Lambda = None  # Diagonal matrix for FITC
        self.alpha = None
        
    def _create_kernel(self):
        """Create the kernel based on type"""
        if self.kernel_type == 'rbf':
            return ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=self.noise_level)
        elif self.kernel_type == 'matern':
            return ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=self.noise_level)
        else:
            return ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=self.noise_level)
            
    def _compute_kernel_matrix(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute kernel matrix between two sets of points"""
        if self.use_gpu and GPU_AVAILABLE:
            return self._compute_kernel_matrix_gpu(X1, X2)
        else:
            return self._compute_kernel_matrix_cpu(X1, X2)
            
    def _compute_kernel_matrix_cpu(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """CPU kernel matrix computation"""
        sqdist = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
        return np.exp(-0.5 * sqdist)
        
    def _compute_kernel_matrix_gpu(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """GPU kernel matrix computation"""
        X1_gpu = cp.asarray(X1)
        X2_gpu = cp.asarray(X2)
        sqdist = cp.sum(X1_gpu**2, axis=1).reshape(-1, 1) + cp.sum(X2_gpu**2, axis=1) - 2 * cp.dot(X1_gpu, X2_gpu.T)
        K = cp.exp(-0.5 * sqdist)
        return cp.asnumpy(K)
        
    def fit(self, X: np.ndarray, y: np.ndarray, y_var: Optional[np.ndarray] = None) -> 'FITCGPSurrogate':
        """Fit the FITC GP approximation"""
        start_time = time.time()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Select inducing points
        self.X_inducing = self._select_inducing_points(X_scaled)
        
        # Compute kernel matrices
        self.K_mm = self._compute_kernel_matrix(self.X_inducing, self.X_inducing)
        K_nm = self._compute_kernel_matrix(X_scaled, self.X_inducing)
        
        # Compute diagonal of full kernel matrix
        K_nn_diag = np.diag(self._compute_kernel_matrix(X_scaled, X_scaled))
        
        # Compute Lambda (diagonal matrix for FITC)
        K_mm_inv_K_mn = np.linalg.solve(self.K_mm, K_nm.T)
        self.Lambda = K_nn_diag - np.sum(K_nm * K_mm_inv_K_mn.T, axis=1)
        self.Lambda = np.maximum(self.Lambda, 1e-6)  # Ensure positive
        
        # Add noise to diagonal
        self.Lambda += self.noise_level
        
        # Compute FITC approximation
        Lambda_inv = 1.0 / self.Lambda
        K_mm_inv = np.linalg.inv(self.K_mm)
        
        # Compute Q matrix for FITC
        Q = K_nm.T @ (Lambda_inv[:, None] * K_nm) + K_mm_inv
        
        # Solve for alpha
        try:
            self.alpha = np.linalg.solve(Q, K_nm.T @ (Lambda_inv * y))
        except np.linalg.LinAlgError:
            # Add jitter if singular
            Q += np.eye(Q.shape[0]) * 1e-6
            self.alpha = np.linalg.solve(Q, K_nm.T @ (Lambda_inv * y))
        
        self.is_fitted = True
        
        # Evaluate performance
        y_pred = self.predict(X_scaled)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        elapsed = time.time() - start_time
        logger.info(f"FITC GP fitted in {elapsed:.3f}s - R²: {r2:.4f}, MSE: {mse:.4f}")
        
        return self
        
    def predict(self, X: np.ndarray, return_std: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Make predictions using FITC approximation"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        X_scaled = self.scaler.transform(X)
        
        # Compute kernel between test points and inducing points
        K_star = self._compute_kernel_matrix(X_scaled, self.X_inducing)
        
        # Predict mean
        mean_pred = K_star @ self.alpha
        
        if return_std:
            # Compute variance (simplified FITC variance)
            K_star_star = np.diag(self._compute_kernel_matrix(X_scaled, X_scaled))
            K_mm_inv = np.linalg.inv(self.K_mm)
            var_pred = K_star_star - np.sum(K_star * (K_mm_inv @ K_star.T).T, axis=1) + self.noise_level
            var_pred = np.maximum(var_pred, 1e-6)
            std_pred = np.sqrt(var_pred)
            return mean_pred, std_pred
        else:
            return mean_pred
            
    def sample_y(self, X: np.ndarray, n_samples: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        """Sample from the FITC GP model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before sampling")
            
        X_scaled = self.scaler.transform(X)
        
        # Get predictions
        mean_pred, std_pred = self.predict(X_scaled, return_std=True)
        
        # Sample from normal distribution
        if random_state is not None:
            local_rng = np.random.RandomState(random_state)
        else:
            local_rng = self.np_random
            
        samples = []
        for _ in range(n_samples):
            sample = local_rng.normal(mean_pred, std_pred)
            samples.append(sample)
            
        return np.array(samples).T


class VFEGPSurrogate(SparseGPBase):
    """Variational Free Energy (VFE) GP approximation"""
    
    def __init__(self, 
                 n_inducing_points: int = 50,
                 kernel_type: str = 'rbf',
                 noise_level: float = 0.1,
                 random_state: int = 42,
                 use_gpu: bool = True,
                 max_iter: int = 100):
        super().__init__(n_inducing_points, random_state)
        self.kernel_type = kernel_type
        self.noise_level = noise_level
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.max_iter = max_iter
        
        # Initialize kernel
        self.kernel = self._create_kernel()
        
        # Storage for VFE approximation
        self.X_inducing = None
        self.K_mm = None
        self.K_mm_inv = None
        self.alpha = None
        
    def _create_kernel(self):
        """Create the kernel based on type"""
        if self.kernel_type == 'rbf':
            return ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=self.noise_level)
        elif self.kernel_type == 'matern':
            return ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=self.noise_level)
        else:
            return ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=self.noise_level)
            
    def _compute_kernel_matrix(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute kernel matrix between two sets of points"""
        if self.use_gpu and GPU_AVAILABLE:
            return self._compute_kernel_matrix_gpu(X1, X2)
        else:
            return self._compute_kernel_matrix_cpu(X1, X2)
            
    def _compute_kernel_matrix_cpu(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """CPU kernel matrix computation"""
        sqdist = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
        return np.exp(-0.5 * sqdist)
        
    def _compute_kernel_matrix_gpu(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """GPU kernel matrix computation"""
        X1_gpu = cp.asarray(X1)
        X2_gpu = cp.asarray(X2)
        sqdist = cp.sum(X1_gpu**2, axis=1).reshape(-1, 1) + cp.sum(X2_gpu**2, axis=1) - 2 * cp.dot(X1_gpu, X2_gpu.T)
        K = cp.exp(-0.5 * sqdist)
        return cp.asnumpy(K)
        
    def fit(self, X: np.ndarray, y: np.ndarray, y_var: Optional[np.ndarray] = None) -> 'VFEGPSurrogate':
        """Fit the VFE GP approximation"""
        start_time = time.time()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Select inducing points
        self.X_inducing = self._select_inducing_points(X_scaled)
        
        # Compute kernel matrices
        self.K_mm = self._compute_kernel_matrix(self.X_inducing, self.X_inducing)
        K_nm = self._compute_kernel_matrix(X_scaled, self.X_inducing)
        
        # Add jitter to K_mm for numerical stability
        self.K_mm += np.eye(self.K_mm.shape[0]) * 1e-6
        
        # Compute inverse of K_mm
        self.K_mm_inv = np.linalg.inv(self.K_mm)
        
        # VFE optimization (simplified - in practice would optimize inducing points)
        # For now, use fixed inducing points and solve for alpha
        
        # Compute VFE approximation
        K_mm_inv_K_mn = self.K_mm_inv @ K_nm.T
        Q = K_nm.T @ K_nm + self.K_mm * self.noise_level
        
        # Solve for alpha
        try:
            self.alpha = np.linalg.solve(Q, K_nm.T @ y)
        except np.linalg.LinAlgError:
            # Add jitter if singular
            Q += np.eye(Q.shape[0]) * 1e-6
            self.alpha = np.linalg.solve(Q, K_nm.T @ y)
        
        self.is_fitted = True
        
        # Evaluate performance
        y_pred = self.predict(X_scaled)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        elapsed = time.time() - start_time
        logger.info(f"VFE GP fitted in {elapsed:.3f}s - R²: {r2:.4f}, MSE: {mse:.4f}")
        
        return self
        
    def predict(self, X: np.ndarray, return_std: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Make predictions using VFE approximation"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        X_scaled = self.scaler.transform(X)
        
        # Compute kernel between test points and inducing points
        K_star = self._compute_kernel_matrix(X_scaled, self.X_inducing)
        
        # Predict mean
        mean_pred = K_star @ self.alpha
        
        if return_std:
            # Compute variance (simplified VFE variance)
            K_star_star = np.diag(self._compute_kernel_matrix(X_scaled, X_scaled))
            var_pred = K_star_star - np.sum(K_star * (self.K_mm_inv @ K_star.T).T, axis=1) + self.noise_level
            var_pred = np.maximum(var_pred, 1e-6)
            std_pred = np.sqrt(var_pred)
            return mean_pred, std_pred
        else:
            return mean_pred
            
    def sample_y(self, X: np.ndarray, n_samples: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        """Sample from the VFE GP model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before sampling")
            
        X_scaled = self.scaler.transform(X)
        
        # Get predictions
        mean_pred, std_pred = self.predict(X_scaled, return_std=True)
        
        # Sample from normal distribution
        if random_state is not None:
            local_rng = np.random.RandomState(random_state)
        else:
            local_rng = self.np_random
            
        samples = []
        for _ in range(n_samples):
            sample = local_rng.normal(mean_pred, std_pred)
            samples.append(sample)
            
        return np.array(samples).T


class SparseGPFactory:
    """Factory for creating different types of sparse GP models"""
    
    @staticmethod
    def create_sparse_gp(sparse_type: str, **kwargs) -> SparseGPBase:
        """Create a sparse GP model of the specified type"""
        
        if sparse_type.lower() == 'nystrom':
            return NystromGPSurrogate(**kwargs)
        elif sparse_type.lower() == 'fitc':
            return FITCGPSurrogate(**kwargs)
        elif sparse_type.lower() == 'vfe':
            return VFEGPSurrogate(**kwargs)
        else:
            raise ValueError(f"Unknown sparse GP type: {sparse_type}")
            
    @staticmethod
    def get_available_sparse_gps() -> List[str]:
        """Get list of available sparse GP types"""
        return ['nystrom', 'fitc', 'vfe'] 