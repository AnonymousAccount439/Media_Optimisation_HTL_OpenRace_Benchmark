import numpy as np
import pandas as pd
from typing import Tuple, List
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from scipy.stats import qmc
import warnings
warnings.filterwarnings('ignore')
import os

# Find the project root directory (where Data folder is located)
def find_project_root():
    """Find the project root directory by looking for the Data folder"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Walk up the directory tree to find the Data folder
    while current_dir != os.path.dirname(current_dir):  # Stop at filesystem root
        # Check for both "Data" and "Extras/Data" directories
        data_dir = os.path.join(current_dir, "Data")
        extras_data_dir = os.path.join(current_dir, "Extras", "Data")
        if os.path.exists(data_dir) and os.path.isdir(data_dir):
            return current_dir
        elif os.path.exists(extras_data_dir) and os.path.isdir(extras_data_dir):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    
    # If not found, assume current directory is project root
    return os.path.dirname(os.path.abspath(__file__))

# Set the data directory to be relative to project root
PROJECT_ROOT = find_project_root()
# Check for both "Data" and "Extras/Data" directories
DATA_DIR = os.path.join(PROJECT_ROOT, "Data")
if not os.path.exists(DATA_DIR):
    DATA_DIR = os.path.join(PROJECT_ROOT, "Extras", "Data")

# If Data directory is missing, allow synthetic datasets to work and only warn.
if not os.path.exists(DATA_DIR):
    warnings.warn(
        f"Data directory not found at: {DATA_DIR}. Real datasets will be unavailable; synthetic datasets remain usable.",
        RuntimeWarning,
    )


class Dataset:
    """Base class for datasets used in hide-the-label competitions"""
    
    def __init__(self, name: str):
        self.name = name
        self.X = None
        self.y = None
        self.y_var = None
        self.feature_names = None
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load and return the dataset"""
        raise NotImplementedError
        
    def get_info(self) -> dict:
        """Get information about the dataset"""
        return {
            'name': self.name,
            'n_samples': len(self.X) if self.X is not None else 0,
            'n_features': self.X.shape[1] if self.X is not None else 0,
            'feature_names': self.feature_names,
            'y_range': (np.min(self.y), np.max(self.y)) if self.y is not None else (None, None)
        }


class RealDataset(Dataset):
    """Dataset loaded from CSV files"""
    
    def __init__(self, name: str, csv_path: str, feature_cols: List[str] = None, 
                 target_col: str = 'Label', variance_col: str = 'Label_var',
                 drop_cols: List[str] = None, transform_target: callable = None):
        super().__init__(name)
        self.csv_path = csv_path
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.variance_col = variance_col
        self.drop_cols = drop_cols or []
        self.transform_target = transform_target
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load data from CSV file"""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(
                f"CSV file for dataset '{self.name}' not found at: {self.csv_path}. "
                f"Ensure the 'Data' directory exists at {DATA_DIR} and contains the required file."
            )
        df = pd.read_csv(self.csv_path)
        
        # Drop unwanted columns
        for col in self.drop_cols:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)
        
        # Extract target and variance
        y = df[self.target_col]
        y_var = df[self.variance_col] if self.variance_col in df.columns else pd.Series(np.zeros(len(df)))
        
        # Extract features
        if self.feature_cols:
            X = df[self.feature_cols]
        else:
            # Use all columns except target and variance
            exclude_cols = {self.target_col, self.variance_col}
            X = df.drop(columns=[col for col in exclude_cols if col in df.columns])
        
        # Apply target transformation if specified
        if self.transform_target:
            y = self.transform_target(y)
        
        self.X = X.values
        self.y = y.values
        self.y_var = y_var.values
        self.feature_names = X.columns.tolist()
        
        return self.X, self.y, self.y_var


class SyntheticDataset(Dataset):
    """Dataset generated from a Gaussian Process"""
    
    def __init__(self, name: str, n_features: int, n_samples: int = 200, 
                 noise_level: float = 0.1, random_state: int = 42):
        super().__init__(name)
        self.n_features = n_features
        self.n_samples = n_samples
        self.noise_level = noise_level
        self.random_state = random_state
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate synthetic data using GP sampling"""
        np.random.seed(self.random_state)
        
        # Generate Latin Hypercube design
        sampler = qmc.LatinHypercube(d=self.n_features, seed=self.random_state)
        X = sampler.random(self.n_samples)
        
        # Create GP with Matern kernel
        kernel = 1.0 * Matern(length_scale=0.3, nu=2.5) + WhiteKernel(noise_level=1e-6)
        gp = GaussianProcessRegressor(kernel=kernel, random_state=self.random_state)
        
        # Fit GP on a smaller design to sample function
        X_design = sampler.random(50)
        y_design = np.zeros(50)
        gp.fit(X_design, y_design)
        
        # Sample function values
        y = gp.sample_y(X, 1, random_state=self.random_state).flatten()
        
        # Normalize to reasonable range and add noise
        y = (y - np.min(y)) / (np.max(y) - np.min(y))  # Scale to [0, 1]
        y = y * 10  # Scale to [0, 10] range
        
        # Add heteroscedastic noise
        y_var = np.full(len(y), self.noise_level**2)
        noise = np.random.normal(0, self.noise_level, len(y))
        y = y + noise
        
        self.X = X
        self.y = y
        self.y_var = y_var
        self.feature_names = [f'x{i}' for i in range(self.n_features)]
        
        return self.X, self.y, self.y_var


def get_available_datasets() -> dict:
    """Get all available datasets"""
    datasets = {
        # Real biological datasets
        'MOBO_dataset_rat_myocyte': RealDataset(
            name='MOBO_dataset_rat_myocyte',
            csv_path=os.path.join(DATA_DIR, 'MOBO_dataset_rat_myocyte.csv'),
            target_col='Label',
            variance_col='Label_var',
            drop_cols=['Unnamed: 0', 'Fidelity']
        ),
        
        'DBO_dataset_rat_myocyte': RealDataset(
            name='DBO_dataset_rat_myocyte',
            csv_path=os.path.join(DATA_DIR, 'DBO_dataset_rat_myocyte.csv'),
            target_col='Label',
            variance_col='Label_var'
        ),
        
        'df_Human_Hela_regular_mode': RealDataset(
            name='df_Human_Hela_regular_mode',
            csv_path=os.path.join(DATA_DIR, 'df_Human_Hela_regular_mode.csv'),
            target_col='mean A450',
            variance_col='sd A450',
            drop_cols=['Unnamed: 0', 'Round'],
            transform_target=lambda y: y  # Variance needs squaring later
        ),
        
        'df_Human_Hela_timesaving_mode': RealDataset(
            name='df_Human_Hela_timesaving_mode',
            csv_path=os.path.join(DATA_DIR, 'df_Human_Hela_timesaving_mode.csv'),
            target_col='mean A450',
            variance_col='sd A450',
            drop_cols=['Unnamed: 0', 'Round'],
            transform_target=lambda y: y
        ),
        
        'df_Human_T_Cell_Expanded': RealDataset(
            name='df_Human_T_Cell_Expanded',
            csv_path=os.path.join(DATA_DIR, 'df_Human_T_Cell_Expanded.csv'),
            feature_cols=[
                "bME", "LS1000", "PY", "ITS", "ALB", "MEMAA", "ARG", "GLU",
                "CB6", "IL-2", "IL-12", "IL-18", "IL-21", "VS"
            ],
            target_col='Cell count',
            variance_col=None,  # Will create dummy variance
            transform_target=lambda y_df: np.log1p(y_df / y_df.index.map(
                lambda i: pd.read_csv(os.path.join(DATA_DIR, 'df_Human_T_Cell_Expanded.csv')).iloc[i]['PosCont FC']
            ))
        ),
        
        'df_Human_TF_Cell_Expanded': RealDataset(
            name='df_Human_TF_Cell_Expanded',
            csv_path=os.path.join(DATA_DIR, 'df_Human_TF_Cell_Expanded.csv'),
            feature_cols=[
                "CHIR99021", "SP600125", "Dexameth", "rhGM-CSF", "rhSCF",
                "rhIGF-1", "AA", "Y27632", "ALB", "FN", "GL", "CH",
                "ITS", "bME", "PY"
            ],
            target_col='normalized cell count',
            variance_col=None,
            transform_target=lambda y: np.log1p(y)
        ),
        
        # Synthetic datasets
        'synthetic_2d': SyntheticDataset(
            name='synthetic_2d',
            n_features=2,
            n_samples=200,
            noise_level=0.1,
            random_state=42
        ),
        
        'synthetic_5d': SyntheticDataset(
            name='synthetic_5d',
            n_features=5,
            n_samples=200,
            noise_level=0.1,
            random_state=42
        ),
        
        'synthetic_10d': SyntheticDataset(
            name='synthetic_10d',
            n_features=10,
            n_samples=200,
            noise_level=0.1,
            random_state=42
        )
    }
    
    return datasets


def load_dataset(dataset_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dataset]:
    """Load a specific dataset by name"""
    datasets = get_available_datasets()
    
    if dataset_name not in datasets:
        available = list(datasets.keys())
        raise ValueError(f"Dataset '{dataset_name}' not found. Available datasets: {available}")
    
    dataset = datasets[dataset_name]
    X, y, y_var = dataset.load_data()
    
    # Handle special cases for variance transformation
    if dataset_name in ['df_Human_Hela_regular_mode', 'df_Human_Hela_timesaving_mode']:
        y_var = y_var ** 2  # Convert standard deviation to variance
    elif dataset_name in ['df_Human_T_Cell_Expanded', 'df_Human_TF_Cell_Expanded']:
        y_var = np.zeros(len(y))  # Create dummy variance for datasets without it
    
    return X, y, y_var, dataset


def print_dataset_info(dataset_name: str = None):
    """Print information about available datasets"""
    datasets = get_available_datasets()
    
    if dataset_name:
        if dataset_name not in datasets:
            print(f"Dataset '{dataset_name}' not found.")
            return
        dataset = datasets[dataset_name]
        try:
            dataset.load_data()
            info = dataset.get_info()
            print(f"\nDataset: {info['name']}")
            print(f"Samples: {info['n_samples']}")
            print(f"Features: {info['n_features']}")
            print(f"Target range: {info['y_range']}")
            if info['feature_names']:
                print(f"Feature names: {info['feature_names']}")
        except Exception as e:
            print(f"Error loading dataset: {e}")
    else:
        print("Available datasets:")
        for name in datasets:
            print(f"  - {name}")


if __name__ == "__main__":
    # Test loading datasets
    print_dataset_info()
    
    # Test loading a specific dataset
    try:
        X, y, y_var, dataset = load_dataset('df_Human_Hela_timesaving_mode')
        print(f"\nLoaded {dataset.name}: X.shape={X.shape}, y.shape={y.shape}")
    except Exception as e:
        print(f"Error: {e}") 

    print(dataset)