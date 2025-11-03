# data_loader.py
import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List

# --- CRITICAL FIX: Import image processing libraries ---
from scipy.signal import spectrogram
from PIL import Image
import warnings
# ---------------------------------------------------

# Suppress warnings that may arise during image resizing/interpolation
warnings.filterwarnings("ignore", category=UserWarning)

# Define column names based on our preprocessing
LABEL_COL = 'anomaly'
MAX_VARS = 38 # We will enforce this
OUTPUT_SPATIAL_SIZE_DEFAULT = 224 # ViT default (224, 224)
RANDOM_SEED = 42 # For reproducible subsampling

# =============================================================================
# --- Time-to-Image functions (Unchanged) ---
# =============================================================================

def calculate_gasf(series: np.ndarray, output_size: int) -> np.ndarray:
    """
    Simulated implementation of Gramian Angular Summation Field (GASF).
    """
    # Normalize to [-1, 1] for best GAF simulation
    min_val = np.min(series)
    max_val = np.max(series)
    if max_val > min_val:
        normalized_series = 2 * (series - min_val) / (max_val - min_val) - 1
    else:
        normalized_series = np.zeros_like(series)
    
    # Create a dummy T x T correlation matrix (simulating GAF output)
    dummy_gaf = np.outer(normalized_series, normalized_series)  
    
    # Resize to fixed spatial dimension (H x W)
    dummy_gaf_resized = np.array(Image.fromarray(dummy_gaf).resize(
        (output_size, output_size), Image.Resampling.BILINEAR
    ))

    # Final normalization (0 to 1)
    min_res, max_res = np.min(dummy_gaf_resized), np.max(dummy_gaf_resized)
    if max_res > min_res:
        return (dummy_gaf_resized - min_res) / (max_res - min_res)
    return np.zeros((output_size, output_size), dtype=np.float32)


def calculate_spectrogram(series: np.ndarray, output_size: int) -> np.ndarray:
    """
    Calculates the Spectrogram (power spectral density) for a 1D time series.
    """
    Fs = 1.0  
    nperseg = min(len(series) // 2, 64) # Segment length for FFT
    noverlap = nperseg // 2
    
    if nperseg < 4: nperseg = 4 # Ensure minimum segment size

    try:
        f, t, Sxx = spectrogram(series, fs=Fs, nperseg=nperseg, noverlap=noverlap)
        
        # --- FIX: Safe log10 calculation ---
        Sxx_safe = Sxx + 1e-10
        Sxx_clipped = np.clip(Sxx_safe, a_min=1e-10, a_max=None)
        Sxx_db = 10 * np.log10(Sxx_clipped)
        # --- END FIX ---
        
        Sxx_resized = np.array(Image.fromarray(Sxx_db).resize(
            (output_size, output_size), Image.Resampling.BILINEAR
        ))
        
        # Normalize the resized Spectrogram (0 to 1)
        min_val, max_val = np.min(Sxx_resized), np.max(Sxx_resized)
        if max_val > min_val:
            return (Sxx_resized - min_val) / (max_val - min_val)
        return np.zeros((output_size, output_size), dtype=np.float32)

    except Exception as e:
        print(f"Spectrogram calculation error: {e}")
        return np.zeros((output_size, output_size), dtype=np.float32)

# =============================================================================
# --- End of Time-to-Image functions ---
# =============================================================================


class TimeSeriesDataset(Dataset):
    """
    --- FAST LOADER ---
    This dataset performs the time-to-image (GAF + Spectrogram) conversion
    inside its __getitem__ method...
    """
    
    def __init__(self, 
                 data_path: str, 
                 window_size: int, 
                 step_size: int, 
                 max_vars: int = MAX_VARS,
                 output_spatial_size: int = OUTPUT_SPATIAL_SIZE_DEFAULT,
                 file_subsample_percent: float = 1.0,
                 train_mode: bool = True):
        
        self.window_size = window_size
        self.step_size = step_size
        self.train_mode = train_mode
        self.max_vars = max_vars
        self.output_spatial_size = output_spatial_size
        
        self.data, self.labels, self.feature_cols = self._load_data(data_path, file_subsample_percent)
        self.num_vars = len(self.feature_cols)
        
        if self.num_vars != self.max_vars:
            print(f"Warning: Data loader expected {self.max_vars} features, but found {self.num_vars}.")
            print("This can happen if data_preprocessing.py is out of sync.")
            self.data = self.data[:, :self.max_vars]
            self.feature_cols = self.feature_cols[:self.max_vars]
            self.num_vars = self.max_vars
            print(f"Data truncated to {self.num_vars} features.")

        # --- CRITICAL FIX FOR 'nan' LOSS ---
        # We must clean the data (e.g., from CSV loading) *before* scaling
        self.data = np.nan_to_num(self.data, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize features
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        try:
            self.data = self.scaler.fit_transform(self.data)
        except Exception as e:
            # This can happen if a column is all-zero
            print(f"Warning: Scaler failed with error {e}. Will clean NaNs manually.")
            
        # Clean *again* just in case fit_transform produced NaNs (e.g., from 0/0)
        self.data = np.nan_to_num(self.data, nan=0.0, posinf=0.0, neginf=0.0)
        # --- END FIX ---

        # Create window indices
        self.indices = self._create_window_indices()
        
        self.m_i_actual = self.num_vars
        if 'padding_1' in self.feature_cols:
            self.m_i_actual = self.feature_cols.index('padding_1')
        print(f"Dataset {data_path} loaded. True variables (M_i): {self.m_i_actual}")

    def _load_data(self, data_path: str, file_subsample_percent: float) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Loads data from the preprocessed directory.
        --- FIX: Now subsamples the FILE LIST if loading from a directory ---
        """
        all_data_df = []
        
        if os.path.isdir(data_path):
            all_csv_files = sorted(glob.glob(os.path.join(data_path, '*.csv')))
            if not all_csv_files:
                raise FileNotFoundError(f"No .csv files found in directory: {data_path}")
            
            csv_files_to_load = all_csv_files
            
            # --- CRITICAL FIX: Subsample the file list to prevent OOM ---
            if file_subsample_percent < 1.0:
                print(f"--- INFO: Subsampling to {file_subsample_percent * 100:.1f}% of data files. ---")
                num_files_to_load = int(np.ceil(len(all_csv_files) * file_subsample_percent))
                
                np.random.seed(RANDOM_SEED)
                shuffled_files = np.random.permutation(all_csv_files)
                csv_files_to_load = shuffled_files[:num_files_to_load]
            # --- END FIX ---
            
            print(f"Loading {len(csv_files_to_load)} files from {data_path}...")
            for f in csv_files_to_load:
                try:
                    df_chunk = pd.read_csv(f)
                    all_data_df.append(df_chunk)
                except Exception as e:
                    print(f"Warning: Could not load file {f}. Error: {e}")
                    
        elif os.path.isfile(data_path):
            print(f"Loading single file: {data_path}...")
            try:
                df_single = pd.read_csv(data_path)
                all_data_df.append(df_single)
            except Exception as e:
                raise FileNotFoundError(f"Error loading file {data_path}: {e}")
        else:
            raise FileNotFoundError(f"Data path not found: {data_path}")

        if not all_data_df:
            raise ValueError("No data loaded. Check data paths and file formats.")

        df = pd.concat(all_data_df, ignore_index=True)

        if LABEL_COL not in df.columns:
            raise ValueError(f"Label column '{LABEL_COL}' not found in processed data.")
            
        labels = df[LABEL_COL].values
        
        feature_cols = [col for col in df.columns if col != LABEL_COL]
        
        if len(feature_cols) > self.max_vars:
            print(f"Warning: Loaded data has {len(feature_cols)} features. Truncating to {self.max_vars}.")
            feature_cols = feature_cols[:self.max_vars]
        elif len(feature_cols) < self.max_vars:
             raise ValueError(f"Data has {len(feature_cols)} features, but config requires {self.max_vars}. Run preprocessing again.")
            
        data = df[feature_cols].values
        
        return data, labels, feature_cols

    def _create_window_indices(self) -> List[Tuple[int, int]]:
        """Generates a list of (start, end) tuples for windowing."""
        indices = []
        full_length = len(self.data)
        
        start = 0
        while start + self.window_size <= full_length:
            indices.append((start, start + self.window_size))
            start += self.step_size
        return indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """
        --- "FAST LOADER" __getitem__ (Unchanged) ---
        """
        start, end = self.indices[idx]
        
        original_window_data = self.data[start:end] 
        window_labels = self.labels[start:end]
        window_label = 1 if np.sum(window_labels) > 0 else 0
        m_i_actual = self.m_i_actual
        
        window_features = []
        for m in range(self.max_vars): 
            series = original_window_data[:, m]
            
            if m < m_i_actual:
                gasf_image = calculate_gasf(series, self.output_spatial_size)
                spectrogram_image = calculate_spectrogram(series, self.output_spatial_size)
            else:
                gasf_image = np.zeros((self.output_spatial_size, self.output_spatial_size), dtype=np.float32)
                spectrogram_image = np.zeros((self.output_spatial_size, self.output_spatial_size), dtype=np.float32)
            
            window_features.append(gasf_image)
            window_features.append(spectrogram_image)
        
        image_array = np.stack(window_features, axis=0) 

        image_tensor = torch.tensor(image_array, dtype=torch.float32)
        original_window_tensor = torch.tensor(original_window_data, dtype=torch.float32)
        
        return image_tensor, original_window_tensor, window_label, m_i_actual