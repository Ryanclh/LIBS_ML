"""
LIBS Smoothing Filter Comparison Script
========================================
Run this BEFORE finalizing libs_preprocessing.py

Methodology Justification:
- Schmid 2022 [233] (197 citations): "SG filters should be replaced"
- SG has poor noise suppression above cutoff frequency
- Modified Sinc (MS) provides better stopband rejection

Decision Rule from Context Feeder:
"Test both SG AND Modified Sinc filter. If alternative improves accuracy >2%, use it."

This script provides quantitative comparison to make an informed decision.
"""

import numpy as np
import pandas as pd
from scipy. signal import savgol_filter
from scipy.fft import fft, fftfreq
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, normalize
from scipy import sparse
from scipy.sparse. linalg import spsolve
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import warnings
from dataclasses import dataclass
import os

warnings.filterwarnings('ignore')


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_real_data(
    filepath: str,
    *,
    wavelengths_in_first_column: bool = False,
    sample_names_in_first_row: bool = True,
    label_column:  str = None,
    label_column_index: int = -1,
    label_map_csv:  str = None,
    label_map_sample_col: str = "sample",
    label_map_label_col: str = "label",
    wavelength_range:  Tuple[float, float] = (200.0, 950.0),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Robust loader for different CSV layouts. 

    Handles two common cases:
    1) (samples x features) - rows = samples, cols = wavelengths (default)
       - last column may be labels (use label_column or label_column_index)
    2) (features x samples) - rows = wavelengths, cols = sample names (your case)
       - set wavelengths_in_first_column=True and sample_names_in_first_row=True

    Returns: 
      spectra :  np.ndarray, shape (n_samples, n_features)
      labels  : np.ndarray, shape (n_samples,) or None if not provided
      wavelengths : np.ndarray, shape (n_features,)

    If labels are not present in the CSV, you can provide label_map_csv which must
    contain two columns (sample name, label). The function will map sample names to labels.
    """
    import os
    import pandas as pd
    import numpy as np

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    print(f"\n{'='*60}")
    print("LOADING REAL LIBS DATA")
    print(f"{'='*60}")
    print(f"File: {filepath}")

    # Case A: wavelengths in first column (rows = wavelengths, cols = samples)
    if wavelengths_in_first_column:
        print("Layout:  Wavelengths in rows (first column), Samples in columns (first row)")
        
        # Read CSV with first column as index and first row as headers
        # header=0 means first row is column names (sample names)
        # index_col=0 means first column is row index (wavelengths)
        df = pd.read_csv(filepath, header=0, index_col=0)
        
        print(f"Raw DataFrame shape: {df.shape} (rows=wavelengths, cols=samples)")
        print(f"First 3 column names: {df.columns[: 3]. tolist()}")
        print(f"First 3 index values: {df.index[:3].tolist()}")
        
        # Extract sample names from column headers
        sample_names = df.columns. astype(str).tolist()
        n_samples = len(sample_names)
        
        # Extract wavelengths from row index
        try:
            # Try direct conversion
            wavelengths = df.index.astype(float).values
            print(f"✓ Wavelengths parsed:  {wavelengths[0]:.3f} - {wavelengths[-1]:. 3f} nm ({len(wavelengths)} points)")
        except (ValueError, TypeError) as e:
            print(f"⚠️  Warning: Could not parse index as wavelengths: {e}")
            print(f"   Attempting to clean and convert...")
            try:
                # Try cleaning:  strip whitespace and convert
                wavelengths = pd.to_numeric(df.index.str.strip(), errors='coerce').values
                if np.any(np.isnan(wavelengths)):
                    print(f"⚠️  Warning: Some wavelengths failed to parse, using fallback range")
                    n_features = df.shape[0]
                    wavelengths = np.linspace(wavelength_range[0], wavelength_range[1], n_features)
                else:
                    print(f"✓ Wavelengths cleaned: {wavelengths[0]:.3f} - {wavelengths[-1]:.3f} nm")
            except Exception as e2:
                print(f"❌ Failed to parse wavelengths: {e2}")
                print(f"   Using fallback:  evenly spaced {wavelength_range[0]} - {wavelength_range[1]} nm")
                n_features = df.shape[0]
                wavelengths = np.linspace(wavelength_range[0], wavelength_range[1], n_features)
        
        # Convert DataFrame to spectra array
        # Transpose:  (wavelengths, samples) -> (samples, wavelengths)
        try:
            spectra = df.values.T. astype(float)
            print(f"✓ Spectra array shape: {spectra.shape} (samples, features)")
        except ValueError as e:
            print(f"❌ Failed to convert spectra to float: {e}")
            print(f"   This usually means non-numeric values in the spectral data.")
            print(f"   First few values of first column: {df.iloc[:5, 0]. tolist()}")
            raise ValueError(f"Cannot convert spectral data to numeric values.  Check CSV format.")
        
        # Handle labels
        labels = None
        
        # 1) Check if a specific column in df contains labels (unlikely in this layout)
        if label_column is not None and label_column in df.columns:
            print(f"⚠️  Found label column '{label_column}' in data (unusual for this layout)")
            labels = df[label_column].values
            # Remove label column from spectra
            df_no_label = df.drop(columns=[label_column])
            spectra = df_no_label.values.T.astype(float)
            sample_names = df_no_label.columns.astype(str).tolist()
        
        # 2) Use label map CSV to map sample names to labels
        if labels is None and label_map_csv is not None:
            if os.path.exists(label_map_csv):
                print(f"Loading label mapping from: {label_map_csv}")
                map_df = pd.read_csv(label_map_csv)
                print(f"  Label map shape: {map_df.shape}")
                print(f"  Label map columns:  {map_df.columns.tolist()}")
                
                # Build mapping dictionary
                if label_map_sample_col not in map_df.columns:
                    raise ValueError(f"Sample column '{label_map_sample_col}' not found in label map.  Available:  {map_df.columns.tolist()}")
                if label_map_label_col not in map_df.columns:
                    raise ValueError(f"Label column '{label_map_label_col}' not found in label map. Available: {map_df. columns.tolist()}")
                
                mapping = dict(zip(
                    map_df[label_map_sample_col].astype(str), 
                    map_df[label_map_label_col]. astype(str)
                ))
                
                # Map sample names to labels
                labels = np.array([mapping. get(s, "Unknown") for s in sample_names])
                
                # Check for unmapped samples
                n_unknown = np.sum(labels == "Unknown")
                if n_unknown > 0:
                    print(f"⚠️  Warning: {n_unknown}/{n_samples} samples have no label mapping (labeled as 'Unknown')")
                    print(f"   First few unmapped samples: {[s for s, l in zip(sample_names, labels) if l == 'Unknown'][:5]}")
                else:
                    print(f"✓ All {n_samples} samples successfully mapped to labels")
            else:
                raise FileNotFoundError(f"Label map file not found: {label_map_csv}")
        
        # 3) Fallback: use sample names as identifiers
        if labels is None: 
            print("⚠️  No labels provided.  Using sample names as identifiers.")
            print("   Classification will not work without proper labels!")
            labels = np.array(sample_names)
        
        # Final summary
        print(f"\n{'='*60}")
        print("DATA LOADING SUMMARY")
        print(f"{'='*60}")
        print(f"Spectra shape:      {spectra.shape} (samples, features)")
        print(f"Wavelengths:        {len(wavelengths)} points ({wavelengths. min():.3f} - {wavelengths.max():.3f} nm)")
        print(f"Labels:            {len(labels)} samples")
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f"Unique labels:      {len(unique_labels)}")
        for lbl, cnt in zip(unique_labels, counts):
            print(f"  - {lbl}: {cnt} samples")
        print(f"{'='*60}\n")
        
        return spectra, labels, wavelengths

    # Case B: samples in rows and wavelengths are columns (original default)
    print("Layout: Samples in rows, Wavelengths in columns")
    df = pd.read_csv(filepath)
    
    # If label_column is provided by name
    if label_column is not None and label_column in df.columns:
        labels = df[label_column].values
        df_spectra = df.drop(columns=[label_column])
    else:
        # Use index-based label column if present
        if label_column_index is not None and -df. shape[1] <= label_column_index < df.shape[1]:
            labels = df.iloc[:, label_column_index].values
            df_spectra = df.drop(df.columns[label_column_index], axis=1)
        else:
            # No labels in file
            labels = None
            df_spectra = df

    # Try to parse column headers as wavelengths, else generate wavelengths
    try:
        wavelengths = df_spectra.columns.astype(float).values
        print(f"✓ Wavelengths from headers: {wavelengths[0]:.3f} - {wavelengths[-1]:.3f} nm")
    except Exception: 
        n_features = df_spectra. shape[1]
        wavelengths = np.linspace(wavelength_range[0], wavelength_range[1], n_features)
        print(f"✓ Wavelengths generated: {wavelengths[0]:.3f} - {wavelengths[-1]:.3f} nm")

    spectra = df_spectra.values.astype(float)

    # If labels not present in CSV but a label_map is provided
    if labels is None and label_map_csv is not None:
        if os.path. exists(label_map_csv):
            map_df = pd.read_csv(label_map_csv)
            mapping = dict(zip(
                map_df[label_map_sample_col].astype(str), 
                map_df[label_map_label_col].astype(str)
            ))
            
            # Find sample name column (non-numeric)
            sample_name_col = None
            for col in df. columns:
                if df[col].dtype == object: 
                    sample_name_col = col
                    break
            
            if sample_name_col: 
                sample_names = df[sample_name_col].astype(str).values
                labels = np.array([mapping.get(s, "Unknown") for s in sample_names])
            else:
                raise ValueError("No sample-name column found to map labels.  Provide labels in CSV or check label_map_csv.")
        else:
            raise FileNotFoundError(f"Label map file not found: {label_map_csv}")

    print(f"\nSpectra shape: {spectra.shape}")
    print(f"Wavelengths:  {len(wavelengths)} points")
    if labels is not None:
        print(f"Labels: {dict(zip(*np.unique(labels, return_counts=True)))}")
    
    return spectra, labels, wavelengths

def generate_synthetic_data(
    n_samples: int = 324,
    n_features: int = 12981,
    wavelength_range: Tuple[float, float] = (200, 950),
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic LIBS data for testing the pipeline.
    
    Parameters
    ----------
    n_samples : int
        Number of samples to generate (default 324 per methodology)
    n_features : int
        Number of spectral features (default 12,981 per methodology)
    wavelength_range : Tuple[float, float]
        Wavelength range in nm
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    spectra : np.ndarray
    labels : np.ndarray
    wavelengths : np.ndarray
    """
    print(f"\n{'='*60}")
    print("GENERATING SYNTHETIC LIBS DATA")
    print(f"{'='*60}")
    
    np.random.seed(random_state)
    wavelengths = np.linspace(wavelength_range[0], wavelength_range[1], n_features)
    
    spectra = np.zeros((n_samples, n_features))
    labels = np.repeat(['High', 'Medium', 'Low'], n_samples // 3 + 1)[:n_samples]
    
    # NIST database peak positions for N, P, K (from Context Feeder)
    # Nitrogen peaks: 742.4, 744.2, 746.8 nm
    # Phosphorus peaks: 253.5, 255.3 nm
    # Potassium peaks: 766.5, 769.9 nm
    peak_positions = [253.5, 255.3, 742.4, 744.2, 746.8, 766.5, 769.9]
    
    for i in range(n_samples):
        # Baseline (Bremsstrahlung continuum) - physics-based
        baseline = 1000 * np.exp(-0.005 * (wavelengths - 200))
        
        # Generate peaks with class-dependent intensities
        peak_intensities = np.random. uniform(500, 2000, len(peak_positions))
        
        if labels[i] == 'High':
            peak_intensities *= 1.5
        elif labels[i] == 'Low':
            peak_intensities *= 0.6
        
        for pos, intensity in zip(peak_positions, peak_intensities):
            # Gaussian peak profile (typical for LIBS)
            peak = intensity * np.exp(-0.5 * ((wavelengths - pos) / 0.5)**2)
            spectra[i] += peak
        
        spectra[i] += baseline
        spectra[i] += np.random.normal(0, 50, n_features)  # Sensor noise
    
    print(f"Generated:  {n_samples} samples × {n_features} features")
    print(f"Wavelength range: {wavelengths.min():.1f} - {wavelengths. max():.1f} nm")
    print(f"Classes: {np.unique(labels)}")
    print(f"Class distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
    print(f"{'='*60}\n")
    
    return spectra, labels, wavelengths


# ============================================================================
# FILTER IMPLEMENTATIONS
# ============================================================================

def savitzky_golay(spectrum: np.ndarray, window: int = 11, order: int = 3) -> np.ndarray:
    """Standard Savitzky-Golay filter [228]."""
    return savgol_filter(spectrum, window, order)


def modified_sinc(spectrum: np.ndarray, fc: float = 0.1, M: int = 11) -> np.ndarray:
    """
    Modified Sinc (Windowed-Sinc) low-pass filter [233].
    
    Schmid 2022: Better stopband rejection than SG.
    
    Parameters
    ----------
    spectrum : np. ndarray
        Input signal
    fc : float
        Normalized cutoff frequency (0 < fc < 0.5)
    M : int
        Filter length (odd)
    """
    if M % 2 == 0:
        M += 1
    
    n = np.arange(M)
    h = np.sinc(2 * fc * (n - (M - 1) / 2))
    
    # Blackman window for better stopband attenuation
    window = np. blackman(M)
    h = h * window
    h = h / np.sum(h)
    
    return np.convolve(spectrum, h, mode='same')


def modified_sinc_hamming(spectrum: np.ndarray, fc: float = 0.1, M: int = 11) -> np.ndarray:
    """Modified Sinc with Hamming window (alternative)."""
    if M % 2 == 0:
        M += 1
    
    n = np.arange(M)
    h = np.sinc(2 * fc * (n - (M - 1) / 2))
    window = np.hamming(M)
    h = h * window
    h = h / np.sum(h)
    
    return np.convolve(spectrum, h, mode='same')


def gaussian_filter(spectrum: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Gaussian smoothing filter (additional comparison)."""
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(spectrum, sigma=sigma)


# ============================================================================
# BASELINE CORRECTION (Same as main pipeline)
# ============================================================================

def polynomial_baseline(spectrum: np.ndarray, degree: int = 5) -> np.ndarray:
    """Polynomial baseline correction."""
    x = np.arange(len(spectrum))
    coeffs = np.polyfit(x, spectrum, degree)
    baseline = np.polyval(coeffs, x)
    return baseline

def apply_baseline_correction_batch(
    spectra: np.ndarray,
    degree: int = 5,
    verbose: bool = True
) -> np.ndarray:
    """Apply Polynomial baseline correction to all spectra."""
    n_samples = spectra.shape[0]
    n_features = spectra.shape[1]
    corrected = np.zeros_like(spectra)
    
    if verbose:
        print(f"Applying Polynomial (degree={degree}) baseline correction to {n_samples} spectra...")
    
    x = np.arange(n_features)
    
    for i in range(n_samples):
        # Fit polynomial to spectrum
        coeffs = np.polyfit(x, spectra[i], degree)
        baseline = np.polyval(coeffs, x)
        
        # Subtract baseline (ensure non-negative)
        corrected[i] = np.maximum(spectra[i] - baseline, 0)
        
        if verbose and (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{n_samples} spectra...")
    
    if verbose:
        print(f"✓ Baseline correction complete")
    
    return corrected

# ============================================================================
# EVALUATION METRICS
# ============================================================================

@dataclass
class FilterMetrics:
    """Container for filter evaluation metrics."""
    name: str
    noise_reduction_pct: float
    peak_preservation_pct: float
    snr_improvement_pct: float
    frequency_attenuation_db: float
    classification_accuracy: float
    computation_time_ms: float


def calculate_noise_reduction(original: np.ndarray, filtered: np.ndarray,
                              noise_region: slice = slice(100, 300)) -> float:
    """Calculate noise reduction percentage in a flat region."""
    noise_before = np.std(original[noise_region])
    noise_after = np.std(filtered[noise_region])
    
    if noise_before == 0:
        return 0.0
    
    return (noise_before - noise_after) / noise_before * 100


def calculate_peak_preservation(original: np. ndarray, filtered: np.ndarray) -> float:
    """Calculate peak height preservation percentage."""
    # Find top 10 peaks in original
    peak_indices = np.argsort(original)[-10:]
    
    original_heights = original[peak_indices]
    filtered_heights = filtered[peak_indices]
    
    # Avoid division by zero
    valid = original_heights > 0
    if not np.any(valid):
        return 100.0
    
    preservation = np.mean(filtered_heights[valid] / original_heights[valid]) * 100
    return min(preservation, 100.0)  # Cap at 100%


def calculate_snr(spectrum: np.ndarray, signal_region: slice = slice(500, 600),
                  noise_region: slice = slice(100, 200)) -> float:
    """Calculate Signal-to-Noise Ratio."""
    signal = np.max(spectrum[signal_region]) - np.mean(spectrum[noise_region])
    noise = np.std(spectrum[noise_region])
    
    if noise == 0:
        return 0.0
    
    return signal / noise


def calculate_stopband_attenuation(original: np.ndarray, filtered: np.ndarray) -> float:
    """
    Calculate stopband attenuation in dB.
    
    This is the key metric from Schmid 2022 [233] - MS should show
    better high-frequency suppression than SG.
    """
    # Compute FFT
    n = len(original)
    fft_original = np.abs(fft(original))[:n//2]
    fft_filtered = np.abs(fft(filtered))[:n//2]
    
    # High-frequency region (stopband) - upper 20%
    stopband = slice(int(0.4 * n//2), n//2)
    
    # Avoid log(0)
    fft_original[fft_original == 0] = 1e-10
    fft_filtered[fft_filtered == 0] = 1e-10
    
    # Attenuation in dB
    attenuation = 20 * np.log10(
        np.mean(fft_original[stopband]) / np.mean(fft_filtered[stopband])
    )
    
    return attenuation


# ============================================================================
# COMPREHENSIVE FILTER COMPARISON
# ============================================================================

def compare_filters(
    spectra:  np.ndarray,
    wavelengths: np.ndarray,
    verbose: bool = True
) -> Dict[str, FilterMetrics]:
    """
    Comprehensive comparison of smoothing filters using SPECTRAL QUALITY METRICS ONLY.
    
    No classification required - compares filters based on: 
    - Noise reduction (%)
    - Peak preservation (%)
    - SNR improvement (%)
    - Stopband attenuation (dB) - Key metric from Schmid 2022 [233]
    - Computation time (ms)
    
    Parameters
    ----------
    spectra : np. ndarray
        Baseline-corrected spectra (n_samples, n_features)
    wavelengths : np.ndarray
        Wavelength values
    verbose : bool
        Print progress
        
    Returns
    -------
    results : Dict[str, FilterMetrics]
        Metrics for each filter method
    """
    import time
    
    # Filter configurations to test
    filter_configs = {
        'SG_w7_o2': lambda x: savitzky_golay(x, window=7, order=2),
        'SG_w11_o3': lambda x: savitzky_golay(x, window=11, order=3),  # Default
        'SG_w15_o3': lambda x: savitzky_golay(x, window=15, order=3),
        'SG_w21_o4': lambda x: savitzky_golay(x, window=21, order=4),
        'MS_fc0. 05_M11': lambda x: modified_sinc(x, fc=0.05, M=11),
        'MS_fc0.10_M11': lambda x: modified_sinc(x, fc=0.10, M=11),  # Default
        'MS_fc0.10_M15': lambda x: modified_sinc(x, fc=0.10, M=15),
        'MS_fc0.15_M11': lambda x: modified_sinc(x, fc=0.15, M=11),
        'MS_Hamming_fc0.10':  lambda x: modified_sinc_hamming(x, fc=0.10, M=11),
        'Gaussian_s1. 0':  lambda x: gaussian_filter(x, sigma=1.0),
        'Gaussian_s1.5': lambda x: gaussian_filter(x, sigma=1.5),
        'Gaussian_s2.0': lambda x: gaussian_filter(x, sigma=2.0),
    }
    
    results = {}
    n_samples = spectra.shape[0]
    
    if verbose:
        print("\n" + "="*70)
        print("SMOOTHING FILTER COMPARISON (Spectral Quality Metrics)")
        print("Methodology: Schmid 2022 [233] - Comparing SG vs Modified Sinc")
        print("="*70)
        print("\nNote: Classification disabled - using spectral quality only")
        print("      (Provide label_map. csv for classification-based comparison)\n")
    
    for name, filter_func in filter_configs.items():
        if verbose:
            print(f"Evaluating:  {name}...")
        
        # Apply filter to all spectra
        start_time = time.time()
        filtered = np.apply_along_axis(filter_func, axis=1, arr=spectra)
        comp_time = (time.time() - start_time) / n_samples * 1000  # ms per spectrum
        
        # Calculate metrics on representative samples
        noise_reductions = []
        peak_preservations = []
        snr_improvements = []
        attenuations = []
        
        for i in range(min(50, n_samples)):  # Sample 50 spectra
            orig = spectra[i]
            filt = filtered[i]
            
            noise_reductions.append(calculate_noise_reduction(orig, filt))
            peak_preservations.append(calculate_peak_preservation(orig, filt))
            
            snr_orig = calculate_snr(orig)
            snr_filt = calculate_snr(filt)
            if snr_orig > 0:
                snr_improvements.append((snr_filt - snr_orig) / snr_orig * 100)
            
            attenuations.append(calculate_stopband_attenuation(orig, filt))
        
        results[name] = FilterMetrics(
            name=name,
            noise_reduction_pct=np.mean(noise_reductions),
            peak_preservation_pct=np.mean(peak_preservations),
            snr_improvement_pct=np.mean(snr_improvements) if snr_improvements else 0,
            frequency_attenuation_db=np.mean(attenuations),
            classification_accuracy=0.0,  # Not applicable without labels
            computation_time_ms=comp_time
        )
        
        if verbose:
            m = results[name]
            print(f"  Noise Reduction:     {m.noise_reduction_pct:+.1f}%")
            print(f"  Peak Preservation:  {m.peak_preservation_pct:.1f}%")
            print(f"  SNR Improvement:    {m.snr_improvement_pct:+.1f}%")
            print(f"  Stopband Atten.:     {m.frequency_attenuation_db:.1f} dB")
            print(f"  Computation Time:   {m.computation_time_ms:.3f} ms")
    
    return results

def generate_recommendation(results: Dict[str, FilterMetrics]) -> Tuple[str, str]:
    """
    Generate recommendation based on SPECTRAL QUALITY metrics only.
    
    Scoring weights based on LIBS best practices:
    - Peak preservation: 40% (CRITICAL - LIBS analysis depends on peak heights)
    - Noise reduction:  30% (Important for low-concentration detection)
    - Stopband attenuation: 20% (Schmid 2022 [233] key metric)
    - SNR improvement: 10% (Overall quality indicator)
    
    Returns
    -------
    recommended_filter : str
        Name of recommended filter
    justification : str
        Detailed justification for the recommendation
    """
    # Separate SG and MS results
    sg_results = {k: v for k, v in results.items() if k.startswith('SG')}
    ms_results = {k: v for k, v in results.items() if k.startswith('MS')}
    gaussian_results = {k: v for k, v in results.items() if k.startswith('Gaussian')}
    
    # Calculate weighted scores for all filters
    scores = {}
    for name, m in results.items():
        # Weighted combination (higher is better for all metrics)
        score = (
            m.peak_preservation_pct * 0.40 +      # CRITICAL for LIBS
            m.noise_reduction_pct * 0.30 +        # Important
            m.frequency_attenuation_db * 0.20 +   # Schmid 2022 [233]
            m.snr_improvement_pct * 0.10          # Overall quality
        )
        scores[name] = score
    
    # Find best in each category
    best_sg = max(sg_results.values(), key=lambda x: scores[x. name])
    best_ms = max(ms_results.values(), key=lambda x: scores[x. name])
    best_gaussian = max(gaussian_results.values(), key=lambda x: scores[x.name])
    
    # Overall best
    recommended = max(scores, key=scores.get)
    best_metrics = results[recommended]
    
    # Build justification
    justification_parts = []
    
    justification_parts.append(f"Best SG: {best_sg.name} (score={scores[best_sg.name]:.2f})")
    justification_parts.append(f"Best MS: {best_ms.name} (score={scores[best_ms.name]:.2f})")
    justification_parts.append(f"Best Gaussian: {best_gaussian.name} (score={scores[best_gaussian.name]:.2f})")
    justification_parts.append("")
    
    # Compare SG vs MS (Schmid 2022 [233] question)
    sg_vs_ms_stopband = best_ms.frequency_attenuation_db - best_sg.frequency_attenuation_db
    justification_parts.append(f"Stopband attenuation (MS vs SG): {sg_vs_ms_stopband:+.1f} dB")
    justification_parts.append("")
    
    # Decision logic based on spectral quality
    justification_parts.append(f"RECOMMENDATION: {recommended}")
    justification_parts.append("")
    justification_parts.append("SPECTRAL QUALITY METRICS:")
    justification_parts.append(f"- Peak Preservation:     {best_metrics.peak_preservation_pct:.1f}% (weight: 40%)")
    justification_parts.append(f"- Noise Reduction:      {best_metrics.noise_reduction_pct:+.1f}% (weight: 30%)")
    justification_parts.append(f"- Stopband Attenuation: {best_metrics.frequency_attenuation_db:.1f} dB (weight: 20%)")
    justification_parts.append(f"- SNR Improvement:      {best_metrics.snr_improvement_pct:+.1f}% (weight: 10%)")
    justification_parts.append(f"- Computation Time:     {best_metrics.computation_time_ms:.3f} ms/spectrum")
    justification_parts.append("")
    
    # Interpretation based on context
    if recommended. startswith('MS'):
        justification_parts. append("INTERPRETATION:")
        justification_parts.append("Modified Sinc filter shows best overall spectral quality.")
        justification_parts.append("This supports Schmid 2022 [233]:  'SG filters should be replaced'")
        justification_parts. append(f"Stopband rejection advantage: {sg_vs_ms_stopband:+.1f} dB over SG")
        justification_parts.append("")
        justification_parts.append("BENEFIT for TinyML:")
        justification_parts.append("Better noise suppression → cleaner features → potentially higher accuracy")
    elif recommended.startswith('SG'):
        justification_parts.append("INTERPRETATION:")
        justification_parts.append("Savitzky-Golay filter provides best balance of quality and simplicity.")
        justification_parts.append("Acceptable per standard LIBS practice [228]")
        if abs(sg_vs_ms_stopband) < 3:
            justification_parts.append(f"Stopband difference vs MS is small ({sg_vs_ms_stopband:+.1f} dB)")
        justification_parts.append("")
        justification_parts.append("BENEFIT for TinyML:")
        justification_parts.append("Lower computational cost → faster inference → lower power consumption")
    else:
        justification_parts. append("INTERPRETATION:")
        justification_parts. append("Gaussian filter provides best spectral quality for this dataset.")
        justification_parts. append("")
        justification_parts.append("BENEFIT for TinyML:")
        justification_parts.append("Simple convolution → easy MCU implementation")
    
    justification_parts.append("")
    justification_parts.append("NOTE:  Classification accuracy comparison requires label_map.csv")
    justification_parts.append("      This recommendation is based on spectral quality metrics only.")
    
    return recommended, "\n".join(justification_parts)

def plot_filter_comparison(
    spectrum:  np.ndarray,
    wavelengths: np.ndarray,
    results: Dict[str, FilterMetrics],
    recommended:  str,
    save_path: str = None
) -> None:
    """Visualize filter comparison results (spectral quality metrics only)."""
    
    # Changed to 2x2 layout with 3 active plots (removed classification chart)
    fig = plt.figure(figsize=(16, 10))
    
    # Plot 1: Spectral comparison of best filters
    ax1 = fig. add_subplot(2, 2, 1)
    ax1.plot(wavelengths, spectrum, 'k-', alpha=0.3, linewidth=0.5, label='Original')
    
    colors = {'SG': 'blue', 'MS': 'red', 'Gaussian': 'green'}
    for name in ['SG_w11_o3', 'MS_fc0. 10_M11', 'Gaussian_s1. 5']: 
        if name. startswith('SG'):
            filtered = savitzky_golay(spectrum, 11, 3)
            c = colors['SG']
        elif name. startswith('MS'):
            filtered = modified_sinc(spectrum, 0.10, 11)
            c = colors['MS']
        else:
            filtered = gaussian_filter(spectrum, 1.5)
            c = colors['Gaussian']
        
        ax1.plot(wavelengths, filtered, color=c, linewidth=1,
                label=f'{name}', alpha=0.8)
    
    ax1.set_xlabel('Wavelength (nm)', fontsize=11)
    ax1.set_ylabel('Intensity', fontsize=11)
    ax1.set_title('Spectral Comparison (Default Parameters)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Frequency domain comparison
    ax2 = fig.add_subplot(2, 2, 2)
    n = len(spectrum)
    freqs = fftfreq(n, d=1)[: n//2]
    
    fft_orig = np.abs(fft(spectrum))[:n//2]
    fft_sg = np.abs(fft(savitzky_golay(spectrum, 11, 3)))[:n//2]
    fft_ms = np.abs(fft(modified_sinc(spectrum, 0.10, 11)))[:n//2]
    
    ax2.semilogy(freqs, fft_orig, 'k-', alpha=0.3, label='Original', linewidth=0.8)
    ax2.semilogy(freqs, fft_sg, 'b-', label='SG_w11_o3', alpha=0.8, linewidth=1.2)
    ax2.semilogy(freqs, fft_ms, 'r-', label='MS_fc0.10_M11', alpha=0.8, linewidth=1.2)
    ax2.axvline(x=0.1, color='gray', linestyle='--', alpha=0.7, label='Cutoff (fc=0.1)')
    
    ax2.set_xlabel('Normalized Frequency', fontsize=11)
    ax2.set_ylabel('Magnitude (log scale)', fontsize=11)
    ax2.set_title('Frequency Response (Stopband Attenuation)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, which='both')
    
    # Plot 3: Spectral Quality Metrics Comparison (replaces classification chart)
    ax3 = fig.add_subplot(2, 2, 3)
    
    # Select top 6 filters by overall quality score
    scores = {}
    for name, m in results.items():
        score = (
            m.peak_preservation_pct * 0.40 +
            m.noise_reduction_pct * 0.30 +
            m.frequency_attenuation_db * 0.20 +
            m.snr_improvement_pct * 0.10
        )
        scores[name] = score
    
    # Get top 6 filters
    top_filters = sorted(scores.items(), key=lambda x: -x[1])[:6]
    filter_names = [f[0] for f in top_filters]
    filter_scores = [f[1] for f in top_filters]
    
    # Color code by type
    colors_bar = []
    for name in filter_names:
        if name == recommended:
            colors_bar. append('green')
        elif name.startswith('SG'):
            colors_bar.append('blue')
        elif name.startswith('MS'):
            colors_bar.append('red')
        else:
            colors_bar.append('orange')
    
    bars = ax3.barh(filter_names, filter_scores, color=colors_bar, alpha=0.7, edgecolor='black')
    
    # Highlight recommended filter
    for bar, name in zip(bars, filter_names):
        if name == recommended:
            bar.set_edgecolor('darkgreen')
            bar.set_linewidth(3)
    
    ax3.set_xlabel('Weighted Quality Score', fontsize=11)
    ax3.set_title(f'Overall Spectral Quality Ranking\n(Recommended: {recommended})', 
                  fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Add score values on bars
    for bar, score in zip(bars, filter_scores):
        width = bar.get_width()
        ax3.text(width, bar.get_y() + bar.get_height()/2, 
                f'{score:.1f}',
                ha='left', va='center', fontsize=9, fontweight='bold')
    
    # Plot 4: Multi-metric radar comparison
    ax4 = fig. add_subplot(2, 2, 4)
    
    # Select representative filters for detailed comparison
    rep_filters = ['SG_w11_o3', 'MS_fc0.10_M11', 'Gaussian_s1.5']
    metrics_names = ['Noise\nReduction', 'Peak\nPreservation', 'SNR\nImprovement', 'Stopband\nAttenuation']
    
    x = np.arange(len(metrics_names))
    width = 0.25
    
    for i, fname in enumerate(rep_filters):
        if fname not in results:
            continue
        m = results[fname]
        
        # Normalize metrics to 0-100 scale for comparison
        values = [
            m.noise_reduction_pct,
            m.peak_preservation_pct,
            m.snr_improvement_pct,
            min(m.frequency_attenuation_db * 5, 100)  # Scale stopband to similar range
        ]
        
        color = 'blue' if fname. startswith('SG') else ('red' if fname.startswith('MS') else 'green')
        ax4.bar(x + i*width, values, width, label=fname, alpha=0.7, color=color, edgecolor='black')
    
    ax4.set_xticks(x + width)
    ax4.set_xticklabels(metrics_names, fontsize=10)
    ax4.set_ylabel('Metric Value (%)', fontsize=11)
    ax4.set_title('Detailed Metric Comparison (Top 3 Filters)', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right')
    ax4.axhline(y=95, color='gray', linestyle='--', alpha=0.5, label='95% threshold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim(0, 105)
    
    # Overall title
    plt.suptitle('LIBS Smoothing Filter Comparison - Spectral Quality Analysis\nSchmid 2022 [233]:  Modified Sinc vs Savitzky-Golay Evaluation',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt. savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Figure saved to:  {save_path}")
    
    plt.show()

def print_results_table(results: Dict[str, FilterMetrics]) -> None:
    """Print formatted results table."""
    
    # Check if tabulate is available
    try: 
        from tabulate import tabulate
        use_tabulate = True
    except ImportError:
        use_tabulate = False
    
    headers = ['Filter', 'Noise Red.', 'Peak Pres.', 'SNR Impr.',
               'Stopband Att.', 'Accuracy', 'Time']
    
    rows = []
    for name, m in sorted(results.items(), key=lambda x: -x[1].classification_accuracy):
        rows.append([
            name,
            f"{m.noise_reduction_pct:+.1f}%",
            f"{m.peak_preservation_pct:.1f}%",
            f"{m.snr_improvement_pct:+.1f}%",
            f"{m.frequency_attenuation_db:.1f} dB",
            f"{m.classification_accuracy:.2f}%",
            f"{m. computation_time_ms:.3f} ms"
        ])
    
    print("\n" + "="*90)
    print("FILTER COMPARISON RESULTS (Sorted by Classification Accuracy)")
    print("="*90)
    
    if use_tabulate:
        print(tabulate(rows, headers=headers, tablefmt='grid'))
    else:
        # Manual table formatting
        col_widths = [20, 12, 12, 12, 14, 12, 12]
        header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
        print(header_line)
        print("-" * len(header_line))
        for row in rows:
            print(" | ".join(str(c).ljust(w) for c, w in zip(row, col_widths)))


def save_recommendation_report(
    results: Dict[str, FilterMetrics],
    recommended: str,
    justification: str,
    filepath: str = 'filter_recommendation.txt'
) -> None:
    """Save recommendation report to file."""
    
    # FIX: Add explicit UTF-8 encoding to handle Unicode characters (arrows, special symbols)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("LIBS SMOOTHING FILTER RECOMMENDATION REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write("METHODOLOGY REFERENCE:\n")
        f.write("- Schmid 2022 [233]:  'SG filters should be replaced'\n")
        f.write("- Decision Rule: 'If alternative improves accuracy >2%, use it'\n\n")
        
        f. write("RESULTS SUMMARY:\n")
        f.write("-"*70 + "\n")
        
        headers = ['Filter', 'Accuracy', 'Noise Red.', 'Peak Pres.', 'Stopband']
        f.write(f"{headers[0]:<25} {headers[1]:<12} {headers[2]:<12} {headers[3]:<12} {headers[4]:<12}\n")
        f.write("-"*70 + "\n")
        
        for name, m in sorted(results.items(), key=lambda x: -x[1].classification_accuracy):
            # Note: classification_accuracy is 0.0 for spectral-quality-only mode
            f.write(f"{name:<25} {m.classification_accuracy:.2f}%       "
                   f"{m.noise_reduction_pct:+.1f}%       "
                   f"{m. peak_preservation_pct:.1f}%       "
                   f"{m.frequency_attenuation_db:.1f} dB\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("RECOMMENDATION\n")
        f.write("="*70 + "\n\n")
        f.write(justification)
        f.write("\n\n")
        
        f. write("="*70 + "\n")
        f.write("ACTION REQUIRED\n")
        f.write("="*70 + "\n")
        f.write(f"\nUpdate libs_preprocessing.py with:\n\n")
        
        if recommended. startswith('SG'):
            params = recommended.replace('SG_w', '').replace('_o', ',').split(',')
            f.write(f"  smoothing_method = 'savgol'\n")
            f.write(f"  sg_window = {params[0]}\n")
            f.write(f"  sg_order = {params[1]}\n")
        elif recommended.startswith('MS'):
            if 'Hamming' in recommended:
                f.write(f"  # Use Modified Sinc with Hamming window\n")
                f.write(f"  # Modify apply_modified_sinc_filter to use Hamming\n")
            params = recommended.replace('MS_fc', '').replace('_M', ',').replace('_Hamming', '').split(',')
            f.write(f"  smoothing_method = 'sinc'\n")
            f.write(f"  sinc_fc = {params[0]}\n")
            if len(params) > 1 and params[1]: 
                f.write(f"  sinc_M = {params[1]}\n")
            else:
                f.write(f"  sinc_M = 11\n")
        else:
            # Gaussian
            sigma = recommended.split('_s')[1]
            f.write(f"  # Add Gaussian option to pipeline\n")
            f.write(f"  # Update preprocess_libs_spectra() to include:\n")
            f.write(f"  # smoothing_method = 'gaussian'\n")
            f.write(f"  gaussian_sigma = {sigma}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("IMPLEMENTATION GUIDE FOR libs_preprocessing.py\n")
        f.write("="*70 + "\n\n")
        
        f.write("1.  Locate the preprocess_libs_spectra() function\n")
        f.write("2. Update default parameters based on recommendation above\n")
        f.write("3. Run preprocessing pipeline with new filter settings\n")
        f.write("4. Validate on your 320 samples (Clay_1... Clay_320)\n")
        f.write("5. Proceed to Phase 2: Feature Selection using NIST peaks\n\n")
        
        f. write("NIST SPECTRAL LINES FOR NPK (197. 001-973.5891 nm range):\n")
        f.write("-"*70 + "\n")
        f.write("Nitrogen (N):   742.364, 744.229, 746.831 nm (primary)\n")
        f.write("                821.634, 824.238, 868.028 nm (secondary)\n")
        f.write("Phosphorus (P): 253.565, 255.328 nm (YOUR RANGE - use these! )\n")
        f.write("                213.618, 214.914 nm (below 197 nm - not captured)\n")
        f.write("Potassium (K):  766.491, 769.896 nm (primary - very strong)\n")
        f.write("                404.414, 404.721 nm (secondary)\n\n")
        
        f. write("NOTE: Your wavelength range (197.001-973.5891 nm) captures:\n")
        f.write("  - All N lines (742-868 nm)\n")
        f.write("  - All K lines (404-769 nm)\n")
        f.write("  - P 253-255 nm lines (Al-Juboori et al. used these)\n")
        f.write("  - Missing P 213-214 nm (below spectrometer range)\n\n")
    
    print(f"\n✓ Recommendation report saved to: {filepath}")
    
# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__": 
    print("="*70)
    print("LIBS SMOOTHING FILTER COMPARISON")
    print("Schmid 2022 [233]:  Evaluating SG vs Modified Sinc")
    print("="*70)
    
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    USE_REAL_DATA = True  # Set to False to use synthetic test data
    
    REAL_DATA_CONFIG = {
        # UPDATE THIS PATH to your actual CSV file
        "filepath": r"D:\Ryan\UM\Academic\Y4S1\KIE4002 FINAL YEAR PROJECT\Irish_Soil_DataSet\Irish_Soil_DataSet2.csv",
        
        # Your data layout:  First column = wavelengths, First row = sample names
        "wavelengths_in_first_column": True,
        "sample_names_in_first_row": True,
        
        # Label mapping (optional - set to None if not available)
        "label_map_csv": None,  # Set path if you have label mapping
        "label_map_sample_col": "sample",
        "label_map_label_col": "label",
        
        # Wavelength range (197. 001 - 973.5891 nm per your data)
        "wavelength_range": (197.001, 973.5891),
        
        "label_column":  None,
        "label_column_index": -1,
    }
    
    # =========================================================================
    # LOAD DATA
    # =========================================================================
    if USE_REAL_DATA: 
        cfg = REAL_DATA_CONFIG
        try:
            spectra_raw, labels, wavelengths = load_real_data(
                cfg["filepath"],
                wavelengths_in_first_column=cfg. get("wavelengths_in_first_column", False),
                sample_names_in_first_row=cfg.get("sample_names_in_first_row", True),
                label_column=cfg.get("label_column", None),
                label_column_index=cfg.get("label_column_index", -1),
                label_map_csv=cfg.get("label_map_csv", None),
                label_map_sample_col=cfg. get("label_map_sample_col", "sample"),
                label_map_label_col=cfg.get("label_map_label_col", "label"),
                wavelength_range=cfg.get("wavelength_range", (200.0, 950.0)),
            )
            print("\n✓ Real data loaded successfully")
        except Exception as e: 
            print(f"\n❌ Failed to load real data: {e}")
            print("\nDEBUGGING TIPS:")
            print("1. Check CSV has wavelengths in first column (197.001-973.5891)")
            print("2. Check CSV has sample names in first row (Clay_1, Clay_2, ... )")
            print("3. Ensure no extra header rows/columns")
            print("\nFalling back to synthetic data.. .\n")
            USE_REAL_DATA = False
    
    if not USE_REAL_DATA: 
        spectra_raw, labels, wavelengths = generate_synthetic_data(
            n_samples=324,
            n_features=12981,
            wavelength_range=(200, 950),
            random_state=42
        )
    
    # =========================================================================
    # APPLY BASELINE CORRECTION
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 1: Baseline Correction (Polynomial degree 5)")
    print("="*70)
    print("Removing Bremsstrahlung continuum background...")
    
    spectra_baseline = apply_baseline_correction_batch(
        spectra_raw,
        degree=5,
        verbose=True
    )
    
    # =========================================================================
    # RUN FILTER COMPARISON (NO CLASSIFICATION)
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 2: Filter Comparison (Spectral Quality Only)")
    print("="*70)
    
    # Modified function signature - no labels parameter
    results = compare_filters(spectra_baseline, wavelengths, verbose=True)
    
    # =========================================================================
    # DISPLAY RESULTS
    # =========================================================================
    print_results_table(results)
    
    # Generate recommendation
    recommended, justification = generate_recommendation(results)
    
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    print(justification)
    
    # Save report
    save_recommendation_report(results, recommended, justification)
    
    # Visualize
    plot_filter_comparison(
        spectra_baseline[0],
        wavelengths,
        results,
        recommended,
        save_path='filter_comparison.png'
    )
    
    # =========================================================================
    # SUMMARY & NEXT STEPS
    # =========================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"✓ Analyzed {spectra_baseline.shape[0]} spectra")
    print(f"✓ Tested {len(results)} filter configurations")
    print(f"✓ Recommended:  {recommended}")
    print("\nFiles generated:")
    print("  - filter_recommendation.txt (detailed report)")
    print("  - filter_comparison.png (visualization)")
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1. Review 'filter_recommendation.txt' for detailed justification")
    print("2. Update 'libs_preprocessing.py' with recommended filter:")
    print(f"   → Recommended: {recommended}")
    print("3. (Optional) Create label_map.csv for classification-based comparison")
    print("4. Proceed to preprocessing pipeline with selected filter")
    print("="*70)
    print("="*70)
    print("LIBS SMOOTHING FILTER COMPARISON")
    print("Schmid 2022 [233]:  Evaluating SG vs Modified Sinc")
    print("="*70)
    
    # -------------------- REPLACE the existing "DATA LOADING OPTIONS" + "LOAD DATA" block --------------------
    # (paste this in the main section of filter_comparison.py, replacing the previous USE_REAL_DATA / REAL_DATA_CONFIG / loading code)

    # Set this True to load your real CSV, False to use synthetic data
    USE_REAL_DATA = True  # <-- change to False to use synthetic test data

    # Example configuration for your dataset:
    # - first column = wavelengths (197.001 -> 973.5891)
    # - first row = sample names ("Clay_1", "Clay_2", ...)
    # - 320 samples, 12981 wavelength rows
    # Configuration for your dataset layout: 
    # CSV structure:  First column = wavelengths (197.001-973.5891), First row = sample names (Clay_1, Clay_2, ...)
    REAL_DATA_CONFIG = {
        # UPDATE THIS PATH to your actual file location
        "filepath": r"D:\Ryan\UM\Academic\Y4S1\KIE4002 FINAL YEAR PROJECT\Irish_Soil_DataSet\Irish_Soil_DataSet.csv",
        
        # Your data layout flags
        "wavelengths_in_first_column": True,   # Wavelengths are in rows (first column)
        "sample_names_in_first_row": True,     # Sample names are in columns (first row)
        
        # Label mapping (REQUIRED for classification)
        # Create a CSV file with two columns: sample,label
        # Example content:
        #   sample,label
        #   Clay_1,High
        #   Clay_2,Medium
        #   Clay_3,Low
        #   ...
        "label_map_csv": r"D:\Ryan\UM\Academic\Y4S1\KIE4002 FINAL YEAR PROJECT\Data\label_map.csv",
        "label_map_sample_col":  "sample",      # Column name containing sample IDs
        "label_map_label_col": "label",        # Column name containing class labels (High/Medium/Low)
        
        # Wavelength range (your actual range)
        "wavelength_range": (197.001, 973.5891),
        
        # Leave these as None for your layout
        "label_column": None,
        "label_column_index": -1,
    }

    # Load either real data (using config.get to avoid KeyError) or synthetic data
    if USE_REAL_DATA:
        cfg = REAL_DATA_CONFIG
        try:
            spectra_raw, labels, wavelengths = load_real_data(
                cfg["filepath"],
                wavelengths_in_first_column=cfg.get("wavelengths_in_first_column", False),
                sample_names_in_first_row=cfg.get("sample_names_in_first_row", True),
                label_column=cfg.get("label_column", None),
                label_column_index=cfg.get("label_column_index", -1),
                label_map_csv=cfg.get("label_map_csv", None),
                label_map_sample_col=cfg.get("label_map_sample_col", "sample"),
                label_map_label_col=cfg.get("label_map_label_col", "label"),
                wavelength_range=cfg.get("wavelength_range", (200.0, 950.0)),
            )
        except Exception as e:
            print(f"\n❌ Failed to load real data: {e}")
            print("Falling back to synthetic data for debugging.\n")
            spectra_raw, labels, wavelengths = generate_synthetic_data(
                n_samples=324,
                n_features=12981,
                wavelength_range=(200, 950),
                random_state=42
            )
    else:
        spectra_raw, labels, wavelengths = generate_synthetic_data(
            n_samples=324,
            n_features=12981,
            wavelength_range=(200, 950),
            random_state=42
        )

    # Quick checks and guidance
    print(f"\nLoaded spectra shape: {spectra_raw.shape}")
    print(f"Wavelengths length:   {wavelengths.shape}")
    # If labels look like sample IDs (e.g., "Clay_1") rather than class labels ("High"/"Low"), warn the user
    unique_labels = np.unique(labels)
    if len(unique_labels) > 20 or any(s.startswith("Clay_") or s.startswith("Sample_") for s in unique_labels.astype(str)[:5]):
        print("\n⚠️  Note: The 'labels' array appears to contain sample IDs, not class labels.")
        print("    If you want to run classification-based comparison, provide a label-map CSV")
        print("    (see REAL_DATA_CONFIG['label_map_csv']) mapping sample name -> class (e.g. Clay_1 -> High).")
        print("    You can still run frequency / smoothing comparisons without class labels.\n")
    # -------------------------------------------------------------------------------------------------------
    
    # =========================================================================
    # APPLY BASELINE CORRECTION (Required before filter comparison)
    # =========================================================================
    
    print("\nApplying Polynomial baseline correction (degree 5)...")
    print("(This removes Bremsstrahlung continuum background)")
    
    spectra_baseline = apply_baseline_correction_batch(
        spectra_raw,
        degree=5,
        verbose=True
    )
    
    # =========================================================================
    # RUN FILTER COMPARISON
    # =========================================================================
    
    results = compare_filters(spectra_baseline, labels, wavelengths, verbose=True)
    
    # =========================================================================
    # DISPLAY RESULTS
    # =========================================================================
    
    print_results_table(results)
    
    # Generate recommendation
    recommended, justification = generate_recommendation(results)
    
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    print(justification)
    
    # Save report
    save_recommendation_report(results, recommended, justification)
    
    # Visualize
    plot_filter_comparison(
        spectra_baseline[0],
        wavelengths,
        results,
        recommended,
        save_path='filter_comparison.png'
    )
    
    # =========================================================================
    # NEXT STEPS
    # =========================================================================
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1. Review 'filter_recommendation.txt' for detailed justification")
    print("2. Update 'libs_preprocessing.py' with recommended parameters:")
    print(f"   - Recommended filter: {recommended}")
    print("3. Re-run preprocessing pipeline with optimal filter")
    print("4. Proceed to Phase 2: Feature Selection")
    print("="*70)