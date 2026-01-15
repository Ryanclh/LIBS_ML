"""
LIBS Spectral Data Preprocessing Pipeline
==========================================
Phase 1 Implementation for Embedded ML Soil Analysis Project

Dataset: Irish Soil Dataset (320 samples × 12,981 wavelengths)
- NO CLASSIFICATION LABELS (qualitative analysis only at this stage)
- Labels will be added later via external label mapping

Methods justified by academic literature:
- AsLS Baseline Correction:   r=0.93, 19% RMSE improvement [221]
- Savitzky-Golay Filter:  Standard LIBS practice [228]
- L2 Normalization: Bias=0.000, RMSE=0.002 [237]
"""

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict
import warnings
import os

# ============================================================================
# STEP 1: DATA LOADING (Label-Free)
# ============================================================================

def load_irish_soil_data(
    filepath: str = r"D:\Ryan\UM\Academic\Y4S1\KIE4002 FINAL YEAR PROJECT\Irish_Soil_DataSet\Irish_Soil_DataSet2.csv"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: 
    """
    Load Irish Soil LIBS spectral data from CSV file.
    
    Expected format:
    - First column:  Wavelengths (197.001 - 973.5891 nm) as row index
    - First row: Sample names (Clay_1, Clay_2, ..., Clay_320) as column headers
    - Data:  12,981 rows (wavelengths) × 320 columns (samples)
    
    NO LABELS INCLUDED - Classification labels must be added separately
    via label_map.csv when available.
    
    Parameters
    ----------
    filepath : str
        Path to the Irish Soil CSV file
        
    Returns
    -------
    spectra : np.ndarray
        Shape (320, 12981) - raw spectral intensities (samples × features)
    sample_names : np.ndarray
        Shape (320,) - sample identifiers (Clay_1, Clay_2, ...)
    wavelengths : np.ndarray
        Shape (12981,) - wavelength values in nm (197.001 - 973.5891)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}\n"
                              f"Please verify the file path is correct.")
    
    print(f"\n{'='*70}")
    print("LOADING IRISH SOIL LIBS DATASET")
    print(f"{'='*70}")
    print(f"File: {filepath}")
    
    # Read CSV with first column as index (wavelengths)
    df = pd.read_csv(filepath, index_col=0)
    
    print(f"Raw CSV shape: {df.shape} (rows=wavelengths, cols=samples)")
    print(f"First 3 sample names: {df.columns[:3].tolist()}")
    print(f"First 3 wavelengths: {df.index[:3].tolist()}")
    
    # Extract sample names from column headers
    sample_names = df.columns.astype(str).values
    n_samples = len(sample_names)
    
    # Extract wavelengths from row index
    try:
        wavelengths = df.index.astype(float).values
        print(f"✓ Wavelengths parsed:  {wavelengths[0]:.3f} - {wavelengths[-1]:.3f} nm ({len(wavelengths)} points)")
    except (ValueError, TypeError) as e:
        raise ValueError(f"Failed to parse wavelengths from index: {e}\n"
                        f"Expected numeric wavelength values in first column.")
    
    # Convert to spectra array:  Transpose (wavelengths, samples) → (samples, wavelengths)
    try:
        spectra = df.values.T.astype(float)
        print(f"✓ Spectra array shape: {spectra.shape} (samples, features)")
    except ValueError as e:
        raise ValueError(f"Failed to convert spectra to numeric values: {e}\n"
                        f"Check CSV contains only numeric spectral data.")
    
    # Summary
    print(f"\n{'='*70}")
    print("DATA LOADING SUMMARY")
    print(f"{'='*70}")
    print(f"Samples:        {n_samples} (Clay_1 ... Clay_{n_samples})")
    print(f"Features:      {len(wavelengths)} wavelengths")
    print(f"Wavelengths:   {wavelengths.min():.3f} - {wavelengths.max():.3f} nm")
    print(f"Intensity range: {spectra.min():.2f} - {spectra.max():.2f}")
    print(f"\n⚠️  NO LABELS LOADED - This is qualitative preprocessing only")
    print(f"   For classification, provide label_map.csv later:")
    print(f"   Format: sample,label")
    print(f"   Example: Clay_1,High")
    print(f"{'='*70}\n")
    
    return spectra, sample_names, wavelengths


# ============================================================================
# STEP 2: BASELINE CORRECTION - Asymmetric Least Squares (AsLS)
# Justification: r=0.93, 19% RMSE improvement [221]
# ============================================================================

def asls_baseline(
    spectrum: np.ndarray,
    lam: float = 1e6,
    p: float = 0.01,
    max_iter: int = 10
) -> np.ndarray:
    """
    Asymmetric Least Squares baseline correction.
    
    Removes Bremsstrahlung continuum background from LIBS spectra.
    
    Justification: 
    - LIBS soil carbon study achieved r=0.93, 19% RMSE improvement [221]
    - Physics-based removal of thermal radiation background
    
    Parameters
    ----------
    spectrum : np.ndarray
        Single spectrum (1D array)
    lam : float
        Smoothness parameter (λ).Higher = smoother baseline. 
        Recommended: 1e6 for LIBS (validated by baseline_comparison.py)
    p : float
        Asymmetry parameter. Lower = baseline follows valleys better.
        Recommended: 0.01 for LIBS
    max_iter : int
        Maximum iterations for convergence
        
    Returns
    -------
    baseline :  np.ndarray
        Estimated baseline to subtract from spectrum
    """
    L = len(spectrum)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
    D = lam * D.dot(D.T)
    
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    
    for _ in range(max_iter):
        W.setdiag(w)
        Z = W + D
        z = spsolve(Z, w * spectrum)
        w_new = p * (spectrum > z) + (1 - p) * (spectrum < z)
        if np.allclose(w, w_new):
            break
        w = w_new
    
    return z

def polynomial_baseline(
    spectrum: np.ndarray,
    degree: int = 5
) -> np.ndarray:
    """
    Polynomial fitting baseline correction.
    
    Justification: 
    - Lowest computational cost for MCU deployment
    - baseline_comparison.py validation:  Best peak preservation + edge stability
    
    Parameters
    ----------
    spectrum : np.ndarray
        Single spectrum (1D array)
    degree : int
        Polynomial degree (default: 5, validated optimal)
        
    Returns
    -------
    baseline : np.ndarray
        Estimated baseline to subtract from spectrum
    """
    x = np.arange(len(spectrum))
    # Iterative fitting to find lower envelope
    baseline = spectrum.copy()
    
    for _ in range(3):  # 3 iterations to converge to lower envelope
        coeffs = np.polyfit(x, baseline, degree)
        baseline_new = np.polyval(coeffs, x)
        # Keep points below fitted curve
        baseline = np.minimum(baseline, baseline_new)
    
    return baseline_new

def generate_nutrient_labels(sample_names:  np.ndarray) -> np.ndarray:
    """
    Generate nutrient level labels (High/Medium/Low) from sample names.
    
    Mapping based on Soil Type as Proxy for Nutrient Holding Capacity: 
    - Clay/ClayLoam: HIGH nutrient retention (High CEC)
    - Garden_top/Chalk: MEDIUM fertility
    - Sandy/Gravel: LOW nutrient retention (leaches nutrients)
    - Contaminated samples:  Excluded (may distort nutrient signals)
    
    Parameters
    ----------
    sample_names :  np.ndarray
        Sample identifiers (e.g., Clay_1, Sandy_15, TPH_10)
        
    Returns
    -------
    labels : np.ndarray
        Nutrient level labels:  2=High, 1=Medium, 0=Low, -1=Excluded
    """
    labels = []
    
    for name in sample_names:
        n = str(name).lower()
        
        # HIGH NUTRIENT RETENTION (Clay soils - High CEC)
        if 'clay' in n and 'loam' not in n:
            labels.append(2)  # High
        elif 'clayloam' in n:
            labels.append(2)  # High
        
        # MEDIUM NUTRIENT RETENTION
        elif 'garden' in n: 
            labels.append(1)  # Medium
        elif 'chalk_' in n and 't' not in n:  # Chalk but NOT ChalkT
            labels.append(1)  # Medium
        
        # LOW NUTRIENT RETENTION (Sandy/Gravel - low CEC)
        elif 'sandy' in n: 
            labels.append(0)  # Low
        elif 'gravel' in n:
            labels.append(0)  # Low
        
        # CONTAMINATED - EXCLUDE from nutrient training
        # Contaminants suppress nutrient signals, confusing classification
        elif 'tph' in n or 'arsenic' in n or 'heavymetals' in n or 'chalkt' in n: 
            labels.append(-1)  # Exclude
        
        else:
            labels.append(-1)  # Unknown - exclude by default
    
    labels = np.array(labels)
    
    # Summary statistics
    n_high = np.sum(labels == 2)
    n_medium = np.sum(labels == 1)
    n_low = np.sum(labels == 0)
    n_excluded = np.sum(labels == -1)
    
    print(f"\n{'='*70}")
    print("NUTRIENT LEVEL LABELS GENERATED")
    print(f"{'='*70}")
    print(f"Mapping:  Soil Type → Nutrient Retention Capacity")
    print(f"  High (2):    {n_high} samples (Clay, ClayLoam)")
    print(f"  Medium (1):  {n_medium} samples (Garden_top, Chalk)")
    print(f"  Low (0):     {n_low} samples (Sandy, Gravel)")
    print(f"  Excluded:     {n_excluded} samples (Contaminated)")
    print(f"\nTotal usable:  {n_high + n_medium + n_low} samples")
    print(f"{'='*70}")
    
    return labels

def generate_contaminant_labels(sample_names: np.ndarray) -> np.ndarray:
    """
    Generate contaminant detection labels (Clean/Contaminated) from sample names.
    
    Binary Classification: 
    - Clean (0): Uncontaminated soils (Clay, ClayLoam, Chalk, Gravel, Sandy, Garden_top)
    - Contaminated (1): TPH (diesel), Arsenic, HeavyMetals, ChalkT (treated)
    
    Parameters
    ----------
    sample_names : np.ndarray
        Sample identifiers
        
    Returns
    -------
    labels : np.ndarray
        Binary labels: 0=Clean, 1=Contaminated
    """
    labels = []
    
    for name in sample_names: 
        n = str(name).lower()
        
        # CONTAMINATED CLASS
        if 'tph' in n:
            labels.append(1)  # Total Petroleum Hydrocarbons (Diesel)
        elif 'arsenic' in n:
            labels.append(1)  # Heavy Metal (As)
        elif 'heavymetals' in n: 
            labels.append(1)  # Heavy Metals mix (Pb, Cd, Cr)
        elif 'chalkt' in n:
            labels.append(1)  # Chalk Treated (assuming T = Treated/Contaminated)
        
        # CLEAN CLASS (Control)
        elif 'clay' in n and 'loam' not in n:
            labels.append(0)
        elif 'clayloam' in n:
            labels.append(0)
        elif 'chalk_' in n:  # Non-treated chalk
            labels.append(0)
        elif 'gravel' in n:
            labels.append(0)
        elif 'sandy' in n:
            labels.append(0)
        elif 'garden' in n:
            labels.append(0)
        
        else:
            labels.append(0)  # Default to clean if unsure
    
    labels = np.array(labels)
    
    # Summary statistics
    n_clean = np.sum(labels == 0)
    n_contaminated = np.sum(labels == 1)
    
    print(f"\n{'='*70}")
    print("CONTAMINANT DETECTION LABELS GENERATED")
    print(f"{'='*70}")
    print(f"Binary Classification:")
    print(f"  Clean (0):        {n_clean} samples")
    print(f"  Contaminated (1): {n_contaminated} samples")
    print(f"\nClass balance:  {n_contaminated/(n_clean+n_contaminated)*100:.1f}% contaminated")
    print(f"{'='*70}")
    
    return labels

def stratified_split_with_labels(
    spectra: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.30,
    random_state: int = 42,
    exclude_negative:  bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform stratified train-test split with optional exclusion of -1 labels.
    
    Parameters
    ----------
    spectra : np.ndarray
        Preprocessed spectra
    labels : np.ndarray
        Class labels (may include -1 for excluded samples)
    test_size : float
        Proportion for test set (default 0.30 = 30%)
    random_state : int
        Random seed for reproducibility
    exclude_negative :  bool
        If True, remove samples with label=-1 before splitting
        
    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
        Split datasets
    """
    from sklearn.model_selection import train_test_split
    
    # Filter out excluded samples if requested
    if exclude_negative: 
        valid_mask = labels != -1
        spectra = spectra[valid_mask]
        labels = labels[valid_mask]
        print(f"Excluded {np.sum(~valid_mask)} samples with label=-1")
    
    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        spectra, labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state
    )
    
    print(f"\n{'='*70}")
    print("STRATIFIED SPLIT COMPLETE")
    print(f"{'='*70}")
    print(f"Training:   {X_train.shape[0]} samples ({100*(1-test_size):.0f}%)")
    print(f"Testing:   {X_test.shape[0]} samples ({100*test_size:.0f}%)")
    
    # Class distribution
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    
    print(f"\nTrain class distribution:")
    for cls, cnt in zip(unique_train, counts_train):
        print(f"  Class {cls}: {cnt} samples ({cnt/len(y_train)*100:.1f}%)")
    
    print(f"\nTest class distribution:")
    for cls, cnt in zip(unique_test, counts_test):
        print(f"  Class {cls}: {cnt} samples ({cnt/len(y_test)*100:.1f}%)")
    
    print(f"{'='*70}")
    
    return X_train, X_test, y_train, y_test

def apply_baseline_correction(
    spectra: np.ndarray,
    method: str = 'polynomial',  # CHANGED from 'asls'
    degree: int = 5,              # NEW parameter
    lam: float = 1e6,             # Keep for AsLS option
    p: float = 0.01,              # Keep for AsLS option
    verbose: bool = True
) -> np.ndarray:
    """
    Apply baseline correction to all spectra.
    
    DEFAULT METHOD:  Polynomial (degree=5)
    Justification:  baseline_comparison.py showed best peak recovery + edge stability
    
    Parameters
    ----------
    spectra :  np.ndarray
        Shape (n_samples, n_features)
    method : str
        'polynomial' (RECOMMENDED) or 'asls' (legacy)
    degree : int
        Polynomial degree (default: 5, validated by baseline_comparison.py)
    lam : float
        AsLS smoothness parameter (if method='asls')
    p : float
        AsLS asymmetry parameter (if method='asls')
    verbose : bool
        Print progress
        
    Returns
    -------
    corrected :  np.ndarray
        Baseline-corrected spectra
    """
    n_samples = spectra.shape[0]
    corrected = np.zeros_like(spectra)
    
    if verbose:
        method_name = f"Polynomial (degree={degree})" if method == 'polynomial' else f"AsLS (λ={lam:.0e}, p={p})"
        print(f"Applying {method_name} baseline correction...")
    
    for i in range(n_samples):
        if method == 'polynomial':
            baseline = polynomial_baseline(spectra[i], degree=degree)
        elif method == 'asls': 
            baseline = asls_baseline(spectra[i], lam=lam, p=p)
        else:
            raise ValueError(f"Unknown method: {method}.Use 'polynomial' or 'asls'")
        
        corrected[i] = spectra[i] - baseline
        
        if verbose and (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{n_samples} spectra processed")
    
    # Ensure no negative values (physical constraint)
    corrected = np.maximum(corrected, 0)
    
    if verbose:
        print(f"✓ Baseline correction complete: {n_samples} spectra")
    
    return corrected
# ============================================================================
# STEP 3: NOISE REDUCTION
# Default:  Savitzky-Golay [228]
# Alternative: Modified Sinc (from filter_comparison.py recommendation)
# ============================================================================

def apply_savgol_filter(
    spectra: np.ndarray,
    window_length: int = 7,
    polyorder: int = 2
) -> np.ndarray:
    """
    Apply Savitzky-Golay smoothing filter.
    
    Justification:  Standard LIBS practice [228]
    
    Parameters from filter_comparison.py optimization:
    - window_length:  7 (optimal for LIBS spectral resolution)
    - polyorder: 2 (preserves peak shape)
    
    Parameters
    ----------
    spectra :  np.ndarray
        Shape (n_samples, n_features)
    window_length : int
        Filter window (must be odd)
    polyorder : int
        Polynomial order for fitting
        
    Returns
    -------
    smoothed : np.ndarray
        Smoothed spectra
    """
    if window_length % 2 == 0:
        window_length += 1
        warnings.warn(f"Window length must be odd. Adjusted to {window_length}")
    
    smoothed = np.apply_along_axis(
        lambda x: savgol_filter(x, window_length, polyorder),
        axis=1,
        arr=spectra
    )
    
    print(f"✓ Savitzky-Golay filter applied: window={window_length}, order={polyorder}")
    
    return smoothed

def apply_gaussian_filter(
    spectra: np.ndarray,
    sigma: float = 1.5
) -> np.ndarray:
    """
    Apply Gaussian smoothing filter to all spectra.
    
    Parameters
    ----------
    spectra : np.ndarray
        Shape (n_samples, n_features)
    sigma : float
        Standard deviation for Gaussian kernel
        
    Returns
    -------
    smoothed : np.ndarray
        Smoothed spectra
    """
    smoothed = np.apply_along_axis(
        lambda x: gaussian_filter1d(x, sigma=sigma),
        axis=1,
        arr=spectra
    )
    
    print(f"✓ Gaussian filter applied: sigma={sigma}")
    
    return smoothed

def modified_sinc_filter(
    spectrum: np.ndarray,
    fc: float = 0.1,
    M: int = 11
) -> np.ndarray:
    """
    Modified Sinc (Windowed-Sinc) low-pass filter.
    
    Alternative to SG if filter_comparison.py showed >2% improvement.
    
    Parameters
    ----------
    spectrum : np.ndarray
        Single spectrum
    fc : float
        Normalized cutoff frequency (0 < fc < 0.5)
    M : int
        Filter length (odd)
        
    Returns
    -------
    filtered : np.ndarray
        Filtered spectrum
    """
    if M % 2 == 0:
        M += 1
    
    n = np.arange(M)
    h = np.sinc(2 * fc * (n - (M - 1) / 2))
    window = np.blackman(M)
    h = h * window
    h = h / np.sum(h)
    
    return np.convolve(spectrum, h, mode='same')

def apply_modified_sinc_filter(
    spectra: np.ndarray,
    fc: float = 0.1,
    M: int = 11
) -> np.ndarray:
    """Apply Modified Sinc filter to all spectra."""
    smoothed = np.apply_along_axis(
        lambda x: modified_sinc_filter(x, fc=fc, M=M),
        axis=1,
        arr=spectra
    )
    
    print(f"✓ Modified Sinc filter applied: fc={fc}, M={M}")
    
    return smoothed


# ============================================================================
# STEP 4: NORMALIZATION
# Default: L2-Norm [237]:  Bias=0.000, RMSE=0.002
# ============================================================================

def apply_l2_normalization(spectra: np.ndarray) -> np.ndarray:
    """
    Apply L2 (Euclidean) normalization to each spectrum.
    
    Justification: 
    - 2025 validation study:  Bias=0.000, RMSE=0.002 [237]
    - Corrects for shot-to-shot laser energy fluctuations
    - Superior to SNV (RMSE=0.220 - worst performer)
    
    From normalization_comparison.py: L2-Norm is recommended method.
    
    Parameters
    ----------
    spectra : np.ndarray
        Shape (n_samples, n_features)
        
    Returns
    -------
    normalized : np.ndarray
        L2-normalized spectra (each row has unit norm)
    """
    normalized = normalize(spectra, norm='l2', axis=1)
    
    # Verify normalization
    norms = np.linalg.norm(normalized, axis=1)
    assert np.allclose(norms, 1.0), "L2 normalization failed"
    
    print(f"✓ L2 normalization applied: all spectra scaled to unit norm")
    
    return normalized


# ============================================================================
# COMPLETE PREPROCESSING PIPELINE
# ============================================================================

def preprocess_irish_soil_spectra(
    spectra: np.ndarray,
    sample_names: np.ndarray,
    wavelengths: np.ndarray,
    # UPDATED DEFAULTS based on comparison script results
    baseline_method: str = 'polynomial',      # CHANGED from asls
    baseline_degree: int = 5,                 # NEW:  Polynomial degree
    asls_lam: float = 1e6,                    # Keep for legacy/alternative
    asls_p: float = 0.01,                     # Keep for legacy/alternative
    smoothing_method: str = 'savgol',
    sg_window: int = 7,                       # CHANGED from 11
    sg_order: int = 2,                        # CHANGED from 3
    sinc_fc: float = 0.1,                     # Keep for alternative
    sinc_M: int = 11,                         # Keep for alternative
    verbose: bool = True
) -> Dict[str, np.ndarray]: 
    """
    Complete LIBS preprocessing pipeline for Irish Soil Dataset.
    
    OPTIMIZED PARAMETERS (from comparison script validation):
    - Baseline:  Polynomial degree=5 (best peak recovery + edge stability)
    - Smoothing: SG window=7, order=2 (optimal noise reduction)
    - Normalization: L2-Norm (Bias=0.000, RMSE=0.002 [237])
    
    Pipeline:  Raw → Baseline Correction → Noise Reduction → Normalization
    
    Parameters
    ----------
    spectra : np.ndarray
        Raw spectral data (n_samples, n_features)
    sample_names : np.ndarray
        Sample identifiers
    wavelengths : np.ndarray
        Wavelength values in nm
    baseline_method : str
        'polynomial' (RECOMMENDED, validated) or 'asls' (legacy)
    baseline_degree : int
        Polynomial degree (default:  5, from baseline_comparison.py)
    asls_lam : float
        AsLS smoothness (if baseline_method='asls')
    asls_p : float
        AsLS asymmetry (if baseline_method='asls')
    smoothing_method : str
        'savgol' (RECOMMENDED, validated) or 'sinc' (alternative)
    sg_window : int
        SG window length (default: 7, from filter_comparison.py)
    sg_order : int
        SG polynomial order (default: 2, from filter_comparison.py)
    sinc_fc : float
        Modified Sinc cutoff (if smoothing_method='sinc')
    sinc_M : int
        Modified Sinc filter length (if smoothing_method='sinc')
    verbose : bool
        Print progress messages
        
    Returns
    -------
    results : dict
        Dictionary containing: 
        - 'raw': Original spectra
        - 'baseline_corrected': After baseline correction
        - 'smoothed': After noise reduction
        - 'normalized':  FINAL preprocessed (READY FOR FEATURE SELECTION)
        - 'sample_names': Sample identifiers
        - 'wavelengths': Wavelength values
    """
    results = {
        'raw': spectra.copy(),
        'sample_names': sample_names,
        'wavelengths': wavelengths
    }
    
    if verbose:
        print("\n" + "="*70)
        print("IRISH SOIL LIBS PREPROCESSING PIPELINE")
        print("="*70)
        print(f"Input:  {spectra.shape[0]} samples × {spectra.shape[1]} features")
        method_str = f"Poly(deg={baseline_degree})" if baseline_method == 'polynomial' else f"AsLS(λ={asls_lam:.0e})"
        print(f"Methods: {method_str} → SG(w={sg_window},o={sg_order}) → L2-Norm")
        print("="*70)
    
    # Step 1: Baseline Correction
    if verbose:
        print("\n[1/3] Baseline Correction")
        if baseline_method == 'polynomial': 
            print(f"      Method: Polynomial (degree={baseline_degree})")
            print(f"      Justification: baseline_comparison.py validation")
            print(f"                     Best peak recovery + edge stability")
        else:
            print(f"      Method: AsLS (λ={asls_lam:.0e}, p={asls_p})")
            print(f"      Justification: r=0.93, 19% RMSE improvement [221]")
    
    results['baseline_corrected'] = apply_baseline_correction(
        spectra, 
        method=baseline_method,
        degree=baseline_degree,
        lam=asls_lam, 
        p=asls_p, 
        verbose=verbose
    )
    
    # Step 2: Noise Reduction
    if verbose:
        print(f"\n[2/3] Noise Reduction")
    
    if smoothing_method == 'savgol':
        if verbose:
            print(f"      Method: Savitzky-Golay (window={sg_window}, order={sg_order})")
            print(f"      Justification: filter_comparison.py validation")
            print(f"                     Optimal noise reduction + peak preservation")
        results['smoothed'] = apply_savgol_filter(
            results['baseline_corrected'],
            window_length=sg_window,
            polyorder=sg_order
        )
    elif smoothing_method == 'sinc': 
        if verbose:
            print(f"      Method: Modified Sinc (fc={sinc_fc}, M={sinc_M})")
            print(f"      Justification: Better stopband rejection [233]")
        results['smoothed'] = apply_modified_sinc_filter(
            results['baseline_corrected'],
            fc=sinc_fc,
            M=sinc_M
        )
    else:
        raise ValueError(f"Unknown smoothing method: {smoothing_method}.Use 'savgol' or 'sinc'")
    
    # Step 3: Normalization [237]
    if verbose:
        print("\n[3/3] Normalization (L2-Norm)")
        print(f"      Justification: Bias=0.000, RMSE=0.002 [237]")
        print(f"                     normalization_comparison.py validation")
    
    results['normalized'] = apply_l2_normalization(results['smoothed'])
    
    if verbose:
        print("\n" + "="*70)
        print("✓ PREPROCESSING COMPLETE")
        print(f"  Input shape:   {spectra.shape}")
        print(f"  Output shape: {results['normalized'].shape}")
        print(f"  Baseline:      Polynomial degree={baseline_degree}")
        print(f"  Smoothing:    SG window={sg_window}, order={sg_order}")
        print(f"  Normalization: L2-Norm (unit vector)")
        print("="*70)
    
    return results

def preprocess_libs_spectra(
    spectra: np.ndarray,
    poly_degree: int = 5,
    sg_window: int = 11,
    sg_order: int = 3,
    use_gaussian: bool = False,
    gaussian_sigma: float = 1.5,
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """
    Complete LIBS preprocessing pipeline.
    
    Order: Baseline Correction → Noise Reduction → Normalization
    
    Parameters
    ----------
    spectra : np.ndarray
        Raw spectral data (n_samples, n_features)
    poly_degree : int
        Polynomial degree for baseline correction
    sg_window : int
        Savitzky-Golay window length
    sg_order : int
        Savitzky-Golay polynomial order
    use_gaussian : bool
        If True, use Gaussian filter instead of Savitzky-Golay
    gaussian_sigma : float
        Gaussian filter sigma (if use_gaussian=True)
    verbose : bool
        Print progress messages
        
    Returns
    -------
    results : dict
        Dictionary containing intermediate and final results:
        - 'raw': Original spectra
        - 'baseline_corrected': After Polynomial correction
        - 'smoothed': After noise reduction
        - 'normalized': Final preprocessed spectra
    """
    results = {'raw': spectra.copy()}
    
    if verbose:
        print("\n" + "="*60)
        print("LIBS PREPROCESSING PIPELINE")
        print("="*60)
    
    # Step 1: Baseline Correction
    if verbose:
        print(f"\n[1/3] Applying Polynomial Baseline Correction (degree={poly_degree})...")
    results['baseline_corrected'] = apply_baseline_correction(
        spectra, degree=poly_degree, verbose=verbose
    )
    
    # Step 2: Noise Reduction
    if verbose:
        print("\n[2/3] Applying Noise Reduction Filter...")
    if use_gaussian:
        results['smoothed'] = apply_gaussian_filter(
            results['baseline_corrected'], sigma=gaussian_sigma
        )
    else:
        results['smoothed'] = apply_savgol_filter(
            results['baseline_corrected'], 
            window_length=sg_window, 
            polyorder=sg_order
        )
    
    # Step 3: Normalization
    if verbose:
        print("\n[3/3] Applying L2 Normalization...")
    results['normalized'] = apply_l2_normalization(results['smoothed'])
    
    if verbose:
        print("\n" + "="*60)
        print("✓ PREPROCESSING COMPLETE")
        print(f"  Input shape:  {spectra.shape}")
        print(f"  Output shape: {results['normalized'].shape}")
        print("="*60)
    
    return results

# ============================================================================
# VALIDATION FUNCTIONS (Label-Free)
# ============================================================================

def validate_preprocessing_quality(
    raw:  np.ndarray,
    processed: np.ndarray,
    wavelengths: np.ndarray,
    sample_idx: int = 0
) -> Dict[str, float]:
    """
    Validate preprocessing quality WITHOUT requiring labels.
    
    Metrics:
    - SNR improvement
    - Baseline flatness
    - Peak preservation (correlation)
    - Normalization verification
    
    Parameters
    ----------
    raw : np.ndarray
        Raw spectrum
    processed : np.ndarray
        Preprocessed spectrum
    wavelengths : np.ndarray
        Wavelength values
    sample_idx : int
        Sample index for reporting
        
    Returns
    -------
    metrics : dict
        Validation metrics
    """
    metrics = {}
    
    # 1.Signal-to-Noise Ratio improvement
    flat_region = slice(100, 200)
    noise_raw = np.std(raw[flat_region])
    noise_processed = np.std(processed[flat_region])
    
    signal_raw = np.max(raw) - np.median(raw)
    signal_processed = np.max(processed) - np.median(processed)
    
    snr_raw = signal_raw / noise_raw if noise_raw > 0 else 0
    snr_processed = signal_processed / noise_processed if noise_processed > 0 else 0
    
    metrics['snr_raw'] = snr_raw
    metrics['snr_processed'] = snr_processed
    metrics['snr_improvement_pct'] = (snr_processed - snr_raw) / snr_raw * 100 if snr_raw > 0 else 0
    
    # 2.Baseline flatness
    baseline_region = slice(0, 100)
    metrics['baseline_std_raw'] = np.std(raw[baseline_region])
    metrics['baseline_std_processed'] = np.std(processed[baseline_region])
    
    # 3.Peak preservation (correlation)
    metrics['peak_correlation'] = np.corrcoef(raw, processed)[0, 1]
    
    # 4.Normalization verification
    metrics['l2_norm'] = np.linalg.norm(processed)
    metrics['is_normalized'] = np.isclose(metrics['l2_norm'], 1.0)
    
    # 5.Dynamic range
    metrics['dynamic_range_raw'] = np.max(raw) - np.min(raw)
    metrics['dynamic_range_processed'] = np.max(processed) - np.min(processed)
    
    print(f"\n{'='*60}")
    print(f"VALIDATION METRICS (Sample {sample_idx})")
    print(f"{'='*60}")
    print(f"SNR Improvement:       {metrics['snr_improvement_pct']:+.1f}%")
    print(f"Baseline Std (raw):   {metrics['baseline_std_raw']:.4f}")
    print(f"Baseline Std (proc):  {metrics['baseline_std_processed']:.6f}")
    print(f"Peak Correlation:     {metrics['peak_correlation']:.4f}")
    print(f"L2 Norm:              {metrics['l2_norm']:.6f} (normalized: {metrics['is_normalized']})")
    print(f"{'='*60}")
    
    return metrics


def plot_preprocessing_stages(
    results: Dict[str, np.ndarray],
    sample_idx: int = 0,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize all preprocessing stages.
    
    Parameters
    ----------
    results : dict
        Output from preprocess_irish_soil_spectra() containing 'wavelengths'
    sample_idx : int
        Sample index to plot (default: 0 = first sample)
    save_path : str, optional
        Path to save figure
    """
    # Extract wavelengths and sample name from results
    wavelengths = results['wavelengths']
    sample_name = results['sample_names'][sample_idx]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    stages = ['raw', 'baseline_corrected', 'smoothed', 'normalized']
    titles = [
        'Step 0: Raw Spectrum',
        'Step 1: After Polynomial(deg=5) Baseline Correction',
        'Step 2: After Savitzky-Golay(w=7,o=2) Smoothing',
        'Step 3: After L2 Normalization (FINAL)'
    ]
    
    for ax, stage, title in zip(axes.flat, stages, titles):
        spectrum = results[stage][sample_idx]
        ax.plot(wavelengths, spectrum, 'b-', linewidth=0.5)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Intensity (a.u.)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Statistics annotation
        stats_text = f"Max: {np.max(spectrum):.4f}\nMean: {np.mean(spectrum):.4f}\nStd: {np.std(spectrum):.4f}"
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'Irish Soil LIBS Preprocessing - {sample_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Figure saved to:   {save_path}")
    
    plt.show()

def plot_sample_comparison(
    spectra: np.ndarray,
    sample_names: np.ndarray,
    wavelengths:  np.ndarray,
    n_samples: int = 10,
    save_path: Optional[str] = None
) -> None:
    """
    Plot multiple preprocessed spectra for visual inspection.
    
    Parameters
    ----------
    spectra : np.ndarray
        Preprocessed spectra
    sample_names : np.ndarray
        Sample identifiers
    wavelengths : np.ndarray
        Wavelength values
    n_samples : int
        Number of samples to plot
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot first n samples
    for i in range(min(n_samples, len(spectra))):
        ax.plot(wavelengths, spectra[i], alpha=0.5, linewidth=0.5, label=sample_names[i])
    
    # Plot mean spectrum
    mean_spectrum = np.mean(spectra, axis=0)
    ax.plot(wavelengths, mean_spectrum, 'k-', linewidth=2, label='Mean Spectrum')
    
    ax.set_xlabel('Wavelength (nm)', fontsize=12)
    ax.set_ylabel('Normalized Intensity', fontsize=12)
    ax.set_title('Preprocessed Irish Soil LIBS Spectra', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Figure saved to:  {save_path}")
    
    plt.show()


# ============================================================================
# SAVE/LOAD FUNCTIONS
# ============================================================================

def save_preprocessed_data(
    results: Dict[str, np.ndarray],
    nutrient_labels: Optional[np.ndarray] = None,
    contaminant_labels: Optional[np.ndarray] = None,
    output_path: str = "irish_soil_preprocessed.npz"
) -> None:
    """
    Save preprocessed data AND labels to compressed NumPy file.
    
    Parameters
    ----------
    results : dict
        Output from preprocess_irish_soil_spectra()
    nutrient_labels : np.ndarray, optional
        Nutrient level labels (High/Med/Low)
    contaminant_labels : np.ndarray, optional
        Contaminant detection labels (Clean/Contaminated)
    output_path : str
        Output file path
    """
    save_dict = {
        'raw':  results['raw'],
        'baseline_corrected': results['baseline_corrected'],
        'smoothed': results['smoothed'],
        'normalized': results['normalized'],
        'sample_names': results['sample_names'],
        'wavelengths': results['wavelengths']
    }
    
    # Add labels if provided
    if nutrient_labels is not None:
        save_dict['nutrient_labels'] = nutrient_labels
    if contaminant_labels is not None: 
        save_dict['contaminant_labels'] = contaminant_labels
    
    np.savez_compressed(output_path, **save_dict)
    
    print(f"\n{'='*70}")
    print("PREPROCESSED DATA SAVED")
    print(f"{'='*70}")
    print(f"File: {output_path}")
    print(f"Size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    print(f"\nContents:")
    print(f"  - raw:  Raw spectra (320, 12981)")
    print(f"  - baseline_corrected: After AsLS")
    print(f"  - smoothed: After noise reduction")
    print(f"  - normalized:  FINAL preprocessed (320, 12981)")
    print(f"  - sample_names: Sample identifiers")
    print(f"  - wavelengths: 197.001-973.5891 nm")
    
    if nutrient_labels is not None: 
        print(f"  - nutrient_labels: High/Med/Low (Objective 1)")
    if contaminant_labels is not None: 
        print(f"  - contaminant_labels: Clean/Contaminated (Objective 2)")
    
    print(f"\n{'='*70}")
    print("READY FOR PHASE 2: Feature Selection")
    print(f"{'='*70}")
    print("\nObjective 1 (NPK Classification):")
    print("  - Target: ≥85% accuracy")
    print("  - Method:  ANOVA F-test → RF Gini Importance → RF Classifier")
    print("  - Use: nutrient_labels (exclude -1 samples)")
    print("\nObjective 2 (Contaminant Detection):")
    print("  - Target: ≥90% accuracy")
    print("  - Method: Same feature selection pipeline")
    print("  - Use: contaminant_labels (all samples)")
    print(f"{'='*70}")

def load_preprocessed_data(filepath: str = "irish_soil_preprocessed.npz") -> Dict[str, np.ndarray]:
    """
    Load preprocessed data from compressed NumPy file.
    
    Parameters
    ----------
    filepath :  str
        Path to preprocessed data file
        
    Returns
    -------
    results :  dict
        Dictionary containing all preprocessing stages
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Preprocessed data not found:  {filepath}")
    
    data = np.load(filepath, allow_pickle=True)
    
    results = {
        'raw': data['raw'],
        'baseline_corrected': data['baseline_corrected'],
        'smoothed':  data['smoothed'],
        'normalized': data['normalized'],
        'sample_names': data['sample_names'],
        'wavelengths': data['wavelengths']
    }
    
    print(f"✓ Loaded preprocessed data from: {filepath}")
    print(f"  Shape: {results['normalized'].shape}")
    
    return results

# ============================================================================
# MAIN EXECUTION - PHASE 1: DATA PREPROCESSING & PREPARATION
# ============================================================================

if __name__ == "__main__": 
    print("="*70)
    print("PHASE 1: DATA PREPROCESSING & PREPARATION")
    print("Irish Soil LIBS Dataset (320 samples × 12,981 features)")
    print("="*70)
    print("\nPipeline:")
    print("  1.Data Loading")
    print("  2.Baseline Correction (Polynomial degree=5)")
    print("  3.Noise Reduction (Savitzky-Golay window=7, order=2)")
    print("  4.L2 Normalization")
    print("  5.Label Generation (Nutrient & Contaminant)")
    print("  6.Stratified Train-Test Split (70/30)")
    print("="*70)
    
    # =========================================================================
    # STEP 1: LOAD IRISH SOIL DATASET (320 samples × 12,981 features)
    # =========================================================================
    try:
        spectra_raw, sample_names, wavelengths = load_irish_soil_data()
        
        # Verify dataset dimensions
        assert spectra_raw.shape == (320, 12981), f"Expected (320, 12981), got {spectra_raw.shape}"
        print(f"\n✓ Dataset verified: {spectra_raw.shape[0]} samples × {spectra_raw.shape[1]} features")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\nPlease verify:")
        print("  1.File path is correct in load_irish_soil_data() function")
        print("  2.CSV file exists:  Irish_Soil_DataSet2.csv")
        print("  3.File format:  wavelengths in rows, samples in columns")
        exit(1)
    except AssertionError as e:
        print(f"\n❌ Dataset dimension mismatch: {e}")
        print("   Expected: 320 samples (Clay_1...Clay_320) × 12,981 wavelengths")
        exit(1)
    
    # =========================================================================
    # STEP 2: RUN PREPROCESSING PIPELINE
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 2: PREPROCESSING PIPELINE")
    print("="*70)
    print("Configuration (validated by comparison scripts):")
    print("  • Baseline:     Polynomial degree=5")
    print("  • Smoothing:   Savitzky-Golay window=7, order=2")
    print("  • Normalization: L2-Norm (unit vector)")
    print("="*70)
    
    results = preprocess_irish_soil_spectra(
        spectra_raw,
        sample_names,
        wavelengths,
        baseline_method='polynomial',
        baseline_degree=5,
        smoothing_method='savgol',
        sg_window=7,
        sg_order=2,
        verbose=True
    )
    
    # =========================================================================
    # STEP 3: GENERATE LABELS (From Sample Names)
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 3: LABEL GENERATION")
    print("="*70)
    
    # Generate labels for BOTH objectives
    nutrient_labels = generate_nutrient_labels(sample_names)
    contaminant_labels = generate_contaminant_labels(sample_names)
    
    # =========================================================================
    # STEP 4: STRATIFIED TRAIN-TEST SPLIT (70% Train / 30% Test)
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 4: STRATIFIED TRAIN-TEST SPLIT")
    print("="*70)
    print("Split ratio: 70% Training / 30% Testing")
    print("Method: Stratified sampling (preserves class distribution)")
    
    # --- OBJECTIVE 1: NUTRIENT CLASSIFICATION (High/Med/Low) ---
    print("\n" + "-"*70)
    print("OBJECTIVE 1: NUTRIENT CLASSIFICATION (High/Med/Low)")
    print("-"*70)
    
    X_train_nut, X_test_nut, y_train_nut, y_test_nut = stratified_split_with_labels(
        spectra=results['normalized'],
        labels=nutrient_labels,
        test_size=0.30,
        random_state=42,
        exclude_negative=True  # Exclude contaminated samples (-1)
    )
    
    print(f"\n✓ Nutrient dataset ready:")
    print(f"  Training:    {X_train_nut.shape[0]} samples ({X_train_nut.shape[1]} features)")
    print(f"  Testing:    {X_test_nut.shape[0]} samples ({X_test_nut.shape[1]} features)")
    
    # --- OBJECTIVE 2: CONTAMINANT DETECTION (Clean/Contaminated) ---
    print("\n" + "-"*70)
    print("OBJECTIVE 2: CONTAMINANT DETECTION (Clean/Contaminated)")
    print("-"*70)
    
    X_train_con, X_test_con, y_train_con, y_test_con = stratified_split_with_labels(
        spectra=results['normalized'],
        labels=contaminant_labels,
        test_size=0.30,
        random_state=42,
        exclude_negative=False  # Use all samples
    )
    
    print(f"\n✓ Contaminant dataset ready:")
    print(f"  Training:    {X_train_con.shape[0]} samples ({X_train_con.shape[1]} features)")
    print(f"  Testing:    {X_test_con.shape[0]} samples ({X_test_con.shape[1]} features)")
    
    # =========================================================================
    # STEP 5: QUALITY VALIDATION
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 5: PREPROCESSING QUALITY VALIDATION")
    print("="*70)
    
    metrics = validate_preprocessing_quality(
        raw=results['raw'][0],
        processed=results['normalized'][0],
        wavelengths=wavelengths,
        sample_idx=0
    )
    
    # =========================================================================
    # STEP 6: VISUALIZATION
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 6: VISUALIZATION")
    print("="*70)
    
    # Plot preprocessing stages for first sample
    plot_preprocessing_stages(
        results,
        sample_idx=0,
        save_path='preprocessing_stages_Clay1.png'
    )
    
    # Plot comparison of first 10 samples
    plot_sample_comparison(
        results['normalized'],
        sample_names,
        wavelengths,
        n_samples=10,
        save_path='sample_comparison_10samples.png'
    )
    
    # =========================================================================
    # STEP 7: SAVE PREPROCESSED DATA + LABELS + SPLITS
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 7: SAVING PREPROCESSED DATA & SPLITS")
    print("="*70)
    
    # Save complete preprocessed dataset with labels
    save_dict = {
        # Preprocessing stages
        'raw': results['raw'],
        'baseline_corrected': results['baseline_corrected'],
        'smoothed': results['smoothed'],
        'normalized': results['normalized'],
        
        # Metadata
        'sample_names': sample_names,
        'wavelengths': wavelengths,
        
        # Labels
        'nutrient_labels':  nutrient_labels,
        'contaminant_labels': contaminant_labels,
        
        # OBJECTIVE 1: Nutrient Classification Splits
        'X_train_nutrient': X_train_nut,
        'X_test_nutrient': X_test_nut,
        'y_train_nutrient': y_train_nut,
        'y_test_nutrient': y_test_nut,
        
        # OBJECTIVE 2: Contaminant Detection Splits
        'X_train_contaminant':  X_train_con,
        'X_test_contaminant': X_test_con,
        'y_train_contaminant': y_train_con,
        'y_test_contaminant': y_test_con,
    }
    
    output_file = "irish_soil_preprocessed_phase1.npz"
    np.savez_compressed(output_file, **save_dict)
    
    file_size_mb = os.path.getsize(output_file) / 1024 / 1024
    
    print(f"\n✓ All data saved to:  {output_file}")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"\n  Contents:")
    print(f"    • Preprocessing stages:   raw, baseline_corrected, smoothed, normalized")
    print(f"    • Metadata:             sample_names, wavelengths")
    print(f"    • Labels:               nutrient_labels, contaminant_labels")
    print(f"    • Nutrient splits:      X_train/test_nutrient, y_train/test_nutrient")
    print(f"    • Contaminant splits:   X_train/test_contaminant, y_train/test_contaminant")
    
    # =========================================================================
    # PHASE 1 COMPLETION SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("✓ PHASE 1 COMPLETE:  DATA PREPROCESSING & PREPARATION")
    print("="*70)
    
    print("\nDataset Summary:")
    print(f"  Total samples:         320")
    print(f"  Total features:       12,981 wavelengths (197.001-973.5891 nm)")
    print(f"  Preprocessing:         Poly(5) → SG(7,2) → L2-Norm")
    
    print("\nObjective 1: Nutrient Classification (High/Med/Low)")
    print(f"  Usable samples:       {len(y_train_nut) + len(y_test_nut)}")
    print(f"  Training set:         {len(y_train_nut)} samples (70%)")
    print(f"  Testing set:          {len(y_test_nut)} samples (30%)")
    print(f"  Target accuracy:      ≥85%")
    
    print("\nObjective 2: Contaminant Detection (Clean/Contaminated)")
    print(f"  Total samples:        {len(y_train_con) + len(y_test_con)}")
    print(f"  Training set:          {len(y_train_con)} samples (70%)")
    print(f"  Testing set:          {len(y_test_con)} samples (30%)")
    print(f"  Target accuracy:      ≥90%")
    
    print("\n" + "="*70)
    print("NEXT PHASE: FEATURE SELECTION")
    print("="*70)
    print("\nPhase 2 Objectives:")
    print("  1. NIST Peak Identification:")
    print("     - Nitrogen (N):   742.4, 744.2, 746.8 nm")
    print("     - Phosphorus (P): 253.5, 255.3 nm")
    print("     - Potassium (K):  766.5, 769.9 nm")
    print("     - Heavy Metals:   Pb, Cd, Cr characteristic lines")
    
    print("\n  2.Statistical Feature Selection:")
    print("     - ANOVA F-test:    12,981 → ~500 features")
    print("     - Random Forest:   500 → ~100 features (Gini importance)")
    
    print("\n  3.Load preprocessed data:")
    print(f"     data = np.load('{output_file}')")
    print(f"     X_train = data['X_train_nutrient']")
    print(f"     y_train = data['y_train_nutrient']")
    
    print("\n  4.Expected outcome:")
    print("     - Feature reduction: 99.2% (12,981 → ~100)")
    print("     - Accuracy improvement: +12-20 pp over baseline (73.01%)")
    print("="*70)
    
    print("\n✓ Preprocessing pipeline validated and ready for Phase 2")
    print(f"✓ Generated files:")
    print(f"    • {output_file}")
    print(f"    • preprocessing_stages_Clay1.png")
    print(f"    • sample_comparison_10samples.png")
    print("="*70)
