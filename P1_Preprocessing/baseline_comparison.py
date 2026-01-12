"""
LIBS Baseline Correction Method Comparison Script
==================================================
Run this to compare baseline correction methods before finalizing preprocessing

Methodology Justification:
- AsLS: r=0.93, RMSE improved 19% [221]
- airPLS:  "Had little effect" vs AsLS (Mars LIBS study)
- SNIP: Common in spectroscopy, simple parameter tuning
- Polynomial:  Lowest computational cost for MCU

Decision Rule:
"Select method with best baseline flatness AND peak preservation"
"""

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter
from scipy.ndimage import grey_opening, median_filter
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
import warnings
import time
from dataclasses import dataclass
import os
import sys
sys.path.append(os.path.dirname(__file__))

# Try to import label generation functions
try:
    from generate_labels import get_nutrient_label, get_contaminant_label
    ENABLE_CLASSIFICATION = True
except ImportError: 
    print("⚠️ Warning: Could not import label generation functions from generate_labels.py")
    print("   Classification evaluation will be disabled.")
    ENABLE_CLASSIFICATION = False
warnings.filterwarnings('ignore')


# ============================================================================
# BASELINE CORRECTION METHODS
# ============================================================================

def asls_baseline(spectrum: np.ndarray, lam: float = 1e6, p: float = 0.01, 
                  max_iter: int = 10) -> np.ndarray:
    """
    Asymmetric Least Squares baseline correction. 
    
    Justification: r=0.93, 19% RMSE improvement [221]
    
    Parameters
    ----------
    spectrum : np. ndarray
        Input spectrum
    lam : float
        Smoothness parameter (higher = smoother baseline)
    p : float
        Asymmetry parameter (lower = follows valleys better)
    max_iter : int
        Maximum iterations
    
    Returns
    -------
    baseline : np.ndarray
        Estimated baseline
    """
    L = len(spectrum)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
    D = lam * D. dot(D. T)
    
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


def airpls_baseline(spectrum: np.ndarray, lam: float = 1e6, 
                    max_iter: int = 15) -> np.ndarray:
    """
    Adaptive Iteratively Reweighted Penalized Least Squares (airPLS).
    
    Justification:  Adaptive version of AsLS, mentioned in Mars LIBS study
    
    Parameters
    ----------
    spectrum : np.ndarray
        Input spectrum
    lam :  float
        Smoothness parameter
    max_iter : int
        Maximum iterations
        
    Returns
    -------
    baseline : np.ndarray
        Estimated baseline
    """
    L = len(spectrum)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
    D = lam * D.dot(D.T)
    
    w = np.ones(L)
    for _ in range(max_iter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + D
        z = spsolve(Z, w * spectrum)
        
        # Adaptive weights
        d = spectrum - z
        dssn = np.abs(d[d < 0].sum())
        if dssn < 0.001 * (abs(spectrum).sum()):
            break
        
        # Update weights adaptively
        w[d >= 0] = 0  # Baseline should be below spectrum
        w[d < 0] = np.exp(_ * np.abs(d[d < 0]) / dssn)
        w[0] = np.exp(_ * (d[d < 0]).max() / dssn)
        w[-1] = w[0]
    
    return z


def snip_baseline(spectrum: np.ndarray, iterations: int = 50) -> np.ndarray:
    """
    Statistics-sensitive Non-linear Iterative Peak-clipping (SNIP).
    
    Justification: Common in X-ray and LIBS spectroscopy, simple parameters
    
    Parameters
    ----------
    spectrum : np.ndarray
        Input spectrum
    iterations :  int
        Number of clipping iterations (controls baseline smoothness)
        
    Returns
    -------
    baseline : np. ndarray
        Estimated baseline
    """
    # Work in log space for numerical stability
    spectrum_log = np.log(np.log(np.sqrt(spectrum + 1) + 1) + 1)
    
    for m in range(1, iterations + 1):
        for i in range(m, len(spectrum_log) - m):
            # Rolling ball clipping
            a = spectrum_log[i]
            b = (spectrum_log[i - m] + spectrum_log[i + m]) / 2
            if b < a:
                spectrum_log[i] = b
    
    # Transform back
    baseline = (np.exp(np.exp(spectrum_log) - 1) - 1) ** 2 - 1
    return np.maximum(baseline, 0)


def polynomial_baseline(spectrum: np.ndarray, degree: int = 5) -> np.ndarray:
    """
    Polynomial fitting baseline correction.
    
    Justification:  Lowest computational cost, suitable for MCU deployment
    
    Parameters
    ----------
    spectrum : np.ndarray
        Input spectrum
    degree :  int
        Polynomial degree (3-7 typical)
        
    Returns
    -------
    baseline : np. ndarray
        Estimated baseline
    """
    x = np.arange(len(spectrum))
    # Fit polynomial to lower envelope (iterative approach)
    baseline = spectrum.copy()
    
    for _ in range(3):  # 3 iterations to find lower envelope
        coeffs = np.polyfit(x, baseline, degree)
        baseline_new = np.polyval(coeffs, x)
        # Keep points below fitted curve
        baseline = np.minimum(baseline, baseline_new)
    
    return baseline_new

def calculate_edge_stability(corrected:  np.ndarray, 
                            edge_size: int = 50) -> Tuple[float, float, float]: 
    """
    Calculate edge effect stability for baseline-corrected spectrum.
    
    Polynomials are notorious for "wagging" (oscillating) at endpoints. 
    This test quantifies edge instability vs center stability.
    
    Parameters
    ----------
    corrected : np.ndarray
        Baseline-corrected spectrum
    edge_size : int
        Number of points at each edge to test (default 50)
        
    Returns
    -------
    left_std : float
        Standard deviation of first edge_size points
    right_std :  float
        Standard deviation of last edge_size points
    ratio : float
        Max(left_std, right_std) / center_std
        Values > 2. 0 indicate significant edge instability
    """
    if len(corrected) < 3 * edge_size:
        # Too short for meaningful edge analysis
        return 0.0, 0.0, 1.0
    
    # Extract regions
    left_edge = corrected[:edge_size]
    right_edge = corrected[-edge_size:]
    center = corrected[edge_size:-edge_size]
    
    # Calculate standard deviations
    left_std = np.std(left_edge)
    right_std = np.std(right_edge)
    center_std = np.std(center)
    
    # Edge instability ratio
    # Higher ratio = edges are more unstable than center = BAD
    max_edge_std = max(left_std, right_std)
    ratio = max_edge_std / (center_std + 1e-10)  # Avoid division by zero
    
    return left_std, right_std, ratio


def morphological_baseline(spectrum: np.ndarray, window: int = 50) -> np.ndarray:
    """
    Morphological opening for baseline estimation.
    
    Justification: Simple, fast, parameter-free except window size
    
    Parameters
    ----------
    spectrum : np.ndarray
        Input spectrum
    window : int
        Structuring element size
        
    Returns
    -------
    baseline :  np.ndarray
        Estimated baseline
    """
    # Morphological opening = erosion followed by dilation
    baseline = grey_opening(spectrum, size=window)
    return baseline


def rolling_ball_baseline(spectrum: np.ndarray, radius: int = 50) -> np.ndarray:
    """
    Rolling ball baseline estimation.
    
    Justification: Used in ImageJ, conceptually simple
    
    Parameters
    ----------
    spectrum : np. ndarray
        Input spectrum
    radius : int
        Ball radius
        
    Returns
    -------
    baseline : np. ndarray
        Estimated baseline
    """
    # Approximate rolling ball with minimum filter
    from scipy.ndimage import minimum_filter
    baseline = minimum_filter(spectrum, size=2*radius+1, mode='nearest')
    
    # Smooth the baseline
    baseline = savgol_filter(baseline, min(51, len(baseline)//10*2+1), 3)
    
    return baseline


def median_baseline(spectrum: np.ndarray, window: int = 101) -> np.ndarray:
    """
    Median filter baseline (simple alternative).
    
    Justification: Robust to outliers, very simple
    
    Parameters
    ----------
    spectrum : np.ndarray
        Input spectrum
    window : int
        Median filter window size
        
    Returns
    -------
    baseline : np. ndarray
        Estimated baseline
    """
    if window % 2 == 0:
        window += 1
    baseline = median_filter(spectrum, size=window, mode='nearest')
    return baseline

# Around line 190, add synthetic spectrum generation and recovery test functions:

def generate_synthetic_ground_truth(
    base_spectrum: np.ndarray,
    wavelengths: np.ndarray,
    peak_position_nm: float = 500.0,
    peak_height:  float = 1000.0,
    peak_width_nm:  float = 0.5
) -> Tuple[np.ndarray, float, int]:
    """
    Generate synthetic spectrum with known Gaussian peak on real baseline.
    
    Strategy:
    1. Take a "clean" region of real spectrum (no peaks, just baseline)
    2. Add synthetic Gaussian peak with known height
    3. Use this as ground truth to test baseline correction accuracy
    
    Parameters
    ----------
    base_spectrum :  np.ndarray
        Real spectrum to use as baseline template
    wavelengths : np. ndarray
        Wavelength values in nm
    peak_position_nm : float
        Where to insert synthetic peak (nm)
    peak_height : float
        True height of synthetic peak (ground truth)
    peak_width_nm : float
        FWHM of Gaussian peak in nm
        
    Returns
    -------
    synthetic_spectrum : np.ndarray
        Spectrum with synthetic peak added
    true_peak_height : float
        Ground truth peak height (for recovery calculation)
    peak_index : int
        Index of peak maximum (for measurement)
    """
    # Find clean baseline region (first 20% of spectrum, typically flat)
    clean_region = slice(0, len(base_spectrum) // 5)
    baseline_level = np.median(base_spectrum[clean_region])
    
    # Create synthetic spectrum starting from baseline level
    synthetic = np.ones_like(base_spectrum) * baseline_level
    
    # Add slight noise to make it realistic
    synthetic += np.random. normal(0, baseline_level * 0.02, len(synthetic))
    
    # Find index closest to desired peak position
    peak_index = np.argmin(np.abs(wavelengths - peak_position_nm))
    
    # Generate Gaussian peak
    sigma = peak_width_nm / 2.355  # Convert FWHM to sigma
    for i in range(len(wavelengths)):
        wl = wavelengths[i]
        gaussian = peak_height * np.exp(-0.5 * ((wl - peak_position_nm) / sigma) ** 2)
        synthetic[i] += gaussian
    
    return synthetic, peak_height, peak_index


def calculate_peak_recovery_rate(
    corrected_spectrum: np.ndarray,
    true_peak_height: float,
    peak_index: int,
    search_window:  int = 5
) -> float:
    """
    Calculate how well baseline correction preserved synthetic peak height.
    
    Recovery Rate = (Measured Height / True Height) × 100%
    
    - 100% = Perfect preservation
    - >100% = Over-estimation (baseline under-corrected)
    - <100% = Under-estimation (baseline over-corrected, peak distorted)
    
    Parameters
    ----------
    corrected_spectrum : np.ndarray
        Baseline-corrected spectrum
    true_peak_height : float
        Known ground truth peak height
    peak_index : int
        Expected location of peak
    search_window :  int
        Search ±window around peak_index for maximum
        
    Returns
    -------
    recovery_rate : float
        Percentage of true peak height recovered
    """
    # Search for maximum in vicinity (account for slight shifts)
    start = max(0, peak_index - search_window)
    end = min(len(corrected_spectrum), peak_index + search_window + 1)
    
    measured_height = np.max(corrected_spectrum[start: end])
    
    # Calculate recovery rate
    recovery_rate = (measured_height / true_peak_height) * 100.0
    
    return recovery_rate


def test_synthetic_ground_truth(
    baseline_func,
    base_spectrum: np.ndarray,
    wavelengths: np. ndarray,
    n_tests: int = 5
) -> Tuple[float, float]: 
    """
    Run multiple synthetic peak tests and return average recovery rate.
    
    Tests peaks at different positions to avoid bias from specific regions.
    
    Parameters
    ----------
    baseline_func : callable
        Baseline correction function to test
    base_spectrum : np. ndarray
        Real spectrum template
    wavelengths : np. ndarray
        Wavelength values
    n_tests : int
        Number of synthetic peaks to test at different positions
        
    Returns
    -------
    mean_recovery :  float
        Average recovery rate across tests
    std_recovery : float
        Standard deviation (consistency measure)
    """
    # Test peaks at different wavelength positions
    wl_min, wl_max = wavelengths. min(), wavelengths.max()
    test_positions = np.linspace(wl_min + 100, wl_max - 100, n_tests)
    
    recovery_rates = []
    
    for pos in test_positions:
        # Generate synthetic spectrum with peak at this position
        synthetic, true_height, peak_idx = generate_synthetic_ground_truth(
            base_spectrum, wavelengths, peak_position_nm=pos,
            peak_height=1000.0, peak_width_nm=0.5
        )
        
        # Apply baseline correction
        try:
            baseline = baseline_func(synthetic)
            corrected = np.maximum(synthetic - baseline, 0)
            
            # Measure recovery
            recovery = calculate_peak_recovery_rate(corrected, true_height, peak_idx)
            recovery_rates. append(recovery)
        except Exception: 
            # If method fails, record 0% recovery
            recovery_rates.append(0.0)
    
    return np.mean(recovery_rates), np.std(recovery_rates)

# ============================================================================
# EVALUATION METRICS
# ============================================================================

@dataclass
class BaselineMetrics:
    """Container for baseline correction evaluation metrics."""
    name: str
    baseline_flatness: float          # Lower is better (std of baseline region)
    peak_preservation: float          # Higher is better (correlation)
    residual_std: float               # Lower is better
    peak_distortion: float            # Lower is better (% height change)
    computation_time_ms: float        # Lower is better
    parameter_complexity: int         # Lower is better (1-3 scale)
    edge_stability_left: float        # std of first 50 points (lower=better)
    edge_stability_right: float       # std of last 50 points (lower=better)
    edge_stability_ratio: float       # edge_std / center_std (lower=better)
    peak_recovery_rate: float         # % recovery of synthetic peak (higher=better, 100%=ideal)
    classification_accuracy: float    # NEW: Classification accuracy with RF classifier (higher=better)


def calculate_baseline_flatness(corrected:  np.ndarray, 
                                baseline_region: slice = slice(0, 500)) -> float:
    """
    Calculate flatness of corrected spectrum baseline region.
    
    Lower std = flatter baseline = better correction
    """
    return np.std(corrected[baseline_region])


def calculate_peak_preservation(original: np.ndarray, 
                               corrected: np.ndarray) -> float:
    """
    Calculate correlation between original and corrected peaks.
    
    Higher correlation = better peak preservation
    """
    # Normalize both to [0, 1] for fair comparison
    orig_norm = (original - original.min()) / (original.max() - original.min() + 1e-10)
    corr_norm = (corrected - corrected.min()) / (corrected.max() - corrected.min() + 1e-10)
    
    correlation = np.corrcoef(orig_norm, corr_norm)[0, 1]
    return correlation


def calculate_peak_distortion(original: np. ndarray, corrected: np.ndarray) -> float:
    """
    Calculate percentage change in peak heights.
    
    Lower = better peak preservation
    """
    # Find top 10 peaks in original
    from scipy.signal import find_peaks
    peaks_orig, _ = find_peaks(original, prominence=np.std(original))
    
    if len(peaks_orig) == 0:
        return 0.0
    
    # Get top 10 by height
    if len(peaks_orig) > 10:
        peak_heights = original[peaks_orig]
        top_10_idx = np.argsort(peak_heights)[-10:]
        peaks_orig = peaks_orig[top_10_idx]
    
    # Calculate height change
    heights_orig = original[peaks_orig]
    heights_corr = corrected[peaks_orig]
    
    # Percentage change
    distortion = np.abs((heights_corr - heights_orig) / (heights_orig + 1e-10)) * 100
    return np.mean(distortion)


def calculate_residual_quality(spectrum: np.ndarray, baseline: np.ndarray) -> float:
    """
    Calculate standard deviation of residual. 
    
    Lower = baseline fits better
    """
    residual = spectrum - baseline
    return np.std(residual)


# ============================================================================
# COMPREHENSIVE COMPARISON
# ============================================================================

def compare_baseline_methods(
    spectra: np.ndarray,
    wavelengths: np.ndarray,
    sample_names: np.ndarray = None,  # NEW: Optional for classification
    objective: str = 'nutrient',      # NEW:  'nutrient' or 'contaminant'
    verbose: bool = True
) -> Dict[str, BaselineMetrics]:
    """
    Comprehensive comparison INCLUDING optional classification evaluation.
    
    All existing metrics preserved: 
    - Baseline flatness
    - Peak preservation
    - Edge stability (polynomial oscillation test)
    - Peak recovery (synthetic ground truth test)
    - NEW: Classification accuracy (if sample_names provided)
    
    Parameters
    ----------
    spectra : np.ndarray
        Raw spectra (n_samples, n_features)
    wavelengths : np.ndarray
        Wavelength values
    sample_names : np.ndarray, optional
        Sample identifiers for label generation (enables classification)
    objective : str
        'nutrient' (NPK High/Med/Low) or 'contaminant' (Clean/Contaminated)
    verbose : bool
        Print progress
        
    Returns
    -------
    results : Dict[str, BaselineMetrics]
        All metrics INCLUDING classification_accuracy
    """
    
    # NEW: Generate labels if sample names provided
    y_encoded = None
    if sample_names is not None and ENABLE_CLASSIFICATION:
        if objective == 'nutrient':
            labels_raw = np.array([get_nutrient_label(name) for name in sample_names])
            valid_mask = labels_raw != -1
            spectra = spectra[valid_mask]
            labels = labels_raw[valid_mask]
            print(f"\n✓ Classification enabled:  Nutrient classification (High/Med/Low)")
            print(f"  Usable samples: {len(labels)} (excluded {np.sum(~valid_mask)} contaminated)")
        elif objective == 'contaminant': 
            labels = np.array([get_contaminant_label(name) for name in sample_names])
            print(f"\n✓ Classification enabled: Contaminant detection (Clean/Contaminated)")
            print(f"  Total samples: {len(labels)}")
        else:
            raise ValueError(f"Unknown objective: {objective}")
        
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_encoded = le.fit_transform(labels)
    else:
        if verbose and sample_names is None:
            print("\n⚠️ Sample names not provided - classification disabled")
            print("   All other metrics will be calculated\n")
    
    # Method configurations (UNCHANGED - all existing methods preserved)
    methods = {
        'AsLS_default': {
            'func': lambda x: asls_baseline(x, lam=1e6, p=0.01),
            'params': 2,
            'description': 'AsLS (lam=1e6, p=0.01) - RECOMMENDED [221]'
        },
        'AsLS_smooth': {
            'func': lambda x: asls_baseline(x, lam=1e7, p=0.01),
            'params': 2,
            'description': 'AsLS (lam=1e7, p=0.01) - Smoother'
        },
        'AsLS_aggressive': {
            'func': lambda x: asls_baseline(x, lam=1e6, p=0.001),
            'params': 2,
            'description': 'AsLS (lam=1e6, p=0.001) - More asymmetric'
        },
        'airPLS':  {
            'func': lambda x: airpls_baseline(x, lam=1e6),
            'params': 1,
            'description': 'airPLS (lam=1e6) - Adaptive'
        },
        'SNIP_50': {
            'func': lambda x: snip_baseline(x, iterations=50),
            'params':  1,
            'description':  'SNIP (iterations=50) - Standard'
        },
        'SNIP_100': {
            'func': lambda x: snip_baseline(x, iterations=100),
            'params': 1,
            'description': 'SNIP (iterations=100) - Smoother'
        },
        'Polynomial_5': {
            'func': lambda x: polynomial_baseline(x, degree=5),
            'params': 1,
            'description': 'Polynomial (degree=5) - Low cost'
        },
        'Polynomial_7': {
            'func': lambda x: polynomial_baseline(x, degree=7),
            'params': 1,
            'description': 'Polynomial (degree=7)'
        },
        'Morphological':  {
            'func': lambda x: morphological_baseline(x, window=50),
            'params': 1,
            'description': 'Morphological opening (window=50)'
        },
        'RollingBall': {
            'func':  lambda x: rolling_ball_baseline(x, radius=50),
            'params': 1,
            'description': 'Rolling ball (radius=50)'
        },
        'Median':  {
            'func': lambda x: median_baseline(x, window=101),
            'params': 1,
            'description': 'Median filter (window=101) - Robust'
        },
    }
    
    results = {}
    n_samples = spectra.shape[0]
    
    if verbose:
        print("\n" + "="*70)
        mode = "WITH CLASSIFICATION" if y_encoded is not None else "SPECTRAL QUALITY ONLY"
        print(f"BASELINE CORRECTION METHOD COMPARISON ({mode})")
        print("All metrics:  Flatness, Peak Preservation, Edge Stability,")
        print("             Peak Recovery, Computation Time" + (" + Classification" if y_encoded is not None else ""))
        print("="*70)
        print(f"Testing on {n_samples} spectra...")
    
    # Select representative spectrum for synthetic testing
    test_spectrum_idx = n_samples // 2
    base_spectrum_template = spectra[test_spectrum_idx]
    
    for name, config in methods.items():
        if verbose:
            print(f"\nEvaluating:  {name}...")
            print(f"  {config['description']}")
        
        # Metrics accumulators (ALL EXISTING METRICS PRESERVED)
        flatness_scores = []
        preservation_scores = []
        residual_scores = []
        distortion_scores = []
        times = []
        edge_left_scores = []
        edge_right_scores = []
        edge_ratio_scores = []
        
        # Test on subset of spectra
        test_samples = min(50, n_samples)
        
        for i in range(test_samples):
            original = spectra[i]
            
            start_time = time.time()
            try:
                baseline = config['func'](original)
                elapsed_ms = (time.time() - start_time) * 1000
            except Exception as e:
                if verbose:
                    print(f"  ⚠️ Error:  {e}")
                baseline = np.zeros_like(original)
                elapsed_ms = 0
            
            corrected = np.maximum(original - baseline, 0)
            
            # Calculate all existing metrics
            flatness_scores.append(calculate_baseline_flatness(corrected))
            preservation_scores.append(calculate_peak_preservation(original, corrected))
            residual_scores.append(calculate_residual_quality(original, baseline))
            distortion_scores. append(calculate_peak_distortion(original, corrected))
            times.append(elapsed_ms)
            
            # Edge stability metrics
            left_std, right_std, ratio = calculate_edge_stability(corrected, edge_size=50)
            edge_left_scores.append(left_std)
            edge_right_scores.append(right_std)
            edge_ratio_scores.append(ratio)
        
        # Synthetic Ground Truth Test (PRESERVED)
        if verbose:
            print(f"  Running synthetic peak recovery test...")
        
        mean_recovery, std_recovery = test_synthetic_ground_truth(
            config['func'],
            base_spectrum_template,
            wavelengths,
            n_tests=5
        )
        
        # NEW: Classification accuracy evaluation (if labels available)
        classification_accuracy = 0.0
        if y_encoded is not None:
            # Apply baseline correction to all spectra
            corrected_all = np.zeros_like(spectra)
            for i in range(len(spectra)):
                try:
                    baseline = config['func'](spectra[i])
                    corrected_all[i] = np.maximum(spectra[i] - baseline, 0)
                except: 
                    corrected_all[i] = spectra[i]
            
            # Normalize for classification
            from sklearn.preprocessing import normalize
            corrected_norm = normalize(corrected_all, norm='l2', axis=1)
            
            # Random Forest classifier (Context Feeder [302]:  94.9%, 128KB)
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import StratifiedKFold, cross_val_score
            
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                random_state=42,
                n_jobs=-1
            )
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(rf, corrected_norm, y_encoded, cv=cv, scoring='accuracy')
            classification_accuracy = np.mean(cv_scores) * 100
        
        # Store results (ALL METRICS)
        results[name] = BaselineMetrics(
            name=name,
            baseline_flatness=np.mean(flatness_scores),
            peak_preservation=np.mean(preservation_scores),
            residual_std=np.mean(residual_scores),
            peak_distortion=np.mean(distortion_scores),
            computation_time_ms=np. mean(times),
            parameter_complexity=config['params'],
            edge_stability_left=np. mean(edge_left_scores),
            edge_stability_right=np.mean(edge_right_scores),
            edge_stability_ratio=np.mean(edge_ratio_scores),
            peak_recovery_rate=mean_recovery,
            classification_accuracy=classification_accuracy  # NEW
        )
        
        if verbose:
            m = results[name]
            print(f"  Baseline Flatness:     {m.baseline_flatness:.4f} (lower=better)")
            print(f"  Peak Preservation:    {m.peak_preservation:.4f} (higher=better)")
            print(f"  Edge Stability:       L={m.edge_stability_left:.4f}, R={m. edge_stability_right:.4f}, Ratio={m.edge_stability_ratio:.2f}")
            if m.edge_stability_ratio > 2.0:
                print(f"  ⚠️ WARNING: High edge instability")
            print(f"  Peak Recovery:        {m. peak_recovery_rate:.2f}% (±{std_recovery:.2f}%)")
            if m.peak_recovery_rate < 95.0 or m.peak_recovery_rate > 105.0:
                print(f"  ⚠️ WARNING: Peak height deviation")
            if y_encoded is not None:
                print(f"  Classification:        {m.classification_accuracy:.2f}% ✓")
            print(f"  Computation Time:     {m.computation_time_ms:.3f} ms")
            print()
    
    return results


def generate_recommendation(results: Dict[str, BaselineMetrics]) -> Tuple[str, str]:
    """
    Generate recommendation INCLUDING synthetic peak recovery validation.
    
    Scoring weights:
    - Baseline flatness: 20%
    - Peak preservation: 30% (CRITICAL for LIBS)
    - Edge stability: 15%
    - Peak recovery: 25% (NEW - quantitative accuracy validation)
    - Peak distortion: 5%
    - Computation time:  5%
    """
    
    methods = list(results.keys())
    
    # Normalize metrics
    flatness_vals = np.array([results[m].baseline_flatness for m in methods])
    distortion_vals = np.array([results[m].peak_distortion for m in methods])
    time_vals = np.array([results[m].computation_time_ms for m in methods])
    preservation_vals = np.array([results[m].peak_preservation for m in methods])
    edge_ratio_vals = np.array([results[m].edge_stability_ratio for m in methods])
    recovery_vals = np.array([results[m].peak_recovery_rate for m in methods])
    
    # Score calculations
    flatness_scores = 100 * (1 - (flatness_vals - flatness_vals.min()) / (flatness_vals.max() - flatness_vals.min() + 1e-10))
    preservation_scores = 100 * preservation_vals
    distortion_scores = 100 * (1 - (distortion_vals - distortion_vals.min()) / (distortion_vals.max() - distortion_vals.min() + 1e-10))
    time_scores = 100 * (1 - (time_vals - time_vals.min()) / (time_vals.max() - time_vals.min() + 1e-10))
    edge_scores = 100 * (1 - np.clip((edge_ratio_vals - 1.0) / 3.0, 0, 1))
    
    # NEW: Peak recovery scoring (100% = ideal, penalize deviation)
    recovery_scores = 100 * (1 - np.abs(recovery_vals - 100.0) / 100.0)
    
    # Weighted combination (UPDATED WEIGHTS)
    scores = {}
    for i, method in enumerate(methods):
        score = (
            flatness_scores[i] * 0.20 +      # Reduced from 25%
            preservation_scores[i] * 0.30 +  # Reduced from 35%
            distortion_scores[i] * 0.05 +    # Reduced from 15%
            edge_scores[i] * 0.15 +          # Same
            recovery_scores[i] * 0.25 +      # NEW: 25% weight - CRITICAL
            time_scores[i] * 0.05            # Reduced from 10%
        )
        scores[method] = score
    
    # Find best
    recommended = max(scores, key=scores. get)
    asls_methods = {k: v for k, v in results.items() if k.startswith('AsLS')}
    best_asls = max(asls_methods, key=lambda x: scores[x]) if asls_methods else None
    
    best_metrics = results[recommended]
    
    justification = f"""
RECOMMENDATION:  {recommended}

JUSTIFICATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PERFORMANCE METRICS:
  Baseline Flatness:      {best_metrics.baseline_flatness:.6f} (lower=better)
  Peak Preservation:    {best_metrics.peak_preservation:.4f} (higher=better, max=1.0)
  Peak Distortion:      {best_metrics.peak_distortion:.2f}% (lower=better)
  Edge Stability Ratio:   {best_metrics.edge_stability_ratio:.2f} (lower=better, <2.0 ideal)
  Peak Recovery Rate:   {best_metrics.peak_recovery_rate:.2f}% (100% = ideal)
  Computation Time:     {best_metrics.computation_time_ms:.3f} ms/spectrum

WEIGHTED SCORE: {scores[recommended]:.2f}/100
  20% Baseline flatness
  30% Peak preservation (CRITICAL for LIBS)
  15% Edge stability (polynomial oscillation check)
  25% Peak recovery (quantitative accuracy validation)
  5% Peak distortion
  5% Computation time

"""
    
    # Peak recovery interpretation
    if 95.0 <= best_metrics. peak_recovery_rate <= 105.0:
        justification += f"""
✓ EXCELLENT QUANTITATIVE ACCURACY: 
  Peak recovery = {best_metrics.peak_recovery_rate:.2f}% (within 5% of true height)
  Synthetic ground truth test confirms reliable peak height preservation. 
  Suitable for quantitative LIBS analysis (elemental concentration estimation).
  
"""
    elif best_metrics.peak_recovery_rate < 95.0:
        justification += f"""
⚠️ PEAK HEIGHT UNDER-ESTIMATION:
  Peak recovery = {best_metrics.peak_recovery_rate:.2f}% (<95%)
  Baseline correction may be over-aggressive, distorting true peak heights.
  
  IMPACT: 
  - Qualitative classification (High/Med/Low) should be unaffected
  - Quantitative analysis (ppm concentration) may underestimate by {100 - best_metrics.peak_recovery_rate:.1f}%
  - Consider using {[m for m in methods if 95 <= results[m].peak_recovery_rate <= 105][0] if any(95 <= results[m].peak_recovery_rate <= 105 for m in methods) else best_asls} for quantitative work
  
"""
    else:  # > 105%
        justification += f"""
⚠️ PEAK HEIGHT OVER-ESTIMATION:
  Peak recovery = {best_metrics. peak_recovery_rate:.2f}% (>105%)
  Baseline under-corrected, leaving residual background under peaks.
  
  IMPACT: 
  - Qualitative classification acceptable but may confuse weak peaks
  - Quantitative analysis may overestimate by {best_metrics.peak_recovery_rate - 100:.1f}%
  - Recommend stricter baseline correction parameters
  
"""
    
    # Edge stability (keep existing logic)
    if best_metrics. edge_stability_ratio > 2.5:
        justification += f"""
⚠️ EDGE EFFECT WARNING:
  Edge ratio = {best_metrics.edge_stability_ratio:.2f} - restrict analysis to 250-900nm
  
"""
    elif best_metrics.edge_stability_ratio > 2.0:
        justification += f"""
⚠️ MODERATE EDGE INSTABILITY:  Monitor endpoint behavior. 
  
"""
    else:
        justification += f"""
✓ EXCELLENT EDGE STABILITY: Full 200-950nm range usable.
  
"""
    
    # Method-specific interpretation
    if recommended. startswith('AsLS'):
        justification += f"""
INTERPRETATION: 
Confirms literature justification [221]:  AsLS r=0.93, 19% RMSE improvement. 
Peak recovery {best_metrics.peak_recovery_rate:.1f}% validates quantitative reliability. 

DEPLOYMENT for TinyML:
  - Time:  {best_metrics.computation_time_ms:.2f} ms (acceptable for MCU)
  - Proven for LIBS soil analysis
  - Edge stability:  {best_metrics.edge_stability_ratio:.2f}
  
"""
    elif recommended.startswith('Polynomial'):
        justification += f"""
INTERPRETATION:
Polynomial shows best overall performance. 

CRITICAL VALIDATION:
  - Peak recovery: {best_metrics.peak_recovery_rate:.1f}% 
"""
        if 95 <= best_metrics.peak_recovery_rate <= 105:
            justification += f"    ✓ Excellent quantitative accuracy - UNEXPECTED for polynomials!\n"
            justification += f"    This should be highlighted in thesis as superior to literature expectations.\n"
        else:
            justification += f"    ⚠️ {abs(best_metrics.peak_recovery_rate - 100):.1f}% deviation from true height\n"
        
        if best_metrics.edge_stability_ratio > 2.0:
            justification += f"""
  - Edge effects present (ratio={best_metrics.edge_stability_ratio:.2f})
  - RECOMMENDATION: Use for qualitative classification only, or restrict to 250-900nm
  - Alternative for quantitative work: {best_asls} (recovery={results[best_asls].peak_recovery_rate:.1f}%)
"""
        else:
            justification += f"  - Edge stability excellent:  Full range usable\n"
        
        justification += f"""
COMPUTATIONAL ADVANTAGE:
  - {best_metrics.computation_time_ms/results[best_asls].computation_time_ms:.1f}x faster than AsLS
  - Simplest MCU implementation (polyfit + polyval)
"""
    
    return recommended, justification
    """
    Generate recommendation based on metrics INCLUDING edge stability.
    
    Scoring:
    - Baseline flatness: 25% (key indicator)
    - Peak preservation: 35% (CRITICAL for LIBS)
    - Peak distortion: 15% (avoid artifacts)
    - Edge stability: 15% (NEW - critical for polynomial methods)
    - Computation time: 10% (MCU constraint)
    """
    
    # Separate SG and MS results
    methods = list(results.keys())
    
    # Normalize metrics to 0-100 scale
    flatness_vals = np.array([results[m].baseline_flatness for m in methods])
    distortion_vals = np.array([results[m].peak_distortion for m in methods])
    time_vals = np.array([results[m].computation_time_ms for m in methods])
    preservation_vals = np.array([results[m].peak_preservation for m in methods])
    edge_ratio_vals = np.array([results[m].edge_stability_ratio for m in methods])
    
    # Normalize (avoid division by zero)
    flatness_scores = 100 * (1 - (flatness_vals - flatness_vals.min()) / (flatness_vals.max() - flatness_vals.min() + 1e-10))
    preservation_scores = 100 * preservation_vals
    distortion_scores = 100 * (1 - (distortion_vals - distortion_vals.min()) / (distortion_vals.max() - distortion_vals.min() + 1e-10))
    time_scores = 100 * (1 - (time_vals - time_vals.min()) / (time_vals.max() - time_vals.min() + 1e-10))
    
    # Edge stability:  penalize ratios > 2.0 heavily
    edge_scores = 100 * (1 - np.clip((edge_ratio_vals - 1.0) / 3.0, 0, 1))  # Ratio 1.0 → 100%, Ratio 4.0 → 0%
    
    # Weighted combination (updated weights)
    scores = {}
    for i, method in enumerate(methods):
        score = (
            flatness_scores[i] * 0.25 +      # Reduced from 30%
            preservation_scores[i] * 0.35 +   # Reduced from 40%
            distortion_scores[i] * 0.15 +     # Reduced from 20%
            edge_scores[i] * 0.15 +           # NEW: 15% weight
            time_scores[i] * 0.10             # Same
        )
        scores[method] = score
    
    # Find best overall and best AsLS variant
    recommended = max(scores, key=scores.get)
    asls_methods = {k: v for k, v in results.items() if k.startswith('AsLS')}
    best_asls = max(asls_methods, key=lambda x: scores[x]) if asls_methods else None
    
    # Build justification
    best_metrics = results[recommended]
    
    justification = f"""
RECOMMENDATION: {recommended}

JUSTIFICATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PERFORMANCE METRICS:
  Baseline Flatness:      {best_metrics.baseline_flatness:.6f} (lower=better)
  Peak Preservation:    {best_metrics.peak_preservation:.4f} (higher=better, max=1.0)
  Peak Distortion:      {best_metrics.peak_distortion:.2f}% (lower=better)
  Edge Stability (L/R):  {best_metrics.edge_stability_left:.6f} / {best_metrics.edge_stability_right:.6f}
  Edge Stability Ratio:  {best_metrics.edge_stability_ratio:.2f} (lower=better, <2.0 ideal)
  Computation Time:     {best_metrics.computation_time_ms:.3f} ms/spectrum
  Parameter Complexity: {best_metrics.parameter_complexity} parameters

WEIGHTED SCORE: {scores[recommended]:.2f}/100
  25% Baseline flatness
  35% Peak preservation (CRITICAL for LIBS)
  15% Peak distortion
  15% Edge stability (polynomial oscillation check)
  10% Computation time (MCU constraint)

"""
    
    # Edge stability interpretation
    if best_metrics.edge_stability_ratio > 2.5:
        justification += f"""
⚠️  EDGE EFFECT WARNING: 
  Edge instability ratio = {best_metrics.edge_stability_ratio:.2f} (>2.5 is HIGH)
  This indicates significant oscillations at spectrum endpoints.
  
  IMPACT: 
  - Wavelength regions near 200nm and 950nm may be unreliable
  - Recommend restricting analysis to 250-900nm effective range
  - Consider edge trimming in preprocessing pipeline
  
"""
    elif best_metrics.edge_stability_ratio > 2.0:
        justification += f"""
⚠️  MODERATE EDGE INSTABILITY:
  Edge instability ratio = {best_metrics.edge_stability_ratio:.2f}
  Endpoints show 2x more variation than center region. 
  Acceptable for most applications but monitor edge behavior.
  
"""
    else:
        justification += f"""
✓ EXCELLENT EDGE STABILITY: 
  Edge instability ratio = {best_metrics.edge_stability_ratio:.2f} (<2.0)
  Baseline correction is stable across entire wavelength range (200-950nm).
  No edge trimming required.
  
"""
    
    # Contextualize recommendation
    if recommended.startswith('AsLS'):
        justification += f"""
INTERPRETATION: 
This confirms the literature justification [221]: 
  - AsLS achieved r=0.93, 19% RMSE improvement for LIBS soil carbon
  - Physics-based method (removes Bremsstrahlung continuum)
  - Optimal balance of flatness, peak preservation, AND edge stability

DEPLOYMENT for TinyML:
  - Computation time: {best_metrics.computation_time_ms:.2f} ms acceptable for MCU
  - 2 parameters (lambda, p) - moderate tuning complexity
  - Proven in LIBS applications
  - Edge stability:  {best_metrics.edge_stability_ratio:.2f} indicates reliable full-spectrum coverage

"""
        if best_asls != recommended:
            alt_metrics = results[best_asls]
            justification += f"NOTE: Alternative AsLS config '{best_asls}' scored {scores[best_asls]:.2f}/100\n"
            justification += f"      Edge ratio: {alt_metrics.edge_stability_ratio:.2f}\n"
            justification += f"      Consider testing both for your specific data.\n"
    
    elif recommended.startswith('Polynomial'):
        justification += f"""
INTERPRETATION: 
Polynomial baseline shows best overall performance for your data. 

ADVANTAGES for TinyML:
  - LOWEST computational cost ({best_metrics.computation_time_ms:.2f} ms)
  - Simple implementation (np.polyfit + np.polyval)
  - Single parameter (polynomial degree)
  - Easy to port to C for MCU

"""
        # Critical edge stability check for polynomials
        if best_metrics.edge_stability_ratio > 2.0:
            justification += f"""
CRITICAL LIMITATION - EDGE EFFECTS:
  Polynomial methods are known for endpoint oscillations ("Runge's phenomenon")
  Your data shows edge ratio = {best_metrics.edge_stability_ratio:.2f}
  
  REQUIRED ACTIONS:
  1. Restrict wavelength analysis to 250-900nm (trim 50 points each end)
  2. Document effective range in thesis:  "Polynomial baseline valid 250-900nm"
  3. If N, P, K peaks fall in 200-250nm or 900-950nm, MUST use AsLS instead
  
  ALTERNATIVE: Use {best_asls} (edge ratio:  {results[best_asls].edge_stability_ratio:.2f})
  
"""
        else:
            justification += f"""
EDGE STABILITY VALIDATION:
  Surprisingly, polynomial shows excellent edge stability (ratio={best_metrics.edge_stability_ratio:.2f})
  This is UNUSUAL for polynomial fitting and should be highlighted in thesis.
  Full 200-950nm range appears usable.
  
"""
        
        justification += f"""
TRADE-OFF vs AsLS:
  - {best_asls}:  Score={scores[best_asls]:.2f}, Edge ratio={results[best_asls].edge_stability_ratio:.2f}
  - Polynomial is {best_metrics.computation_time_ms/results[best_asls].computation_time_ms:.1f}x faster
  - If edge effects acceptable, use Polynomial; otherwise use {best_asls}
"""
    
    else:
        # Other methods
        justification += f"""
INTERPRETATION:
{recommended} shows best performance for your data.

COMPARISON TO AsLS [221]:
"""
        if best_asls: 
            asls_metrics = results[best_asls]
            justification += f"  Best AsLS:   {best_asls} (score={scores[best_asls]:.2f})\n"
            justification += f"  Flatness:       {recommended}:  {best_metrics.baseline_flatness:.6f} vs AsLS: {asls_metrics. baseline_flatness:.6f}\n"
            justification += f"  Preservation: {recommended}:  {best_metrics.peak_preservation:.4f} vs AsLS:  {asls_metrics.peak_preservation:.4f}\n"
            justification += f"  Edge Ratio:    {recommended}: {best_metrics.edge_stability_ratio:.2f} vs AsLS: {asls_metrics.edge_stability_ratio:.2f}\n"
            justification += f"  Time:         {recommended}: {best_metrics.computation_time_ms:.2f} ms vs AsLS: {asls_metrics.computation_time_ms:.2f} ms\n"
        
        justification += f"""
CONSIDERATION:
  - AsLS has stronger literature justification for LIBS [221]
  - If scores are close (<5 point difference), recommend AsLS for reproducibility
"""
    
    return recommended, justification

def plot_baseline_comparison(
    spectrum:  np.ndarray,
    wavelengths: np.ndarray,
    results: Dict[str, BaselineMetrics],
    recommended:  str,
    save_path: Optional[str] = None
) -> None:
    """Visualize baseline correction comparison across TWO figures."""
    
    # Select representative methods to plot
    plot_methods = {
        'AsLS_default': ('AsLS (default)', 'blue'),
        'airPLS': ('airPLS', 'red'),
        'SNIP_50': ('SNIP', 'green'),
        'Polynomial_5': ('Polynomial', 'orange'),
        'Morphological':  ('Morphological', 'purple')
    }
    
    # ========================================================================
    # FIGURE 1: SPECTRAL VISUALIZATION (6 plots in 2x3 layout)
    # ========================================================================
    fig1 = plt.figure(figsize=(18, 10))
    fig1.suptitle('LIBS Baseline Correction - Spectral Analysis\nContext Feeder [221]:  AsLS r=0.93, 19% RMSE improvement',
                  fontsize=14, fontweight='bold', y=0.995)
    
    # Plot 1: Baselines overlay
    ax1 = fig1.add_subplot(2, 3, 1)
    ax1.plot(wavelengths, spectrum, 'k-', alpha=0.3, linewidth=0.8, label='Original')
    
    for method, (label, color) in plot_methods.items():
        if method == 'AsLS_default':
            baseline = asls_baseline(spectrum, lam=1e6, p=0.01)
        elif method == 'airPLS':
            baseline = airpls_baseline(spectrum, lam=1e6)
        elif method == 'SNIP_50':
            baseline = snip_baseline(spectrum, iterations=50)
        elif method == 'Polynomial_5': 
            baseline = polynomial_baseline(spectrum, degree=5)
        elif method == 'Morphological': 
            baseline = morphological_baseline(spectrum, window=50)
        else:
            continue
        
        ax1.plot(wavelengths, baseline, color=color, linewidth=1.5,
                label=label, alpha=0.8)
    
    ax1.set_xlabel('Wavelength (nm)', fontsize=11)
    ax1.set_ylabel('Intensity', fontsize=11)
    ax1.set_title('Baseline Comparison', fontweight='bold', fontsize=12)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Corrected spectra
    ax2 = fig1.add_subplot(2, 3, 2)
    
    for method, (label, color) in plot_methods.items():
        if method == 'AsLS_default':
            baseline = asls_baseline(spectrum, lam=1e6, p=0.01)
        elif method == 'airPLS': 
            baseline = airpls_baseline(spectrum, lam=1e6)
        elif method == 'SNIP_50': 
            baseline = snip_baseline(spectrum, iterations=50)
        elif method == 'Polynomial_5':
            baseline = polynomial_baseline(spectrum, degree=5)
        elif method == 'Morphological':
            baseline = morphological_baseline(spectrum, window=50)
        else:
            continue
        
        corrected = np.maximum(spectrum - baseline, 0)
        ax2.plot(wavelengths, corrected, color=color, linewidth=1.2,
                label=label, alpha=0.7)
    
    ax2.set_xlabel('Wavelength (nm)', fontsize=11)
    ax2.set_ylabel('Intensity', fontsize=11)
    ax2.set_title('Corrected Spectra', fontweight='bold', fontsize=12)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Baseline region zoom (flatness check)
    ax3 = fig1.add_subplot(2, 3, 3)
    baseline_region = slice(0, 500)
    wl_region = wavelengths[baseline_region]
    
    for method, (label, color) in plot_methods.items():
        if method == 'AsLS_default':
            baseline = asls_baseline(spectrum, lam=1e6, p=0.01)
        elif method == 'airPLS':
            baseline = airpls_baseline(spectrum, lam=1e6)
        elif method == 'SNIP_50':
            baseline = snip_baseline(spectrum, iterations=50)
        elif method == 'Polynomial_5': 
            baseline = polynomial_baseline(spectrum, degree=5)
        elif method == 'Morphological':
            baseline = morphological_baseline(spectrum, window=50)
        else:
            continue
        
        corrected = np. maximum(spectrum - baseline, 0)
        ax3.plot(wl_region, corrected[baseline_region], color=color,
                linewidth=1.2, label=label, alpha=0.7)
    
    ax3.set_xlabel('Wavelength (nm)', fontsize=11)
    ax3.set_ylabel('Intensity', fontsize=11)
    ax3.set_title('Baseline Region Flatness (First 500 Points)', fontweight='bold', fontsize=12)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Plot 4: Left edge zoom (first 50 points - edge effect test)
    ax4 = fig1.add_subplot(2, 3, 4)
    edge_left = slice(0, 50)
    wl_left = wavelengths[edge_left]
    
    for method, (label, color) in plot_methods.items():
        if method == 'AsLS_default': 
            baseline = asls_baseline(spectrum, lam=1e6, p=0.01)
        elif method == 'airPLS':
            baseline = airpls_baseline(spectrum, lam=1e6)
        elif method == 'SNIP_50':
            baseline = snip_baseline(spectrum, iterations=50)
        elif method == 'Polynomial_5':
            baseline = polynomial_baseline(spectrum, degree=5)
        elif method == 'Morphological': 
            baseline = morphological_baseline(spectrum, window=50)
        else:
            continue
        
        corrected = np.maximum(spectrum - baseline, 0)
        ax4.plot(wl_left, corrected[edge_left], color=color,
                linewidth=1.5, label=label, alpha=0.8, marker='o', markersize=3)
    
    ax4.set_xlabel('Wavelength (nm)', fontsize=11)
    ax4.set_ylabel('Intensity', fontsize=11)
    ax4.set_title('LEFT Edge Stability (First 50 Points @ 200nm)', fontweight='bold', fontsize=12)
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=2)
    
    # Plot 5: Right edge zoom (last 50 points)
    ax5 = fig1.add_subplot(2, 3, 5)
    edge_right = slice(-50, None)
    wl_right = wavelengths[edge_right]
    
    for method, (label, color) in plot_methods.items():
        if method == 'AsLS_default':
            baseline = asls_baseline(spectrum, lam=1e6, p=0.01)
        elif method == 'airPLS':
            baseline = airpls_baseline(spectrum, lam=1e6)
        elif method == 'SNIP_50':
            baseline = snip_baseline(spectrum, iterations=50)
        elif method == 'Polynomial_5': 
            baseline = polynomial_baseline(spectrum, degree=5)
        elif method == 'Morphological':
            baseline = morphological_baseline(spectrum, window=50)
        else:
            continue
        
        corrected = np. maximum(spectrum - baseline, 0)
        ax5.plot(wl_right, corrected[edge_right], color=color,
                linewidth=1.5, label=label, alpha=0.8, marker='o', markersize=3)
    
    ax5.set_xlabel('Wavelength (nm)', fontsize=11)
    ax5.set_ylabel('Intensity', fontsize=11)
    ax5.set_title('RIGHT Edge Stability (Last 50 Points @ 950nm)', fontweight='bold', fontsize=12)
    ax5.legend(loc='best', fontsize=9)
    ax5.grid(True, alpha=0.3)
    ax5.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=2)
    
    # Plot 6: AsLS parameter comparison
    ax6 = fig1.add_subplot(2, 3, 6)
    asls_methods = {k: v for k, v in results.items() if k.startswith('AsLS')}
    asls_names = list(asls_methods.keys())
    
    if asls_names:
        asls_flat = [results[m]. baseline_flatness for m in asls_names]
        asls_pres = [results[m].peak_preservation for m in asls_names]
        asls_edge = [results[m].edge_stability_ratio for m in asls_names]
        
        x_pos = np.arange(len(asls_names))
        width = 0.25
        
        ax6.bar(x_pos - width, [f*1000 for f in asls_flat], width, label='Flatness (×1000)', alpha=0.7, color='blue')
        ax6.bar(x_pos, [p*100 for p in asls_pres], width, label='Preservation (×100)', alpha=0.7, color='green')
        ax6.bar(x_pos + width, [e*10 for e in asls_edge], width, label='Edge Ratio (×10)', alpha=0.7, color='orange')
        
        ax6.set_xlabel('AsLS Configuration', fontsize=11)
        ax6.set_ylabel('Metric Value (scaled)', fontsize=11)
        ax6.set_title('AsLS Parameter Tuning Comparison', fontweight='bold', fontsize=12)
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels([m.replace('AsLS_', '') for m in asls_names], fontsize=10)
        ax6.legend(fontsize=9)
        ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    if save_path:
        path1 = save_path. replace('.png', '_spectral.png')
        fig1.savefig(path1, dpi=150, bbox_inches='tight')
        print(f"\n✓ Figure 1 saved to: {path1}")
    
    plt.show()
    
    # ========================================================================
    # FIGURE 2: PERFORMANCE METRICS (7 plots in 3x3 layout, bottom-center empty)
    # ========================================================================
    fig2 = plt.figure(figsize=(20, 14))
    fig2.suptitle('LIBS Baseline Correction - Performance Metrics\nEdge Stability Test: Ratio <2.0 = Excellent, >2.5 = Problematic | Peak Recovery: 95-105% = Accurate',
                  fontsize=14, fontweight='bold', y=0.995)
    
    methods = list(results.keys())
    
    # Plot 1: Baseline flatness bar chart
    ax1 = fig2.add_subplot(3, 3, 1)
    flatness_vals = [results[m].baseline_flatness for m in methods]
    colors_bar = ['green' if m == recommended else 'blue' for m in methods]
    
    bars = ax1.barh(methods, flatness_vals, color=colors_bar, alpha=0.7, edgecolor='black')
    for bar, method in zip(bars, methods):
        if method == recommended:
            bar.set_edgecolor('darkgreen')
            bar.set_linewidth(3)
    
    ax1.set_xlabel('Baseline Flatness (lower=better)', fontsize=11)
    ax1.set_title('Baseline Flatness Comparison', fontweight='bold', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Adjust y-axis labels if too many methods
    if len(methods) > 10:
        ax1.tick_params(axis='y', labelsize=8)
    
    # Plot 2: Peak preservation bar chart
    ax2 = fig2.add_subplot(3, 3, 2)
    preservation_vals = [results[m].peak_preservation for m in methods]
    
    bars = ax2.barh(methods, preservation_vals, color=colors_bar, alpha=0.7, edgecolor='black')
    for bar, method in zip(bars, methods):
        if method == recommended:
            bar.set_edgecolor('darkgreen')
            bar.set_linewidth(3)
    
    ax2.set_xlabel('Peak Preservation (higher=better)', fontsize=11)
    ax2.set_title('Peak Preservation Comparison', fontweight='bold', fontsize=12)
    ax2.axvline(x=0.95, color='gray', linestyle='--', alpha=0.5, linewidth=2, label='95% threshold')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.legend(fontsize=9)
    
    if len(methods) > 10:
        ax2.tick_params(axis='y', labelsize=8)
    
    # Plot 3: Edge stability ratio (CRITICAL)
    ax3 = fig2.add_subplot(3, 3, 3)
    edge_ratios = [results[m].edge_stability_ratio for m in methods]
    colors_edge = []
    for m, ratio in zip(methods, edge_ratios):
        if m == recommended:
            colors_edge.append('green')
        elif ratio > 2.5:
            colors_edge.append('red')
        elif ratio > 2.0:
            colors_edge.append('orange')
        else:
            colors_edge.append('blue')
    
    bars = ax3.barh(methods, edge_ratios, color=colors_edge, alpha=0.7, edgecolor='black')
    for bar, method in zip(bars, methods):
        if method == recommended:
            bar.set_edgecolor('darkgreen')
            bar.set_linewidth(3)
    
    ax3.set_xlabel('Edge Instability Ratio (lower=better)', fontsize=11)
    ax3.set_title('Edge Effect Test (First/Last 50 Points)', fontweight='bold', fontsize=12)
    ax3.axvline(x=2.0, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Acceptable (2.0)')
    ax3.axvline(x=2.5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Critical (2.5)')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='x')
    
    if len(methods) > 10:
        ax3.tick_params(axis='y', labelsize=8)
    
    # Plot 4: Peak Recovery Rate (Synthetic Ground Truth Validation)
    ax4 = fig2.add_subplot(3, 3, 4)
    recovery_vals = [results[m].peak_recovery_rate for m in methods]
    colors_recovery = []
    for m, recovery in zip(methods, recovery_vals):
        if m == recommended:
            colors_recovery.append('green')
        elif 95.0 <= recovery <= 105.0:
            colors_recovery.append('blue')
        else:
            colors_recovery.append('red')
    
    bars = ax4.barh(methods, recovery_vals, color=colors_recovery, alpha=0.7, edgecolor='black')
    for bar, method in zip(bars, methods):
        if method == recommended:
            bar.set_edgecolor('darkgreen')
            bar.set_linewidth(3)
    
    ax4.set_xlabel('Peak Recovery Rate (%, 100% = ideal)', fontsize=11)
    ax4.set_title('Synthetic Ground Truth Test\nQuantitative Accuracy Validation', fontweight='bold', fontsize=12)
    ax4.axvline(x=95, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='95% threshold')
    ax4.axvline(x=100, color='green', linestyle='-', linewidth=2, alpha=0.7, label='100% ideal')
    ax4.axvline(x=105, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='105% threshold')
    ax4.legend(fontsize=9, loc='lower right')
    ax4.grid(True, alpha=0.3, axis='x')
    
    # Dynamic x-axis limits based on data range
    min_recovery = min(recovery_vals)
    max_recovery = max(recovery_vals)
    x_margin = (max_recovery - min_recovery) * 0.15  # 15% margin
    ax4.set_xlim(max(80, min_recovery - x_margin), min(120, max_recovery + x_margin))
    
    # Add percentage annotations with smart positioning
    for bar, recovery in zip(bars, recovery_vals):
        width = bar.get_width()
        bar_center = bar.get_y() + bar.get_height() / 2
        
        # Check if annotation fits inside bar
        x_range = ax4.get_xlim()[1] - ax4.get_xlim()[0]
        bar_width_percent = (width - ax4.get_xlim()[0]) / x_range
        
        if bar_width_percent > 0.15:  # If bar is >15% of plot width, put text inside
            text_x = width - (x_range * 0.02)  # 2% margin from right edge
            text_ha = 'right'
            text_color = 'white'
        else:  # Otherwise put text outside
            text_x = width + (x_range * 0.01)  # 1% margin from bar end
            text_ha = 'left'
            text_color = 'black'
        
        ax4.text(text_x, bar_center, f'{recovery:.1f}%',
                ha=text_ha, va='center', fontsize=8, fontweight='bold', color=text_color)
    
    if len(methods) > 10:
        ax4.tick_params(axis='y', labelsize=8)
    
    # Plot 5: Computation Time
    ax5 = fig2.add_subplot(3, 3, 5)
    time_vals = [results[m].computation_time_ms for m in methods]
    
    bars = ax5.barh(methods, time_vals, color=colors_bar, alpha=0.7, edgecolor='black')
    for bar, method in zip(bars, methods):
        if method == recommended:
            bar.set_edgecolor('darkgreen')
            bar.set_linewidth(3)
    
    ax5.set_xlabel('Computation Time (ms, lower=better)', fontsize=11)
    ax5.set_title('Computational Cost', fontweight='bold', fontsize=12)
    ax5.grid(True, alpha=0.3, axis='x')
    
    # Dynamic x-axis for computation time
    min_time = min(time_vals)
    max_time = max(time_vals)
    time_margin = (max_time - min_time) * 0.15
    ax5.set_xlim(0, max_time + time_margin)
    
    # Add time annotations with smart positioning
    for bar, time_val in zip(bars, time_vals):
        width = bar.get_width()
        bar_center = bar.get_y() + bar.get_height() / 2
        
        x_range = ax5.get_xlim()[1] - ax5.get_xlim()[0]
        bar_width_percent = width / x_range
        
        if bar_width_percent > 0.15:
            text_x = width - (x_range * 0.02)
            text_ha = 'right'
            text_color = 'white'
        else:
            text_x = width + (x_range * 0.01)
            text_ha = 'left'
            text_color = 'black'
        
        ax5.text(text_x, bar_center, f'{time_val:.2f}',
                ha=text_ha, va='center', fontsize=8, fontweight='bold', color=text_color)
    
    if len(methods) > 10:
        ax5.tick_params(axis='y', labelsize=8)
    
    # Plot 6: Peak Distortion
    ax6 = fig2.add_subplot(3, 3, 6)
    distortion_vals = [results[m].peak_distortion for m in methods]
    
    bars = ax6.barh(methods, distortion_vals, color=colors_bar, alpha=0.7, edgecolor='black')
    for bar, method in zip(bars, methods):
        if method == recommended:
            bar.set_edgecolor('darkgreen')
            bar.set_linewidth(3)
    
    ax6.set_xlabel('Peak Distortion (%, lower=better)', fontsize=11)
    ax6.set_title('Peak Height Distortion', fontweight='bold', fontsize=12)
    ax6.grid(True, alpha=0.3, axis='x')
    
    # Dynamic x-axis
    min_dist = min(distortion_vals)
    max_dist = max(distortion_vals)
    dist_margin = (max_dist - min_dist) * 0.15
    ax6.set_xlim(0, max_dist + dist_margin)
    
    # Add annotations
    for bar, dist in zip(bars, distortion_vals):
        width = bar.get_width()
        bar_center = bar.get_y() + bar.get_height() / 2
        
        x_range = ax6.get_xlim()[1] - ax6.get_xlim()[0]
        bar_width_percent = width / x_range
        
        if bar_width_percent > 0.15:
            text_x = width - (x_range * 0.02)
            text_ha = 'right'
            text_color = 'white'
        else:
            text_x = width + (x_range * 0.01)
            text_ha = 'left'
            text_color = 'black'
        
        ax6.text(text_x, bar_center, f'{dist:.2f}%',
                ha=text_ha, va='center', fontsize=8, fontweight='bold', color=text_color)
    
    if len(methods) > 10:
        ax6.tick_params(axis='y', labelsize=8)
    
    # Plot 7: Overall ranking (weighted score)
    ax7 = fig2.add_subplot(3, 3, 7)
    
    # Calculate scores
    flatness_all = np.array([results[m].baseline_flatness for m in methods])
    preservation_all = np.array([results[m].peak_preservation for m in methods])
    distortion_all = np.array([results[m].peak_distortion for m in methods])
    time_all = np.array([results[m].computation_time_ms for m in methods])
    edge_ratio_all = np.array([results[m].edge_stability_ratio for m in methods])
    recovery_all = np.array([results[m].peak_recovery_rate for m in methods])
    
    scores = {}
    for i, method in enumerate(methods):
        flatness_score = 100 * (1 - (flatness_all[i] - flatness_all.min()) / (flatness_all.max() - flatness_all.min() + 1e-10))
        preservation_score = 100 * preservation_all[i]
        distortion_score = 100 * (1 - (distortion_all[i] - distortion_all.min()) / (distortion_all.max() - distortion_all.min() + 1e-10))
        time_score = 100 * (1 - (time_all[i] - time_all.min()) / (time_all.max() - time_all.min() + 1e-10))
        edge_score = 100 * (1 - np.clip((edge_ratio_all[i] - 1.0) / 3.0, 0, 1))
        recovery_score = 100 * (1 - np.abs(recovery_all[i] - 100.0) / 100.0)
        
        scores[method] = (
            flatness_score * 0.20 +
            preservation_score * 0.30 +
            distortion_score * 0.05 +
            edge_score * 0.15 +
            recovery_score * 0.25 +
            time_score * 0.05
        )
    
    sorted_methods = sorted(scores, key=scores.get, reverse=True)
    sorted_scores = [scores[m] for m in sorted_methods]
    colors_rank = ['green' if m == recommended else 'blue' for m in sorted_methods]
    
    bars = ax7.barh(sorted_methods, sorted_scores, color=colors_rank, alpha=0.7, edgecolor='black')
    for bar, method in zip(bars, sorted_methods):
        if method == recommended:
            bar.set_edgecolor('darkgreen')
            bar.set_linewidth(3)
    
    ax7.set_xlabel('Overall Score (0-100)', fontsize=11)
    ax7.set_title(f'Overall Ranking\nRecommended: {recommended}', fontweight='bold', fontsize=12)
    ax7.grid(True, alpha=0.3, axis='x')
    ax7.set_xlim(0, 105)
    
    # Add score annotations with smart positioning
    for bar, score in zip(bars, sorted_scores):
        width = bar.get_width()
        bar_center = bar.get_y() + bar.get_height() / 2
        
        if width > 15:  # If score > 15, put inside
            text_x = width - 2
            text_ha = 'right'
            text_color = 'white'
        else:
            text_x = width + 1
            text_ha = 'left'
            text_color = 'black'
        
        ax7.text(text_x, bar_center, f'{score:.1f}',
                ha=text_ha, va='center', fontsize=9, fontweight='bold', color=text_color)
    
    if len(sorted_methods) > 10:
        ax7.tick_params(axis='y', labelsize=8)
    
    # Plot 8: Multi-metric radar (top 5 methods) - BOTTOM CENTER (position 3,3,8)
    ax8 = fig2.add_subplot(3, 3, 8, projection='polar')
    
    top_5 = sorted_methods[:5]
    
    categories = ['Flatness', 'Preservation', 'Low Distortion', 'Edge Stability', 'Recovery', 'Speed']
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    
    for method in top_5:
        i = methods.index(method)
        values = [
            100 * (1 - (flatness_all[i] - flatness_all.min()) / (flatness_all.max() - flatness_all.min() + 1e-10)),
            100 * preservation_all[i],
            100 * (1 - (distortion_all[i] - distortion_all.min()) / (distortion_all.max() - distortion_all.min() + 1e-10)),
            100 * (1 - np.clip((edge_ratio_all[i] - 1.0) / 3.0, 0, 1)),
            100 * (1 - np.abs(recovery_all[i] - 100.0) / 100.0),
            100 * (1 - (time_all[i] - time_all.min()) / (time_all.max() - time_all.min() + 1e-10))
        ]
        values += values[:1]
        
        linewidth = 3 if method == recommended else 2
        alpha_val = 0.9 if method == recommended else 0.6
        ax8.plot(angles, values, 'o-', linewidth=linewidth, label=method, alpha=alpha_val)
        ax8.fill(angles, values, alpha=0.15)
    
    ax8.set_xticks(angles[:-1])
    ax8.set_xticklabels(categories, fontsize=10)
    ax8.set_ylim(0, 100)
    ax8.set_title('Top 5 Methods\n(Normalized Metrics)', fontweight='bold', fontsize=12, pad=20)
    ax8.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax8.grid(True)
    
    # Plot 9 position (3,3,9) remains empty for better spacing
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    if save_path:
        path2 = save_path.replace('.png', '_metrics.png')
        fig2.savefig(path2, dpi=150, bbox_inches='tight')
        print(f"✓ Figure 2 saved to: {path2}")
    
    plt.show()


def print_results_table(results: Dict[str, BaselineMetrics]) -> None:
    """Print formatted results table with peak recovery."""
    try:
        from tabulate import tabulate
        use_tabulate = True
    except ImportError:
        use_tabulate = False
    
    headers = ['Method', 'Flatness', 'Preservation', 'Edge Ratio', 'Peak Recovery', 'Time(ms)']
    
    rows = []
    for name, m in results.items():
        edge_flag = " ⚠️" if m.edge_stability_ratio > 2.0 else ""
        recovery_flag = ""
        if m.peak_recovery_rate < 95.0 or m.peak_recovery_rate > 105.0:
            recovery_flag = " ⚠️"
        
        rows.append([
            name,
            f"{m.baseline_flatness:.6f}",
            f"{m.peak_preservation:.4f}",
            f"{m.edge_stability_ratio:.2f}{edge_flag}",
            f"{m.peak_recovery_rate:.2f}%{recovery_flag}",
            f"{m.computation_time_ms:.3f}"
        ])
    
    print("\n" + "="*110)
    print("BASELINE CORRECTION COMPARISON RESULTS (with Synthetic Ground Truth Validation)")
    print("="*110)
    print("Edge Ratio:  <2.0 = Excellent | Peak Recovery: 95-105% = Accurate")
    print("="*110)
    
    if use_tabulate:
        print(tabulate(rows, headers=headers, tablefmt='grid'))
    else:
        col_widths = [20, 12, 14, 12, 16, 12]
        header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
        print(header_line)
        print("-" * len(header_line))
        for row in rows:
            print(" | ".join(str(c).ljust(w) for c, w in zip(row, col_widths)))
    """Print formatted results table with edge stability."""
    try:
        from tabulate import tabulate
        use_tabulate = True
    except ImportError:
        use_tabulate = False
    
    headers = ['Method', 'Flatness', 'Preservation', 'Distortion%', 'Edge Ratio', 'Time(ms)', 'Params']
    
    rows = []
    for name, m in results.items():
        # Flag edge instability
        edge_flag = " ⚠️" if m.edge_stability_ratio > 2.0 else ""
        rows.append([
            name,
            f"{m.baseline_flatness:.6f}",
            f"{m.peak_preservation:.4f}",
            f"{m.peak_distortion:.2f}%",
            f"{m.edge_stability_ratio:.2f}{edge_flag}",
            f"{m.computation_time_ms:.3f}",
            f"{m.parameter_complexity}"
        ])
    
    print("\n" + "="*100)
    print("BASELINE CORRECTION COMPARISON RESULTS (with Edge Stability Test)")
    print("="*100)
    print("Edge Stability Ratio: <2.0 = Excellent, 2.0-2.5 = Acceptable, >2.5 = Problematic")
    print("="*100)
    
    if use_tabulate:
        print(tabulate(rows, headers=headers, tablefmt='grid'))
    else:
        col_widths = [20, 12, 14, 14, 12, 12, 8]
        header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
        print(header_line)
        print("-" * len(header_line))
        for row in rows:
            print(" | ".join(str(c).ljust(w) for c, w in zip(row, col_widths)))

def save_recommendation_report(
    results: Dict[str, BaselineMetrics],
    recommended:  str,
    justification: str,
    filepath: str = 'baseline_recommendation.txt'
) -> None:
    """Save recommendation report to file."""
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("LIBS BASELINE CORRECTION RECOMMENDATION REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write("METHODOLOGY REFERENCE:\n")
        f.write("- AsLS: r=0.93, 19% RMSE improvement [221]\n")
        f.write("- Decision Rule: 'Best flatness + peak preservation balance'\n\n")
        
        f.write("RESULTS SUMMARY:\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Method':<25} {'Flatness':<12} {'Preservation':<14} {'Time(ms)':<12}\n")
        f.write("-"*70 + "\n")
        
        for name, m in sorted(results.items(), key=lambda x: x[1].baseline_flatness):
            f.write(f"{name:<25} {m.baseline_flatness: <12.6f} {m.peak_preservation:<14.4f} {m.computation_time_ms:<12.3f}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("RECOMMENDATION\n")
        f.write("="*70 + "\n\n")
        f.write(justification)
        f.write("\n\n")
        
        f.write("="*70 + "\n")
        f.write("ACTION REQUIRED\n")
        f.write("="*70 + "\n\n")
        f.write("Update libs_preprocessing.py:\n\n")
        
        if recommended.startswith('AsLS'):
            if 'default' in recommended:
                f.write("  # Use AsLS with default parameters (RECOMMENDED)\n")
                f.write("  lam = 1e6\n")
                f.write("  p = 0.01\n")
            elif 'smooth' in recommended:
                f.write("  # Use AsLS with smoother baseline\n")
                f.write("  lam = 1e7\n")
                f.write("  p = 0.01\n")
            elif 'aggressive' in recommended:
                f.write("  # Use AsLS with more aggressive asymmetry\n")
                f.write("  lam = 1e6\n")
                f.write("  p = 0.001\n")
        elif recommended.startswith('Polynomial'):
            degree = recommended.split('_')[1]
            f.write(f"  # Use Polynomial baseline (degree={degree})\n")
            f.write(f"  # Lowest computational cost for MCU\n")
            f.write(f"  degree = {degree}\n")
        elif recommended.startswith('SNIP'):
            iterations = recommended.split('_')[1]
            f.write(f"  # Use SNIP baseline (iterations={iterations})\n")
            f.write(f"  iterations = {iterations}\n")
        else:
            f.write(f"  # Use {recommended} method\n")
            f.write(f"  # See implementation in baseline_comparison.py\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("IMPLEMENTATION NOTES\n")
        f.write("="*70 + "\n\n")
        f.write("1. Copy the recommended baseline function from baseline_comparison.py\n")
        f.write("2. Replace the asls_baseline() function in libs_preprocessing.py\n")
        f.write("3. Update parameters in apply_baseline_correction()\n")
        f.write("4. Run filter_comparison.py to select optimal smoothing filter\n")
        f.write("5. Proceed with full preprocessing pipeline\n\n")
    
    print(f"\n✓ Recommendation report saved to: {filepath}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__": 
    print("="*70)
    print("LIBS BASELINE CORRECTION METHOD COMPARISON")
    print("Context Feeder [221]: AsLS r=0.93, 19% RMSE improvement")
    print("="*70)
    
    USE_REAL_DATA = True
    
    REAL_DATA_CONFIG = {
        "filepath": r"D:\Ryan\UM\Academic\Y4S1\KIE4002 FINAL YEAR PROJECT\Irish_Soil_DataSet\Irish_Soil_DataSet2.csv",
        "wavelengths_in_first_column": True,
        "sample_names_in_first_row": True,
        "wavelength_range": (197.001, 973.5891),
    }
    
    # Load data
    if USE_REAL_DATA: 
        try:
            from libs_preprocessing import load_irish_soil_data
            
            spectra_raw, sample_names, wavelengths = load_irish_soil_data(
                REAL_DATA_CONFIG["filepath"]
            )
            
        except Exception as e:
            print(f"\n⚠️ Could not load real data: {e}")
            print("Falling back to synthetic data.. .\n")
            USE_REAL_DATA = False
    
    if not USE_REAL_DATA: 
        print("\nGenerating synthetic data...")
        np.random.seed(42)
        n_samples, n_features = 324, 12981
        wavelengths = np.linspace(197.001, 973.5891, n_features)
        sample_names = np.array([f"Sample_{i+1}" for i in range(n_samples)])
        
        spectra_raw = np.zeros((n_samples, n_features))
        for i in range(n_samples):
            baseline = 1000 * np.exp(-0.005 * (wavelengths - 200))
            peak_positions = [253.5, 255.3, 742.4, 766.5]
            for pos in peak_positions:
                intensity = np.random.uniform(500, 2000)
                peak = intensity * np.exp(-0.5 * ((wavelengths - pos) / 0.5)**2)
                spectra_raw[i] += peak
            spectra_raw[i] += baseline
            spectra_raw[i] += np.random.normal(0, 50, n_features)
    
    # Run comparison for BOTH objectives (if classification enabled)
    if ENABLE_CLASSIFICATION and sample_names is not None:
        print("\n" + "="*70)
        print("COMPARISON 1: NUTRIENT CLASSIFICATION")
        print("="*70)
        results_nutrient = compare_baseline_methods(
            spectra_raw, wavelengths, sample_names,
            objective='nutrient', verbose=True
        )
        
        print("\n" + "="*70)
        print("COMPARISON 2: CONTAMINANT DETECTION")
        print("="*70)
        results_contaminant = compare_baseline_methods(
            spectra_raw, wavelengths, sample_names,
            objective='contaminant', verbose=True
        )
        
        # Generate recommendations for BOTH
        print("\n" + "="*70)
        print("NUTRIENT CLASSIFICATION RECOMMENDATION")
        print("="*70)
        recommended_nut, justification_nut = generate_recommendation(results_nutrient)
        print(justification_nut)
        
        print("\n" + "="*70)
        print("CONTAMINANT DETECTION RECOMMENDATION")
        print("="*70)
        recommended_con, justification_con = generate_recommendation(results_contaminant)
        print(justification_con)
        
        # Save both reports
        save_recommendation_report(results_nutrient, recommended_nut, justification_nut,
                                  'baseline_recommendation_nutrient.txt')
        save_recommendation_report(results_contaminant, recommended_con, justification_con,
                                  'baseline_recommendation_contaminant.txt')
        
        # Visualize nutrient results
        plot_baseline_comparison(spectra_raw[0], wavelengths, results_nutrient,
                                recommended_nut, save_path='baseline_comparison_nutrient.png')
        
    else:
        # Fallback:  spectral quality only
        print("\n⚠️ Running spectral quality comparison only")
        results = compare_baseline_methods(spectra_raw, wavelengths, verbose=True)
        
        print_results_table(results)
        recommended, justification = generate_recommendation(results)
        
        print("\n" + "="*70)
        print("RECOMMENDATION")
        print("="*70)
        print(justification)
        
        save_recommendation_report(results, recommended, justification)
        plot_baseline_comparison(spectra_raw[0], wavelengths, results, recommended,
                                save_path='baseline_comparison. png')
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1. Review recommendation reports")
    print("2. Update libs_preprocessing.py with optimal baseline method")
    print("3. Proceed to Phase 2: Feature Selection")
    print("="*70)
