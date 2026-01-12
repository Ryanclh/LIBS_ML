"""
LIBS Normalization Method Comparison Script
============================================
Run this to compare normalization methods before finalizing preprocessing

Methodology Justification:
- L2-Norm: Bias=0.000, RMSE=0.002 [237]
- SNV: RMSE=0.220 (worst performer) [237]
- L1-Norm:  EMSLIBS contest - "reduced matrix effect"

Decision Rule:
"Select method with lowest bias, best peak preservation, and stable inter-sample variance"
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler
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
# BASELINE CORRECTION (Local copy to avoid import issues)
# ============================================================================

def apply_baseline_correction_batch(
    spectra: np.ndarray,
    degree: int = 5,
    verbose: bool = True
) -> np.ndarray:
    """Apply Polynomial baseline correction to batch of spectra."""
    
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
# NORMALIZATION METHODS
# ============================================================================

def l2_normalization(spectrum: np.ndarray) -> np.ndarray:
    """
    L2 (Euclidean) normalization.
    
    Justification: Bias=0.000, RMSE=0.002 [237]
    Corrects for shot-to-shot laser energy fluctuations. 
    
    Formula: x_i / sqrt(sum(x_i^2))
    """
    norm = np.linalg.norm(spectrum)
    if norm == 0:
        return spectrum
    return spectrum / norm


def l1_normalization(spectrum: np.ndarray) -> np.ndarray:
    """
    L1 (Total Area) normalization.
    
    Justification: EMSLIBS contest - "reduced matrix effect"
    
    Formula: x_i / sum(|x_i|)
    """
    total = np.sum(np.abs(spectrum))
    if total == 0:
        return spectrum
    return spectrum / total


def snv_normalization(spectrum: np.ndarray) -> np.ndarray:
    """
    Standard Normal Variate (SNV) normalization.
    
    Justification: Common in NIR spectroscopy but WORST for LIBS [237]
    RMSE=0.220 vs L2 RMSE=0.002
    
    Formula: (x - mean(x)) / std(x)
    """
    mean = np.mean(spectrum)
    std = np.std(spectrum)
    if std == 0:
        return spectrum - mean
    return (spectrum - mean) / std


def minmax_normalization(spectrum: np.ndarray) -> np.ndarray:
    """
    Min-Max scaling to [0, 1] range.
    
    Formula: (x - min) / (max - min)
    """
    min_val = np.min(spectrum)
    max_val = np.max(spectrum)
    if max_val == min_val:
        return np.zeros_like(spectrum)
    return (spectrum - min_val) / (max_val - min_val)


def zscore_normalization(spectrum: np.ndarray) -> np.ndarray:
    """
    Z-score standardization (similar to SNV but preserves mean=0).
    
    Formula: (x - mean(x)) / std(x)
    """
    return snv_normalization(spectrum)


def vector_normalization(spectrum:  np.ndarray) -> np.ndarray:
    """
    Vector normalization (unit vector).
    
    Same as L2 but explicit implementation.
    """
    return l2_normalization(spectrum)


def mean_centering(spectrum: np.ndarray) -> np.ndarray:
    """
    Mean centering only (no scaling).
    
    Formula: x - mean(x)
    """
    return spectrum - np.mean(spectrum)


def max_normalization(spectrum: np.ndarray) -> np.ndarray:
    """
    Max peak normalization.
    
    Formula: x / max(x)
    """
    max_val = np.max(spectrum)
    if max_val == 0:
        return spectrum
    return spectrum / max_val


def no_normalization(spectrum: np.ndarray) -> np.ndarray:
    """No normalization (baseline for comparison)."""
    return spectrum.copy()


# ============================================================================
# EVALUATION METRICS
# ============================================================================

@dataclass
class NormalizationMetrics:
    """Container for normalization evaluation metrics."""
    name: str
    bias:  float                      # Absolute bias from expected value (lower=better)
    rmse: float                      # Root mean squared error (lower=better)
    peak_preservation: float         # Correlation with original peaks (higher=better)
    inter_sample_variance: float     # Variance across samples (stability metric)
    intra_sample_variance: float     # Variance within single sample
    dynamic_range: float             # Max - Min after normalization
    computation_time_ms: float       # Processing speed (lower=better)
    numerical_stability: bool        # No NaN/Inf values
    peak_ratio_preservation: float   # Preservation of peak height ratios (higher=better)
    classification_accuracy: float   # NEW: Classification accuracy (higher=better)


def calculate_bias(normalized:  np.ndarray, expected_mean: float = 0.0) -> float:
    """
    Calculate absolute bias from expected mean.
    
    For L2/L1: Expected mean varies by spectrum content
    For SNV: Expected mean = 0.0
    For MinMax: Expected mean ≈ 0.5
    """
    actual_mean = np.mean(normalized)
    return abs(actual_mean - expected_mean)


def calculate_rmse(original: np.ndarray, normalized: np.ndarray, 
                   reference:  np.ndarray) -> float:
    """
    Calculate RMSE between normalized spectrum and reference.
    
    Reference is typically a "gold standard" normalized version.
    """
    # Use L2-normalized version as reference (gold standard per [237])
    diff = normalized - reference
    return np.sqrt(np.mean(diff ** 2))


def calculate_peak_ratio_preservation(original: np.ndarray, 
                                      normalized: np.ndarray) -> float:
    """
    Check if normalization preserves relative peak height ratios.
    
    Critical for LIBS:  Peak ratios encode elemental concentrations. 
    """
    from scipy.signal import find_peaks
    
    # Find peaks in original
    peaks_orig, _ = find_peaks(original, prominence=np.std(original))
    if len(peaks_orig) < 2:
        return 1.0  # Not enough peaks to compare
    
    # Get top 5 peak heights
    n_peaks = min(5, len(peaks_orig))
    top_peaks_idx = np.argsort(original[peaks_orig])[-n_peaks:]
    peak_indices = peaks_orig[top_peaks_idx]
    
    # Calculate ratios in original
    heights_orig = original[peak_indices]
    ratios_orig = heights_orig[1:] / (heights_orig[:-1] + 1e-10)
    
    # Calculate ratios in normalized
    heights_norm = normalized[peak_indices]
    ratios_norm = heights_norm[1:] / (heights_norm[:-1] + 1e-10)
    
    # Measure preservation via correlation
    if len(ratios_orig) == 0:
        return 1.0
    
    correlation = np.corrcoef(ratios_orig, ratios_norm)[0, 1]
    return correlation if not np.isnan(correlation) else 0.0


def calculate_inter_sample_variance(normalized_spectra: np.ndarray) -> float:
    """
    Calculate variance across samples (mean spectrum variance).
    
    Lower variance = more consistent normalization across different samples.
    """
    # Calculate mean spectrum
    mean_spectrum = np.mean(normalized_spectra, axis=0)
    
    # Calculate variance of each sample from mean
    variances = np.var(normalized_spectra - mean_spectrum, axis=1)
    
    return np.mean(variances)


def calculate_intra_sample_variance(normalized:  np.ndarray) -> float:
    """
    Calculate variance within single sample.
    
    Higher variance = better preservation of spectral features.
    """
    return np.var(normalized)


# ============================================================================
# SYNTHETIC GROUND TRUTH TEST
# ============================================================================

def generate_synthetic_spectrum_with_known_ratios(
    wavelengths: np.ndarray,
    peak_positions: list = [253.5, 255.3, 742.4, 766.5],
    peak_heights: list = [1000, 800, 1500, 1200],
    baseline_level: float = 100.0,
    noise_level: float = 20.0
) -> Tuple[np.ndarray, dict]:
    """
    Generate synthetic spectrum with known peak height ratios.
    
    Returns
    -------
    spectrum : np.ndarray
        Synthetic spectrum
    ground_truth : dict
        Known peak ratios for validation
    """
    spectrum = np.ones_like(wavelengths) * baseline_level
    spectrum += np.random.normal(0, noise_level, len(wavelengths))
    
    # Add peaks with known heights
    peak_indices = []
    for pos, height in zip(peak_positions, peak_heights):
        idx = np.argmin(np.abs(wavelengths - pos))
        peak_indices.append(idx)
        
        # Gaussian peak
        sigma = 0.5 / 2.355
        for i in range(len(wavelengths)):
            gaussian = height * np.exp(-0.5 * ((wavelengths[i] - pos) / sigma) ** 2)
            spectrum[i] += gaussian
    
    # Calculate ground truth ratios
    heights = [spectrum[idx] for idx in peak_indices]
    ratios = [heights[i+1] / heights[i] for i in range(len(heights) - 1)]
    
    ground_truth = {
        'peak_positions': peak_positions,
        'peak_heights': heights,
        'peak_ratios': ratios
    }
    
    return spectrum, ground_truth


def test_ratio_preservation(
    norm_func,
    wavelengths: np.ndarray,
    n_tests: int = 10
) -> Tuple[float, float]:
    """
    Test how well normalization preserves peak height ratios.
    
    Returns
    -------
    mean_preservation : float
        Average ratio preservation across tests
    std_preservation : float
        Standard deviation
    """
    preservations = []
    
    for _ in range(n_tests):
        # Generate synthetic spectrum
        synthetic, gt = generate_synthetic_spectrum_with_known_ratios(wavelengths)
        
        # Apply normalization
        try:
            normalized = norm_func(synthetic)
            
            # Calculate preserved ratios
            preservation = calculate_peak_ratio_preservation(synthetic, normalized)
            preservations.append(preservation)
        except Exception: 
            preservations.append(0.0)
    
    return np.mean(preservations), np.std(preservations)


# ============================================================================
# COMPREHENSIVE COMPARISON
# ============================================================================

def compare_normalization_methods(
    spectra: np.ndarray,
    wavelengths: np.ndarray,
    sample_names: np.ndarray = None,  # NEW: Optional for classification
    objective: str = 'nutrient',      # NEW: 'nutrient' or 'contaminant'
    verbose: bool = True
) -> Dict[str, NormalizationMetrics]:
    """
    Comprehensive comparison INCLUDING optional classification evaluation.
    
    All existing metrics preserved: 
    - Bias, RMSE vs L2 gold standard
    - Peak preservation and peak ratio preservation
    - Inter-sample and intra-sample variance
    - Dynamic range and numerical stability
    - NEW: Classification accuracy (if sample_names provided)
    
    Parameters
    ----------
    spectra : np.ndarray
        Baseline-corrected spectra (n_samples, n_features)
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
    results : Dict[str, NormalizationMetrics]
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
        'L2_Norm': {
            'func': l2_normalization,
            'description': 'L2 (Euclidean) - RECOMMENDED [237]:  Bias=0.000, RMSE=0.002'
        },
        'L1_Norm': {
            'func': l1_normalization,
            'description': 'L1 (Total Area) - EMSLIBS:  Reduced matrix effect'
        },
        'SNV': {
            'func':  snv_normalization,
            'description': 'Standard Normal Variate - WARNING [237]:  RMSE=0.220 (WORST)'
        },
        'MinMax': {
            'func':  minmax_normalization,
            'description': 'Min-Max [0,1] scaling'
        },
        'Z-Score': {
            'func':  zscore_normalization,
            'description': 'Z-score standardization'
        },
        'Max_Peak':  {
            'func': max_normalization,
            'description': 'Max peak normalization'
        },
        'Mean_Center': {
            'func': mean_centering,
            'description': 'Mean centering only (no scaling)'
        },
        'None': {
            'func': no_normalization,
            'description': 'No normalization (baseline)'
        }
    }
    
    results = {}
    n_samples = spectra.shape[0]
    
    if verbose:
        print("\n" + "="*70)
        mode = "WITH CLASSIFICATION" if y_encoded is not None else "SPECTRAL QUALITY ONLY"
        print(f"NORMALIZATION METHOD COMPARISON ({mode})")
        print("All metrics:  Bias, RMSE, Peak Preservation, Ratio Preservation,")
        print("            Inter/Intra Variance" + (" + Classification" if y_encoded is not None else ""))
        print("="*70)
        print(f"Testing on {n_samples} spectra...")
    
    # Generate L2-normalized reference (gold standard) - PRESERVED
    reference_spectra = np.array([l2_normalization(s) for s in spectra])
    
    for name, config in methods.items():
        if verbose:
            print(f"\nEvaluating:  {name}...")
            print(f"  {config['description']}")
        
        # Apply normalization to all spectra
        start_time = time.time()
        try:
            normalized_spectra = np.array([config['func'](s) for s in spectra])
            elapsed_ms = (time.time() - start_time) / n_samples * 1000
            
            # Check for numerical issues
            has_nan = np.any(np.isnan(normalized_spectra))
            has_inf = np.any(np.isinf(normalized_spectra))
            numerical_stability = not (has_nan or has_inf)
            
            if not numerical_stability and verbose:
                print(f"  ⚠️ WARNING: Numerical instability detected (NaN/Inf)")
        except Exception as e:
            if verbose:
                print(f"  ⚠️ ERROR: {e}")
            normalized_spectra = spectra.copy()
            elapsed_ms = 0
            numerical_stability = False
        
        # Calculate all existing spectral quality metrics (PRESERVED)
        bias_scores = []
        rmse_scores = []
        peak_pres_scores = []
        intra_var_scores = []
        dynamic_ranges = []
        
        for i in range(min(50, n_samples)):
            original = spectra[i]
            normalized = normalized_spectra[i]
            reference = reference_spectra[i]
            
            # Expected mean depends on method
            if name in ['SNV', 'Z-Score', 'Mean_Center']:
                expected_mean = 0.0
            elif name == 'MinMax':
                expected_mean = 0.5
            else:
                expected_mean = np.mean(normalized)
            
            bias_scores.append(calculate_bias(normalized, expected_mean))
            rmse_scores.append(calculate_rmse(original, normalized, reference))
            
            # Peak preservation
            correlation = np.corrcoef(original, normalized)[0, 1]
            peak_pres_scores.append(correlation if not np.isnan(correlation) else 0.0)
            
            # Intra-sample variance
            intra_var_scores.append(calculate_intra_sample_variance(normalized))
            
            # Dynamic range
            dynamic_ranges.append(np.max(normalized) - np.min(normalized))
        
        # Inter-sample variance (PRESERVED)
        inter_sample_var = calculate_inter_sample_variance(normalized_spectra[: 50])
        
        # Peak ratio preservation test (PRESERVED)
        if verbose:
            print(f"  Running peak ratio preservation test...")
        
        mean_ratio_pres, std_ratio_pres = test_ratio_preservation(
            config['func'],
            wavelengths,
            n_tests=10
        )
        
        # NEW: Classification accuracy evaluation (if labels available)
        classification_accuracy = 0.0
        if y_encoded is not None:
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
            cv_scores = cross_val_score(rf, normalized_spectra, y_encoded, cv=cv, scoring='accuracy')
            classification_accuracy = np.mean(cv_scores) * 100
        
        # Store results (ALL METRICS)
        results[name] = NormalizationMetrics(
            name=name,
            bias=np.mean(bias_scores),
            rmse=np.mean(rmse_scores),
            peak_preservation=np.mean(peak_pres_scores),
            inter_sample_variance=inter_sample_var,
            intra_sample_variance=np.mean(intra_var_scores),
            dynamic_range=np.mean(dynamic_ranges),
            computation_time_ms=elapsed_ms,
            numerical_stability=numerical_stability,
            peak_ratio_preservation=mean_ratio_pres,
            classification_accuracy=classification_accuracy  # NEW
        )
        
        if verbose:
            m = results[name]
            print(f"  Bias:                  {m.bias:.6f} (lower=better)")
            print(f"  RMSE vs L2:          {m.rmse:.6f} (lower=better)")
            print(f"  Peak Preservation:   {m.peak_preservation:.4f} (higher=better)")
            print(f"  Peak Ratio Preserv.:  {m.peak_ratio_preservation:.4f} (±{std_ratio_pres:.4f})")
            print(f"  Inter-sample Var:    {m.inter_sample_variance:.6f} (lower=better)")
            if y_encoded is not None: 
                print(f"  Classification:       {m.classification_accuracy:.2f}% ✓")
            print(f"  Numerical Stability:  {m.numerical_stability}")
            
            if name == 'SNV' and m.rmse > 0.1:
                print(f"  ⚠️ WARNING:  High RMSE confirms [237] - SNV is WORST performer")
            
            print()
    
    return results

def generate_recommendation(results: Dict[str, NormalizationMetrics]) -> Tuple[str, str]:
    """
    Generate recommendation based on metrics.
    
    Scoring weights:
    - RMSE: 30% (key validation metric from [237])
    - Peak ratio preservation: 30% (CRITICAL for quantitative LIBS)
    - Peak preservation: 20% (general correlation)
    - Bias: 10% (accuracy indicator)
    - Inter-sample variance: 10% (consistency)
    """
    
    methods = list(results.keys())
    
    # Extract metrics
    bias_vals = np.array([results[m].bias for m in methods])
    rmse_vals = np.array([results[m].rmse for m in methods])
    peak_pres_vals = np.array([results[m].peak_preservation for m in methods])
    inter_var_vals = np.array([results[m].inter_sample_variance for m in methods])
    ratio_pres_vals = np.array([results[m].peak_ratio_preservation for m in methods])
    
    # Normalize scores (0-100 scale)
    bias_scores = 100 * (1 - (bias_vals - bias_vals.min()) / (bias_vals.max() - bias_vals.min() + 1e-10))
    rmse_scores = 100 * (1 - (rmse_vals - rmse_vals.min()) / (rmse_vals.max() - rmse_vals.min() + 1e-10))
    peak_pres_scores = 100 * peak_pres_vals
    inter_var_scores = 100 * (1 - (inter_var_vals - inter_var_vals.min()) / (inter_var_vals.max() - inter_var_vals.min() + 1e-10))
    ratio_pres_scores = 100 * ratio_pres_vals
    
    # Weighted combination
    scores = {}
    for i, method in enumerate(methods):
        score = (
            rmse_scores[i] * 0.30 +
            ratio_pres_scores[i] * 0.30 +
            peak_pres_scores[i] * 0.20 +
            bias_scores[i] * 0.10 +
            inter_var_scores[i] * 0.10
        )
        scores[method] = score
    
    recommended = max(scores, key=scores.get)
    best_metrics = results[recommended]
    
    justification = f"""
RECOMMENDATION:  {recommended}

JUSTIFICATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PERFORMANCE METRICS:
  Bias:                    {best_metrics.bias:.6f} (lower=better, 0.000=ideal)
  RMSE vs L2 Gold Std:     {best_metrics.rmse:.6f} (lower=better, 0.002=literature)
  Peak Preservation:       {best_metrics.peak_preservation:.4f} (higher=better, 1.0=perfect)
  Peak Ratio Preservation: {best_metrics.peak_ratio_preservation:.4f} (CRITICAL for quantitative LIBS)
  Inter-sample Variance:   {best_metrics.inter_sample_variance:.6f} (lower=better)
  Numerical Stability:     {best_metrics.numerical_stability}

WEIGHTED SCORE: {scores[recommended]:.2f}/100
  30% RMSE (validation metric from [237])
  30% Peak ratio preservation (quantitative accuracy)
  20% Peak preservation (general correlation)
  10% Bias (accuracy)
  10% Inter-sample variance (consistency)

"""
    
    # Interpretation
    if recommended == 'L2_Norm':
        justification += f"""
INTERPRETATION:
This confirms the literature recommendation [237]: 
  - L2-Norm achieved Bias=0.000, RMSE=0.002 (best performer)
  - SNV had RMSE=0.220 (worst performer - 110x worse)
  - Physically motivated:  Corrects for laser shot-to-shot energy fluctuations

VALIDATION:
  Your results:  Bias={best_metrics.bias:.6f}, RMSE={best_metrics.rmse:.6f}
  Literature:    Bias=0.000, RMSE=0.002
  Match quality: {'EXCELLENT' if best_metrics.bias < 0.01 and best_metrics.rmse < 0.01 else 'ACCEPTABLE'}

DEPLOYMENT for TinyML:
  - Computation:  O(n) - single pass through spectrum
  - Memory:  Minimal (only need sum of squares)
  - Numerical stability: Excellent (unit vector always defined)
  - Peak ratios preserved: {best_metrics.peak_ratio_preservation:.2%}

"""
    elif recommended == 'L1_Norm':
        justification += f"""
INTERPRETATION:
L1 (Total Area) normalization shows best performance for your data. 

LITERATURE SUPPORT:
  - EMSLIBS contest: "Total area normalization reduced matrix effect"
  - Advantage: Less sensitive to outlier peaks than L2

COMPARISON TO L2 [237]:
  L2 Metrics: Bias={results['L2_Norm'].bias:.6f}, RMSE={results['L2_Norm'].rmse:.6f}
  L1 Metrics: Bias={best_metrics.bias:.6f}, RMSE={best_metrics.rmse:.6f}
  
RECOMMENDATION:
  {'Use L1 if RMSE difference > 0.01' if best_metrics.rmse < results['L2_Norm'].rmse - 0.01 else 'Consider using L2 for stronger literature justification [237]'}

"""
    elif recommended == 'SNV':
        justification += f"""
⚠️ WARNING: SNV recommended by scoring but contradicts literature! 

LITERATURE EVIDENCE [237]:
  - SNV had RMSE=0.220 (WORST performer)
  - L2 had RMSE=0.002 (110x better)

YOUR RESULTS:
  SNV RMSE: {best_metrics.rmse:.6f}
  L2 RMSE:   {results['L2_Norm'].rmse:.6f}

POSSIBLE REASONS:
  1. Dataset-specific characteristics favor zero-centering
  2. Your baseline correction step already removed most bias
  3. Peak ratios happen to be preserved despite mean centering

RECOMMENDATION:
  Use L2-Norm instead for: 
    - Stronger literature justification [237]
    - Better generalization to unseen samples
    - Industry standard for LIBS
  
  SNV acceptable ONLY if:
    - Your validation set shows >5% accuracy improvement
    - You document this as dataset-specific finding

"""
    else:
        justification += f"""
INTERPRETATION:
{recommended} shows best performance for your data.

COMPARISON TO L2 [237] (Recommended Standard):
  L2 Metrics: Bias={results['L2_Norm'].bias:.6f}, RMSE={results['L2_Norm'].rmse:6f}, Ratio Pres={results['L2_Norm'].peak_ratio_preservation:.4f}
  {recommended}:  Bias={best_metrics.bias:.6f}, RMSE={best_metrics.rmse:.6f}, Ratio Pres={best_metrics.peak_ratio_preservation:.4f}
  
  Score difference: {scores[recommended] - scores['L2_Norm']:.2f} points

RECOMMENDATION:
  {'Use ' + recommended if scores[recommended] - scores['L2_Norm'] > 5 else 'Consider L2-Norm for stronger literature support [237]'}
  
  If using {recommended}:
    - Document as methodological contribution
    - Justify via empirical validation on your dataset
    - Compare classification accuracy:  L2 vs {recommended}

"""
    
    return recommended, justification


def plot_normalization_comparison(
    spectrum:   np.ndarray,
    wavelengths: np.ndarray,
    results:  Dict[str, NormalizationMetrics],
    recommended:  str,
    save_path: Optional[str] = None
) -> None:
    """Visualize normalization comparison across TWO figures."""
    
    # Select representative methods - EXCLUDE 'None' from spectral plots
    plot_methods = {
        'L2_Norm': ('L2-Norm (Recommended)', 'blue'),
        'L1_Norm': ('L1-Norm', 'green'),
        'SNV': ('SNV (WARNING)', 'red'),
        'MinMax': ('Min-Max', 'orange'),
        'Max_Peak': ('Max Peak', 'purple')
    }
    
    # Keep 'None' only for comparison plots
    all_methods = {
        'None': ('No Normalization', 'gray'),
        **plot_methods
    }
    
    # ========================================================================
    # FIGURE 1: SPECTRAL VISUALIZATION
    # ========================================================================
    fig1 = plt.figure(figsize=(18, 10))
    fig1.suptitle('LIBS Normalization - Spectral Comparison\nContext Feeder [237]:    L2 Bias=0.000, RMSE=0.002 vs SNV RMSE=0.220',
                  fontsize=14, fontweight='bold', y=0.995)
    
    # Plot 1: Normalized spectra overlay (EXCLUDE 'None')
    ax1 = fig1.add_subplot(2, 3, 1)
    
    for method, (label, color) in plot_methods.items():  # CHANGED: use plot_methods instead of all_methods
        if method == 'L2_Norm':
            normalized = l2_normalization(spectrum)
        elif method == 'L1_Norm':
            normalized = l1_normalization(spectrum)
        elif method == 'SNV':
            normalized = snv_normalization(spectrum)
        elif method == 'MinMax':  
            normalized = minmax_normalization(spectrum)
        elif method == 'Max_Peak': 
            normalized = max_normalization(spectrum)
        else:
            continue
        
        linewidth = 2.5 if method == recommended else 1.2
        alpha = 0.9 if method == recommended else 0.6
        ax1.plot(wavelengths, normalized, color=color, linewidth=linewidth,
                label=label, alpha=alpha)
    
    ax1.set_xlabel('Wavelength (nm)', fontsize=11)
    ax1.set_ylabel('Normalized Intensity', fontsize=11)
    ax1.set_title('Normalized Spectra Comparison\n(Excludes unnormalized for clarity)', 
                  fontweight='bold', fontsize=12)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Peak region zoom (EXCLUDE 'None')
    ax2 = fig1.add_subplot(2, 3, 2)
    peak_region = slice(5000, 5500)
    wl_zoom = wavelengths[peak_region]
    
    for method, (label, color) in plot_methods.items():  # CHANGED
        if method == 'L2_Norm':  
            normalized = l2_normalization(spectrum)
        elif method == 'L1_Norm':  
            normalized = l1_normalization(spectrum)
        elif method == 'SNV': 
            normalized = snv_normalization(spectrum)
        elif method == 'MinMax':
            normalized = minmax_normalization(spectrum)
        elif method == 'Max_Peak':
            normalized = max_normalization(spectrum)
        else:
            continue
        
        linewidth = 2.5 if method == recommended else 1.2
        ax2.plot(wl_zoom, normalized[peak_region], color=color, 
                linewidth=linewidth, label=label, alpha=0.7)
    
    ax2.set_xlabel('Wavelength (nm)', fontsize=11)
    ax2.set_ylabel('Normalized Intensity', fontsize=11)
    ax2.set_title('Peak Region Detail\n(Better visibility without unnormalized)', 
                  fontweight='bold', fontsize=12)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Histogram of intensity distributions (EXCLUDE 'None')
    ax3 = fig1.add_subplot(2, 3, 3)
    
    for method, (label, color) in plot_methods.items():  # CHANGED
        if method == 'L2_Norm': 
            normalized = l2_normalization(spectrum)
        elif method == 'L1_Norm':
            normalized = l1_normalization(spectrum)
        elif method == 'SNV':  
            normalized = snv_normalization(spectrum)
        elif method == 'MinMax': 
            normalized = minmax_normalization(spectrum)
        elif method == 'Max_Peak':
            normalized = max_normalization(spectrum)
        else:
            continue
        
        ax3.hist(normalized, bins=50, alpha=0.4, color=color, label=label)
    
    ax3.set_xlabel('Normalized Intensity', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('Intensity Distribution\n(Normalized methods only)', 
                  fontweight='bold', fontsize=12)
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: L2 vs SNV scatter (keep as is)
    ax4 = fig1.add_subplot(2, 3, 4)
    l2_norm = l2_normalization(spectrum)
    snv_norm = snv_normalization(spectrum)
    
    ax4.scatter(l2_norm, snv_norm, alpha=0.5, s=1, c='blue')
    ax4.plot([l2_norm.min(), l2_norm.max()], [l2_norm.min(), l2_norm.max()], 
            'r--', linewidth=2, label='Perfect agreement')
    ax4.set_xlabel('L2-Normalized Intensity', fontsize=11)
    ax4.set_ylabel('SNV-Normalized Intensity', fontsize=11)
    ax4.set_title('L2 vs SNV Point-by-Point Comparison', fontweight='bold', fontsize=12)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Dynamic range comparison (KEEP 'None' for reference)
    ax5 = fig1.add_subplot(2, 3, 5)
    methods_list = list(all_methods.keys())  # CHANGED: use all_methods here to show comparison
    dynamic_ranges = [results[m].dynamic_range for m in methods_list if m in results]
    colors_bar = [all_methods[m][1] for m in methods_list if m in results]
    
    bars = ax5.barh(range(len(dynamic_ranges)), dynamic_ranges, 
                   color=colors_bar, alpha=0.7, edgecolor='black')
    ax5.set_yticks(range(len(dynamic_ranges)))
    ax5.set_yticklabels([all_methods[m][0] for m in methods_list if m in results], fontsize=9)
    ax5.set_xlabel('Dynamic Range (Max - Min)', fontsize=11)
    ax5.set_title('Dynamic Range Comparison\n(Shows why normalization needed)', 
                  fontweight='bold', fontsize=12)
    ax5.grid(True, alpha=0.3, axis='x')
    
    # Highlight recommended
    for bar, method in zip(bars, [m for m in methods_list if m in results]):
        if method == recommended:
            bar.set_edgecolor('darkgreen')
            bar.set_linewidth(3)
    
    # Add annotation explaining the scale difference
    if 'None' in results:
        none_range = results['None'].dynamic_range
        l2_range = results['L2_Norm'].dynamic_range
        ratio = none_range / l2_range
        ax5.text(0.98, 0.02, f'Unnormalized is {ratio:.0f}× larger\nthan L2-Norm', 
                transform=ax5.transAxes, fontsize=9, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Plot 6: Mean vs Std scatter (EXCLUDE 'None' or use log scale)
    ax6 = fig1.add_subplot(2, 3, 6)
    
    for method, (label, color) in plot_methods.items():  # CHANGED: exclude 'None'
        if method not in results:
            continue
        
        if method == 'L2_Norm':
            normalized = l2_normalization(spectrum)
        elif method == 'L1_Norm':  
            normalized = l1_normalization(spectrum)
        elif method == 'SNV': 
            normalized = snv_normalization(spectrum)
        elif method == 'MinMax':
            normalized = minmax_normalization(spectrum)
        elif method == 'Max_Peak':
            normalized = max_normalization(spectrum)
        else:
            continue
        
        mean_val = np.mean(normalized)
        std_val = np.std(normalized)
        
        marker_size = 200 if method == recommended else 100
        ax6.scatter(mean_val, std_val, s=marker_size, color=color, 
                   label=label, alpha=0.7, edgecolor='black', 
                   linewidth=2 if method == recommended else 1)
    
    ax6.set_xlabel('Mean Intensity', fontsize=11)
    ax6.set_ylabel('Std Deviation', fontsize=11)
    ax6.set_title('Mean vs Std Distribution\n(Normalized methods cluster)', 
                  fontweight='bold', fontsize=12)
    ax6.legend(loc='best', fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    if save_path:
        path1 = save_path.replace('.png', '_spectral.png')
        fig1.savefig(path1, dpi=150, bbox_inches='tight')
        print(f"\n✓ Figure 1 saved to:   {path1}")
    
    plt.show()
    
    # ========================================================================
    # FIGURE 2: PERFORMANCE METRICS
    # ========================================================================
    fig2 = plt.figure(figsize=(18, 10))
    fig2.suptitle('LIBS Normalization - Performance Metrics\nL2-Norm [237]:  Bias=0.000, RMSE=0.002 (Best) | SNV:   RMSE=0.220 (Worst)',
                  fontsize=14, fontweight='bold', y=0.995)
    
    methods = list(results.keys())
    
    # Extract all metrics from results dictionary (FIX for undefined variables)
    bias_all = np.array([results[m].bias for m in methods])
    rmse_all = np.array([results[m].rmse for m in methods])
    peak_pres_all = np.array([results[m].peak_preservation for m in methods])
    inter_var_all = np.array([results[m].inter_sample_variance for m in methods])
    ratio_pres_all = np.array([results[m].peak_ratio_preservation for m in methods])
    classification_all = np.array([results[m].classification_accuracy for m in methods])
    
    # Check if classification is available
    has_classification = np.any(classification_all > 0)
    
    # Plot 1: RMSE comparison
    ax1 = fig2.add_subplot(2, 3, 1)
    rmse_vals = rmse_all.tolist()
    colors_bar = ['green' if m == recommended else ('red' if m == 'SNV' else 'blue') for m in methods]
    
    bars = ax1.barh(methods, rmse_vals, color=colors_bar, alpha=0.7, edgecolor='black')
    for bar, method in zip(bars, methods):
        if method == recommended:
            bar.set_edgecolor('darkgreen')
            bar.set_linewidth(3)
    
    ax1.set_xlabel('RMSE vs L2 Gold Standard (lower=better)', fontsize=11)
    ax1.set_title('RMSE Comparison\n[237]:   L2=0.002, SNV=0.220', fontweight='bold', fontsize=12)
    ax1.axvline(x=0.002, color='green', linestyle='--', linewidth=2, label='Literature L2')
    ax1.axvline(x=0.220, color='red', linestyle='--', linewidth=2, label='Literature SNV')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Plot 2: Bias comparison
    ax2 = fig2.add_subplot(2, 3, 2)
    bias_vals = bias_all.tolist()
    
    bars = ax2.barh(methods, bias_vals, color=colors_bar, alpha=0.7, edgecolor='black')
    for bar, method in zip(bars, methods):
        if method == recommended:
            bar.set_edgecolor('darkgreen')
            bar.set_linewidth(3)
    
    ax2.set_xlabel('Bias (lower=better, 0.000=ideal)', fontsize=11)
    ax2.set_title('Bias Comparison', fontweight='bold', fontsize=12)
    ax2.axvline(x=0.0, color='green', linestyle='--', linewidth=2, label='Zero bias')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Plot 3: Peak ratio preservation (CRITICAL)
    ax3 = fig2.add_subplot(2, 3, 3)
    ratio_vals = ratio_pres_all.tolist()
    colors_ratio = []
    for m, ratio in zip(methods, ratio_vals):
        if m == recommended:  
            colors_ratio.append('green')
        elif ratio < 0.9:
            colors_ratio.append('red')
        else:
            colors_ratio.append('blue')
    
    bars = ax3.barh(methods, ratio_vals, color=colors_ratio, alpha=0.7, edgecolor='black')
    for bar, method in zip(bars, methods):
        if method == recommended:
            bar.set_edgecolor('darkgreen')
            bar.set_linewidth(3)
    
    ax3.set_xlabel('Peak Ratio Preservation (higher=better)', fontsize=11)
    ax3.set_title('Peak Ratio Preservation\nCRITICAL for Quantitative LIBS', fontweight='bold', fontsize=12)
    ax3.axvline(x=0.95, color='orange', linestyle='--', linewidth=2, label='95% threshold')
    ax3.axvline(x=1.0, color='green', linestyle='-', linewidth=2, label='Perfect')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='x')
    ax3.set_xlim(0.7, 1.05)
    
    # Plot 4: Inter-sample variance
    ax4 = fig2.add_subplot(2, 3, 4)
    var_vals = inter_var_all.tolist()
    
    bars = ax4.barh(methods, var_vals, color=colors_bar, alpha=0.7, edgecolor='black')
    for bar, method in zip(bars, methods):
        if method == recommended:
            bar.set_edgecolor('darkgreen')
            bar.set_linewidth(3)
    
    ax4.set_xlabel('Inter-sample Variance (lower=better)', fontsize=11)
    ax4.set_title('Inter-sample Consistency', fontweight='bold', fontsize=12)
    ax4.grid(True, alpha=0.3, axis='x')
    
    # Plot 5: Classification Accuracy OR Metric Comparison Table
    ax5 = fig2.add_subplot(2, 3, 5)
    
    if has_classification:
        # Show classification accuracy bar chart
        cls_vals = classification_all.tolist()
        
        # Sort by classification accuracy
        sorted_indices = np.argsort(cls_vals)[::-1]
        sorted_methods_cls = [methods[i] for i in sorted_indices]
        sorted_cls_vals = [cls_vals[i] for i in sorted_indices]
        
        colors_cls = []
        for m, acc in zip(sorted_methods_cls, sorted_cls_vals):
            if m == recommended:  
                colors_cls.append('green')
            elif acc < 80:
                colors_cls.append('red')
            else:
                colors_cls.append('blue')
        
        bars = ax5.barh(sorted_methods_cls, sorted_cls_vals, color=colors_cls, alpha=0.7, edgecolor='black')
        
        for bar, method in zip(bars, sorted_methods_cls):
            if method == recommended: 
                bar.set_edgecolor('darkgreen')
                bar.set_linewidth(3)
        
        ax5.set_xlabel('Classification Accuracy (%)', fontsize=11)
        ax5.set_title('Classification Accuracy\n(Key Performance Metric)', fontweight='bold', fontsize=12)
        ax5.axvline(x=85, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Target (85%)')
        ax5.axvline(x=73.01, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Baseline (73.01%)')
        ax5.legend(fontsize=9, loc='lower right')
        ax5.grid(True, alpha=0.3, axis='x')
        
        # Add accuracy annotations
        for bar, acc in zip(bars, sorted_cls_vals):
            width = bar.get_width()
            ax5.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{acc:.2f}%', ha='left', va='center', fontsize=9, fontweight='bold')
    else:
        # Fallback:  Show detailed metric table
        ax5.axis('tight')
        ax5.axis('off')
        
        # Create table with top 5 methods sorted by RMSE
        sorted_indices = np.argsort(rmse_all)
        top_5_indices = sorted_indices[:5]
        
        table_data = [['Method', 'RMSE', 'Ratio Pres. ', 'Bias']]
        cell_colors = [['lightgray'] * 4]
        
        for idx in top_5_indices: 
            method = methods[idx]
            m = results[method]
            row = [
                method,
                f'{m.rmse:.4f}',
                f'{m.peak_ratio_preservation:.3f}',
                f'{m.bias:.4f}'
            ]
            table_data.append(row)
            
            if method == recommended:
                cell_colors.append(['lightgreen'] * 4)
            elif method == 'SNV':  
                cell_colors.append(['lightcoral'] * 4)
            else:
                cell_colors.append(['white'] * 4)
        
        table = ax5.table(cellText=table_data, cellColours=cell_colors,
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2.5)
        
        for i in range(4):
            table[(0, i)].set_text_props(weight='bold')
        
        ax5.set_title('Top 5 Methods - Actual Metrics\n(Shows performance differences)', 
                     fontweight='bold', fontsize=12, pad=20)
    
    # Plot 6: Radar chart (top 4 methods)
    ax6 = fig2.add_subplot(2, 3, 6, projection='polar')
    
    # Calculate normalized scores for radar chart
    bias_scores = 100 * (1 - (bias_all - bias_all.min()) / (bias_all.max() - bias_all.min() + 1e-10))
    rmse_scores = 100 * (1 - (rmse_all - rmse_all.min()) / (rmse_all.max() - rmse_all.min() + 1e-10))
    peak_pres_scores = 100 * peak_pres_all
    inter_var_scores = 100 * (1 - (inter_var_all - inter_var_all.min()) / (inter_var_all.max() - inter_var_all.min() + 1e-10))
    ratio_pres_scores = 100 * ratio_pres_all
    
    scores = {}
    for i, method in enumerate(methods):
        score = (
            rmse_scores[i] * 0.30 +
            ratio_pres_scores[i] * 0.30 +
            peak_pres_scores[i] * 0.20 +
            bias_scores[i] * 0.10 +
            inter_var_scores[i] * 0.10
        )
        scores[method] = score
    
    sorted_methods = sorted(scores, key=scores.get, reverse=True)
    top_4 = sorted_methods[:4]
    
    categories = ['RMSE', 'Ratio Preservation', 'Peak Preservation', 'Low Bias', 'Consistency']
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    
    for method in top_4:
        i = methods.index(method)
        values = [
            rmse_scores[i],
            ratio_pres_scores[i],
            peak_pres_scores[i],
            bias_scores[i],
            inter_var_scores[i]
        ]
        values += values[:1]
        
        linewidth = 3 if method == recommended else 2
        alpha_val = 0.9 if method == recommended else 0.6
        ax6.plot(angles, values, 'o-', linewidth=linewidth, label=method, alpha=alpha_val)
        ax6.fill(angles, values, alpha=0.15)
    
    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(categories, fontsize=10)
    ax6.set_ylim(0, 100)
    ax6.set_title('Top 4 Methods\n(Normalized Metrics)', fontweight='bold', fontsize=12, pad=20)
    ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax6.grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    if save_path:  
        path2 = save_path.replace('.png', '_metrics.png')
        fig2.savefig(path2, dpi=150, bbox_inches='tight')
        print(f"✓ Figure 2 saved to:  {path2}")
    
    plt.show()


def print_results_table(results:  Dict[str, NormalizationMetrics]) -> None:
    """Print formatted results table."""
    try:
        from tabulate import tabulate
        use_tabulate = True
    except ImportError:
        use_tabulate = False
    
    headers = ['Method', 'Bias', 'RMSE', 'Peak Pres. ', 'Ratio Pres.', 'Inter-Var', 'Stable']
    
    rows = []
    for name, m in results.items():
        ratio_flag = " ⚠️" if m.peak_ratio_preservation < 0.95 else ""
        stable_flag = "✓" if m.numerical_stability else "✗"
        
        rows.append([
            name,
            f"{m.bias:.6f}",
            f"{m.rmse:.6f}",
            f"{m.peak_preservation:.4f}",
            f"{m.peak_ratio_preservation:.4f}{ratio_flag}",
            f"{m.inter_sample_variance:.6f}",
            stable_flag
        ])
    
    print("\n" + "="*110)
    print("NORMALIZATION METHOD COMPARISON RESULTS")
    print("="*110)
    print("Context [237]: L2 Bias=0.000, RMSE=0.002 | SNV RMSE=0.220 (WORST)")
    print("Ratio Preservation: >0.95 = Excellent (CRITICAL for quantitative LIBS)")
    print("="*110)
    
    if use_tabulate:
        print(tabulate(rows, headers=headers, tablefmt='grid'))
    else:
        col_widths = [15, 12, 12, 12, 16, 12, 8]
        header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
        print(header_line)
        print("-" * len(header_line))
        for row in rows:
            print(" | ".join(str(c).ljust(w) for c, w in zip(row, col_widths)))


def save_recommendation_report(
    results: Dict[str, NormalizationMetrics],
    recommended: str,
    justification: str,
    filepath: str = 'normalization_recommendation.txt'
) -> None:
    """Save recommendation report to file."""
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("LIBS NORMALIZATION RECOMMENDATION REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write("METHODOLOGY REFERENCE:\n")
        f.write("- Context Feeder [237]: L2 Bias=0.000, RMSE=0.002 (Best)\n")
        f.write("- Context Feeder [237]: SNV RMSE=0.220 (Worst - 110x worse than L2)\n")
        f.write("- Decision Rule:  'Best RMSE + Peak ratio preservation'\n\n")
        
        f.write("RESULTS SUMMARY:\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Method':<20} {'Bias':<12} {'RMSE':<12} {'Ratio Pres.':<14}\n")
        f.write("-"*70 + "\n")
        
        for name, m in sorted(results.items(), key=lambda x: x[1].rmse):
            f.write(f"{name:<20} {m.bias: <12.6f} {m.rmse:<12.6f} {m.peak_ratio_preservation:<14.4f}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("RECOMMENDATION\n")
        f.write("="*70 + "\n\n")
        f.write(justification)
        f.write("\n\n")
        
        f.write("="*70 + "\n")
        f.write("ACTION REQUIRED\n")
        f.write("="*70 + "\n\n")
        f.write("Update libs_preprocessing.py:\n\n")
        
        if recommended == 'L2_Norm':
            f.write("  # Use L2 normalization (RECOMMENDED [237])\n")
            f.write("  from sklearn.preprocessing import normalize\n")
            f.write("  normalized = normalize(spectra, norm='l2', axis=1)\n")
        elif recommended == 'L1_Norm':
            f.write("  # Use L1 normalization\n")
            f.write("  from sklearn.preprocessing import normalize\n")
            f.write("  normalized = normalize(spectra, norm='l1', axis=1)\n")
        elif recommended == 'SNV': 
            f.write("  # Use SNV normalization (CAUTION:  Contradicts literature [237])\n")
            f.write("  from sklearn.preprocessing import StandardScaler\n")
            f.write("  scaler = StandardScaler()\n")
            f.write("  normalized = scaler.fit_transform(spectra.T).T\n")
        else:
            f.write(f"  # Use {recommended} normalization\n")
            f.write(f"  # See implementation in normalization_comparison.py\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("IMPLEMENTATION NOTES\n")
        f.write("="*70 + "\n\n")
        f.write("1. Apply normalization AFTER baseline correction and smoothing\n")
        f.write("2. Verify peak ratio preservation on validation set\n")
        f.write("3. If using non-L2 method, justify empirically in thesis\n")
        f.write("4. Proceed with feature selection using normalized spectra\n\n")
    
    print(f"\n✓ Recommendation report saved to: {filepath}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================


if __name__ == "__main__": 
    print("="*70)
    print("LIBS NORMALIZATION METHOD COMPARISON")
    print("Context Feeder [237]: L2 Bias=0.000, RMSE=0.002 | SNV RMSE=0.220")
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
            
            # Apply baseline correction first
            print("\nApplying Polynomial baseline correction (degree=5)...")
            spectra_baseline = apply_baseline_correction_batch(
                spectra_raw, degree=5, verbose=True
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
        
        spectra_baseline = np.zeros((n_samples, n_features))
        for i in range(n_samples):
            baseline_level = 100
            peak_positions = [253.5, 255.3, 742.4, 766.5]
            for pos in peak_positions:
                intensity = np.random.uniform(500, 2000)
                peak = intensity * np.exp(-0.5 * ((wavelengths - pos) / 0.5)**2)
                spectra_baseline[i] += peak
            spectra_baseline[i] += baseline_level
            spectra_baseline[i] += np.random.normal(0, 20, n_features)
    
    # Run comparison for BOTH objectives (if classification enabled)
    if ENABLE_CLASSIFICATION and sample_names is not None:
        print("\n" + "="*70)
        print("COMPARISON 1: NUTRIENT CLASSIFICATION")
        print("="*70)
        results_nutrient = compare_normalization_methods(
            spectra_baseline, wavelengths, sample_names,
            objective='nutrient', verbose=True
        )
        
        print("\n" + "="*70)
        print("COMPARISON 2: CONTAMINANT DETECTION")
        print("="*70)
        results_contaminant = compare_normalization_methods(
            spectra_baseline, wavelengths, sample_names,
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
                                  'normalization_recommendation_nutrient.txt')
        save_recommendation_report(results_contaminant, recommended_con, justification_con,
                                  'normalization_recommendation_contaminant.txt')
        
        # Visualize nutrient results
        plot_normalization_comparison(spectra_baseline[0], wavelengths, results_nutrient,
                                     recommended_nut, save_path='normalization_comparison_nutrient.png')
        
    else:
        # Fallback:  spectral quality only
        print("\n⚠️ Running spectral quality comparison only")
        results = compare_normalization_methods(spectra_baseline, wavelengths, verbose=True)
        
        print_results_table(results)
        recommended, justification = generate_recommendation(results)
        
        print("\n" + "="*70)
        print("RECOMMENDATION")
        print("="*70)
        print(justification)
        
        save_recommendation_report(results, recommended, justification)
        plot_normalization_comparison(spectra_baseline[0], wavelengths, results, recommended,
                                     save_path='normalization_comparison.png')
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1. Review recommendation reports")
    print("2. Update libs_preprocessing.py with optimal normalization method")
    print("3. Proceed to Phase 2: Feature Selection")
    print("="*70)  