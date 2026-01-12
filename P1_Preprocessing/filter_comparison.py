"""
LIBS Smoothing Filter Comparison Script
========================================
Run this BEFORE finalizing libs_preprocessing.py

Methodology Justification:
- Schmid 2022 [233] (197 citations): "SG filters should be replaced"
- SG has poor noise suppression above cutoff frequency
- Modified Sinc (MS) provides better stopband rejection

Decision Rule from Context Feeder:
"Test both SG AND Modified Sinc filter.If MS improves accuracy >2%, use it."

Updated:  2026-01-12
Changes: 
- Baseline correction:  Polynomial → AsLS (r=0.93, 19% RMSE improvement)
- Added 2% decision threshold for MS vs SG comparison
- Fixed data loading for wavelength-in-rows CSV format
- Integrated classification-based comparison
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.fft import fft, fftfreq
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.ndimage import gaussian_filter1d
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, normalize
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import warnings
from dataclasses import dataclass
import os
import sys
import time

# Import label generation from preprocessing pipeline
sys.path.append(os.path.dirname(__file__))
try:
    from libs_preprocessing import generate_nutrient_labels, generate_contaminant_labels
    LABELS_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: Could not import label generation functions.")
    print("   Classification-based comparison will be skipped.")
    LABELS_AVAILABLE = False

warnings.filterwarnings('ignore')


# ============================================================================
# BASELINE CORRECTION: AsLS (METHODOLOGY-COMPLIANT)
# ============================================================================

def baseline_als(spectrum: np.ndarray, lam: float = 1e6, p: float = 0.01, 
                 niter: int = 10) -> np.ndarray:
    """
    Asymmetric Least Squares (AsLS) baseline correction.
    
    Reference: [221] - Achieved r=0.93, 19% RMSE improvement over alternatives
    
    Parameters
    ----------
    spectrum : np.ndarray
        Input LIBS spectrum
    lam : float
        Smoothness parameter (typical:  1e5 - 1e7)
    p : float
        Asymmetry parameter (typical: 0.001 - 0.1)
    niter : int
        Maximum iterations
        
    Returns
    -------
    baseline : np.ndarray
        Estimated baseline (Bremsstrahlung continuum)
    """
    L = len(spectrum)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    D = lam * D.dot(D.transpose())
    
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    
    for i in range(niter):
        W.setdiag(w)
        Z = W + D
        z = spsolve(Z, w * spectrum)
        w = p * (spectrum > z) + (1 - p) * (spectrum < z)
    
    return z


def apply_baseline_correction_batch(
    spectra: np.ndarray,
    method: str = 'asls',
    lam: float = 1e6,
    p: float = 0.01,
    verbose: bool = True
) -> np.ndarray:
    """
    Apply AsLS baseline correction to all spectra.
    
    Context Feeder Evidence:
    - AsLS + spectral ranges:  r = 0.93, RMSE = 0.21 (19% better than alternatives)
    - Removes Bremsstrahlung continuum background
    
    Parameters
    ----------
    spectra : np.ndarray, shape (n_samples, n_features)
        Raw LIBS spectra
    method :  str
        'asls' (Asymmetric Least Squares) - METHODOLOGY STANDARD
    lam : float
        AsLS smoothness (default 1e6)
    p : float
        AsLS asymmetry (default 0.01)
    verbose : bool
        Print progress
        
    Returns
    -------
    corrected :  np.ndarray
        Baseline-corrected spectra (non-negative)
    """
    n_samples = spectra.shape[0]
    corrected = np.zeros_like(spectra)
    
    if verbose:
        print(f"Applying AsLS baseline correction (λ={lam:.0e}, p={p}) to {n_samples} spectra...")
        print("Reference: [221] - Pearson's r=0.93, 19% RMSE improvement")
    
    for i in range(n_samples):
        baseline = baseline_als(spectra[i], lam=lam, p=p, niter=10)
        corrected[i] = np.maximum(spectra[i] - baseline, 0)
        
        if verbose and (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{n_samples} spectra...")
    
    if verbose:
        print(f"✓ Baseline correction complete")
    
    return corrected


# ============================================================================
# DATA LOADING (FIXED FOR YOUR CSV FORMAT)
# ============================================================================

def load_irish_soil_data(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load Irish Soil Dataset CSV.
    
    Expected format:
    - First row: Sample names (Clay_1, Clay_2, ..., Clay_320)
    - First column: Wavelengths (197.001 - 973.5891 nm, 12,981 rows)
    - Data:  Spectral intensity values
    
    Returns
    -------
    spectra : np.ndarray, shape (320, 12981)
        Spectral data (samples × features)
    sample_names : np.ndarray, shape (320,)
        Sample identifiers (e.g., "Clay_1")
    wavelengths : np.ndarray, shape (12981,)
        Wavelength values in nm
    """
    print(f"\n{'='*70}")
    print("LOADING IRISH SOIL DATASET (Real LIBS Data)")
    print(f"{'='*70}")
    print(f"File: {filepath}")
    
    # Read CSV with first column as wavelengths, first row as sample names
    df = pd.read_csv(filepath, index_col=0)
    
    # Extract wavelengths from row index
    try:
        wavelengths = df.index.astype(float).values
        print(f"✓ Wavelengths:  {wavelengths[0]:.3f} - {wavelengths[-1]:.3f} nm ({len(wavelengths)} points)")
    except ValueError: 
        print("⚠️  Warning: Could not parse wavelengths from index, using fallback")
        wavelengths = np.linspace(197.001, 973.5891, df.shape[0])
    
    # Extract sample names from column headers
    sample_names = df.columns.astype(str).values
    
    # Transpose:  (wavelengths, samples) → (samples, wavelengths)
    spectra = df.values.T.astype(float)
    
    print(f"✓ Loaded:  {spectra.shape[0]} samples × {spectra.shape[1]} features")
    print(f"✓ Sample names (first 5): {sample_names[:5].tolist()}")
    print(f"{'='*70}\n")
    
    return spectra, sample_names, wavelengths


# ============================================================================
# FILTER IMPLEMENTATIONS
# ============================================================================

def savitzky_golay(spectrum: np.ndarray, window: int = 11, order: int = 3) -> np.ndarray:
    """
    Standard Savitzky-Golay filter.
    
    Reference: [228] - LIBS standard practice
    """
    return savgol_filter(spectrum, window, order)


def modified_sinc(spectrum: np.ndarray, fc: float = 0.1, M: int = 11) -> np.ndarray:
    """
    Modified Sinc (Windowed-Sinc) low-pass filter.
    
    Reference: [233] Schmid 2022 (197 citations) - Better stopband rejection than SG
    
    Parameters
    ----------
    spectrum : np.ndarray
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
    window = np.blackman(M)
    h = h * window
    h = h / np.sum(h)
    
    return np.convolve(spectrum, h, mode='same')


def gaussian_filter(spectrum: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Gaussian smoothing filter (additional comparison)."""
    return gaussian_filter1d(spectrum, sigma=sigma)


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


def calculate_peak_preservation(original: np.ndarray, filtered: np.ndarray) -> float:
    """Calculate peak height preservation percentage."""
    peak_indices = np.argsort(original)[-10:]
    
    original_heights = original[peak_indices]
    filtered_heights = filtered[peak_indices]
    
    valid = original_heights > 0
    if not np.any(valid):
        return 100.0
    
    preservation = np.mean(filtered_heights[valid] / original_heights[valid]) * 100
    return min(preservation, 100.0)


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
    
    Key metric from Schmid 2022 [233] - MS should show better high-frequency suppression than SG.
    """
    n = len(original)
    fft_original = np.abs(fft(original))[:n//2]
    fft_filtered = np.abs(fft(filtered))[:n//2]
    
    # High-frequency region (stopband) - upper 20%
    stopband = slice(int(0.4 * n//2), n//2)
    
    # Avoid log(0)
    fft_original[fft_original == 0] = 1e-10
    fft_filtered[fft_filtered == 0] = 1e-10
    
    attenuation = 20 * np.log10(
        np.mean(fft_original[stopband]) / np.mean(fft_filtered[stopband])
    )
    
    return attenuation


# ============================================================================
# COMPREHENSIVE FILTER COMPARISON
# ============================================================================

def compare_filters(
    spectra: np.ndarray,
    wavelengths: np.ndarray,
    sample_names: np.ndarray,
    objective:  str = 'nutrient',
    verbose: bool = True
) -> Dict[str, FilterMetrics]: 
    """
    Comprehensive filter comparison WITH classification validation.
    
    Parameters
    ----------
    spectra : np.ndarray, shape (n_samples, n_features)
        Baseline-corrected spectra
    wavelengths : np.ndarray
        Wavelength values
    sample_names : np.ndarray
        Sample identifiers for label generation
    objective : str
        'nutrient' (NPK:  High/Med/Low) or 'contaminant' (Clean/Contaminated)
    verbose : bool
        Print progress
        
    Returns
    -------
    results : Dict[str, FilterMetrics]
        Metrics including classification accuracy
    """
    # Generate labels based on objective
    if LABELS_AVAILABLE:
        if objective == 'nutrient': 
            labels_raw = generate_nutrient_labels(sample_names)
            valid_mask = labels_raw != -1
            spectra = spectra[valid_mask]
            labels = labels_raw[valid_mask]
            print(f"\n✓ Objective:  Nutrient Classification (High/Med/Low)")
            print(f"  Usable samples: {len(labels)} (excluded {np.sum(~valid_mask)} contaminated)")
        elif objective == 'contaminant': 
            labels = generate_contaminant_labels(sample_names)
            print(f"\n✓ Objective: Contaminant Detection (Clean/Contaminated)")
            print(f"  Total samples: {len(labels)}")
        else:
            raise ValueError(f"Unknown objective: {objective}")
        
        le = LabelEncoder()
        y_encoded = le.fit_transform(labels)
    else:
        print("\n⚠️  Classification labels not available - using spectral quality metrics only")
        y_encoded = None
    
    # Filter configurations
    filter_configs = {
        'SG_w7_o3': lambda x: savitzky_golay(x, window=7, order=2),  # LIBS standard
        'SG_w11_o3': lambda x: savitzky_golay(x, window=11, order=3),
        'MS_fc0.10_M11': lambda x: modified_sinc(x, fc=0.10, M=11),   # Schmid 2022 alternative
        'MS_fc0.10_M15': lambda x:  modified_sinc(x, fc=0.10, M=15),
        'Gaussian_s1.5': lambda x: gaussian_filter(x, sigma=1.5),
    }
    
    results = {}
    n_samples = spectra.shape[0]
    
    if verbose:
        print("\n" + "="*70)
        print("SMOOTHING FILTER COMPARISON")
        print("Methodology: Schmid 2022 [233] + Classification Validation")
        print("="*70)
    
    for name, filter_func in filter_configs.items():
        if verbose:
            print(f"\nEvaluating:  {name}...")
        
        # Apply filter
        start_time = time.time()
        filtered = np.apply_along_axis(filter_func, axis=1, arr=spectra)
        comp_time = (time.time() - start_time) / n_samples * 1000
        
        # Spectral quality metrics
        noise_reductions = []
        peak_preservations = []
        snr_improvements = []
        attenuations = []
        
        for i in range(min(50, n_samples)):
            orig = spectra[i]
            filt = filtered[i]
            
            noise_reductions.append(calculate_noise_reduction(orig, filt))
            peak_preservations.append(calculate_peak_preservation(orig, filt))
            
            snr_orig = calculate_snr(orig)
            snr_filt = calculate_snr(filt)
            if snr_orig > 0:
                snr_improvements.append((snr_filt - snr_orig) / snr_orig * 100)
            
            attenuations.append(calculate_stopband_attenuation(orig, filt))
        
        # Classification accuracy evaluation
        if y_encoded is not None:
            filtered_norm = normalize(filtered, norm='l2', axis=1)
            
            # Random Forest (as per methodology [302]:  94.9%, 128KB)
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                random_state=42,
                n_jobs=-1
            )
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(rf, filtered_norm, y_encoded, cv=cv, scoring='accuracy')
            classification_acc = np.mean(cv_scores) * 100
        else:
            classification_acc = 0.0
        
        results[name] = FilterMetrics(
            name=name,
            noise_reduction_pct=np.mean(noise_reductions),
            peak_preservation_pct=np.mean(peak_preservations),
            snr_improvement_pct=np.mean(snr_improvements) if snr_improvements else 0,
            frequency_attenuation_db=np.mean(attenuations),
            classification_accuracy=classification_acc,
            computation_time_ms=comp_time
        )
        
        if verbose: 
            m = results[name]
            print(f"  Noise Reduction:      {m.noise_reduction_pct:+.1f}%")
            print(f"  Peak Preservation:   {m.peak_preservation_pct:.1f}%")
            print(f"  Stopband Atten.:      {m.frequency_attenuation_db:.1f} dB")
            if y_encoded is not None: 
                print(f"  Classification:       {m.classification_accuracy:.2f}%")
    
    return results


def generate_recommendation(results: Dict[str, FilterMetrics]) -> Tuple[str, str]:
    """
    Generate recommendation based on 2% DECISION THRESHOLD.
    
    Decision Rule from Context Feeder: 
    "If Modified Sinc improves accuracy >2% over SG, use it. Otherwise, use SG for lower cost."
    
    Returns
    -------
    recommended_filter : str
        Name of recommended filter
    justification : str
        Detailed justification
    """
    # Separate by filter type
    sg_results = {k: v for k, v in results.items() if k.startswith('SG')}
    ms_results = {k: v for k, v in results.items() if k.startswith('MS')}
    
    # Find best SG and best MS
    best_sg = max(sg_results.values(), key=lambda x: x.classification_accuracy)
    best_ms = max(ms_results.values(), key=lambda x: x.classification_accuracy)
    
    # Calculate improvement
    accuracy_improvement = best_ms.classification_accuracy - best_sg.classification_accuracy
    stopband_improvement = best_ms.frequency_attenuation_db - best_sg.frequency_attenuation_db
    
    # DECISION RULE: 2% threshold
    if accuracy_improvement > 2.0:
        recommended = best_ms.name
        decision_reason = f"Modified Sinc EXCEEDS 2% accuracy threshold (+{accuracy_improvement:.2f} pp)"
    else:
        recommended = best_sg.name
        decision_reason = f"SG performs within 2% (+{accuracy_improvement:.2f} pp) - use for lower computational cost"
    
    # Build justification
    justification_parts = []
    
    justification_parts.append("COMPARISON:  Modified Sinc vs Savitzky-Golay")
    justification_parts.append("="*70)
    justification_parts.append(f"Best SG: {best_sg.name}")
    justification_parts.append(f"  Accuracy:             {best_sg.classification_accuracy:.2f}%")
    justification_parts.append(f"  Stopband Atten.:     {best_sg.frequency_attenuation_db:.1f} dB")
    justification_parts.append(f"  Computation Time:    {best_sg.computation_time_ms:.3f} ms")
    justification_parts.append("")
    justification_parts.append(f"Best MS: {best_ms.name}")
    justification_parts.append(f"  Accuracy:            {best_ms.classification_accuracy:.2f}%")
    justification_parts.append(f"  Stopband Atten.:     {best_ms.frequency_attenuation_db:.1f} dB")
    justification_parts.append(f"  Computation Time:    {best_ms.computation_time_ms:.3f} ms")
    justification_parts.append("")
    justification_parts.append("DIFFERENCE (MS - SG):")
    justification_parts.append(f"  Accuracy:             {accuracy_improvement:+.2f} pp")
    justification_parts.append(f"  Stopband Atten.:     {stopband_improvement:+.1f} dB")
    justification_parts.append("")
    justification_parts.append("="*70)
    justification_parts.append(f"DECISION: {recommended}")
    justification_parts.append("="*70)
    justification_parts.append(f"Reason: {decision_reason}")
    justification_parts.append("")
    justification_parts.append("CONTEXT FEEDER REFERENCE:")
    justification_parts.append("- Schmid 2022 [233] (197 citations): 'SG filters should be replaced'")
    justification_parts.append("- Decision Rule: 'If MS improves accuracy >2%, use it'")
    justification_parts.append("")
    
    if recommended.startswith('MS'):
        justification_parts.append("INTERPRETATION:")
        justification_parts.append("✓ Modified Sinc shows >2% accuracy improvement")
        justification_parts.append("✓ Supports Schmid 2022's recommendation")
        justification_parts.append(f"✓ Stopband rejection improved by {stopband_improvement:+.1f} dB")
        justification_parts.append("")
        justification_parts.append("TRADE-OFF:")
        time_overhead = best_ms.computation_time_ms - best_sg.computation_time_ms
        justification_parts.append(f"  Computation time overhead: {time_overhead:+.3f} ms ({time_overhead/best_sg.computation_time_ms*100:+.1f}%)")
        justification_parts.append("  ✓ Acceptable for MCU deployment (still << 50 ms target)")
    else:
        justification_parts.append("INTERPRETATION:")
        justification_parts.append("✓ SG performance is within 2% of Modified Sinc")
        justification_parts.append("✓ Use SG for lower computational cost")
        justification_parts.append("✓ Compatible with standard LIBS practice [228]")
        justification_parts.append("")
        justification_parts.append("JUSTIFICATION:")
        justification_parts.append(f"  Accuracy difference: {abs(accuracy_improvement):.2f} pp (< 2% threshold)")
        justification_parts.append(f"  Computation advantage: {best_sg.computation_time_ms:.3f} ms vs {best_ms.computation_time_ms:.3f} ms")
    
    return recommended, "\n".join(justification_parts)


def save_recommendation_report(
    results: Dict[str, FilterMetrics],
    recommended:  str,
    justification: str,
    objective: str,
    filepath: str
) -> None:
    """Save recommendation report to file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("LIBS SMOOTHING FILTER RECOMMENDATION REPORT\n")
        f.write(f"Objective: {objective.upper()}\n")
        f.write("="*70 + "\n\n")
        
        f.write("METHODOLOGY REFERENCE:\n")
        f.write("- AsLS Baseline: [221] r=0.93, 19% RMSE improvement\n")
        f.write("- Schmid 2022 [233]: 'SG filters should be replaced'\n")
        f.write("- Decision Rule: 'If MS improves accuracy >2%, use it'\n\n")
        
        f.write("RESULTS SUMMARY:\n")
        f.write("-"*70 + "\n")
        
        headers = ['Filter', 'Accuracy', 'Noise Red.', 'Peak Pres.', 'Stopband', 'Time']
        f.write(f"{headers[0]:<20} {headers[1]:<12} {headers[2]:<12} {headers[3]:<12} {headers[4]:<12} {headers[5]:<10}\n")
        f.write("-"*70 + "\n")
        
        for name, m in sorted(results.items(), key=lambda x: -x[1].classification_accuracy):
            f.write(f"{name:<20} {m.classification_accuracy:.2f}%       "
                   f"{m.noise_reduction_pct:+.1f}%       "
                   f"{m.peak_preservation_pct:.1f}%       "
                   f"{m.frequency_attenuation_db:.1f} dB      "
                   f"{m.computation_time_ms:.3f} ms\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("RECOMMENDATION\n")
        f.write("="*70 + "\n\n")
        f.write(justification)
        f.write("\n\n")
        
        f.write("="*70 + "\n")
        f.write("ACTION REQUIRED:  Update libs_preprocessing.py\n")
        f.write("="*70 + "\n\n")
        
        if recommended.startswith('SG'):
            params = recommended.replace('SG_w', '').replace('_o', ',').split('_')
            window = params[0]
            order = params[1] if len(params) > 1 else '3'
            f.write(f"smoothing_method = 'savgol'\n")
            f.write(f"sg_window = {window}\n")
            f.write(f"sg_order = {order}\n")
        elif recommended.startswith('MS'):
            params = recommended.replace('MS_fc', '').replace('_M', ',').split(',')
            fc = params[0]
            M = params[1] if len(params) > 1 else '11'
            f.write(f"smoothing_method = 'sinc'\n")
            f.write(f"sinc_fc = {fc}\n")
            f.write(f"sinc_M = {M}\n")
        
        f.write("\n" + "="*70 + "\n")
    
    print(f"\n✓ Recommendation report saved to: {filepath}")


def plot_filter_comparison(
    spectrum:  np.ndarray,
    wavelengths: np.ndarray,
    results: Dict[str, FilterMetrics],
    recommended: str,
    save_path: str = None
) -> None:
    """Visualize filter comparison with 2% decision threshold."""
    
    fig = plt.figure(figsize=(16, 10))
    
    # Get SG and MS results
    sg_filters = {k: v for k, v in results.items() if k.startswith('SG')}
    ms_filters = {k: v for k, v in results.items() if k.startswith('MS')}
    
    best_sg = max(sg_filters.values(), key=lambda x: x.classification_accuracy)
    best_ms = max(ms_filters.values(), key=lambda x: x.classification_accuracy)
    
    # Plot 1: Spectral comparison
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(wavelengths, spectrum, 'k-', alpha=0.3, linewidth=0.5, label='Original')
    
    # Apply filters
    sg_filtered = savitzky_golay(spectrum, window=11, order=3)
    ms_filtered = modified_sinc(spectrum, fc=0.10, M=11)
    
    ax1.plot(wavelengths, sg_filtered, 'b-', linewidth=1.2, label=f'{best_sg.name} (SG)', alpha=0.8)
    ax1.plot(wavelengths, ms_filtered, 'r-', linewidth=1.2, label=f'{best_ms.name} (MS)', alpha=0.8)
    
    ax1.set_xlabel('Wavelength (nm)', fontsize=11)
    ax1.set_ylabel('Intensity', fontsize=11)
    ax1.set_title('Spectral Comparison: SG vs Modified Sinc', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Frequency domain
    ax2 = fig.add_subplot(2, 2, 2)
    n = len(spectrum)
    freqs = fftfreq(n, d=1)[: n//2]
    
    fft_orig = np.abs(fft(spectrum))[:n//2]
    fft_sg = np.abs(fft(sg_filtered))[:n//2]
    fft_ms = np.abs(fft(ms_filtered))[:n//2]
    
    ax2.semilogy(freqs, fft_orig, 'k-', alpha=0.3, label='Original', linewidth=0.8)
    ax2.semilogy(freqs, fft_sg, 'b-', label='SG', alpha=0.8, linewidth=1.2)
    ax2.semilogy(freqs, fft_ms, 'r-', label='MS', alpha=0.8, linewidth=1.2)
    
    ax2.set_xlabel('Normalized Frequency', fontsize=11)
    ax2.set_ylabel('Magnitude (log scale)', fontsize=11)
    ax2.set_title('Frequency Response (Stopband Attenuation)', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')
    
    # Plot 3: Accuracy comparison with 2% threshold
    ax3 = fig.add_subplot(2, 2, 3)
    
    filter_names = [best_sg.name, best_ms.name]
    accuracies = [best_sg.classification_accuracy, best_ms.classification_accuracy]
    colors = ['blue' if f.startswith('SG') else 'red' for f in filter_names]
    
    # Highlight recommended
    for i, f in enumerate(filter_names):
        if f == recommended:
            colors[i] = 'green'
    
    bars = ax3.bar(filter_names, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add 2% threshold line
    baseline_acc = best_sg.classification_accuracy
    ax3.axhline(y=baseline_acc, color='blue', linestyle='--', linewidth=2, label='SG Baseline', alpha=0.7)
    ax3.axhline(y=baseline_acc + 2, color='red', linestyle='--', linewidth=2, label='2% Threshold', alpha=0.7)
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax3.set_ylabel('Classification Accuracy (%)', fontsize=11)
    ax3.set_title(f'Accuracy Comparison (2% Decision Threshold)\nRecommended: {recommended}', 
                  fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Multi-metric comparison
    ax4 = fig.add_subplot(2, 2, 4)
    
    metrics_names = ['Noise\nReduction', 'Peak\nPreservation', 'Stopband\nAttenuation']
    sg_values = [
        best_sg.noise_reduction_pct,
        best_sg.peak_preservation_pct,
        best_sg.frequency_attenuation_db * 5  # Scale for visibility
    ]
    ms_values = [
        best_ms.noise_reduction_pct,
        best_ms.peak_preservation_pct,
        best_ms.frequency_attenuation_db * 5
    ]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    ax4.bar(x - width/2, sg_values, width, label='SG', alpha=0.7, color='blue', edgecolor='black')
    ax4.bar(x + width/2, ms_values, width, label='MS', alpha=0.7, color='red', edgecolor='black')
    
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics_names, fontsize=10)
    ax4.set_ylabel('Metric Value (%)', fontsize=11)
    ax4.set_title('Spectral Quality Metrics Comparison', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('LIBS Smoothing Filter Comparison\nSchmid 2022 [233]: Modified Sinc vs Savitzky-Golay (2% Decision Rule)',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Figure saved to: {save_path}")
    
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("LIBS SMOOTHING FILTER COMPARISON")
    print("Schmid 2022 [233]:  Evaluating SG vs Modified Sinc (2% Threshold)")
    print("="*70)
    
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    REAL_DATA_CONFIG = {
        "filepath": r"D:\Ryan\UM\Academic\Y4S1\KIE4002 FINAL YEAR PROJECT\Irish_Soil_DataSet\Irish_Soil_DataSet2.csv",
    }
    
    # =========================================================================
    # LOAD DATA
    # =========================================================================
    try:
        spectra_raw, sample_names, wavelengths = load_irish_soil_data(
            REAL_DATA_CONFIG["filepath"]
        )
    except Exception as e:
        print(f"\n❌ ERROR loading data: {e}")
        print("Please check file path and format.")
        exit(1)
    
    # =========================================================================
    # APPLY AsLS BASELINE CORRECTION
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 1: AsLS Baseline Correction")
    print("="*70)
    print("Reference: [221] Pearson's r=0.93, 19% RMSE improvement")
    
    spectra_baseline = apply_baseline_correction_batch(
        spectra_raw,
        method='asls',
        lam=1e6,
        p=0.01,
        verbose=True
    )
    
    # =========================================================================
    # RUN FILTER COMPARISON FOR BOTH OBJECTIVES
    # =========================================================================
    
    # NUTRIENT CLASSIFICATION
    print("\n" + "="*70)
    print("COMPARISON 1: NUTRIENT CLASSIFICATION (NPK)")
    print("="*70)
    results_nutrient = compare_filters(
        spectra_baseline,
        wavelengths,
        sample_names,
        objective='nutrient',
        verbose=True
    )
    
    recommended_nut, justification_nut = generate_recommendation(results_nutrient)
    
    print("\n" + "="*70)
    print("NUTRIENT CLASSIFICATION RECOMMENDATION")
    print("="*70)
    print(justification_nut)
    
    save_recommendation_report(
        results_nutrient,
        recommended_nut,
        justification_nut,
        'nutrient',
        'filter_recommendation_nutrient.txt'
    )
    
    plot_filter_comparison(
        spectra_baseline[0],
        wavelengths,
        results_nutrient,
        recommended_nut,
        save_path='filter_comparison_nutrient.png'
    )
    
    # CONTAMINANT DETECTION
    print("\n" + "="*70)
    print("COMPARISON 2: CONTAMINANT DETECTION")
    print("="*70)
    results_contaminant = compare_filters(
        spectra_baseline,
        wavelengths,
        sample_names,
        objective='contaminant',
        verbose=True
    )
    
    recommended_con, justification_con = generate_recommendation(results_contaminant)
    
    print("\n" + "="*70)
    print("CONTAMINANT DETECTION RECOMMENDATION")
    print("="*70)
    print(justification_con)
    
    save_recommendation_report(
        results_contaminant,
        recommended_con,
        justification_con,
        'contaminant',
        'filter_recommendation_contaminant.txt'
    )
    
    plot_filter_comparison(
        spectra_baseline[0],
        wavelengths,
        results_contaminant,
        recommended_con,
        save_path='filter_comparison_contaminant.png'
    )
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"✓ Analyzed {spectra_baseline.shape[0]} spectra")
    print(f"✓ Baseline:  AsLS (r=0.93, 19% RMSE improvement)")
    print(f"✓ Decision Rule: 2% accuracy threshold (Schmid 2022)")
    print("\nRecommendations:")
    print(f"  Nutrient Classification:   {recommended_nut}")
    print(f"  Contaminant Detection:    {recommended_con}")
    print("\nGenerated Files:")
    print("  - filter_recommendation_nutrient.txt")
    print("  - filter_recommendation_contaminant.txt")
    print("  - filter_comparison_nutrient.png")
    print("  - filter_comparison_contaminant.png")
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1.Review recommendation reports")
    print("2.Update libs_preprocessing.py with recommended filters")
    print("3.Proceed to Phase 2: Feature Selection (NIST peaks + RF Gini)")
    print("="*70)