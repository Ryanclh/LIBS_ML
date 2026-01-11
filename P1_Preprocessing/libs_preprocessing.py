"""
LIBS Spectral Data Preprocessing Pipeline
==========================================
Phase 1 Implementation for Embedded ML Soil Analysis Project

Methods justified by academic literature:
- AsLS Baseline Correction:  r=0.93, 19% RMSE improvement [221]
- Savitzky-Golay Filter: Standard LIBS practice [228]
- L2 Normalization: Bias=0.000, RMSE=0.002 [237]
"""

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse. linalg import spsolve
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict
import warnings

# ============================================================================
# STEP 1: DATA LOADING & STRATIFIED SPLITTING
# ============================================================================

def load_libs_data(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load LIBS spectral data from CSV file. 
    
    Expected format:
    - Rows: 324 samples
    - Columns: wavelength values (12,981 features) + label column
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file containing LIBS spectra
        
    Returns
    -------
    spectra : np.ndarray
        Shape (n_samples, n_features) - raw spectral intensities
    labels : np.ndarray
        Shape (n_samples,) - class labels (High/Medium/Low or 0/1/2)
    wavelengths : np.ndarray
        Shape (n_features,) - wavelength values in nm
    """
    df = pd.read_csv(filepath)
    
    # Assume last column is label, first row might be wavelengths
    # Adjust based on your actual data format
    if df.columns[0]. replace('.', '').isdigit():
        # First row contains wavelength headers
        wavelengths = df. columns[:-1].astype(float).values
        labels = df.iloc[:, -1].values
        spectra = df.iloc[:, :-1].values. astype(float)
    else:
        # Wavelengths need to be inferred (200-950nm range)
        n_features = df.shape[1] - 1
        wavelengths = np.linspace(200, 950, n_features)
        labels = df.iloc[:, -1].values
        spectra = df.iloc[:, :-1].values. astype(float)
    
    print(f"Loaded:  {spectra.shape[0]} samples × {spectra.shape[1]} features")
    print(f"Wavelength range: {wavelengths.min():.1f} - {wavelengths.max():.1f} nm")
    print(f"Class distribution: {np.unique(labels, return_counts=True)}")
    
    return spectra, labels, wavelengths


def stratified_split(
    spectra: np.ndarray,
    labels: np.ndarray,
    test_size:  float = 0.30,
    random_state: int = 42
) -> Tuple[np.ndarray, np. ndarray, np.ndarray, np.ndarray]:
    """
    Perform stratified train-test split (70-30 as per methodology).
    
    Ensures equal distribution of High/Medium/Low classes in both sets.
    
    Parameters
    ----------
    spectra : np. ndarray
        Raw spectral data
    labels : np.ndarray
        Class labels
    test_size : float
        Proportion for test set (default 0.30 = 30%)
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
        Split datasets
    """
    X_train, X_test, y_train, y_test = train_test_split(
        spectra, labels,
        test_size=test_size,
        stratify=labels,  # Ensures balanced class distribution
        random_state=random_state
    )
    
    print(f"\nStratified Split Complete:")
    print(f"  Training:  {X_train.shape[0]} samples ({100*(1-test_size):.0f}%)")
    print(f"  Testing:   {X_test.shape[0]} samples ({100*test_size:.0f}%)")
    print(f"  Train class distribution: {np.unique(y_train, return_counts=True)}")
    print(f"  Test class distribution:   {np.unique(y_test, return_counts=True)}")
    
    return X_train, X_test, y_train, y_test


# ============================================================================
# STEP 2: BASELINE CORRECTION - Asymmetric Least Squares (AsLS)
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
        Smoothness parameter (λ). Higher = smoother baseline. 
        Typical range: 1e4 to 1e7 for LIBS
    p : float
        Asymmetry parameter.  Lower = baseline follows valleys better.
        Typical range: 0.001 to 0.1
    max_iter : int
        Maximum iterations for convergence
        
    Returns
    -------
    baseline : np.ndarray
        Estimated baseline to subtract from spectrum
        
    References
    ----------
    Eilers, P.H.C., Boelens, H.F.M. (2005). Baseline correction with 
    asymmetric least squares smoothing. 
    """
    L = len(spectrum)
    # Construct second-order difference matrix
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
    D = lam * D. dot(D. T)
    
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    
    for _ in range(max_iter):
        W. setdiag(w)
        Z = W + D
        z = spsolve(Z, w * spectrum)
        w_new = p * (spectrum > z) + (1 - p) * (spectrum < z)
        if np.allclose(w, w_new):
            break
        w = w_new
    
    return z


def apply_baseline_correction(
    spectra: np.ndarray,
    lam: float = 1e6,
    p: float = 0.01,
    verbose: bool = True
) -> np.ndarray:
    """
    Apply AsLS baseline correction to all spectra.
    
    Parameters
    ----------
    spectra :  np.ndarray
        Shape (n_samples, n_features)
    lam : float
        Smoothness parameter
    p : float
        Asymmetry parameter
    verbose : bool
        Print progress
        
    Returns
    -------
    corrected :  np.ndarray
        Baseline-corrected spectra
    """
    n_samples = spectra.shape[0]
    corrected = np.zeros_like(spectra)
    
    for i in range(n_samples):
        baseline = asls_baseline(spectra[i], lam=lam, p=p)
        corrected[i] = spectra[i] - baseline
        
        if verbose and (i + 1) % 50 == 0:
            print(f"  Baseline correction:  {i+1}/{n_samples} spectra processed")
    
    # Ensure no negative values (physical constraint)
    corrected = np. maximum(corrected, 0)
    
    if verbose:
        print(f"✓ Baseline correction complete: {n_samples} spectra")
    
    return corrected


# ============================================================================
# STEP 3: NOISE REDUCTION - Savitzky-Golay Filter
# ============================================================================

def apply_savgol_filter(
    spectra: np.ndarray,
    window_length: int = 11,
    polyorder: int = 3
) -> np.ndarray:
    """
    Apply Savitzky-Golay smoothing filter. 
    
    Justification: 
    - Standard LIBS practice for noise reduction [228]
    - Preserves peak height and shape while reducing high-frequency noise
    
    Parameters
    ----------
    spectra : np.ndarray
        Shape (n_samples, n_features)
    window_length :  int
        Length of filter window (must be odd). Default 11.
    polyorder : int
        Polynomial order for fitting. Default 3.
        
    Returns
    -------
    smoothed : np.ndarray
        Smoothed spectra
        
    Notes
    -----
    Window=11, polyorder=3 is optimal for LIBS spectral resolution.
    Larger windows provide more smoothing but may distort narrow peaks.
    """
    if window_length % 2 == 0:
        window_length += 1
        warnings.warn(f"Window length must be odd.  Adjusted to {window_length}")
    
    smoothed = np.apply_along_axis(
        lambda x: savgol_filter(x, window_length, polyorder),
        axis=1,
        arr=spectra
    )
    
    print(f"✓ Savitzky-Golay filter applied: window={window_length}, order={polyorder}")
    
    return smoothed


def apply_gaussian_filter(
    spectra:  np.ndarray,
    sigma: float = 1.5
) -> np.ndarray:
    """
    Apply Gaussian smoothing filter (alternative to Savitzky-Golay).
    
    Use for comparison as per methodology recommendation to test
    alternative to SG filter.
    
    Parameters
    ----------
    spectra :  np.ndarray
        Shape (n_samples, n_features)
    sigma : float
        Standard deviation of Gaussian kernel
        
    Returns
    -------
    smoothed : np. ndarray
        Smoothed spectra
    """
    smoothed = np. apply_along_axis(
        lambda x: gaussian_filter1d(x, sigma=sigma),
        axis=1,
        arr=spectra
    )
    
    print(f"✓ Gaussian filter applied: sigma={sigma}")
    
    return smoothed


# ============================================================================
# STEP 4: NORMALIZATION - L2 (Euclidean) Norm
# ============================================================================

def apply_l2_normalization(spectra: np.ndarray) -> np.ndarray:
    """
    Apply L2 (Euclidean) normalization to each spectrum.
    
    Justification:
    - 2025 validation study:  Bias=0.000, RMSE=0.002 [237]
    - Corrects for shot-to-shot laser energy fluctuations
    - Superior to SNV (which had RMSE=0.220 - worst performer)
    
    Parameters
    ----------
    spectra : np. ndarray
        Shape (n_samples, n_features)
        
    Returns
    -------
    normalized : np.ndarray
        L2-normalized spectra (each row has unit norm)
    """
    normalized = normalize(spectra, norm='l2', axis=1)
    
    # Verify normalization
    norms = np.linalg. norm(normalized, axis=1)
    assert np.allclose(norms, 1.0), "L2 normalization failed"
    
    print(f"✓ L2 normalization applied:  all spectra scaled to unit norm")
    
    return normalized


# ============================================================================
# COMPLETE PREPROCESSING PIPELINE
# ============================================================================

def preprocess_libs_spectra(
    spectra: np.ndarray,
    asls_lam: float = 1e6,
    asls_p: float = 0.01,
    sg_window: int = 11,
    sg_order: int = 3,
    use_gaussian:  bool = False,
    gaussian_sigma: float = 1.5,
    verbose: bool = True
) -> Dict[str, np.ndarray]: 
    """
    Complete LIBS preprocessing pipeline.
    
    Order:  Baseline Correction → Noise Reduction → Normalization
    
    Parameters
    ----------
    spectra : np.ndarray
        Raw spectral data (n_samples, n_features)
    asls_lam : float
        AsLS smoothness parameter
    asls_p : float
        AsLS asymmetry parameter
    sg_window : int
        Savitzky-Golay window length
    sg_order :  int
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
        - 'baseline_corrected': After AsLS
        - 'smoothed': After noise reduction
        - 'normalized':  Final preprocessed spectra
    """
    results = {'raw': spectra. copy()}
    
    if verbose:
        print("\n" + "="*60)
        print("LIBS PREPROCESSING PIPELINE")
        print("="*60)
    
    # Step 1: Baseline Correction
    if verbose:
        print("\n[1/3] Applying AsLS Baseline Correction...")
    results['baseline_corrected'] = apply_baseline_correction(
        spectra, lam=asls_lam, p=asls_p, verbose=verbose
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
        print(f"  Input shape:   {spectra.shape}")
        print(f"  Output shape: {results['normalized'].shape}")
        print("="*60)
    
    return results


# ============================================================================
# VALIDATION & TESTING FUNCTIONS
# ============================================================================

def validate_preprocessing(
    raw:  np.ndarray,
    processed: np.ndarray,
    wavelengths: np.ndarray,
    sample_idx: int = 0
) -> Dict[str, float]:
    """
    Validate preprocessing quality with quantitative metrics.
    
    Parameters
    ----------
    raw : np.ndarray
        Raw spectrum
    processed : np.ndarray
        Preprocessed spectrum
    wavelengths :  np.ndarray
        Wavelength values
    sample_idx :  int
        Sample index for reporting
        
    Returns
    -------
    metrics : dict
        Validation metrics
    """
    metrics = {}
    
    # 1. Signal-to-Noise Ratio (SNR) improvement
    # Estimate noise as standard deviation in a flat region
    flat_region = slice(100, 200)  # Adjust based on your data
    noise_raw = np.std(raw[flat_region])
    noise_processed = np.std(processed[flat_region])
    
    signal_raw = np.max(raw) - np.median(raw)
    signal_processed = np.max(processed) - np.median(processed)
    
    snr_raw = signal_raw / noise_raw if noise_raw > 0 else 0
    snr_processed = signal_processed / noise_processed if noise_processed > 0 else 0
    
    metrics['snr_raw'] = snr_raw
    metrics['snr_processed'] = snr_processed
    metrics['snr_improvement'] = (snr_processed - snr_raw) / snr_raw * 100 if snr_raw > 0 else 0
    
    # 2. Baseline flatness (standard deviation of baseline region)
    baseline_region = slice(0, 100)
    metrics['baseline_std_raw'] = np.std(raw[baseline_region])
    metrics['baseline_std_processed'] = np.std(processed[baseline_region])
    
    # 3. Peak preservation (correlation between raw and processed)
    # High correlation means peaks are preserved
    metrics['peak_correlation'] = np.corrcoef(raw, processed)[0, 1]
    
    # 4. Normalization verification
    metrics['l2_norm'] = np.linalg.norm(processed)
    metrics['is_normalized'] = np.isclose(metrics['l2_norm'], 1.0)
    
    # 5. Dynamic range
    metrics['dynamic_range_raw'] = np.max(raw) - np.min(raw)
    metrics['dynamic_range_processed'] = np.max(processed) - np.min(processed)
    
    print(f"\n{'='*50}")
    print(f"VALIDATION METRICS (Sample {sample_idx})")
    print(f"{'='*50}")
    print(f"SNR Improvement:      {metrics['snr_improvement']: +.1f}%")
    print(f"Baseline Std (raw):  {metrics['baseline_std_raw']:.4f}")
    print(f"Baseline Std (proc): {metrics['baseline_std_processed']:.6f}")
    print(f"Peak Correlation:    {metrics['peak_correlation']:.4f}")
    print(f"L2 Norm:             {metrics['l2_norm']:.6f} (normalized: {metrics['is_normalized']})")
    print(f"{'='*50}")
    
    return metrics


def compare_smoothing_methods(
    spectra_baseline: np.ndarray,
    wavelengths: np.ndarray,
    sample_idx: int = 0
) -> Dict[str, np.ndarray]:
    """
    Compare Savitzky-Golay vs Gaussian smoothing as recommended in methodology.
    
    The context document notes:  "Test both SG AND Modified Sinc filter. 
    If alternative improves accuracy >2%, use it."
    
    Parameters
    ----------
    spectra_baseline : np.ndarray
        Baseline-corrected spectra
    wavelengths : np.ndarray
        Wavelength values
    sample_idx : int
        Sample to visualize
        
    Returns
    -------
    results : dict
        Smoothed spectra from each method
    """
    sample = spectra_baseline[sample_idx]
    
    results = {
        'original': sample,
        'savgol_7_2': savgol_filter(sample, 7, 2),
        'savgol_11_3': savgol_filter(sample, 11, 3),  # Default
        'savgol_15_3': savgol_filter(sample, 15, 3),
        'gaussian_1. 0': gaussian_filter1d(sample, sigma=1.0),
        'gaussian_1.5': gaussian_filter1d(sample, sigma=1.5),
        'gaussian_2.0': gaussian_filter1d(sample, sigma=2.0),
    }
    
    # Calculate noise reduction for each
    print("\n" + "="*60)
    print("SMOOTHING METHOD COMPARISON")
    print("="*60)
    print(f"{'Method':<20} {'Noise Std':<12} {'Peak Max':<12} {'Peak Preserved'}")
    print("-"*60)
    
    flat_region = slice(100, 200)
    original_noise = np.std(sample[flat_region])
    original_peak = np.max(sample)
    
    for name, smoothed in results.items():
        noise = np.std(smoothed[flat_region])
        peak = np.max(smoothed)
        peak_preserved = peak / original_peak * 100
        print(f"{name:<20} {noise:<12.6f} {peak:<12.4f} {peak_preserved:. 1f}%")
    
    print("="*60)
    print("Recommendation: Choose method with lowest noise AND >95% peak preservation")
    
    return results


def plot_preprocessing_stages(
    results: Dict[str, np.ndarray],
    wavelengths: np.ndarray,
    sample_idx: int = 0,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize all preprocessing stages for validation.
    
    Parameters
    ----------
    results : dict
        Output from preprocess_libs_spectra()
    wavelengths : np. ndarray
        Wavelength values in nm
    sample_idx : int
        Sample index to plot
    save_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    stages = ['raw', 'baseline_corrected', 'smoothed', 'normalized']
    titles = [
        'Step 0: Raw Spectrum',
        'Step 1: After AsLS Baseline Correction',
        'Step 2: After Savitzky-Golay Smoothing',
        'Step 3: After L2 Normalization (FINAL)'
    ]
    
    for ax, stage, title in zip(axes.flat, stages, titles):
        spectrum = results[stage][sample_idx]
        ax. plot(wavelengths, spectrum, 'b-', linewidth=0.5)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Intensity (a.u.)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Add statistics annotation
        stats_text = f"Max: {np.max(spectrum):.4f}\nMean: {np.mean(spectrum):.4f}\nStd: {np.std(spectrum):.4f}"
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', horizontalalignment='right',
                fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'LIBS Preprocessing Pipeline - Sample {sample_idx}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()


def plot_class_comparison(
    spectra: np. ndarray,
    labels: np.ndarray,
    wavelengths: np.ndarray,
    n_samples_per_class: int = 5
) -> None:
    """
    Plot preprocessed spectra grouped by class for visual validation.
    
    This helps verify that preprocessing preserves class-discriminative features.
    
    Parameters
    ----------
    spectra : np.ndarray
        Preprocessed spectra
    labels : np. ndarray
        Class labels
    wavelengths : np.ndarray
        Wavelength values
    n_samples_per_class :  int
        Number of samples to plot per class
    """
    unique_classes = np.unique(labels)
    n_classes = len(unique_classes)
    
    fig, axes = plt.subplots(n_classes, 1, figsize=(14, 4*n_classes), sharex=True)
    if n_classes == 1:
        axes = [axes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    
    for ax, cls, color in zip(axes, unique_classes, colors):
        cls_indices = np.where(labels == cls)[0][:n_samples_per_class]
        
        for idx in cls_indices:
            ax.plot(wavelengths, spectra[idx], color=color, alpha=0.5, linewidth=0.5)
        
        # Plot mean spectrum
        mean_spectrum = np.mean(spectra[labels == cls], axis=0)
        ax.plot(wavelengths, mean_spectrum, color='black', linewidth=2, label=f'Class {cls} Mean')
        
        ax.set_ylabel('Intensity (normalized)')
        ax.set_title(f'Class:  {cls} (n={len(np.where(labels == cls)[0])})')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Wavelength (nm)')
    plt.suptitle('Preprocessed Spectra by Class', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ============================================================================
# MAIN EXECUTION EXAMPLE
# ============================================================================

if __name__ == "__main__": 
    # -------------------------------------------------------------------------
    # OPTION 1: Load your actual data
    # -------------------------------------------------------------------------
    # spectra, labels, wavelengths = load_libs_data("your_libs_data.csv")
    
    # -------------------------------------------------------------------------
    # OPTION 2: Generate synthetic data for testing pipeline
    # -------------------------------------------------------------------------
    print("Generating synthetic LIBS data for pipeline demonstration...")
    
    np.random.seed(42)
    n_samples = 324
    n_features = 12981
    wavelengths = np.linspace(200, 950, n_features)
    
    # Create synthetic spectra with realistic LIBS characteristics
    spectra = np.zeros((n_samples, n_features))
    labels = np.repeat(['High', 'Medium', 'Low'], n_samples // 3 + 1)[:n_samples]
    
    for i in range(n_samples):
        # Baseline (Bremsstrahlung continuum)
        baseline = 1000 * np.exp(-0.005 * (wavelengths - 200))
        
        # Add peaks at N, P, K wavelengths (from NIST database)
        # Nitrogen peaks:  742.4, 744.2, 746.8 nm
        # Phosphorus peaks: 253.5, 255.3 nm
        # Potassium peaks: 766.5, 769.9 nm
        
        peak_positions = [253.5, 255.3, 742.4, 744.2, 746.8, 766.5, 769.9]
        peak_intensities = np.random.uniform(500, 2000, len(peak_positions))
        
        # Modify intensities based on class
        if labels[i] == 'High':
            peak_intensities *= 1.5
        elif labels[i] == 'Low':
            peak_intensities *= 0.6
        
        for pos, intensity in zip(peak_positions, peak_intensities):
            idx = np.argmin(np.abs(wavelengths - pos))
            peak = intensity * np.exp(-0.5 * ((wavelengths - pos) / 0.5)**2)
            spectra[i] += peak
        
        spectra[i] += baseline
        spectra[i] += np.random.normal(0, 50, n_features)  # Noise
    
    print(f"Synthetic data created:  {spectra.shape}")
    
    # -------------------------------------------------------------------------
    # STEP 1: Stratified Split
    # -------------------------------------------------------------------------
    X_train, X_test, y_train, y_test = stratified_split(spectra, labels)
    
    # -------------------------------------------------------------------------
    # STEP 2: Run Preprocessing Pipeline
    # -------------------------------------------------------------------------
    results_train = preprocess_libs_spectra(
        X_train,
        asls_lam=1e6,
        asls_p=0.01,
        sg_window=11,
        sg_order=3,
        verbose=True
    )
    
    # Apply same preprocessing to test set
    results_test = preprocess_libs_spectra(X_test, verbose=False)
    
    # -------------------------------------------------------------------------
    # STEP 3: Validation
    # -------------------------------------------------------------------------
    # Validate preprocessing quality
    metrics = validate_preprocessing(
        raw=X_train[0],
        processed=results_train['normalized'][0],
        wavelengths=wavelengths,
        sample_idx=0
    )
    
    # Compare smoothing methods
    smoothing_comparison = compare_smoothing_methods(
        results_train['baseline_corrected'],
        wavelengths,
        sample_idx=0
    )
    
    # -------------------------------------------------------------------------
    # STEP 4: Visualization
    # -------------------------------------------------------------------------
    plot_preprocessing_stages(results_train, wavelengths, sample_idx=0)
    plot_class_comparison(results_train['normalized'], y_train, wavelengths)
    
    # -------------------------------------------------------------------------
    # STEP 5: Save preprocessed data for Phase 2
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("SAVING PREPROCESSED DATA")
    print("="*60)
    
    np.save('X_train_preprocessed.npy', results_train['normalized'])
    np.save('X_test_preprocessed.npy', results_test['normalized'])
    np.save('y_train. npy', y_train)
    np.save('y_test.npy', y_test)
    np.save('wavelengths.npy', wavelengths)
    
    print("✓ Saved:  X_train_preprocessed.npy")
    print("✓ Saved: X_test_preprocessed.npy")
    print("✓ Saved: y_train.npy")
    print("✓ Saved: y_test.npy")
    print("✓ Saved:  wavelengths.npy")
    print("\nReady for Phase 2: Feature Selection")