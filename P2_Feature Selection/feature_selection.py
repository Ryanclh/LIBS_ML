"""
Feature Selection Comparative Analysis
======================================
Systematic comparison of ANOVA, RFE, and Random Forest Gini Importance
for LIBS spectroscopic feature selection under TinyML constraints.

Author: clh-um
Date: 2026-01-15
FYP:  Embedded Machine Learning for On-Device LIBS-Based Soil Nutrient Estimation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import (
    SelectKBest, f_classif, RFE, RFECV
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, train_test_split
)
from sklearn.preprocessing import normalize, LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
from scipy.stats import ttest_rel
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple
import sys
import os

warnings.filterwarnings('ignore')

# Add preprocessing module path
sys.path.append(os.path.dirname(__file__))

# NIST Atomic Emission Lines (nm) for Physical Validation
NIST_LINES = {
    'N': [746.8312, 744.2294, 868.0279, 821.6337, 862.9245],
    'P': [253.5603, 255.3282, 214.9144, 213.6180],
    'K': [766.4899, 769.8965, 404.7214, 404.4141],
    'Fe': [259.9397, 358.1195, 344.0606, 373.4867],  # Matrix element
    'Ca': [393.3663, 396.8469, 422.6728],  # Matrix element
    'Mg': [285.2127, 279.5528, 383.2307],  # Matrix element
    'Pb': [405.7806, 283.3053, 368.3462],  # Contaminant
    'Cd': [228.8018, 226.5021, 214.4392],  # Contaminant
    'Cr': [267.7161, 283.5629, 357.8687]   # Contaminant
}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_preprocessed_data(
    filepath: str,
    labels_file: str = 'irish_soil_labels.csv'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load preprocessed LIBS spectra and labels.
    
    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
        Preprocessed spectra (Polynomial + SG + L2-Norm)
    y_nutrient : np.ndarray
        Nutrient labels (0=Low, 1=Medium, 2=High)
    y_contaminant : np.ndarray
        Contaminant labels (0=Clean, 1=Contaminated)
    wavelengths : np.ndarray
        Wavelength values (nm)
    """
    print(f"\n{'='*70}")
    print("LOADING PREPROCESSED DATA")
    print(f"{'='*70}")
    
    # Check file extension
    if filepath.endswith('.npz'):
        # Load from NumPy compressed format (Phase 1 output)
        print(f"  Loading from .npz file: {filepath}")
        
        # Load with allow_pickle=True for object arrays (sample names)
        data = np.load(filepath, allow_pickle=True)
        
        # Debug: Print available keys
        print(f"  Available keys in .npz file: {list(data.files)}")
        
        # Based on your file structure, use 'normalized' key
        if 'normalized' in data.files:
            X = data['normalized']
            print(f"  ✓ Loaded normalized spectra (Polynomial + SG + L2)")
        elif 'smoothed' in data.files:
            X = data['smoothed']
            print(f"  ✓ Loaded smoothed spectra (Polynomial + SG)")
            # Apply L2 normalization
            from sklearn.preprocessing import normalize
            X = normalize(X, norm='l2', axis=1)
            print(f"  → Applied L2 normalization")
        elif 'baseline_corrected' in data.files:
            X = data['baseline_corrected']
            print(f"  ✓ Loaded baseline-corrected spectra")
            # Apply SG filter + L2 normalization
            X = apply_savgol_filter(X, window=15, order=3)
            from sklearn.preprocessing import normalize
            X = normalize(X, norm='l2', axis=1)
            print(f"  → Applied SG filter + L2 normalization")
        else:
            raise KeyError(f"Could not find spectra data. Available keys: {list(data.files)}")
        
        # Get wavelengths (check multiple possible keys)
        wavelength_keys = ['wavelengths', 'wavelength', 'wl', 'lambda']
        wavelengths = None
        
        for key in wavelength_keys:
            if key in data.files:
                wavelengths = data[key]
                print(f"  ✓ Found wavelengths under key: '{key}'")
                break
        
        if wavelengths is None:
            # If no wavelengths found, create synthetic ones based on shape
            print(f"  ⚠ Warning: No wavelengths found, creating synthetic range")
            wavelengths = np.linspace(200, 900, X.shape[1])
        
        # Get sample names (check multiple possible keys)
        sample_name_keys = ['sample_names', 'samples', 'sample_ids']
        sample_names = None
        
        for key in sample_name_keys:
            if key in data.files:
                try:
                    sample_names = data[key]
                    # Handle different encoding formats
                    if len(sample_names) > 0:
                        if isinstance(sample_names[0], bytes):
                            sample_names = np.array([s.decode('utf-8') for s in sample_names])
                        elif isinstance(sample_names[0], np.bytes_):
                            sample_names = np.array([s.decode('utf-8') for s in sample_names])
                    print(f"  ✓ Found sample names under key: '{key}'")
                    break
                except Exception as e:
                    print(f"  ⚠ Warning: Could not load sample names from '{key}': {e}")
                    sample_names = None
        
        print(f"\n✓ Loaded spectra: {X.shape[0]} samples × {X.shape[1]} features")
        print(f"  Wavelength range: {wavelengths[0]:.2f} - {wavelengths[-1]:.2f} nm")
        
    elif filepath.endswith('.csv'):
        # Load from CSV (raw data - needs preprocessing)
        print(f"  Loading from .csv file: {filepath}")
        df_spectra = pd.read_csv(filepath, index_col=0)
        
        # Check if wavelengths are in rows or columns
        if len(df_spectra.index) > len(df_spectra.columns):
            print("  → Wavelengths in rows, transposing...")
            df_spectra = df_spectra.T
        
        wavelengths = df_spectra.columns.astype(float).values
        sample_names = df_spectra.index.values
        X_raw = df_spectra.values
        
        print(f"✓ Loaded spectra: {X_raw.shape[0]} samples × {X_raw.shape[1]} features")
        print(f"  Wavelength range: {wavelengths[0]:.2f} - {wavelengths[-1]:.2f} nm")
        
        # Apply preprocessing pipeline (Section 4.1 finalized)
        print("\n[Preprocessing Pipeline]")
        print("  Step 1: Polynomial 5th-order baseline correction...")
        X_baseline = apply_polynomial_baseline(X_raw)
        
        print("  Step 2: Savitzky-Golay filter (w=15, o=3)...")
        X_filtered = apply_savgol_filter(X_baseline, window=15, order=3)
        
        print("  Step 3: L2-Normalization...")
        from sklearn.preprocessing import normalize
        X = normalize(X_filtered, norm='l2', axis=1)
        print("✓ Preprocessing complete")
    
    else:
        raise ValueError(f"Unsupported file format: {filepath}. Use .npz or .csv")
    
    # Load labels from CSV
    if not os.path.isabs(labels_file):
        # Try relative to data file first
        labels_path = os.path.join(os.path.dirname(filepath), labels_file)
        if not os.path.exists(labels_path):
            # Try parent directory
            labels_path = os.path.join(os.path.dirname(os.path.dirname(filepath)), labels_file)
    else:
        labels_path = labels_file
    
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found at: {labels_path}")
    
    print(f"\n  Loading labels from: {labels_path}")
    df_labels = pd.read_csv(labels_path)
    
    # Create label mapping
    label_map = dict(zip(df_labels['Sample_Name'], 
                         zip(df_labels['Nutrient_Label'], 
                             df_labels['Contaminant_Label'])))
    
    # Match labels to samples
    if sample_names is None:
        # If no sample names, assume labels match by index
        print("  ⚠ Warning: No sample names found, matching labels by index")
        y_nutrient = df_labels['Nutrient_Label'].values[:X.shape[0]]
        y_contaminant = df_labels['Contaminant_Label'].values[:X.shape[0]]
    else:
        y_nutrient = []
        y_contaminant = []
        
        for name in sample_names:
            # Handle any remaining byte string conversions
            if isinstance(name, (bytes, np.bytes_)):
                name = name.decode('utf-8')
            
            if name in label_map:
                nut, cont = label_map[name]
                y_nutrient.append(nut)
                y_contaminant.append(cont)
            else:
                y_nutrient.append(-1)
                y_contaminant.append(-1)
        
        y_nutrient = np.array(y_nutrient)
        y_contaminant = np.array(y_contaminant)
    
    print(f"\n✓ Labels loaded:")
    print(f"  Nutrient (valid): {np.sum(y_nutrient >= 0)} samples")
    print(f"  Nutrient distribution: Low={np.sum(y_nutrient==0)}, Med={np.sum(y_nutrient==1)}, High={np.sum(y_nutrient==2)}")
    print(f"  Contaminant: Clean={np.sum(y_contaminant==0)}, Contaminated={np.sum(y_contaminant==1)}")
    print(f"{'='*70}\n")
    
    return X, y_nutrient, y_contaminant, wavelengths

def apply_polynomial_baseline(spectra: np.ndarray, order: int = 5) -> np.ndarray:
    """Polynomial baseline correction (Section 4.1.1)."""
    corrected = np.zeros_like(spectra)
    for i in range(spectra.shape[0]):
        x = np.arange(len(spectra[i]))
        coeffs = np.polyfit(x, spectra[i], order)
        baseline = np.polyval(coeffs, x)
        corrected[i] = np.maximum(spectra[i] - baseline, 0)
    return corrected


def apply_savgol_filter(spectra: np.ndarray, window: int = 15, 
                       order: int = 3) -> np.ndarray:
    """Savitzky-Golay filter (Section 4.1.2)."""
    from scipy.signal import savgol_filter
    if window % 2 == 0:
        window += 1
    return np.apply_along_axis(
        lambda x: savgol_filter(x, window, order), axis=1, arr=spectra
    )

def apply_polynomial_baseline(spectra: np.ndarray, order: int = 5) -> np.ndarray:
    """Polynomial baseline correction (Section 4.1.1)."""
    corrected = np.zeros_like(spectra)
    for i in range(spectra.shape[0]):
        x = np.arange(len(spectra[i]))
        coeffs = np.polyfit(x, spectra[i], order)
        baseline = np.polyval(coeffs, x)
        corrected[i] = np.maximum(spectra[i] - baseline, 0)
    return corrected


def apply_savgol_filter(spectra: np.ndarray, window: int = 15, 
                       order: int = 3) -> np.ndarray:
    """Savitzky-Golay filter (Section 4.1.2)."""
    from scipy.signal import savgol_filter
    if window % 2 == 0:
        window += 1
    return np.apply_along_axis(
        lambda x: savgol_filter(x, window, order), axis=1, arr=spectra
    )


# ============================================================================
# FEATURE SELECTION METHODS
# ============================================================================

@dataclass
class FeatureSelectionResult:
    """Container for feature selection results."""
    method_name: str
    selected_features: np.ndarray  # Indices
    selected_wavelengths: np.ndarray  # Wavelength values (nm)
    training_time: float  # seconds
    memory_footprint: int  # bytes
    accuracy: float
    n_features: int
    feature_importances: np.ndarray = None  # Optional


def select_features_anova(
    X: np.ndarray,
    y:  np.ndarray,
    wavelengths: np.ndarray,
    n_features: int = 87,
    verbose: bool = True
) -> FeatureSelectionResult:
    """
    ANOVA F-test feature selection (Filter-based).
    
    Reference: Univariate statistical test, fast screening.
    """
    if verbose:
        print(f"\n{'='*70}")
        print("METHOD 1: ANOVA F-TEST (Filter-Based)")
        print(f"{'='*70}")
        print(f"Target features: {n_features}")
    
    start_time = time.time()
    
    # ANOVA F-test scoring
    selector = SelectKBest(score_func=f_classif, k=n_features)
    selector.fit(X, y)
    
    selected_indices = selector.get_support(indices=True)
    selected_wavelengths = wavelengths[selected_indices]
    
    # Evaluate accuracy
    X_selected = X[: , selected_indices]
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf, X_selected, y, cv=cv, scoring='accuracy')
    accuracy = np.mean(cv_scores)
    
    training_time = time.time() - start_time
    
    # Memory footprint:  just indices
    memory_footprint = n_features * 4  # 4 bytes per int32 index
    
    if verbose: 
        print(f"  Training time: {training_time:.3f} seconds")
        print(f"  Selected features: {len(selected_indices)}")
        print(f"  5-Fold CV Accuracy: {accuracy*100:.2f}%")
        print(f"  Memory footprint: {memory_footprint} bytes ({memory_footprint/1024:.2f} KB)")
    
    return FeatureSelectionResult(
        method_name='ANOVA F-test',
        selected_features=selected_indices,
        selected_wavelengths=selected_wavelengths,
        training_time=training_time,
        memory_footprint=memory_footprint,
        accuracy=accuracy,
        n_features=len(selected_indices),
        feature_importances=selector.scores_[selected_indices]
    )


def select_features_rfe(
    X: np.ndarray,
    y: np.ndarray,
    wavelengths: np.ndarray,
    n_features: int = 87,
    verbose: bool = True
) -> FeatureSelectionResult: 
    """
    Recursive Feature Elimination with Cross-Validation (Wrapper-based).
    
    Reference: Iterative backward elimination, model-aware.
    """
    if verbose:
        print(f"\n{'='*70}")
        print("METHOD 2: RFE WITH CROSS-VALIDATION (Wrapper-Based)")
        print(f"{'='*70}")
        print(f"Target features: {n_features}")
    
    start_time = time.time()
    
    # Use Linear SVM as estimator (faster than RF for RFE)
    from sklearn.svm import SVC
    estimator = SVC(kernel='linear', random_state=42)
    
    # RFE with target number of features
    selector = RFE(estimator=estimator, n_features_to_select=n_features, step=100)
    selector.fit(X, y)
    
    selected_indices = np.where(selector.support_)[0]
    selected_wavelengths = wavelengths[selected_indices]
    
    # Evaluate with RF (final classifier)
    X_selected = X[:, selected_indices]
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf, X_selected, y, cv=cv, scoring='accuracy')
    accuracy = np.mean(cv_scores)
    
    training_time = time.time() - start_time
    memory_footprint = n_features * 4
    
    if verbose:
        print(f"  Training time:  {training_time:.3f} seconds")
        print(f"  Selected features: {len(selected_indices)}")
        print(f"  5-Fold CV Accuracy: {accuracy*100:.2f}%")
        print(f"  Memory footprint:  {memory_footprint} bytes ({memory_footprint/1024:.2f} KB)")
    
    return FeatureSelectionResult(
        method_name='RFE-CV',
        selected_features=selected_indices,
        selected_wavelengths=selected_wavelengths,
        training_time=training_time,
        memory_footprint=memory_footprint,
        accuracy=accuracy,
        n_features=len(selected_indices),
        feature_importances=selector.ranking_[selected_indices]
    )


def select_features_rf_gini(
    X:  np.ndarray,
    y: np.ndarray,
    wavelengths: np.ndarray,
    n_features: int = 87,
    verbose:  bool = True
) -> FeatureSelectionResult:
    """
    Random Forest Gini Importance (Embedded method).
    
    Reference: Section 4.3 - Captures non-linear interactions,
    NIST-aligned features (N: 746.8nm, P:253.5nm, K:766.5nm).
    """
    if verbose:
        print(f"\n{'='*70}")
        print("METHOD 3: RANDOM FOREST GINI IMPORTANCE (Embedded)")
        print(f"{'='*70}")
        print(f"Target features: {n_features}")
    
    start_time = time.time()
    
    # Train RF for feature importance
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,  # No depth limit for feature importance estimation
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X, y)
    
    # Get Gini importances
    importances = rf.feature_importances_
    selected_indices = np.argsort(importances)[-n_features:][::-1]
    selected_wavelengths = wavelengths[selected_indices]
    
    # Evaluate with depth-limited RF (deployment configuration)
    X_selected = X[:, selected_indices]
    rf_eval = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf_eval, X_selected, y, cv=cv, scoring='accuracy')
    accuracy = np.mean(cv_scores)
    
    training_time = time.time() - start_time
    memory_footprint = n_features * 4
    
    if verbose:
        print(f"  Training time: {training_time:.3f} seconds")
        print(f"  Selected features: {len(selected_indices)}")
        print(f"  5-Fold CV Accuracy: {accuracy*100:.2f}%")
        print(f"  Memory footprint: {memory_footprint} bytes ({memory_footprint/1024:.2f} KB)")
    
    return FeatureSelectionResult(
        method_name='RF Gini',
        selected_features=selected_indices,
        selected_wavelengths=selected_wavelengths,
        training_time=training_time,
        memory_footprint=memory_footprint,
        accuracy=accuracy,
        n_features=len(selected_indices),
        feature_importances=importances[selected_indices]
    )


# ============================================================================
# NIST PHYSICAL VALIDATION
# ============================================================================

def validate_nist_alignment(
    selected_wavelengths: np.ndarray,
    tolerance: float = 0.5,
    task: str = 'nutrient'
) -> Dict: 
    """
    Validate selected features against NIST atomic emission lines.
    
    Parameters
    ----------
    selected_wavelengths : np.ndarray
        Wavelengths selected by feature selection method
    tolerance : float
        Matching tolerance in nm (default ±0.5 nm)
    task : str
        'nutrient' (N/P/K focus) or 'contaminant' (Pb/Cd/Cr focus)
    
    Returns
    -------
    dict
        Validation metrics and matched lines
    """
    if task == 'nutrient': 
        target_elements = ['N', 'P', 'K', 'Fe', 'Ca', 'Mg']
    else: 
        target_elements = ['Pb', 'Cd', 'Cr', 'Fe']
    
    matched_lines = []
    total_nist_lines = 0
    
    for element in target_elements:
        if element not in NIST_LINES:
            continue
        
        nist_lines = NIST_LINES[element]
        total_nist_lines += len(nist_lines)
        
        for nist_wl in nist_lines: 
            # Check if any selected wavelength is within tolerance
            matches = np.abs(selected_wavelengths - nist_wl) < tolerance
            if np.any(matches):
                matched_wl = selected_wavelengths[matches][0]
                matched_lines.append({
                    'element': element,
                    'nist_wavelength': nist_wl,
                    'matched_wavelength': matched_wl,
                    'difference': abs(matched_wl - nist_wl)
                })
    
    n_matched = len(matched_lines)
    match_percentage = (n_matched / total_nist_lines) * 100 if total_nist_lines > 0 else 0
    
    return {
        'n_matched': n_matched,
        'total_nist_lines': total_nist_lines,
        'match_percentage': match_percentage,
        'matched_lines': matched_lines,
        'n_selected': len(selected_wavelengths)
    }


# ============================================================================
# COMPARATIVE ANALYSIS
# ============================================================================

def compare_feature_selection_methods(
    X: np.ndarray,
    y: np.ndarray,
    wavelengths: np.ndarray,
    n_features:  int = 87,
    task: str = 'nutrient'
) -> Dict:
    """
    Comprehensive comparison of all three feature selection methods.
    
    Returns
    -------
    dict
        Comparison results with scoring
    """
    print(f"\n{'='*70}")
    print(f"FEATURE SELECTION COMPARATIVE ANALYSIS")
    print(f"Task: {task.upper()}")
    print(f"Target Features: {n_features}")
    print(f"{'='*70}")
    
    # Run all three methods
    anova_result = select_features_anova(X, y, wavelengths, n_features)
    rfe_result = select_features_rfe(X, y, wavelengths, n_features)
    rf_gini_result = select_features_rf_gini(X, y, wavelengths, n_features)
    
    results = {
        'ANOVA': anova_result,
        'RFE': rfe_result,
        'RF Gini': rf_gini_result
    }
    
    # NIST Validation
    print(f"\n{'='*70}")
    print("NIST PHYSICAL VALIDATION")
    print(f"{'='*70}")
    
    nist_validations = {}
    for method_name, result in results.items():
        nist_val = validate_nist_alignment(
            result.selected_wavelengths, 
            tolerance=0.5, 
            task=task
        )
        nist_validations[method_name] = nist_val
        
        print(f"\n{method_name}:")
        print(f"  Matched NIST lines: {nist_val['n_matched']}/{nist_val['total_nist_lines']}")
        print(f"  Match percentage: {nist_val['match_percentage']:.1f}%")
    
    # Weighted Scoring (Section 4.3.4)
    print(f"\n{'='*70}")
    print("WEIGHTED SCORING SYSTEM")
    print(f"{'='*70}")
    print("Weights: Accuracy 40%, Feature Count 30%, Efficiency 15%, Interpretability 15%")
    
    scores = calculate_weighted_scores(results, nist_validations)
    
    # Print final ranking
    print(f"\n{'='*70}")
    print("FINAL RANKING")
    print(f"{'='*70}")
    sorted_methods = sorted(scores.items(), key=lambda x: x[1]['total_score'], reverse=True)
    
    for rank, (method, score_dict) in enumerate(sorted_methods, 1):
        print(f"\n{rank}.{method}")
        print(f"   Total Score: {score_dict['total_score']:.2f}/100")
        print(f"   - Accuracy:         {score_dict['accuracy_score']:.2f}/40")
        print(f"   - Feature Count:   {score_dict['feature_score']:.2f}/30")
        print(f"   - Efficiency:      {score_dict['efficiency_score']:.2f}/15")
        print(f"   - Interpretability: {score_dict['interpretability_score']:.2f}/15")
    
    return {
        'results': results,
        'nist_validations': nist_validations,
        'scores': scores,
        'recommended':  sorted_methods[0][0]
    }


def calculate_weighted_scores(
    results: Dict[str, FeatureSelectionResult],
    nist_validations: Dict[str, Dict]
) -> Dict: 
    """
    Calculate weighted scores for feature selection methods.
    
    Scoring Criteria:
    - Accuracy:  40% (higher is better)
    - Feature Count: 30% (fewer is better, but target is 87)
    - Computational Efficiency: 15% (faster is better)
    - Interpretability (NIST alignment): 15% (higher is better)
    """
    scores = {}
    
    # Extract metrics
    accuracies = {name: res.accuracy for name, res in results.items()}
    training_times = {name: res.training_time for name, res in results.items()}
    nist_matches = {name: val['match_percentage'] for name, val in nist_validations.items()}
    
    # Normalize metrics (0-100 scale)
    max_acc = max(accuracies.values())
    min_acc = min(accuracies.values())
    
    max_time = max(training_times.values())
    min_time = min(training_times.values())
    
    max_nist = max(nist_matches.values())
    
    for method_name, result in results.items():
        # 1.Accuracy Score (40 points max)
        acc_normalized = (result.accuracy - min_acc) / (max_acc - min_acc) if max_acc > min_acc else 1.0
        accuracy_score = acc_normalized * 40
        
        # 2.Feature Count Score (30 points max)
        # Penalize deviation from target (87 features)
        target_features = 87
        feature_deviation = abs(result.n_features - target_features) / target_features
        feature_score = max(0, (1 - feature_deviation)) * 30
        
        # 3.Computational Efficiency Score (15 points max)
        # Lower time is better
        time_normalized = 1 - ((result.training_time - min_time) / (max_time - min_time)) if max_time > min_time else 1.0
        efficiency_score = time_normalized * 15
        
        # 4.Interpretability Score (15 points max)
        # Based on NIST alignment percentage
        nist_pct = nist_validations[method_name]['match_percentage']
        interpretability_score = (nist_pct / max_nist) * 15 if max_nist > 0 else 0
        
        total_score = accuracy_score + feature_score + efficiency_score + interpretability_score
        
        scores[method_name] = {
            'accuracy_score': accuracy_score,
            'feature_score': feature_score,
            'efficiency_score': efficiency_score,
            'interpretability_score': interpretability_score,
            'total_score': total_score
        }
    
    return scores


# ============================================================================
# VISUALIZATION & REPORTING
# ============================================================================

def plot_comparison_dashboard(
    comparison_results: Dict,
    save_path: str = 'feature_selection_comparison.png'
):
    """
    Generate comprehensive comparison dashboard.
    """
    results = comparison_results['results']
    nist_vals = comparison_results['nist_validations']
    scores = comparison_results['scores']
    
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Accuracy Comparison
    ax1 = fig.add_subplot(2, 3, 1)
    methods = list(results.keys())
    accuracies = [results[m].accuracy * 100 for m in methods]
    colors = ['green' if m == comparison_results['recommended'] else 'steelblue' 
              for m in methods]
    
    bars = ax1.bar(methods, accuracies, color=colors, edgecolor='black', linewidth=2)
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{acc:.2f}%', ha='center', fontweight='bold')
    
    ax1.set_ylabel('5-Fold CV Accuracy (%)', fontsize=11)
    ax1.set_title('Classification Accuracy Comparison', fontsize=12, fontweight='bold')
    ax1.set_ylim([min(accuracies)*0.95, max(accuracies)*1.05])
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: NIST Alignment
    ax2 = fig.add_subplot(2, 3, 2)
    nist_percentages = [nist_vals[m]['match_percentage'] for m in methods]
    
    bars = ax2.bar(methods, nist_percentages, color=colors, edgecolor='black', linewidth=2)
    for bar, pct in zip(bars, nist_percentages):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{pct:.1f}%', ha='center', fontweight='bold')
    
    ax2.set_ylabel('NIST Line Match (%)', fontsize=11)
    ax2.set_title('Physical Validation (NIST Alignment)', fontsize=12, fontweight='bold')
    ax2.set_ylim([0, max(nist_percentages)*1.2])
    ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: Training Time
    ax3 = fig.add_subplot(2, 3, 3)
    times = [results[m].training_time for m in methods]
    
    bars = ax3.bar(methods, times, color=colors, edgecolor='black', linewidth=2)
    for bar, t in zip(bars, times):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{t:.2f}s', ha='center', fontweight='bold')
    
    ax3.set_ylabel('Training Time (seconds)', fontsize=11)
    ax3.set_title('Computational Efficiency', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # Plot 4: Weighted Scores Breakdown
    ax4 = fig.add_subplot(2, 3, 4)
    score_categories = ['Accuracy\n(40%)', 'Features\n(30%)', 'Efficiency\n(15%)', 'Interpret.\n(15%)']
    x = np.arange(len(methods))
    width = 0.2
    
    for i, category in enumerate(['accuracy_score', 'feature_score', 'efficiency_score', 'interpretability_score']):
        values = [scores[m][category] for m in methods]
        ax4.bar(x + i*width, values, width, label=score_categories[i])
    
    ax4.set_xticks(x + width * 1.5)
    ax4.set_xticklabels(methods)
    ax4.set_ylabel('Score', fontsize=11)
    ax4.set_title('Weighted Score Breakdown', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(axis='y', alpha=0.3)
    
    # Plot 5: Total Score Comparison
    ax5 = fig.add_subplot(2, 3, 5)
    total_scores = [scores[m]['total_score'] for m in methods]
    
    bars = ax5.barh(methods, total_scores, color=colors, edgecolor='black', linewidth=2)
    for bar, score in zip(bars, total_scores):
        ax5.text(score + 1, bar.get_y() + bar.get_height()/2,
                f'{score:.1f}/100', va='center', fontweight='bold')
    
    ax5.set_xlabel('Total Weighted Score (0-100)', fontsize=11)
    ax5.set_title(f'Final Ranking (Recommended: {comparison_results["recommended"]})', 
                  fontsize=12, fontweight='bold')
    ax5.set_xlim([0, 105])
    ax5.grid(axis='x', alpha=0.3)
    
    # Plot 6: Memory Footprint
    ax6 = fig.add_subplot(2, 3, 6)
    memories = [results[m].memory_footprint / 1024 for m in methods]  # Convert to KB
    
    bars = ax6.bar(methods, memories, color=colors, edgecolor='black', linewidth=2)
    for bar, mem in zip(bars, memories):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{mem:.2f} KB', ha='center', fontweight='bold')
    
    ax6.set_ylabel('Memory Footprint (KB)', fontsize=11)
    ax6.set_title('TinyML Deployment Feasibility', fontsize=12, fontweight='bold')
    ax6.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Target <1 KB')
    ax6.legend(fontsize=9)
    ax6.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Feature Selection Method Comprehensive Comparison\nScoring:  Accuracy 40%, Feature Count 30%, Efficiency 15%, Interpretability 15%',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Dashboard saved:  {save_path}")
    plt.show()


def generate_comparison_report(
    comparison_results: Dict,
    task: str,
    output_file: str = 'feature_selection_report.txt'
):
    """Generate comprehensive text report for thesis."""
    # Add encoding='utf-8' to handle Unicode characters
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("FEATURE SELECTION COMPARATIVE ANALYSIS REPORT\n")
        f.write(f"Task: {task.upper()}\n")
        f.write("="*70 + "\n\n")
        
        results = comparison_results['results']
        nist_vals = comparison_results['nist_validations']
        scores = comparison_results['scores']
        recommended = comparison_results['recommended']
        
        # Section 1: Performance Summary
        f.write("SECTION 1: CLASSIFICATION PERFORMANCE\n")
        f.write("-"*70 + "\n\n")
        f.write(f"{'Method':<15} {'Accuracy':<12} {'Features':<10} {'Time (s)':<12} {'Memory (KB)':<12}\n")
        f.write("-"*70 + "\n")
        
        for method_name, result in results.items():
            f.write(f"{method_name:<15} {result.accuracy*100:6.2f}%     "
                   f"{result.n_features:<10} {result.training_time:8.3f}    "
                   f"{result.memory_footprint/1024:8.3f}\n")
        
        # Section 2: NIST Validation
        f.write("\n\nSECTION 2: NIST PHYSICAL VALIDATION\n")
        f.write("-"*70 + "\n\n")
        
        for method_name, nist_val in nist_vals.items():
            f.write(f"{method_name}:\n")
            f.write(f"  Total NIST lines checked: {nist_val['total_nist_lines']}\n")
            f.write(f"  Matched lines: {nist_val['n_matched']}\n")
            f.write(f"  Match percentage: {nist_val['match_percentage']:.1f}%\n")
            f.write(f"  Top matched lines:\n")
            
            for match in nist_val['matched_lines'][:5]:
                # Use ASCII arrow instead of Unicode arrow
                f.write(f"    {match['element']} {match['nist_wavelength']:.2f} nm -> "
                       f"{match['matched_wavelength']:.2f} nm (Delta={match['difference']:.3f} nm)\n")
            f.write("\n")
        
        # Section 3: Weighted Scoring
        f.write("\nSECTION 3: WEIGHTED SCORING ANALYSIS\n")
        f.write("-"*70 + "\n")
        f.write("Weights: Accuracy 40%, Feature Count 30%, Efficiency 15%, Interpretability 15%\n\n")
        
        f.write(f"{'Method':<15} {'Accuracy':<12} {'Features':<12} {'Efficiency':<12} {'Interpret.':<12} {'TOTAL':<12}\n")
        f.write("-"*70 + "\n")
        
        for method_name, score_dict in sorted(scores.items(), key=lambda x: x[1]['total_score'], reverse=True):
            f.write(f"{method_name:<15} {score_dict['accuracy_score']:6.2f}/40   "
                   f"{score_dict['feature_score']:6.2f}/30   "
                   f"{score_dict['efficiency_score']:6.2f}/15   "
                   f"{score_dict['interpretability_score']:6.2f}/15   "
                   f"{score_dict['total_score']:6.2f}/100\n")
        
        # Section 4: Recommendation
        f.write("\n\n" + "="*70 + "\n")
        f.write("RECOMMENDATION\n")
        f.write("="*70 + "\n\n")
        f.write(f"SELECTED METHOD: {recommended}\n\n")
        
        best_result = results[recommended]
        best_score = scores[recommended]
        best_nist = nist_vals[recommended]
        
        f.write("JUSTIFICATION:\n")
        f.write(f"  1. Highest total score: {best_score['total_score']:.2f}/100\n")
        f.write(f"  2. Classification accuracy: {best_result.accuracy*100:.2f}%\n")
        f.write(f"  3. NIST alignment: {best_nist['match_percentage']:.1f}% (physical validation)\n")
        f.write(f"  4. Training time: {best_result.training_time:.3f} seconds\n")
        f.write(f"  5. Memory footprint: {best_result.memory_footprint/1024:.3f} KB (TinyML compatible)\n")
        
        f.write("\n" + "="*70 + "\n")
    
    print(f"✓ Report saved: {output_file}")
# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__": 
    print("="*70)
    print("FEATURE SELECTION COMPARATIVE ANALYSIS")
    print("Section 4.3: RF Gini vs ANOVA vs RFE")
    print("="*70)
    
    # Configuration
    DATA_PATH = r"D:\Ryan\UM\Academic\Y4S1\KIE4002 FINAL YEAR PROJECT\Project\P1_Preprocessing\irish_soil_preprocessed.npz"
    LABELS_PATH = r"D:\Ryan\UM\Academic\Y4S1\KIE4002 FINAL YEAR PROJECT\Project\irish_soil_labels.csv"
    N_FEATURES = 87  # Target feature count
    
    # Load data
    X, y_nutrient, y_contaminant, wavelengths = load_preprocessed_data(
        DATA_PATH, LABELS_PATH
    )
    
    # Nutrient Classification (Task A)
    print("\n" + "="*70)
    print("TASK A: NUTRIENT CLASSIFICATION (N/P/K)")
    print("="*70)
    
    # Filter valid nutrient samples
    valid_nutrient = y_nutrient >= 0
    X_nutrient = X[valid_nutrient]
    y_nut = y_nutrient[valid_nutrient]
    
    comparison_nutrient = compare_feature_selection_methods(
        X_nutrient, y_nut, wavelengths, n_features=N_FEATURES, task='nutrient'
    )
    
    # Generate outputs
    plot_comparison_dashboard(
        comparison_nutrient,
        save_path='feature_selection_comparison_nutrient.png'
    )
    
    generate_comparison_report(
        comparison_nutrient,
        task='nutrient',
        output_file='feature_selection_report_nutrient.txt'
    )
    
    # Contamination Detection (Task B)
    print("\n" + "="*70)
    print("TASK B:  CONTAMINATION DETECTION (Pb/Cd/Cr)")
    print("="*70)
    
    comparison_contam = compare_feature_selection_methods(
        X, y_contaminant, wavelengths, n_features=N_FEATURES, task='contaminant'
    )
    
    plot_comparison_dashboard(
        comparison_contam,
        save_path='feature_selection_comparison_contaminant.png'
    )
    
    generate_comparison_report(
        comparison_contam,
        task='contaminant',
        output_file='feature_selection_report_contaminant.txt'
    )
    
    print("\n" + "="*70)
    print("SECTION 4.3 ANALYSIS COMPLETE")
    print("="*70)
    print("\nGenerated Files:")
    print("  - feature_selection_comparison_nutrient.png")
    print("  - feature_selection_comparison_contaminant.png")
    print("  - feature_selection_report_nutrient.txt")
    print("  - feature_selection_report_contaminant.txt")
    print("\nNext:  Proceed to Section 4.4 (Model Training & Validation)")
    print("="*70)