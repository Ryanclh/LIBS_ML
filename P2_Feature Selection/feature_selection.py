"""
Phase 2: Systematic Feature Selection for LIBS Spectral Data
=============================================================
Goal:  Reduce 12,981 features to <100 for TinyML deployment

Three Methods (Research Gap Focus):
1.Filter-Based:     ANOVA F-test (statistical variance)
2.Embedded:        Random Forest Gini Importance (tree-based)
3.Domain-Knowledge: NIST atomic spectral lines (physics-based)

Target Performance: 
- Nutrient Classification: ≥85% accuracy
- Contaminant Detection: ≥90% accuracy
- Feature Reduction: 12,981 → <100 (>99.2% reduction)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import Dict, Tuple, List
import warnings
import os
warnings.filterwarnings('ignore')


# ============================================================================
# NIST ATOMIC SPECTRAL LINES DATABASE (Domain Knowledge)
# ============================================================================

NIST_PEAKS = {
    # NUTRIENT ELEMENTS (NPK)
    'Nitrogen':  {
        'primary': [742.364, 744.229, 746.831],      # N I (neutral)
        'secondary': [821.634, 824.238, 868.028]     # N I
    },
    'Phosphorus': {
        'primary': [253.565, 255.328],               # P I (in your range!)
        'secondary': [213.618, 214.914]              # Below 197 nm - not captured
    },
    'Potassium': {
        'primary': [766.491, 769.896],               # K I (resonance doublet)
        'secondary': [404.414, 404.721]              # K I
    },
    
    # HEAVY METAL CONTAMINANTS
    'Lead': {
        'primary': [405.781, 283.306, 220.353],      # Pb I & Pb II
        'secondary': [261.418, 368.346, 280.199]     # Pb I
    },
    'Cadmium': {
        'primary':  [228.802, 326.106, 361.051],      # Cd I & Cd II
        'secondary':  [467.815, 508.582, 643.847]     # Cd I
    },
    'Chromium':  {
        'primary': [425.435, 427.480, 520.602],      # Cr I & Cr II
        'secondary': [267.716, 283.563, 357.869]     # Cr I & Cr II
    }
}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_phase1_data(filepath: str = "irish_soil_preprocessed_phase1.npz") -> Dict: 
    """
    Load preprocessed data from Phase 1.
    
    Returns
    -------
    data :  dict
        Contains X_train, X_test, y_train, y_test for both objectives
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Phase 1 data not found:  {filepath}\n"
            f"Please run libs_preprocessing.py first to generate preprocessed data."
        )
    
    print(f"\n{'='*70}")
    print("LOADING PHASE 1 PREPROCESSED DATA")
    print(f"{'='*70}")
    print(f"File: {filepath}")
    
    data_raw = np.load(filepath, allow_pickle=True)
    
    data = {
        # Metadata
        'wavelengths': data_raw['wavelengths'],
        'sample_names': data_raw['sample_names'],
        
        # Nutrient Classification (Objective 1)
        'X_train_nutrient': data_raw['X_train_nutrient'],
        'X_test_nutrient': data_raw['X_test_nutrient'],
        'y_train_nutrient': data_raw['y_train_nutrient'],
        'y_test_nutrient': data_raw['y_test_nutrient'],
        
        # Contaminant Detection (Objective 2)
        'X_train_contaminant': data_raw['X_train_contaminant'],
        'X_test_contaminant': data_raw['X_test_contaminant'],
        'y_train_contaminant':  data_raw['y_train_contaminant'],
        'y_test_contaminant': data_raw['y_test_contaminant'],
    }
    
    print(f"\n✓ Data loaded successfully")
    print(f"  Wavelength range: {data['wavelengths'][0]:.3f} - {data['wavelengths'][-1]:.3f} nm")
    print(f"  Total features:    {len(data['wavelengths'])}")
    print(f"\n  Nutrient dataset:")
    print(f"    Train: {data['X_train_nutrient'].shape}")
    print(f"    Test:   {data['X_test_nutrient'].shape}")
    print(f"  Contaminant dataset:")
    print(f"    Train: {data['X_train_contaminant'].shape}")
    print(f"    Test:  {data['X_test_contaminant'].shape}")
    print(f"{'='*70}")
    
    return data


# ============================================================================
# METHOD 1: ANOVA F-TEST (FILTER-BASED)
# ============================================================================

def feature_selection_anova(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    wavelengths: np.ndarray,
    k:  int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """
    ANOVA F-test feature selection.
    
    Justification: [264] +10% accuracy improvement for multiclass
    
    Parameters
    ----------
    X_train, y_train : Training data
    X_test :  Test data
    wavelengths :  Wavelength values for selected features
    k : Number of top features to select
    
    Returns
    -------
    X_train_selected, X_test_selected :  Selected feature subsets
    selected_wavelengths :  Wavelengths of selected features
    selected_indices : Indices of selected features
    """
    print(f"\n{'='*70}")
    print("METHOD 1: ANOVA F-TEST (Filter-Based Statistical)")
    print(f"{'='*70}")
    print(f"Goal: Select top {k} features with highest variance between classes")
    print(f"Justification: [264] - Improved accuracy by >10% for multiclass")
    
    # Apply ANOVA F-test
    selector = SelectKBest(score_func=f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    # Get selected feature indices
    selected_indices = selector.get_support(indices=True)
    selected_wavelengths = wavelengths[selected_indices]
    
    # Get F-scores for ranking
    f_scores = selector.scores_[selected_indices]
    
    print(f"\n✓ Feature selection complete")
    print(f"  Original features:   {X_train.shape[1]}")
    print(f"  Selected features:  {X_train_selected.shape[1]}")
    print(f"  Reduction:           {(1 - k/X_train.shape[1])*100:.2f}%")
    print(f"\n  Top 5 wavelengths by F-score:")
    top_5_idx = np.argsort(f_scores)[-5:][::-1]
    for i, idx in enumerate(top_5_idx, 1):
        print(f"    {i}.{selected_wavelengths[idx]:.3f} nm (F={f_scores[idx]:.2f})")
    
    return X_train_selected, X_test_selected, selected_wavelengths, selected_indices.tolist()


# ============================================================================
# METHOD 2: RANDOM FOREST GINI IMPORTANCE (EMBEDDED)
# ============================================================================

def feature_selection_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    wavelengths: np.ndarray,
    k: int = 100,
    n_estimators: int = 100,
    max_depth: int = 8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """
    Random Forest Gini Importance feature selection.
    
    Justification: [266] - Outperforms univariate tests on spectral data (409 citations)
    "Spectral regions deemed unimportant by ANOVA may be HIGH importance by Gini"
    
    Parameters
    ----------
    X_train, y_train : Training data
    X_test : Test data
    wavelengths : Wavelength values
    k : Number of top features to select
    n_estimators :  Number of trees in forest
    max_depth : Maximum tree depth (for TinyML constraint)
    
    Returns
    -------
    X_train_selected, X_test_selected :  Selected feature subsets
    selected_wavelengths : Wavelengths of selected features
    selected_indices : Indices of selected features
    """
    print(f"\n{'='*70}")
    print("METHOD 2: RANDOM FOREST GINI IMPORTANCE (Embedded Tree-Based)")
    print(f"{'='*70}")
    print(f"Goal: Select top {k} features most useful for RF decision trees")
    print(f"Justification: [266] - Outperforms ANOVA on spectral data (409 citations)")
    print(f"Configuration: {n_estimators} trees, max_depth={max_depth}")
    
    # Train Random Forest
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    
    # Get feature importances
    importances = rf.feature_importances_
    
    # Select top k features
    selected_indices = np.argsort(importances)[-k:][::-1]
    selected_wavelengths = wavelengths[selected_indices]
    
    X_train_selected = X_train[:, selected_indices]
    X_test_selected = X_test[:, selected_indices]
    
    print(f"\n✓ Feature selection complete")
    print(f"  Original features:  {X_train.shape[1]}")
    print(f"  Selected features:  {X_train_selected.shape[1]}")
    print(f"  Reduction:          {(1 - k/X_train.shape[1])*100:.2f}%")
    print(f"\n  Top 5 wavelengths by Gini Importance:")
    for i in range(5):
        idx = selected_indices[i]
        print(f"    {i+1}.{wavelengths[idx]:.3f} nm (Importance={importances[idx]:.4f})")
    
    return X_train_selected, X_test_selected, selected_wavelengths, selected_indices.tolist()


# ============================================================================
# METHOD 3: NIST DOMAIN KNOWLEDGE (HYBRID)
# ============================================================================

def find_nist_peak_indices(
    wavelengths: np.ndarray,
    target_elements: List[str],
    window_nm: float = 0.5
) -> Tuple[List[int], Dict]: 
    """
    Identify feature indices corresponding to NIST atomic spectral lines.
    
    Parameters
    ----------
    wavelengths : np.ndarray
        Array of wavelength values
    target_elements : List[str]
        Elements to search for (e.g., ['Nitrogen', 'Phosphorus', 'Potassium'])
    window_nm : float
        Search window (±0.5 nm tolerance)
    
    Returns
    -------
    selected_indices : List[int]
        Feature indices near NIST peaks
    peak_info : Dict
        Mapping of elements → detected peaks
    """
    selected_indices = []
    peak_info = {}
    
    for element in target_elements:
        if element not in NIST_PEAKS:
            print(f"  ⚠️  {element} not in NIST database, skipping")
            continue
        
        element_peaks = []
        all_peaks = NIST_PEAKS[element]['primary'] + NIST_PEAKS[element].get('secondary', [])
        
        for peak_nm in all_peaks:
            # Find indices within ±window_nm
            mask = np.abs(wavelengths - peak_nm) <= window_nm
            indices = np.where(mask)[0]
            
            if len(indices) > 0:
                # Select index closest to theoretical peak
                closest_idx = indices[np.argmin(np.abs(wavelengths[indices] - peak_nm))]
                selected_indices.append(closest_idx)
                element_peaks.append({
                    'theoretical':  peak_nm,
                    'actual': wavelengths[closest_idx],
                    'index': closest_idx,
                    'deviation': wavelengths[closest_idx] - peak_nm
                })
        
        peak_info[element] = element_peaks
    
    # Remove duplicates
    selected_indices = sorted(list(set(selected_indices)))
    
    return selected_indices, peak_info


def feature_selection_nist(
    X_train: np.ndarray,
    y_train:  np.ndarray,
    X_test: np.ndarray,
    wavelengths: np.ndarray,
    objective:  str = 'nutrient'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], Dict]:
    """
    NIST domain knowledge feature selection.
    
    Justification: Physics-based peak selection from NIST Atomic Spectra Database
    
    Parameters
    ----------
    X_train, y_train : Training data
    X_test : Test data
    wavelengths :  Wavelength values
    objective : 'nutrient' or 'contaminant'
    
    Returns
    -------
    X_train_selected, X_test_selected : Selected feature subsets
    selected_wavelengths : Wavelengths of selected features
    selected_indices : Indices of selected features
    peak_info :  Detected peak information
    """
    print(f"\n{'='*70}")
    print("METHOD 3: NIST DOMAIN KNOWLEDGE (Hybrid Physics-Based)")
    print(f"{'='*70}")
    print(f"Objective: {objective.upper()}")
    print(f"Method: Select features within ±0.5nm of NIST atomic spectral lines")
    
    # Select target elements based on objective
    if objective == 'nutrient':
        target_elements = ['Nitrogen', 'Phosphorus', 'Potassium']
        print(f"Target elements: N, P, K (macronutrients)")
    elif objective == 'contaminant': 
        target_elements = ['Lead', 'Cadmium', 'Chromium']
        print(f"Target elements:  Pb, Cd, Cr (heavy metals)")
    else:
        raise ValueError(f"Unknown objective: {objective}")
    
    # Find NIST peak indices
    selected_indices, peak_info = find_nist_peak_indices(
        wavelengths, 
        target_elements, 
        window_nm=0.5
    )
    
    if len(selected_indices) == 0:
        raise ValueError("No NIST peaks found in wavelength range!")
    
    # Extract features
    selected_wavelengths = wavelengths[selected_indices]
    X_train_selected = X_train[:, selected_indices]
    X_test_selected = X_test[:, selected_indices]
    
    print(f"\n✓ Feature selection complete")
    print(f"  Original features:  {X_train.shape[1]}")
    print(f"  Selected features:  {X_train_selected.shape[1]}")
    print(f"  Reduction:          {(1 - len(selected_indices)/X_train.shape[1])*100:.2f}%")
    
    print(f"\n  Detected NIST Peaks by Element:")
    for element, peaks in peak_info.items():
        print(f"\n    {element}:  {len(peaks)} peaks detected")
        for i, peak in enumerate(peaks[: 3], 1):  # Show first 3
            print(f"      {i}.{peak['actual']:.3f} nm (theory: {peak['theoretical']:.3f} nm, Δ={peak['deviation']:+.3f} nm)")
        if len(peaks) > 3:
            print(f"      ... and {len(peaks)-3} more")
    
    return X_train_selected, X_test_selected, selected_wavelengths, selected_indices, peak_info


# ============================================================================
# EVALUATION & COMPARISON
# ============================================================================

def evaluate_feature_subset(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    method_name: str,
    n_features: int,
    cv_folds: int = 5
) -> Dict:
    """
    Evaluate feature subset with Random Forest classifier.
    
    Parameters
    ----------
    X_train, y_train : Training data
    X_test, y_test : Test data
    method_name : Name of feature selection method
    n_features :  Number of selected features
    cv_folds :  Number of cross-validation folds
    
    Returns
    -------
    results : Dict
        Contains accuracy, CV score, classification report
    """
    print(f"\n  Evaluating {method_name} ({n_features} features)...")
    
    # Train Random Forest (TinyML configuration from [302])
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )
    
    # Cross-validation on training set
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring='accuracy')
    
    # Train on full training set
    rf.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = rf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    results = {
        'method': method_name,
        'n_features': n_features,
        'cv_mean':  np.mean(cv_scores),
        'cv_std': np.std(cv_scores),
        'test_accuracy': test_accuracy,
        'y_pred': y_pred,
        'model':  rf
    }
    
    print(f"    CV Accuracy:    {results['cv_mean']*100:.2f}% (±{results['cv_std']*100:.2f}%)")
    print(f"    Test Accuracy:  {results['test_accuracy']*100:.2f}%")
    
    return results


def compare_all_methods(
    data: Dict,
    objective: str = 'nutrient',
    k_features: int = 100
) -> Dict:
    """
    Run all three feature selection methods and compare performance.
    
    Parameters
    ----------
    data : Dict
        Loaded Phase 1 data
    objective :  str
        'nutrient' or 'contaminant'
    k_features :  int
        Number of features to select (for ANOVA and RF methods)
    
    Returns
    -------
    comparison : Dict
        Results for all three methods
    """
    print(f"\n{'='*70}")
    print(f"COMPARING ALL FEATURE SELECTION METHODS")
    print(f"{'='*70}")
    print(f"Objective: {objective.upper()}")
    print(f"Target features: {k_features} (ANOVA & RF), variable (NIST)")
    print(f"Baseline: Full spectrum ({data['wavelengths'].shape[0]} features)")
    
    # Select appropriate dataset
    if objective == 'nutrient':
        X_train = data['X_train_nutrient']
        X_test = data['X_test_nutrient']
        y_train = data['y_train_nutrient']
        y_test = data['y_test_nutrient']
    elif objective == 'contaminant':
        X_train = data['X_train_contaminant']
        X_test = data['X_test_contaminant']
        y_train = data['y_train_contaminant']
        y_test = data['y_test_contaminant']
    else:
        raise ValueError(f"Unknown objective:  {objective}")
    
    wavelengths = data['wavelengths']
    comparison = {}
    
    # === BASELINE: Full Spectrum ===
    print(f"\n{'='*70}")
    print("BASELINE: Full Spectrum (No Feature Selection)")
    print(f"{'='*70}")
    comparison['baseline'] = evaluate_feature_subset(
        X_train, y_train, X_test, y_test,
        method_name='Baseline (Full)',
        n_features=X_train.shape[1]
    )
    comparison['baseline']['selected_wavelengths'] = wavelengths
    comparison['baseline']['selected_indices'] = list(range(len(wavelengths)))
    
    # === METHOD 1: ANOVA F-TEST ===
    X_train_anova, X_test_anova, wl_anova, idx_anova = feature_selection_anova(
        X_train, y_train, X_test, wavelengths, k=k_features
    )
    comparison['anova'] = evaluate_feature_subset(
        X_train_anova, y_train, X_test_anova, y_test,
        method_name='ANOVA F-test',
        n_features=k_features
    )
    comparison['anova']['selected_wavelengths'] = wl_anova
    comparison['anova']['selected_indices'] = idx_anova
    
    # === METHOD 2: RANDOM FOREST GINI ===
    X_train_rf, X_test_rf, wl_rf, idx_rf = feature_selection_random_forest(
        X_train, y_train, X_test, wavelengths, k=k_features
    )
    comparison['rf_gini'] = evaluate_feature_subset(
        X_train_rf, y_train, X_test_rf, y_test,
        method_name='RF Gini Importance',
        n_features=k_features
    )
    comparison['rf_gini']['selected_wavelengths'] = wl_rf
    comparison['rf_gini']['selected_indices'] = idx_rf
    
    # === METHOD 3: NIST DOMAIN KNOWLEDGE ===
    X_train_nist, X_test_nist, wl_nist, idx_nist, peak_info = feature_selection_nist(
        X_train, y_train, X_test, wavelengths, objective=objective
    )
    comparison['nist'] = evaluate_feature_subset(
        X_train_nist, y_train, X_test_nist, y_test,
        method_name='NIST Domain Knowledge',
        n_features=len(idx_nist)
    )
    comparison['nist']['selected_wavelengths'] = wl_nist
    comparison['nist']['selected_indices'] = idx_nist
    comparison['nist']['peak_info'] = peak_info
    
    # === SUMMARY TABLE ===
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    
    summary_data = []
    for method_key in ['baseline', 'anova', 'rf_gini', 'nist']:
        r = comparison[method_key]
        summary_data.append({
            'Method': r['method'],
            'Features': r['n_features'],
            'Reduction (%)': f"{(1 - r['n_features']/X_train.shape[1])*100:.2f}",
            'CV Accuracy': f"{r['cv_mean']*100:.2f}% (±{r['cv_std']*100:.2f}%)",
            'Test Accuracy': f"{r['test_accuracy']*100:.2f}%"
        })
    
    df_summary = pd.DataFrame(summary_data)
    print(df_summary.to_string(index=False))
    
    # Find best method
    best_method_key = max(comparison.keys(), key=lambda k: comparison[k]['test_accuracy'])
    best_method = comparison[best_method_key]
    
    print(f"\n{'='*70}")
    print(f"🏆 BEST METHOD: {best_method['method']}")
    print(f"{'='*70}")
    print(f"  Test Accuracy:     {best_method['test_accuracy']*100:.2f}%")
    print(f"  Features:         {best_method['n_features']}")
    print(f"  Feature Reduction: {(1 - best_method['n_features']/X_train.shape[1])*100:.2f}%")
    
    comparison['summary_table'] = df_summary
    comparison['best_method'] = best_method_key
    
    return comparison


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_feature_comparison(
    comparison: Dict,
    objective: str,
    wavelengths: np.ndarray,
    save_path: str = None
):
    """
    Visualize feature selection results.
    """
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Accuracy Comparison
    ax1 = fig.add_subplot(2, 2, 1)
    methods = []
    accuracies = []
    colors = []
    
    for key in ['baseline', 'anova', 'rf_gini', 'nist']:
        r = comparison[key]
        methods.append(r['method'])
        accuracies.append(r['test_accuracy'] * 100)
        colors.append('green' if key == comparison['best_method'] else 'skyblue')
    
    bars = ax1.bar(methods, accuracies, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Feature Selection Method Comparison', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 100])
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Rotate x labels
    ax1.set_xticklabels(methods, rotation=15, ha='right')
    
    # Plot 2: Feature Count vs Accuracy
    ax2 = fig.add_subplot(2, 2, 2)
    n_features = [comparison[k]['n_features'] for k in ['baseline', 'anova', 'rf_gini', 'nist']]
    
    ax2.scatter(n_features, accuracies, c=colors, s=200, edgecolor='black', linewidth=2, alpha=0.7)
    
    for i, method in enumerate(methods):
        ax2.annotate(method, (n_features[i], accuracies[i]),
                    xytext=(10, 5), textcoords='offset points',
                    fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))
    
    ax2.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Feature Count vs Accuracy Trade-off', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.set_xscale('log')
    
    # Plot 3: Selected Wavelength Distribution
    ax3 = fig.add_subplot(2, 2, 3)
    
    # Plot full spectrum as background
    ax3.axhspan(0, 1, alpha=0.1, color='gray', label='Full Spectrum')
    
    # Mark selected features for each method
    y_positions = {'anova': 0.75, 'rf_gini':  0.5, 'nist': 0.25}
    colors_map = {'anova': 'blue', 'rf_gini':  'red', 'nist': 'green'}
    
    for key, y_pos in y_positions.items():
        wl_selected = comparison[key]['selected_wavelengths']
        ax3.scatter(wl_selected, [y_pos]*len(wl_selected), 
                   c=colors_map[key], s=20, alpha=0.6, label=comparison[key]['method'])
    
    ax3.set_xlabel('Wavelength (nm)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Method', fontsize=12, fontweight='bold')
    ax3.set_title('Selected Wavelength Distribution', fontsize=14, fontweight='bold')
    ax3.set_yticks(list(y_positions.values()))
    ax3.set_yticklabels([comparison[k]['method'] for k in y_positions.keys()])
    ax3.legend(loc='upper right')
    ax3.grid(alpha=0.3, axis='x')
    
    # Plot 4: Confusion Matrix for Best Method
    ax4 = fig.add_subplot(2, 2, 4)
    best_key = comparison['best_method']
    best_result = comparison[best_key]
    
    cm = confusion_matrix(comparison['baseline']['y_pred'][:len(best_result['y_pred'])], 
                         best_result['y_pred'])
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4, cbar=True)
    ax4.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax4.set_title(f'Confusion Matrix - {best_result["method"]}', fontsize=14, fontweight='bold')
    
    plt.suptitle(f'Phase 2:  Feature Selection Results ({objective.upper()})',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Visualization saved:  {save_path}")
    
    plt.show()


# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_feature_selection_results(
    comparison: Dict,
    objective: str,
    output_path: str = "feature_selection_results.npz"
):
    """
    Save feature selection results for Phase 3.
    """
    best_key = comparison['best_method']
    best_result = comparison[best_key]
    
    save_dict = {
        'objective': objective,
        'best_method': best_key,
        'best_method_name': best_result['method'],
        'selected_indices': np.array(best_result['selected_indices']),
        'selected_wavelengths': best_result['selected_wavelengths'],
        'n_features': best_result['n_features'],
        'test_accuracy': best_result['test_accuracy'],
        'cv_mean': best_result['cv_mean'],
        'cv_std': best_result['cv_std'],
        
        # Save all method results for comparison
        'all_methods': {
            k: {
                'n_features': v['n_features'],
                'test_accuracy':  v['test_accuracy'],
                'cv_mean': v['cv_mean'],
                'selected_indices': v['selected_indices']
            }
            for k, v in comparison.items() if k not in ['summary_table', 'best_method']
        }
    }
    
    # Add NIST peak info if available
    if best_key == 'nist':
        save_dict['nist_peak_info'] = best_result.get('peak_info', {})
    
    np.savez_compressed(output_path, **save_dict)
    
    print(f"\n{'='*70}")
    print("FEATURE SELECTION RESULTS SAVED")
    print(f"{'='*70}")
    print(f"File: {output_path}")
    print(f"Size: {os.path.getsize(output_path) / 1024:.2f} KB")
    print(f"\nContents:")
    print(f"  • Best method:          {best_result['method']}")
    print(f"  • Selected features:   {best_result['n_features']}")
    print(f"  • Test accuracy:       {best_result['test_accuracy']*100:.2f}%")
    print(f"  • Selected indices:    Array of {len(best_result['selected_indices'])} indices")
    print(f"  • Selected wavelengths: {best_result['selected_wavelengths'][0]:.3f} - {best_result['selected_wavelengths'][-1]:.3f} nm")
    print(f"\n✓ Ready for Phase 3: Model Development")
    print(f"{'='*70}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("PHASE 2: SYSTEMATIC FEATURE SELECTION")
    print("Goal: Reduce 12,981 features to <100 for TinyML deployment")
    print("="*70)
    
    # =========================================================================
    # STEP 1: LOAD PHASE 1 DATA
    # =========================================================================
    data = load_phase1_data(r"D:\Ryan\UM\Academic\Y4S1\KIE4002 FINAL YEAR PROJECT\Project\P1_Preprocessing\irish_soil_preprocessed_phase1.npz")
    
    # =========================================================================
    # STEP 2: FEATURE SELECTION FOR NUTRIENT CLASSIFICATION
    # =========================================================================
    print(f"\n{'='*70}")
    print("OBJECTIVE 1: NUTRIENT CLASSIFICATION (NPK)")
    print(f"{'='*70}")
    print("Target: ≥85% accuracy, <100 features")
    
    comparison_nutrient = compare_all_methods(
        data,
        objective='nutrient',
        k_features=100
    )
    
    # Visualize
    plot_feature_comparison(
        comparison_nutrient,
        objective='nutrient',
        wavelengths=data['wavelengths'],
        save_path='feature_selection_nutrient.png'
    )
    
    # Save results
    save_feature_selection_results(
        comparison_nutrient,
        objective='nutrient',
        output_path='feature_selection_nutrient.npz'
    )
    
    # =========================================================================
    # STEP 3: FEATURE SELECTION FOR CONTAMINANT DETECTION
    # =========================================================================
    print(f"\n{'='*70}")
    print("OBJECTIVE 2: CONTAMINANT DETECTION")
    print(f"{'='*70}")
    print("Target: ≥90% accuracy, <100 features")
    
    comparison_contaminant = compare_all_methods(
        data,
        objective='contaminant',
        k_features=100
    )
    
    # Visualize
    plot_feature_comparison(
        comparison_contaminant,
        objective='contaminant',
        wavelengths=data['wavelengths'],
        save_path='feature_selection_contaminant.png'
    )
    
    # Save results
    save_feature_selection_results(
        comparison_contaminant,
        objective='contaminant',
        output_path='feature_selection_contaminant.npz'
    )
    
    # =========================================================================
    # PHASE 2 COMPLETION SUMMARY
    # =========================================================================
    print(f"\n{'='*70}")
    print("✓ PHASE 2 COMPLETE:  SYSTEMATIC FEATURE SELECTION")
    print(f"{'='*70}")
    
    print("\nNutrient Classification Results:")
    best_nut = comparison_nutrient[comparison_nutrient['best_method']]
    print(f"  Best method:        {best_nut['method']}")
    print(f"  Features selected: {best_nut['n_features']}")
    print(f"  Test accuracy:     {best_nut['test_accuracy']*100:.2f}%")
    print(f"  Feature reduction: {(1 - best_nut['n_features']/12981)*100:.2f}%")
    
    print("\nContaminant Detection Results:")
    best_con = comparison_contaminant[comparison_contaminant['best_method']]
    print(f"  Best method:       {best_con['method']}")
    print(f"  Features selected: {best_con['n_features']}")
    print(f"  Test accuracy:     {best_con['test_accuracy']*100:.2f}%")
    print(f"  Feature reduction: {(1 - best_con['n_features']/12981)*100:.2f}%")
    
    print(f"\n{'='*70}")
    print("NEXT PHASE: MODEL DEVELOPMENT (Phase 3)")
    print(f"{'='*70}")
    print("\nPhase 3 Objectives:")
    print("  1.Train Random Forest with selected features")
    print("  2.Compare against PLS-DA baseline (73.01%)")
    print("  3.Optimize hyperparameters (max_depth, n_estimators)")
    print("  4.Prepare for TinyML deployment (Phase 4)")
    print("\n  Load feature selection results:")
    print("    data = np.load('feature_selection_nutrient.npz')")
    print("    selected_indices = data['selected_indices']")
    print(f"{'='*70}")