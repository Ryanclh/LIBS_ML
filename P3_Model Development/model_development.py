"""
Phase 3: Model Development (PC-Based Training)
==============================================
Goal: Train classifiers and prove they beat the PLS-DA baseline

Models:
1.Baseline: PLS-DA with ALL 12,981 features (Al-Juboori replication)
2.Advanced: Random Forest with selected features (Target: ≥85%)
3.Alternative: SVM with selected features (Comparison model)

Evaluation: Accuracy, Model Size, Latency, Constraint Validation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.preprocessing import label_binarize
import pickle
import joblib
import time
import os
import warnings
warnings.filterwarnings('ignore')

# Import from Phase 2 for TinyML metrics
import sys
phase2_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../P2_Feature_Selection'))
if phase2_path not in sys.path:
    sys.path.insert(0, phase2_path)

try:
    from feature_selection import estimate_model_size_rf, estimate_inference_latency_rf  # type: ignore
    PHASE2_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    print("⚠️  Phase 2 module not found - TinyML metrics will be estimated using built-in functions")
    PHASE2_AVAILABLE = False


# ============================================================================
# FALLBACK:  TinyML Estimation Functions (if Phase 2 not available)
# ============================================================================

def estimate_model_size_rf_local(model, n_features):
    """Local implementation of RF model size estimation"""
    n_trees = model.n_estimators
    total_nodes = sum([tree.tree_.node_count for tree in model.estimators_])
    total_leaves = sum([tree.tree_.n_leaves for tree in model.estimators_])
    
    bytes_per_node = 16
    bytes_per_leaf = 8
    model_size_bytes = total_nodes * bytes_per_node + total_leaves * bytes_per_leaf + 1024
    
    return {
        'model_size_kb': model_size_bytes / 1024,
        'total_nodes': total_nodes,
        'ram_kb': (n_features * 4 + n_trees * 4 + 512) / 1024
    }

def estimate_inference_latency_rf_local(model, n_features):
    """Local implementation of RF latency estimation"""
    n_trees = model.n_estimators
    tree_depths = [tree.tree_.max_depth for tree in model.estimators_]
    avg_depth = np.mean(tree_depths)
    
    cycles_preprocessing = n_features * 20
    cycles_per_tree = avg_depth * 75
    cycles_total = cycles_preprocessing + (n_trees * cycles_per_tree)
    
    latency_ms = (cycles_total / (80 * 1e6)) * 1000
    
    return {
        'latency_ms': latency_ms,
        'avg_tree_depth': avg_depth
    }


# ============================================================================
# DATA LOADING
# ============================================================================

def load_phase1_and_phase2_data():
    """
    Load preprocessed data from Phase 1 and feature selection results from Phase 2.
    
    Returns
    -------
    data : dict
        Contains full and selected feature datasets for both objectives
    """
    print(f"\n{'='*70}")
    print("LOADING PHASE 1 & PHASE 2 DATA")
    print(f"{'='*70}")
    
    # Load Phase 1: Full preprocessed data
    phase1_path = "../P1_Preprocessing/irish_soil_preprocessed_phase1.npz"
    if not os.path.exists(phase1_path):
        raise FileNotFoundError(
            f"Phase 1 data not found:  {phase1_path}\n"
            f"Please run libs_preprocessing.py first."
        )
    
    phase1_data = np.load(phase1_path, allow_pickle=True)
    
    print(f"✓ Phase 1 data loaded: {phase1_path}")
    print(f"  Full feature count: {phase1_data['wavelengths'].shape[0]}")
    
    data = {
        'wavelengths': phase1_data['wavelengths'],
        'sample_names': phase1_data['sample_names'],
    }
    
    # For each objective, load full and selected features
    for objective in ['nutrient', 'contaminant']:
        # Full features (for PLS-DA baseline)
        data[f'X_train_full_{objective}'] = phase1_data[f'X_train_{objective}']
        data[f'X_test_full_{objective}'] = phase1_data[f'X_test_{objective}']
        data[f'y_train_{objective}'] = phase1_data[f'y_train_{objective}']
        data[f'y_test_{objective}'] = phase1_data[f'y_test_{objective}']
        
        # Load Phase 2: Selected features
        phase2_path = f"../P2_Feature_Selection/feature_selection_{objective}.npz"
        if os.path.exists(phase2_path):
            phase2_data = np.load(phase2_path, allow_pickle=True)
            
            selected_indices = phase2_data['selected_indices']
            
            # Extract selected features from full dataset
            data[f'X_train_selected_{objective}'] = data[f'X_train_full_{objective}'][:, selected_indices]
            data[f'X_test_selected_{objective}'] = data[f'X_test_full_{objective}'][:, selected_indices]
            data[f'selected_indices_{objective}'] = selected_indices
            data[f'selected_wavelengths_{objective}'] = phase2_data['selected_wavelengths']
            data[f'best_method_{objective}'] = str(phase2_data['best_method'])
            
            print(f"\n✓ Phase 2 ({objective}) loaded:")
            print(f"  Best method: {data[f'best_method_{objective}']}")
            print(f"  Selected features: {len(selected_indices)}")
        else:
            print(f"\n⚠️  Phase 2 ({objective}) not found: {phase2_path}")
            print(f"   Using top 100 features by variance as fallback")
            
            # Fallback:  Use top 100 features by variance
            X_full = data[f'X_train_full_{objective}']
            variances = np.var(X_full, axis=0)
            selected_indices = np.argsort(variances)[-100:]
            
            data[f'X_train_selected_{objective}'] = X_full[:, selected_indices]
            data[f'X_test_selected_{objective}'] = data[f'X_test_full_{objective}'][:, selected_indices]
            data[f'selected_indices_{objective}'] = selected_indices
            data[f'selected_wavelengths_{objective}'] = data['wavelengths'][selected_indices]
            data[f'best_method_{objective}'] = 'Variance (fallback)'
    
    print(f"\n{'='*70}")
    return data


# ============================================================================
# BASELINE:  PLS-DA MODEL
# ============================================================================

class PLSDA: 
    """
    PLS-DA (Partial Least Squares Discriminant Analysis)
    Replicates Al-Juboori et al.baseline approach
    """
    
    def __init__(self, n_components=5):
        """
        Parameters
        ----------
        n_components : int
            Number of PLS components (Al-Juboori used 5 PCs)
        """
        self.n_components = n_components
        self.pls = PLSRegression(n_components=n_components)
        self.lda = LinearDiscriminantAnalysis()
        
    def fit(self, X, y):
        """Train PLS-DA model"""
        # Step 1: PLS dimensionality reduction
        X_pls = self.pls.fit_transform(X, y)[0]
        
        # Step 2: LDA classification on PLS components
        self.lda.fit(X_pls, y)
        
        return self
    
    def predict(self, X):
        """Predict class labels"""
        X_pls = self.pls.transform(X)
        return self.lda.predict(X_pls)
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        X_pls = self.pls.transform(X)
        return self.lda.predict_proba(X_pls)
    
    def score(self, X, y):
        """Calculate accuracy"""
        return accuracy_score(y, self.predict(X))


# ============================================================================
# MODEL TRAINING & EVALUATION
# ============================================================================

def train_baseline_plsda(X_train, y_train, X_test, y_test, n_components=5):
    """
    Train PLS-DA baseline model (Al-Juboori replication).
    
    Parameters
    ----------
    X_train, y_train : Training data (FULL 12,981 features)
    X_test, y_test : Test data
    n_components : int
        Number of PLS components (default: 5, from Al-Juboori)
    
    Returns
    -------
    results : dict
        Model performance metrics
    """
    print(f"\n{'='*70}")
    print("BASELINE MODEL: PLS-DA (Al-Juboori Replication)")
    print(f"{'='*70}")
    print(f"Configuration: {n_components} PLS components + LDA")
    print(f"Training on:  {X_train.shape[0]} samples × {X_train.shape[1]} features")
    print(f"Baseline reference: Al-Juboori et al. 73.01% test accuracy")
    
    # Train model
    plsda = PLSDA(n_components=n_components)
    
    start_time = time.time()
    plsda.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        plsda_cv = PLSDA(n_components=n_components)
        plsda_cv.fit(X_train[train_idx], y_train[train_idx])
        cv_scores.append(plsda_cv.score(X_train[val_idx], y_train[val_idx]))
    
    # Test set evaluation
    y_pred = plsda.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    # Detailed metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0
    )
    
    results = {
        'model':  plsda,
        'model_name': 'PLS-DA Baseline',
        'n_features': X_train.shape[1],
        'n_components': n_components,
        'train_time': train_time,
        'cv_mean':  np.mean(cv_scores),
        'cv_std': np.std(cv_scores),
        'test_accuracy': test_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'y_pred': y_pred,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        
        # TinyML metrics (N/A for PLS-DA - not deployable)
        'model_size_kb': np.nan,
        'latency_ms': np.nan,
        'deployable': False,
        'notes': 'PLS-DA not suitable for MCU deployment (requires matrix operations)'
    }
    
    print(f"\n✓ Training complete ({train_time:.2f}s)")
    print(f"  CV Accuracy:    {results['cv_mean']*100:.2f}% (±{results['cv_std']*100:.2f}%)")
    print(f"  Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"  Precision:     {precision:.4f}")
    print(f"  Recall:        {recall:.4f}")
    print(f"  F1-Score:      {f1:.4f}")
    
    # Compare to Al-Juboori baseline
    baseline_ref = 73.01
    improvement = test_accuracy * 100 - baseline_ref
    print(f"\n  📊 vs Al-Juboori Baseline (73.01%):")
    print(f"     Difference: {improvement:+.2f} pp")
    
    return results


def train_random_forest(X_train, y_train, X_test, y_test, 
                       n_features_original=12981, hyperparameter_tuning=True):
    """
    Train Random Forest classifier with TinyML constraints.
    
    Parameters
    ----------
    X_train, y_train : Training data (selected features only)
    X_test, y_test : Test data
    n_features_original : int
        Original feature count (for reduction ratio)
    hyperparameter_tuning : bool
        Whether to perform grid search (default: True)
    
    Returns
    -------
    results : dict
        Model performance and TinyML metrics
    """
    print(f"\n{'='*70}")
    print("ADVANCED MODEL:  RANDOM FOREST (TinyML-Optimized)")
    print(f"{'='*70}")
    print(f"Training on: {X_train.shape[0]} samples × {X_train.shape[1]} features")
    print(f"Feature reduction: {(1 - X_train.shape[1]/n_features_original)*100:.2f}%")
    print(f"TinyML Evidence: 94.9% accuracy, 128 KB, 2.1 ms [302]")
    
    # Hyperparameter tuning
    if hyperparameter_tuning:
        print(f"\nPerforming Grid Search (5-fold CV)...")
        
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [6, 8, 10],
            'min_samples_split':  [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(
            rf_base, param_grid, cv=5, scoring='accuracy',
            n_jobs=-1, verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        
        print(f"✓ Best parameters found:")
        for param, value in best_params.items():
            print(f"    {param}: {value}")
    else:
        # Use TinyML-validated defaults
        best_params = {
            'n_estimators': 100,
            'max_depth':  8,
            'min_samples_split': 5,
            'min_samples_leaf': 1
        }
        print(f"\nUsing TinyML-validated defaults (from [302]):")
        print(f"  n_estimators=100, max_depth=8")
    
    # Train final model
    rf = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    
    start_time = time.time()
    rf.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    
    # Test set evaluation
    y_pred = rf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    # Detailed metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0
    )
    
    # TinyML metrics
    if PHASE2_AVAILABLE:
        size_info = estimate_model_size_rf(rf, X_train.shape[1])
        latency_info = estimate_inference_latency_rf(rf, X_train.shape[1])
    else:
        size_info = estimate_model_size_rf_local(rf, X_train.shape[1])
        latency_info = estimate_inference_latency_rf_local(rf, X_train.shape[1])
    
    # Inference time measurement (simulated)
    inference_times = []
    for _ in range(100):
        start = time.perf_counter()
        _ = rf.predict(X_test[: 1])
        inference_times.append((time.perf_counter() - start) * 1000)
    
    results = {
        'model':  rf,
        'model_name': 'Random Forest',
        'n_features': X_train.shape[1],
        'best_params': best_params,
        'train_time': train_time,
        'cv_mean': np.mean(cv_scores),
        'cv_std': np.std(cv_scores),
        'test_accuracy': test_accuracy,
        'precision':  precision,
        'recall': recall,
        'f1_score': f1,
        'y_pred': y_pred,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        
        # TinyML metrics
        'model_size_kb': size_info['model_size_kb'],
        'ram_kb': size_info['ram_kb'],
        'latency_ms': latency_info['latency_ms'],
        'latency_measured_mean': np.mean(inference_times),
        'latency_measured_std': np.std(inference_times),
        'avg_tree_depth': latency_info['avg_tree_depth'],
        'total_nodes': size_info.get('total_nodes', 0),
        'reduction_ratio': (1 - X_train.shape[1]/n_features_original) * 100,
        
        # Constraint validation
        'meets_1mb':  size_info['model_size_kb'] <= 1024,
        'meets_50ms': latency_info['latency_ms'] <= 50,
        'deployable': (size_info['model_size_kb'] <= 1024) and (latency_info['latency_ms'] <= 50)
    }
    
    print(f"\n✓ Training complete ({train_time:.2f}s)")
    print(f"\n  📊 ACCURACY METRICS:")
    print(f"     CV Accuracy:   {results['cv_mean']*100:.2f}% (±{results['cv_std']*100:.2f}%)")
    print(f"     Test Accuracy:  {test_accuracy*100:.2f}%")
    print(f"     Precision:     {precision:.4f}")
    print(f"     Recall:        {recall:.4f}")
    print(f"     F1-Score:      {f1:.4f}")
    
    print(f"\n  🔧 TinyML METRICS:")
    print(f"     Model Size:      {results['model_size_kb']:.2f} KB")
    print(f"     RAM Usage:       {results['ram_kb']:.2f} KB")
    print(f"     Latency (est):   {results['latency_ms']:.2f} ms")
    print(f"     Latency (meas):  {results['latency_measured_mean']:.2f} ± {results['latency_measured_std']:.2f} ms")
    print(f"     Tree Depth:      {results['avg_tree_depth']:.1f} (avg)")
    
    print(f"\n  ✔️  CONSTRAINT VALIDATION:")
    print(f"     ≤1 MB Flash:    {'✅ PASS' if results['meets_1mb'] else '❌ FAIL'} ({results['model_size_kb']:.2f} KB)")
    print(f"     ≤50 ms Latency: {'✅ PASS' if results['meets_50ms'] else '❌ FAIL'} ({results['latency_ms']:.2f} ms)")
    print(f"     Deployable:      {'✅ YES' if results['deployable'] else '❌ NO'}")
    
    return results


def train_svm(X_train, y_train, X_test, y_test, n_features_original=12981):
    """
    Train SVM classifier (alternative comparison model).
    
    Parameters
    ----------
    X_train, y_train : Training data (selected features)
    X_test, y_test : Test data
    n_features_original : int
        Original feature count
    
    Returns
    -------
    results : dict
        Model performance metrics
    """
    print(f"\n{'='*70}")
    print("ALTERNATIVE MODEL:  SUPPORT VECTOR MACHINE (SVM)")
    print(f"{'='*70}")
    print(f"Training on: {X_train.shape[0]} samples × {X_train.shape[1]} features")
    print(f"Purpose: Comparison model (expected 5-15 pp lower than RF)")
    
    # Grid search for best kernel
    print(f"\nPerforming Grid Search (5-fold CV)...")
    
    param_grid = {
        'kernel': ['rbf', 'linear'],
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto']
    }
    
    svm_base = SVC(random_state=42, probability=True)
    grid_search = GridSearchCV(
        svm_base, param_grid, cv=5, scoring='accuracy',
        n_jobs=-1, verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    
    print(f"✓ Best parameters found:")
    for param, value in best_params.items():
        print(f"    {param}: {value}")
    
    # Train final model
    svm = SVC(**best_params, random_state=42, probability=True)
    
    start_time = time.time()
    svm.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(svm, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    
    # Test set evaluation
    y_pred = svm.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    # Detailed metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0
    )
    
    # Model size estimation (SVM is typically smaller than RF)
    n_support_vectors = svm.n_support_.sum()
    model_size_bytes = (
        n_support_vectors * X_train.shape[1] * 4 +  # Support vectors (float32)
        n_support_vectors * 4 +                       # Coefficients
        1024                                          # Overhead
    )
    model_size_kb = model_size_bytes / 1024
    
    # Inference latency (SVM is typically faster than RF for small feature sets)
    inference_times = []
    for _ in range(100):
        start = time.perf_counter()
        _ = svm.predict(X_test[:1])
        inference_times.append((time.perf_counter() - start) * 1000)
    
    results = {
        'model':  svm,
        'model_name': 'SVM',
        'n_features': X_train.shape[1],
        'best_params': best_params,
        'n_support_vectors': n_support_vectors,
        'train_time': train_time,
        'cv_mean': np.mean(cv_scores),
        'cv_std': np.std(cv_scores),
        'test_accuracy': test_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'y_pred': y_pred,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        
        # TinyML metrics (estimated)
        'model_size_kb': model_size_kb,
        'ram_kb': (X_train.shape[1] * 4 + 512) / 1024,
        'latency_ms': np.mean(inference_times),
        'latency_measured_mean': np.mean(inference_times),
        'latency_measured_std': np.std(inference_times),
        'reduction_ratio': (1 - X_train.shape[1]/n_features_original) * 100,
        
        # Constraint validation
        'meets_1mb': model_size_kb <= 1024,
        'meets_50ms': np.mean(inference_times) <= 50,
        'deployable': (model_size_kb <= 1024) and (np.mean(inference_times) <= 50)
    }
    
    print(f"\n✓ Training complete ({train_time:.2f}s)")
    print(f"\n  📊 ACCURACY METRICS:")
    print(f"     CV Accuracy:   {results['cv_mean']*100:.2f}% (±{results['cv_std']*100:.2f}%)")
    print(f"     Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"     Precision:     {precision:.4f}")
    print(f"     Recall:         {recall:.4f}")
    print(f"     F1-Score:      {f1:.4f}")
    
    print(f"\n  🔧 TinyML METRICS:")
    print(f"     Model Size:      {results['model_size_kb']:.2f} KB")
    print(f"     Support Vectors: {n_support_vectors}")
    print(f"     RAM Usage:       {results['ram_kb']:.2f} KB")
    print(f"     Latency:          {results['latency_measured_mean']:.2f} ± {results['latency_measured_std']:.2f} ms")
    
    print(f"\n  ✔️  CONSTRAINT VALIDATION:")
    print(f"     ≤1 MB Flash:    {'✅ PASS' if results['meets_1mb'] else '❌ FAIL'} ({results['model_size_kb']:.2f} KB)")
    print(f"     ≤50 ms Latency: {'✅ PASS' if results['meets_50ms'] else '❌ FAIL'} ({results['latency_ms']:.2f} ms)")
    print(f"     Deployable:     {'✅ YES' if results['deployable'] else '❌ NO'}")
    
    return results


# ============================================================================
# COMPREHENSIVE COMPARISON
# ============================================================================

def compare_all_models(data, objective='nutrient'):
    """
    Train and compare all models for a given objective.
    
    Parameters
    ----------
    data : dict
        Loaded Phase 1 and Phase 2 data
    objective : str
        'nutrient' or 'contaminant'
    
    Returns
    -------
    comparison :  dict
        Results for all models
    """
    print(f"\n{'='*70}")
    print(f"MODEL COMPARISON: {objective.upper()} CLASSIFICATION")
    print(f"{'='*70}")
    
    # Get data for this objective
    X_train_full = data[f'X_train_full_{objective}']
    X_test_full = data[f'X_test_full_{objective}']
    X_train_selected = data[f'X_train_selected_{objective}']
    X_test_selected = data[f'X_test_selected_{objective}']
    y_train = data[f'y_train_{objective}']
    y_test = data[f'y_test_{objective}']
    
    print(f"\nDataset Info:")
    print(f"  Training:   {X_train_full.shape[0]} samples")
    print(f"  Testing:   {X_test_full.shape[0]} samples")
    print(f"  Classes:   {len(np.unique(y_train))}")
    print(f"  Full features: {X_train_full.shape[1]}")
    print(f"  Selected features: {X_train_selected.shape[1]}")
    print(f"  Best selection method: {data.get(f'best_method_{objective}', 'N/A')}")
    
    comparison = {}
    
    # Model 1: PLS-DA Baseline (Full Features)
    comparison['plsda'] = train_baseline_plsda(
        X_train_full, y_train, X_test_full, y_test, n_components=5
    )
    
    # Model 2: Random Forest (Selected Features)
    comparison['rf'] = train_random_forest(
        X_train_selected, y_train, X_test_selected, y_test,
        n_features_original=X_train_full.shape[1], hyperparameter_tuning=True
    )
    
    # Model 3: SVM (Selected Features)
    comparison['svm'] = train_svm(
        X_train_selected, y_train, X_test_selected, y_test,
        n_features_original=X_train_full.shape[1]
    )
    
    return comparison, y_test


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_model_comparison(comparison, objective, y_test, save_path=None):
    """
    Generate comprehensive visualization comparing all models.
    """
    fig = plt.figure(figsize=(18, 12))
    
    # Extract data for plotting
    models = list(comparison.keys())
    model_names = [comparison[k]['model_name'] for k in models]
    
    # Define target thresholds
    if objective == 'nutrient': 
        target_accuracy = 85.0
        baseline_ref = 73.01
    else:  # contaminant
        target_accuracy = 90.0
        baseline_ref = 84.09
    
    # Plot 1: Accuracy Comparison
    ax1 = fig.add_subplot(2, 3, 1)
    
    test_accs = [comparison[k]['test_accuracy'] * 100 for k in models]
    colors = ['gray', 'green', 'blue']
    
    bars = ax1.bar(model_names, test_accs, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add target line
    ax1.axhline(y=target_accuracy, color='red', linestyle='--', linewidth=2, 
                label=f'Target: {target_accuracy}%')
    ax1.axhline(y=baseline_ref, color='orange', linestyle=':', linewidth=2,
                label=f'Al-Juboori Baseline: {baseline_ref}%')
    
    # Add value labels
    for bar, acc in zip(bars, test_accs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Model Accuracy Comparison ({objective.title()})', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 100])
    
    # Plot 2: Model Size Comparison
    ax2 = fig.add_subplot(2, 3, 2)
    
    model_sizes = []
    model_names_deployable = []
    for k in models:
        if not np.isnan(comparison[k]['model_size_kb']):
            model_sizes.append(comparison[k]['model_size_kb'])
            model_names_deployable.append(comparison[k]['model_name'])
    
    bars2 = ax2.bar(model_names_deployable, model_sizes, 
                    color=['green', 'blue'], alpha=0.7, edgecolor='black', linewidth=2)
    ax2.axhline(y=1024, color='red', linestyle='--', linewidth=2, label='1 MB Limit')
    
    for bar, size in zip(bars2, model_sizes):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{size:.1f} KB',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.set_ylabel('Model Size (KB)', fontsize=12, fontweight='bold')
    ax2.set_title('Flash Memory Footprint', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_yscale('log')
    
    # Plot 3: Inference Latency
    ax3 = fig.add_subplot(2, 3, 3)
    
    latencies = []
    for k in models:
        if not np.isnan(comparison[k]['latency_ms']):
            latencies.append(comparison[k]['latency_ms'])
    
    bars3 = ax3.bar(model_names_deployable, latencies,
                    color=['green', 'blue'], alpha=0.7, edgecolor='black', linewidth=2)
    ax3.axhline(y=50, color='red', linestyle='--', linewidth=2, label='50 ms Target')
    
    for bar, lat in zip(bars3, latencies):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{lat:.2f} ms',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax3.set_ylabel('Inference Latency (ms)', fontsize=12, fontweight='bold')
    ax3.set_title('Inference Speed', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # Plot 4-6: Confusion Matrices
    for idx, (key, name) in enumerate(zip(models, model_names), 4):
        ax = fig.add_subplot(2, 3, idx)
        
        cm = comparison[key]['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=True)
        
        ax.set_xlabel('Predicted', fontsize=11, fontweight='bold')
        ax.set_ylabel('Actual', fontsize=11, fontweight='bold')
        ax.set_title(f'{name}\nAccuracy: {comparison[key]["test_accuracy"]*100:.2f}%', 
                    fontsize=12, fontweight='bold')
    
    plt.suptitle(f'Phase 3:  Model Development Results ({objective.title()} Classification)',
                fontsize=16, fontweight='bold', y=0.998)
    plt.tight_layout(rect=[0, 0, 1, 0.995])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Visualization saved:  {save_path}")
    
    plt.show()


def generate_summary_table(comparison_nutrient, comparison_contaminant):
    """
    Generate comprehensive summary table for thesis.
    """
    print(f"\n{'='*100}")
    print("COMPREHENSIVE MODEL COMPARISON SUMMARY")
    print(f"{'='*100}")
    
    summary_data = []
    
    for objective, comparison in [('Nutrient', comparison_nutrient), ('Contaminant', comparison_contaminant)]:
        for key, result in comparison.items():
            summary_data.append({
                'Objective': objective,
                'Model': result['model_name'],
                'Features': result['n_features'],
                'Test Acc (%)': f"{result['test_accuracy']*100:.2f}",
                'CV Acc (%)': f"{result['cv_mean']*100:.2f} ± {result['cv_std']*100:.2f}",
                'Precision': f"{result['precision']:.4f}",
                'Recall': f"{result['recall']:.4f}",
                'F1':  f"{result['f1_score']:.4f}",
                'Size (KB)': f"{result['model_size_kb']:.2f}" if not np.isnan(result['model_size_kb']) else 'N/A',
                'Latency (ms)': f"{result['latency_ms']:.2f}" if not np.isnan(result['latency_ms']) else 'N/A',
                'Deployable': '✅' if result.get('deployable', False) else '❌'
            })
    
    df_summary = pd.DataFrame(summary_data)
    print(df_summary.to_string(index=False))
    
    return df_summary


# ============================================================================
# SAVE MODELS
# ============================================================================

def save_best_models(comparison_nutrient, comparison_contaminant, data):
    """
    Save the best performing models for Phase 4 deployment.
    """
    output_dir = 'trained_models'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("SAVING TRAINED MODELS")
    print(f"{'='*70}")
    
    for objective, comparison in [('nutrient', comparison_nutrient), ('contaminant', comparison_contaminant)]:
        # Find best deployable model
        best_key = None
        best_acc = 0
        
        for key, result in comparison.items():
            if result.get('deployable', False) and result['test_accuracy'] > best_acc:
                best_key = key
                best_acc = result['test_accuracy']
        
        if best_key: 
            model_data = {
                'model':  comparison[best_key]['model'],
                'model_name': comparison[best_key]['model_name'],
                'test_accuracy': comparison[best_key]['test_accuracy'],
                'n_features': comparison[best_key]['n_features'],
                'selected_indices': data[f'selected_indices_{objective}'],
                'selected_wavelengths': data[f'selected_wavelengths_{objective}'],
                'best_params': comparison[best_key].get('best_params', {}),
                'model_size_kb': comparison[best_key]['model_size_kb'],
                'latency_ms': comparison[best_key]['latency_ms'],
                'objective': objective
            }
            
            # Save with joblib (better for sklearn models)
            model_path = f"{output_dir}/best_model_{objective}.pkl"
            joblib.dump(model_data, model_path)
            
            print(f"\n✓ {objective.title()} Classification:")
            print(f"  Best model: {model_data['model_name']}")
            print(f"  Test accuracy: {model_data['test_accuracy']*100:.2f}%")
            print(f"  Model size: {model_data['model_size_kb']:.2f} KB")
            print(f"  Saved to: {model_path}")
        else:
            print(f"\n⚠️  No deployable model found for {objective}")
    
    print(f"\n{'='*70}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__": 
    print("="*70)
    print("PHASE 3: MODEL DEVELOPMENT (PC-BASED TRAINING)")
    print("Goal: Train classifiers and prove they beat PLS-DA baseline")
    print("="*70)
    
    # Load data
    data = load_phase1_and_phase2_data()
    
    # =========================================================================
    # NUTRIENT CLASSIFICATION (Objective 1)
    # =========================================================================
    print(f"\n{'='*70}")
    print("OBJECTIVE 1: NUTRIENT CLASSIFICATION (NPK - High/Med/Low)")
    print(f"{'='*70}")
    print("Target: ≥85% accuracy")
    print("Baseline: Al-Juboori et al.73.01% (PLS-DA)")
    
    comparison_nutrient, y_test_nutrient = compare_all_models(data, objective='nutrient')
    
    # Visualize
    plot_model_comparison(
        comparison_nutrient,
        objective='nutrient',
        y_test=y_test_nutrient,
        save_path='model_comparison_nutrient.png'
    )
    
    # =========================================================================
    # CONTAMINANT DETECTION (Objective 2)
    # =========================================================================
    print(f"\n{'='*70}")
    print("OBJECTIVE 2: CONTAMINANT DETECTION (Clean/Contaminated)")
    print(f"{'='*70}")
    print("Target: ≥90% accuracy")
    print("Baseline: Al-Juboori et al.84.09% (max CV)")
    
    comparison_contaminant, y_test_contaminant = compare_all_models(data, objective='contaminant')
    
    # Visualize
    plot_model_comparison(
        comparison_contaminant,
        objective='contaminant',
        y_test=y_test_contaminant,
        save_path='model_comparison_contaminant.png'
    )
    
    # =========================================================================
    # COMPREHENSIVE SUMMARY
    # =========================================================================
    summary_df = generate_summary_table(comparison_nutrient, comparison_contaminant)
    
    # Save summary
    summary_df.to_csv('model_comparison_summary.csv', index=False)
    print(f"\n✓ Summary table saved:  model_comparison_summary.csv")
    
    # Save best models for Phase 4
    save_best_models(comparison_nutrient, comparison_contaminant, data)
    
    # =========================================================================
    # FINAL VERDICT
    # =========================================================================
    print(f"\n{'='*100}")
    print("PHASE 3 COMPLETE:  MODEL TRAINING & VALIDATION")
    print(f"{'='*100}")
    
    # Best RF results
    rf_nut_acc = comparison_nutrient['rf']['test_accuracy'] * 100
    rf_con_acc = comparison_contaminant['rf']['test_accuracy'] * 100
    
    plsda_nut_acc = comparison_nutrient['plsda']['test_accuracy'] * 100
    plsda_con_acc = comparison_contaminant['plsda']['test_accuracy'] * 100
    
    print("\nRandom Forest vs PLS-DA Baseline:")
    print(f"  Nutrient Classification:")
    print(f"    RF:      {rf_nut_acc:.2f}%")
    print(f"    PLS-DA: {plsda_nut_acc:.2f}%")
    print(f"    Improvement: {rf_nut_acc - plsda_nut_acc:+.2f} pp")
    print(f"    Target (≥85%): {'✅ ACHIEVED' if rf_nut_acc >= 85 else '⚠️  BELOW TARGET'}")
    
    print(f"\n  Contaminant Detection:")
    print(f"    RF:     {rf_con_acc:.2f}%")
    print(f"    PLS-DA: {plsda_con_acc:.2f}%")
    print(f"    Improvement: {rf_con_acc - plsda_con_acc:+.2f} pp")
    print(f"    Target (≥90%): {'✅ ACHIEVED' if rf_con_acc >= 90 else '⚠️  BELOW TARGET'}")
    
    print(f"\n{'='*100}")
    print("NEXT PHASE: EMBEDDED DEPLOYMENT (Phase 4)")
    print(f"{'='*100}")
    print("\nPhase 4 Objectives:")
    print("  1.Convert trained RF models to C code (emlearn)")
    print("  2.Deploy on ARM Cortex-M MCU (STM32L4/ESP32)")
    print("  3.Validate accuracy matches PC performance")
    print("  4.Measure actual power consumption")
    print("\n  Load trained models:")
    print("    model = joblib.load('trained_models/best_model_nutrient.pkl')")
    print(f"{'='*100}")