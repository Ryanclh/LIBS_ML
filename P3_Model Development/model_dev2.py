"""
LIBS Soil Classification Model Training and Evaluation
Implements PLS-DA baseline, Random Forest, and SVM classifiers
with stratified cross-validation and comprehensive metrics
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================
TRAIN_TEST_SPLIT_RATIO = 0.90  # 90% train, 10% test (adjust as needed)
CV_FOLDS = 6  # 6-fold cross-validation (adjust as needed)
RANDOM_STATE = 42
N_COMPONENTS_PLS = 10  # Number of PLS components (tune this)

# Random Forest parameters (constrained for MCU deployment)
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 10  # Limited depth for small model size
RF_MIN_SAMPLES_SPLIT = 5

# SVM parameters
SVM_KERNEL = 'rbf'  # 'rbf', 'linear', or 'poly'
SVM_C = 1.0
SVM_GAMMA = 'scale'

# ============================================================================
# LOAD YOUR DATA
# ============================================================================
# Replace with your actual data loading code
# Expected format:  X shape = (n_samples, n_features), y shape = (n_samples,)
# Example: 
# X = np.load('preprocessed_spectra.npy')  # Shape: (324, 12981)
# y = np.load('labels.npy')  # Shape: (324,) with classes like 'High', 'Medium', 'Low'

# For demonstration, assuming you have: 
# X_full:  All 12,981 features
# X_selected: Selected subset (e.g., 65 features from RF Gini importance)
# y: Target labels

print("\n" + "="*80)
print("LOADING DATA FILES")
print("="*80)

# -----------------------------------------------------------------------------
# 1. LOAD FULL FEATURES (12,981 wavelengths) - FOR PLS-DA BASELINE
# -----------------------------------------------------------------------------
print("\n[1/3] Loading full feature dataset (CSV)...")
df_full = pd.read_csv(r"D:\Ryan\UM\Academic\Y4S1\KIE4002 FINAL YEAR PROJECT\Irish_Soil_DataSet\Irish_Soil_DataSet2.csv")

# Remove non-feature columns if present (e.g., 'Sample_ID', 'Label', etc.)
# Adjust column names based on your actual CSV structure
non_feature_cols = ['Sample_ID', 'Label', 'Class', 'P_class', 'N_class', 'K_class']
feature_cols_full = [col for col in df_full. columns if col not in non_feature_cols]

X_full = df_full[feature_cols_full]. values  # Extract only wavelength columns
print(f"✓ Loaded X_full:  {X_full.shape}")
print(f"  First 5 column names: {feature_cols_full[: 5]}")

# -----------------------------------------------------------------------------
# 2. LOAD SELECTED FEATURES (65 features) - FOR RF AND SVM
# -----------------------------------------------------------------------------
print("\n[2/3] Loading preprocessed/selected features (NPZ)...")
npz_data = np.load(r"D:\Ryan\UM\Academic\Y4S1\KIE4002 FINAL YEAR PROJECT\P1_Preprocessing\irish_soil_preprocessed_phase1.npz")

# Debug: Check what keys are available
print(f"  Available keys in NPZ: {list(npz_data.keys())}")

# Extract preprocessed spectra - adjust key based on your Phase 1 output
# Common keys: 'X_preprocessed', 'X_selected', 'spectra', 'features', 'arr_0'
if 'X_preprocessed' in npz_data:
    X_selected = npz_data['X_preprocessed']
elif 'X_selected' in npz_data:
    X_selected = npz_data['X_selected']
elif 'spectra' in npz_data:
    X_selected = npz_data['spectra']
elif 'features' in npz_data:
    X_selected = npz_data['features']
elif 'arr_0' in npz_data:   # Default numpy save name
    X_selected = npz_data['arr_0']
else:
    # Fallback: use first available array
    first_key = list(npz_data. keys())[0]
    X_selected = npz_data[first_key]
    print(f"  ⚠️ Using key '{first_key}' - VERIFY THIS IS CORRECT!")

print(f"✓ Loaded X_selected:  {X_selected.shape}")

# -----------------------------------------------------------------------------
# 3. LOAD LABELS (Phosphorus, Nitrogen, or Potassium classification)
# -----------------------------------------------------------------------------
print("\n[3/3] Loading labels (CSV)...")
df_labels = pd.read_csv(r"D:\Ryan\UM\Academic\Y4S1\KIE4002 FINAL YEAR PROJECT\Project\P1_Preprocessing\irish_soil_labels.csv")

print(f"  Available columns: {list(df_labels. columns)}")
print(f"  First 5 rows:\n{df_labels.head()}")

# Extract the target label column
# Adjust based on your CSV structure:  'P_class', 'N_class', 'K_class', 'label', 'class'
if 'P_class' in df_labels.columns:
    y = df_labels['P_class']. values
    target_nutrient = 'Phosphorus (P)'
elif 'N_class' in df_labels.columns:
    y = df_labels['N_class'].values
    target_nutrient = 'Nitrogen (N)'
elif 'K_class' in df_labels.columns:
    y = df_labels['K_class'].values
    target_nutrient = 'Potassium (K)'
elif 'label' in df_labels.columns:
    y = df_labels['label'].values
    target_nutrient = 'Unknown'
elif 'class' in df_labels.columns:
    y = df_labels['class'].values
    target_nutrient = 'Unknown'
else:
    # Fallback: use first column
    y = df_labels. iloc[:, 0].values
    target_nutrient = 'First Column (VERIFY! )'
    print(f"  ⚠️ Using first column as labels - VERIFY THIS IS CORRECT!")

print(f"✓ Loaded labels: {y.shape}")
print(f"  Target nutrient:  {target_nutrient}")
print(f"  Unique classes: {np.unique(y)}")

# Encode string labels to integers (e.g., 'High' -> 0, 'Low' -> 1, 'Medium' -> 2)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"  Encoded classes: {le.classes_} -> {np.unique(y_encoded)}")

# -----------------------------------------------------------------------------
# 4. DATA VERIFICATION - CRITICAL CHECKS
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("DATA VERIFICATION")
print("="*80)

# Check 1: Sample counts must match
n_samples_full = X_full.shape[0]
n_samples_selected = X_selected.shape[0]
n_samples_labels = y.shape[0]

print(f"\nSample counts:")
print(f"  X_full:      {n_samples_full}")
print(f"  X_selected: {n_samples_selected}")
print(f"  y (labels): {n_samples_labels}")

if not (n_samples_full == n_samples_selected == n_samples_labels):
    raise ValueError(
        f"❌ SAMPLE COUNT MISMATCH!\n"
        f"   X_full: {n_samples_full}, X_selected: {n_samples_selected}, y: {n_samples_labels}\n"
        f"   All datasets must have the same number of samples (324 expected)."
    )
print("✓ Sample counts match!")

# Check 2: Feature counts
print(f"\nFeature counts:")
print(f"  X_full features:      {X_full.shape[1]} (expected ~12,981)")
print(f"  X_selected features: {X_selected.shape[1]} (expected ~65-100)")

if X_full.shape[1] < 1000:
    print(f"  ⚠️ WARNING: X_full has only {X_full.shape[1]} features (expected ~12,981)")
    print(f"     Check if Irish_Soil_DataSet2.csv contains all wavelength columns!")

# Check 3: Class distribution (should be balanced:  108-108-108)
unique, counts = np.unique(y_encoded, return_counts=True)
class_distribution = dict(zip(le.inverse_transform(unique), counts))
print(f"\nClass distribution:")
for class_name, count in class_distribution.items():
    print(f"  {class_name}: {count} samples")

if len(unique) != 3:
    print(f"  ⚠️ WARNING:  Expected 3 classes (High/Medium/Low), found {len(unique)}")

# Check 4: Missing or invalid values
nan_full = np.isnan(X_full).sum()
nan_selected = np.isnan(X_selected).sum()
inf_full = np.isinf(X_full).sum()
inf_selected = np.isinf(X_selected).sum()

if nan_full > 0 or inf_full > 0:
    print(f"  ⚠️ WARNING: X_full contains {nan_full} NaN and {inf_full} Inf values")
if nan_selected > 0 or inf_selected > 0:
    print(f"  ⚠️ WARNING: X_selected contains {nan_selected} NaN and {inf_selected} Inf values")
    
if nan_full == 0 and nan_selected == 0 and inf_full == 0 and inf_selected == 0:
    print("✓ No NaN or Inf values detected")

# Check 5: Data ranges (sanity check)
print(f"\nData ranges:")
print(f"  X_full:      min={X_full.min():.4f}, max={X_full. max():.4f}, mean={X_full.mean():.4f}")
print(f"  X_selected: min={X_selected.min():.4f}, max={X_selected. max():.4f}, mean={X_selected.mean():.4f}")

print("\n" + "="*80)
print("✓ DATA LOADING COMPLETE AND VERIFIED!")
print("="*80)
print(f"\nReady for Phase 3 Model Training:")
print(f"  • Baseline: PLS-DA with {X_full.shape[1]} features")
print(f"  • Advanced: RF & SVM with {X_selected. shape[1]} features")
print(f"  • Target: {target_nutrient} classification")
print(f"  • Target Accuracy: ≥85% (vs baseline 73.01%)")
print("="*80 + "\n")


# For demonstration purposes only:
#print("\n⚠️  PLACEHOLDER DATA - REPLACE WITH YOUR ACTUAL DATA")
#np.random.seed(RANDOM_STATE)
#n_samples = 324
#X_full = np. random.randn(n_samples, 12981)
#X_selected = np.random.randn(n_samples, 65)
#y = np.random.choice(['High', 'Medium', 'Low'], size=n_samples)

# Encode labels to numeric values
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"\nDataset Information:")
print(f"  Full features shape: {X_full.shape}")
print(f"  Selected features shape: {X_selected. shape}")
print(f"  Labels shape: {y.shape}")
print(f"  Classes: {le.classes_}")
print(f"  Class distribution: {np.bincount(y_encoded)}")

# ============================================================================
# DATA SPLITTING WITH STRATIFICATION
# ============================================================================
print("\n" + "="*80)
print("1. DATA SPLITTING")
print("="*80)

# Split for full features (PLS-DA baseline)
X_train_full, X_test_full, y_train, y_test = train_test_split(
    X_full, y_encoded, 
    test_size=(1 - TRAIN_TEST_SPLIT_RATIO),
    stratify=y_encoded,
    random_state=RANDOM_STATE
)

# Split for selected features (RF and SVM)
X_train_sel, X_test_sel, _, _ = train_test_split(
    X_selected, y_encoded,
    test_size=(1 - TRAIN_TEST_SPLIT_RATIO),
    stratify=y_encoded,
    random_state=RANDOM_STATE
)

print(f"\nTrain-Test Split Ratio: {TRAIN_TEST_SPLIT_RATIO*100:.0f}/{(1-TRAIN_TEST_SPLIT_RATIO)*100:.0f}")
print(f"Training set shape (full): {X_train_full.shape}")
print(f"Testing set shape (full):  {X_test_full. shape}")
print(f"Training set shape (selected): {X_train_sel.shape}")
print(f"Testing set shape (selected):  {X_test_sel. shape}")
print(f"\nTraining labels distribution: {np.bincount(y_train)}")
print(f"Testing labels distribution:  {np.bincount(y_test)}")

# ============================================================================
# MODEL 1: PLS-DA BASELINE (Al-Juboori's Approach)
# ============================================================================
print("\n" + "="*80)
print("2. BASELINE MODEL: PLS-DA (All 12,981 Features)")
print("="*80)

# PLS-DA implementation:  PLS + LDA
# Step 1: Apply PLS for dimensionality reduction
pls = PLSRegression(n_components=N_COMPONENTS_PLS, scale=False)
X_train_pls = pls.fit_transform(X_train_full, y_train)[0]
X_test_pls = pls.transform(X_test_full)

# Step 2: Apply LDA for classification
lda = LinearDiscriminantAnalysis()
lda.fit(X_train_pls, y_train)

# Cross-validation on training set
cv_strategy = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# Need to create a pipeline-like scoring
pls_cv = PLSRegression(n_components=N_COMPONENTS_PLS, scale=False)
lda_cv = LinearDiscriminantAnalysis()

cv_scores_plsda = []
for train_idx, val_idx in cv_strategy.split(X_train_full, y_train):
    X_cv_train, X_cv_val = X_train_full[train_idx], X_train_full[val_idx]
    y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]
    
    X_cv_train_pls = pls_cv.fit_transform(X_cv_train, y_cv_train)[0]
    X_cv_val_pls = pls_cv.transform(X_cv_val)
    
    lda_cv.fit(X_cv_train_pls, y_cv_train)
    score = lda_cv.score(X_cv_val_pls, y_cv_val)
    cv_scores_plsda.append(score)

cv_scores_plsda = np.array(cv_scores_plsda)

print(f"\n{CV_FOLDS}-Fold Cross-Validation Results:")
print(f"  Mean CV Accuracy: {cv_scores_plsda.mean():.4f}")
print(f"  Std Dev CV Accuracy: {cv_scores_plsda.std():.4f}")
print(f"  Individual fold scores:  {cv_scores_plsda}")

# Final evaluation on test set
y_pred_plsda = lda. predict(X_test_pls)
accuracy_plsda = accuracy_score(y_test, y_pred_plsda)
conf_matrix_plsda = confusion_matrix(y_test, y_pred_plsda)

print(f"\nTest Set Evaluation:")
print(f"  Test Accuracy: {accuracy_plsda:.4f} ({accuracy_plsda*100:. 2f}%)")
print(f"\nConfusion Matrix:")
print(conf_matrix_plsda)
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_plsda, target_names=le.classes_))

# ============================================================================
# MODEL 2: RANDOM FOREST (Selected Features)
# ============================================================================
print("\n" + "="*80)
print("3. ADVANCED MODEL 1: Random Forest (Selected Features)")
print("="*80)

rf_model = RandomForestClassifier(
    n_estimators=RF_N_ESTIMATORS,
    max_depth=RF_MAX_DEPTH,
    min_samples_split=RF_MIN_SAMPLES_SPLIT,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

# Cross-validation
cv_scores_rf = cross_val_score(
    rf_model, X_train_sel, y_train,
    cv=cv_strategy,
    scoring='accuracy'
)

print(f"\n{CV_FOLDS}-Fold Cross-Validation Results:")
print(f"  Mean CV Accuracy: {cv_scores_rf.mean():.4f}")
print(f"  Std Dev CV Accuracy: {cv_scores_rf.std():.4f}")
print(f"  Individual fold scores: {cv_scores_rf}")

# Train on full training set and evaluate on test set
rf_model.fit(X_train_sel, y_train)
y_pred_rf = rf_model. predict(X_test_sel)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)

print(f"\nTest Set Evaluation:")
print(f"  Test Accuracy: {accuracy_rf:.4f} ({accuracy_rf*100:.2f}%)")
print(f"\nConfusion Matrix:")
print(conf_matrix_rf)
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_rf, target_names=le.classes_))

# Model complexity
print(f"\nModel Complexity:")
print(f"  Number of trees: {RF_N_ESTIMATORS}")
print(f"  Max depth:  {RF_MAX_DEPTH}")
print(f"  Number of features used: {X_train_sel.shape[1]}")

# ============================================================================
# MODEL 3: SUPPORT VECTOR MACHINE (Selected Features)
# ============================================================================
print("\n" + "="*80)
print("4. ADVANCED MODEL 2: SVM (Selected Features)")
print("="*80)

svm_model = SVC(
    kernel=SVM_KERNEL,
    C=SVM_C,
    gamma=SVM_GAMMA,
    random_state=RANDOM_STATE
)

# Cross-validation
cv_scores_svm = cross_val_score(
    svm_model, X_train_sel, y_train,
    cv=cv_strategy,
    scoring='accuracy'
)

print(f"\n{CV_FOLDS}-Fold Cross-Validation Results:")
print(f"  Mean CV Accuracy: {cv_scores_svm.mean():.4f}")
print(f"  Std Dev CV Accuracy:  {cv_scores_svm. std():.4f}")
print(f"  Individual fold scores:  {cv_scores_svm}")

# Train on full training set and evaluate on test set
svm_model.fit(X_train_sel, y_train)
y_pred_svm = svm_model.predict(X_test_sel)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)

print(f"\nTest Set Evaluation:")
print(f"  Test Accuracy: {accuracy_svm:.4f} ({accuracy_svm*100:.2f}%)")
print(f"\nConfusion Matrix:")
print(conf_matrix_svm)
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_svm, target_names=le. classes_))

print(f"\nModel Parameters:")
print(f"  Kernel: {SVM_KERNEL}")
print(f"  C: {SVM_C}")
print(f"  Gamma:  {SVM_GAMMA}")
print(f"  Number of support vectors: {svm_model.n_support_}")

# ============================================================================
# COMPARATIVE SUMMARY
# ============================================================================
print("\n" + "="*80)
print("5. COMPARATIVE SUMMARY")
print("="*80)

summary_data = {
    'Model': ['PLS-DA (Baseline)', 'Random Forest', 'SVM'],
    'Features Used': [12981, X_train_sel.shape[1], X_train_sel.shape[1]],
    'Mean CV Accuracy': [cv_scores_plsda.mean(), cv_scores_rf.mean(), cv_scores_svm.mean()],
    'CV Std Dev': [cv_scores_plsda.std(), cv_scores_rf.std(), cv_scores_svm.std()],
    'Test Accuracy': [accuracy_plsda, accuracy_rf, accuracy_svm],
    'Improvement vs Baseline': [0, accuracy_rf - accuracy_plsda, accuracy_svm - accuracy_plsda]
}

summary_df = pd.DataFrame(summary_data)
print("\n", summary_df.to_string(index=False))

# Find best model
best_model_idx = summary_df['Test Accuracy'].idxmax()
best_model_name = summary_df.loc[best_model_idx, 'Model']
best_accuracy = summary_df.loc[best_model_idx, 'Test Accuracy']

print(f"\n🏆 Best Performing Model: {best_model_name}")
print(f"   Test Accuracy: {best_accuracy:. 4f} ({best_accuracy*100:.2f}%)")

# Check if target metrics are met
print("\n" + "="*80)
print("6. TARGET METRICS VERIFICATION")
print("="*80)

baseline_accuracy = 0.7301  # Al-Juboori et al. baseline
target_accuracy = 0.85  # Primary objective target

print(f"\nBaseline (Al-Juboori et al.): {baseline_accuracy:.4f} (73.01%)")
print(f"Target Accuracy: {target_accuracy:.4f} (85.00%)")
print(f"\nYour Results:")
print(f"  PLS-DA: {accuracy_plsda:.4f} ({accuracy_plsda*100:. 2f}%) - {'✓ PASS' if accuracy_plsda >= baseline_accuracy else '✗ BELOW BASELINE'}")
print(f"  Random Forest: {accuracy_rf:.4f} ({accuracy_rf*100:.2f}%) - {'✓ TARGET MET' if accuracy_rf >= target_accuracy else '✗ BELOW TARGET'}")
print(f"  SVM: {accuracy_svm:.4f} ({accuracy_svm*100:.2f}%) - {'✓ TARGET MET' if accuracy_svm >= target_accuracy else '✗ BELOW TARGET'}")

# ============================================================================
# VISUALIZATION:  CONFUSION MATRICES
# ============================================================================
print("\n" + "="*80)
print("7. GENERATING CONFUSION MATRIX VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# PLS-DA Confusion Matrix
sns.heatmap(conf_matrix_plsda, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_,
            ax=axes[0], cbar_kws={'label': 'Count'})
axes[0].set_title(f'PLS-DA Confusion Matrix\nAccuracy: {accuracy_plsda:.4f}')
axes[0].set_ylabel('True Label')
axes[0].set_xlabel('Predicted Label')

# Random Forest Confusion Matrix
sns. heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Greens',
            xticklabels=le.classes_, yticklabels=le.classes_,
            ax=axes[1], cbar_kws={'label': 'Count'})
axes[1].set_title(f'Random Forest Confusion Matrix\nAccuracy: {accuracy_rf:. 4f}')
axes[1].set_ylabel('True Label')
axes[1].set_xlabel('Predicted Label')

# SVM Confusion Matrix
sns.heatmap(conf_matrix_svm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=le.classes_, yticklabels=le.classes_,
            ax=axes[2], cbar_kws={'label': 'Count'})
axes[2].set_title(f'SVM Confusion Matrix\nAccuracy: {accuracy_svm:.4f}')
axes[2].set_ylabel('True Label')
axes[2].set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Confusion matrices saved as 'confusion_matrices_comparison.png'")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "="*80)
print("8. SAVING RESULTS")
print("="*80)

# Save models
import joblib
joblib.dump((pls, lda), 'models/plsda_model.pkl')
joblib.dump(rf_model, 'models/rf_model.pkl')
joblib.dump(svm_model, 'models/svm_model.pkl')
joblib.dump(le, 'models/label_encoder.pkl')

# Save results to CSV
summary_df.to_csv('results/model_comparison_summary.csv', index=False)

# Save detailed predictions
results_df = pd.DataFrame({
    'True_Label': le.inverse_transform(y_test),
    'PLSDA_Pred': le.inverse_transform(y_pred_plsda),
    'RF_Pred': le.inverse_transform(y_pred_rf),
    'SVM_Pred': le.inverse_transform(y_pred_svm)
})
results_df.to_csv('results/test_predictions.csv', index=False)

print("\n✓ Models saved to 'models/' directory")
print("✓ Summary saved to 'results/model_comparison_summary.csv'")
print("✓ Test predictions saved to 'results/test_predictions. csv'")

print("\n" + "="*80)
print("TRAINING AND EVALUATION COMPLETE")
print("="*80)