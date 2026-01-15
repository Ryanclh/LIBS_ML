import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Load LIBS spectra
spectra_df = pd.read_csv(
    r"D:\Ryan\UM\Academic\Y4S1\KIE4002 FINAL YEAR PROJECT\Irish_Soil_DataSet\Irish_Soil_DataSet2.csv",
    index_col=0
)

# CRITICAL: Verify data orientation
print(f"Raw spectra shape: {spectra_df.shape}")
print(f"Index (samples): {len(spectra_df.index)}")
print(f"Columns (wavelengths): {len(spectra_df.columns)}")

# If wavelengths are in index and samples are in columns, TRANSPOSE
if len(spectra_df.index) > len(spectra_df.columns):
    print("⚠️ Wavelengths in rows - transposing...")
    spectra_df = spectra_df.T

X_raw = spectra_df.values  # Shape: (n_samples, n_wavelengths)
sample_names = spectra_df.index.tolist()

print(f"✓ Spectra shape after verification: {X_raw.shape}")
print(f"  - Samples: {X_raw.shape[0]}")
print(f"  - Wavelengths: {X_raw.shape[1]}")

# Load contaminant labels
labels_df = pd.read_csv(
    r"D:\Ryan\UM\Academic\Y4S1\KIE4002 FINAL YEAR PROJECT\Project\irish_soil_labels.csv"
)
y = labels_df.iloc[:, 1].values  # Extract second column (0 or 1)

# Verify alignment
print(f"\nLabels shape: {y.shape}")
print(f"Class distribution: Clean={sum(y==0)}, Contaminated={sum(y==1)}")

# Check for mismatch
if X_raw.shape[0] != y.shape[0]:
    print(f"❌ ERROR: Sample count mismatch!")
    print(f"   Spectra: {X_raw.shape[0]} samples")
    print(f"   Labels: {y.shape[0]} samples")
    exit(1)

print(f"✓ Alignment verified: {X_raw.shape[0]} samples with labels")

# ============================================================================
# FEATURE EXTRACTION: Spectral Lines
# ============================================================================
print("\n" + "="*60)
print("FEATURE EXTRACTION: Extracting peak-based features")
print("="*60)

X_peaks_fixed = []
max_peaks = 75

for i in range(X_raw.shape[0]):
    spectrum = X_raw[i, :]
    peaks, properties = find_peaks(spectrum, prominence=1500, distance=10)
    
    if len(peaks) > 0:
        peak_intensities = spectrum[peaks]
        peak_intensities_sorted = np.sort(peak_intensities)[::-1]
        
        if len(peak_intensities_sorted) < max_peaks:
            padded = np.pad(peak_intensities_sorted, 
                          (0, max_peaks - len(peak_intensities_sorted)), 
                          mode='constant')
        else:
            padded = peak_intensities_sorted[:max_peaks]
    else:
        padded = np.zeros(max_peaks)
    
    X_peaks_fixed.append(padded)

X_features = np.array(X_peaks_fixed)
print(f"✓ Spectral lines extracted: {X_features.shape}")

# ============================================================================
# DIMENSIONALITY REDUCTION: PCA
# ============================================================================
print("\n" + "="*60)
print("DIMENSIONALITY REDUCTION: PCA on raw spectra")
print("="*60)

# Apply PCA to raw spectra (should be n_samples × n_wavelengths)
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_raw)

print(f"✓ PCA fitted on {X_raw.shape} data")
print(f"✓ Variance explained by 5 PCs: {pca.explained_variance_ratio_.sum():.2%}")
print(f"✓ PCA output shape: {X_pca.shape}")

# ============================================================================
# FEATURE COMBINATION
# ============================================================================
print("\n" + "="*60)
print("COMBINING FEATURES")
print("="*60)

X_combined = np.hstack([X_features, X_pca])
print(f"✓ Combined features shape: {X_combined.shape}")
print(f"  - Spectral lines: {X_features.shape[1]}")
print(f"  - PCA components: {X_pca.shape[1]}")

# Verify shapes before scaling
print(f"\n✓ Before scaling:")
print(f"  - X_combined: {X_combined.shape}")
print(f"  - y: {y.shape}")

assert X_combined.shape[0] == y.shape[0], "Sample count mismatch!"

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)
print(f"✓ After scaling: {X_scaled.shape}")

# ============================================================================
# TRAIN/TEST SPLIT
# ============================================================================
print("\n" + "="*60)
print("TRAIN/TEST SPLIT (90/10 stratified)")
print("="*60)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, 
    test_size=0.10, 
    random_state=42, 
    stratify=y
)

print(f"✓ Training set: {X_train.shape}")
print(f"✓ Test set: {X_test.shape}")
print(f"  - Train class distribution: Clean={sum(y_train==0)}, Contaminated={sum(y_train==1)}")
print(f"  - Test class distribution: Clean={sum(y_test==0)}, Contaminated={sum(y_test==1)}")

# ============================================================================
# MODEL TRAINING: PLS-DA with 6-Fold CV
# ============================================================================
print("\n" + "="*60)
print("MODEL TRAINING: PLS-DA with 6-Fold Cross-Validation")
print("="*60)

pls_da = PLSRegression(n_components=2, scale=False)
skfold = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(skfold.split(X_train, y_train), 1):
    X_cv_train = X_train[train_idx]
    X_cv_val = X_train[val_idx]
    y_cv_train = y_train[train_idx]
    y_cv_val = y_train[val_idx]
    
    pls_da.fit(X_cv_train, y_cv_train)
    
    y_cv_pred = pls_da.predict(X_cv_val)
    y_cv_pred_binary = (y_cv_pred > 0.5).astype(int).ravel()
    
    fold_accuracy = accuracy_score(y_cv_val, y_cv_pred_binary)
    cv_scores.append(fold_accuracy)
    
    print(f"  Fold {fold}: Accuracy = {fold_accuracy:.4f}")

print(f"\n✓ Mean CV Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
print(f"✓ Best CV Accuracy: {np.max(cv_scores):.4f}")
print(f"✓ Worst CV Accuracy: {np.min(cv_scores):.4f}")

# ============================================================================
# FINAL MODEL EVALUATION
# ============================================================================
print("\n" + "="*60)
print("FINAL MODEL EVALUATION ON TEST SET")
print("="*60)

# Train final model on full training set
pls_da_final = PLSRegression(n_components=2, scale=False)
pls_da_final.fit(X_train, y_train)

# Predict on training set
y_train_pred = pls_da_final.predict(X_train)
y_train_pred_binary = np.round(y_train_pred).astype(int).ravel()
y_train_pred_binary = np.clip(y_train_pred_binary, 0, 1)
train_accuracy = accuracy_score(y_train, y_train_pred_binary)

# Predict on test set
y_test_pred = pls_da_final.predict(X_test)
y_test_pred_binary = np.round(y_test_pred).astype(int).ravel()
y_test_pred_binary = np.clip(y_test_pred_binary, 0, 1)
test_accuracy = accuracy_score(y_test, y_test_pred_binary)

print(f"✓ Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"✓ Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# ============================================================================
# CLASSIFICATION REPORT
# ============================================================================
print("\n" + "="*60)
print("CLASSIFICATION REPORT (Test Set)")
print("="*60)

# Debug: Check unique values in predictions
print(f"Unique values in y_test_pred_binary: {np.unique(y_test_pred_binary)}")
print(f"Unique values in y_test: {np.unique(y_test)}")

print(classification_report(y_test, y_test_pred_binary, 
                          target_names=['Clean', 'Contaminated'],
                          labels=[0, 1]))

print("\n" + "="*60)
print("CONFUSION MATRIX (Test Set)")
print("="*60)
cm = confusion_matrix(y_test, y_test_pred_binary, labels=[0, 1])
print(cm)
print(f"  True Negatives (TN): {cm[0, 0]}")
print(f"  False Positives (FP): {cm[0, 1]}")
print(f"  False Negatives (FN): {cm[1, 0]}")
print(f"  True Positives (TP): {cm[1, 1]}")

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\n" + "="*60)
print("GENERATING VISUALIZATIONS")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: CV Accuracy by Fold
ax = axes[0, 0]
fold_numbers = np.arange(1, 7)
ax.plot(fold_numbers, cv_scores, marker='o', linewidth=2, markersize=8, color='#2ecc71')
ax.axhline(y=np.mean(cv_scores), color='r', linestyle='--', label=f'Mean: {np.mean(cv_scores):.4f}')
ax.set_xlabel('Fold Number', fontsize=11)
ax.set_ylabel('Accuracy', fontsize=11)
ax.set_title('6-Fold Cross-Validation Accuracy', fontweight='bold', fontsize=12)
ax.set_ylim([0, 1.05])
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 2: Confusion Matrix
ax = axes[0, 1]
ConfusionMatrixDisplay(cm, display_labels=['Clean', 'Contaminated']).plot(ax=ax, cmap='Blues')
ax.set_title('Confusion Matrix - Test Set', fontweight='bold', fontsize=12)

# Plot 3: Train vs Test Accuracy
ax = axes[1, 0]
models = ['Training', 'Test']
accuracies = [train_accuracy, test_accuracy]
colors = ['#3498db', '#e74c3c']
bars = ax.bar(models, accuracies, color=colors, edgecolor='black', alpha=0.7, width=0.5)
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{acc*100:.2f}%',
           ha='center', va='bottom', fontweight='bold', fontsize=11)
ax.set_ylabel('Accuracy', fontsize=11)
ax.set_title('Training vs Test Accuracy', fontweight='bold', fontsize=12)
ax.set_ylim([0, 1.05])
ax.grid(True, alpha=0.3, axis='y')

# Plot 4: Feature Importance (PLS Loadings)
ax = axes[1, 1]
loadings = pls_da_final.coef_.ravel()
feature_importance = np.abs(loadings)
top_features_idx = np.argsort(feature_importance)[::-1][:15]

feature_labels = [f'Peak_{i}' if i < 75 else f'PC_{i-75}' for i in top_features_idx]
ax.barh(range(len(top_features_idx)), feature_importance[top_features_idx], color='#9b59b6', alpha=0.7)
ax.set_yticks(range(len(top_features_idx)))
ax.set_yticklabels(feature_labels, fontsize=9)
ax.set_xlabel('Absolute Loading', fontsize=11)
ax.set_title('Top 15 Important Features (PLS Loadings)', fontweight='bold', fontsize=12)
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

plt.suptitle('PLS-DA Classification Results: Contamination Detection', 
            fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('classification_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Figure saved as 'classification_results.png'")

# ============================================================================
# TOP FEATURES ANALYSIS
# ============================================================================
print("\n" + "="*60)
print("TOP 10 IMPORTANT FEATURES")
print("="*60)

for i, idx in enumerate(top_features_idx[:10], 1):
    if idx < 75:
        feature_name = f"Spectral Line Peak {idx}"
    else:
        feature_name = f"PCA Component {idx-75}"
    print(f"{i:2d}. {feature_name:30s} | Importance = {feature_importance[idx]:10.6f}")