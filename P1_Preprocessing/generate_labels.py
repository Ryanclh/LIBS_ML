"""
Irish Soil Dataset Label Generation Script
===========================================
Generates labels for both research objectives from sample names

Objective 1: Nutrient Level Classification (High/Med/Low)
- Based on soil type as proxy for nutrient holding capacity (CEC)
- High: Clay, ClayLoam (high CEC)
- Medium: Garden_top, Chalk (medium fertility)
- Low: Sandy, Gravel (low CEC, leaches nutrients)
- Excluded:  Contaminated samples (may distort nutrient signals)

Objective 2: Contaminant Detection (Clean/Contaminated)
- Clean: Uncontaminated soils
- Contaminated: TPH, Arsenic, HeavyMetals, ChalkT
"""

import pandas as pd
import numpy as np
import os


def get_nutrient_label(sample_name:  str) -> int:
    """
    Map sample name to nutrient level label. 
    
    Based on soil type as proxy for Cation Exchange Capacity (CEC):
    - Clay/ClayLoam: High CEC (retains nutrients) → High (2)
    - Garden_top/Chalk: Medium fertility → Medium (1)
    - Sandy/Gravel: Low CEC (leaches nutrients) → Low (0)
    - Contaminated: Excluded from nutrient training → -1
    
    Parameters
    ----------
    sample_name :  str
        Sample identifier (e.g., Clay_1, Sandy_15, TPH_Arsenic_10)
        
    Returns
    -------
    label : int
        2=High, 1=Medium, 0=Low, -1=Excluded
    """
    n = str(sample_name).lower()
    
    # HIGH NUTRIENT RETENTION (High CEC)
    if 'clay' in n and 'loam' not in n:
        return 2  # Clay_1 to Clay_30
    elif 'clayloam' in n:
        return 2  # ClayLoam_1 to ClayLoam_30
    
    # MEDIUM NUTRIENT RETENTION
    elif 'garden' in n: 
        return 1  # Garden_top_1 to Garden_top_30
    elif 'chalk_' in n and 't' not in n. split('_')[-1]. lower():
        return 1  # Chalk_1 to Chalk_30 (NOT ChalkT)
    
    # LOW NUTRIENT RETENTION (Low CEC)
    elif 'sandy' in n: 
        return 0  # Sandy_1 to Sandy_30
    elif 'gravel' in n:
        return 0  # Gravel_1 to Gravel_30
    
    # CONTAMINATED - EXCLUDE from nutrient classification
    # Rationale: Contaminants suppress nutrient spectral signatures
    elif 'tph' in n or 'arsenic' in n or 'heavymetals' in n or 'chalkt' in n: 
        return -1
    
    # Unknown sample type
    else:
        return -1


def get_contaminant_label(sample_name: str) -> int:
    """
    Map sample name to contaminant detection label (binary).
    
    Parameters
    ----------
    sample_name : str
        Sample identifier
        
    Returns
    -------
    label : int
        0=Clean, 1=Contaminated
    """
    n = str(sample_name).lower()
    
    # CONTAMINATED CLASS
    if 'tph' in n:
        return 1  # Total Petroleum Hydrocarbons (Diesel contamination)
    elif 'arsenic' in n: 
        return 1  # Arsenic heavy metal contamination
    elif 'heavymetals' in n: 
        return 1  # Heavy metals mix (Pb, Cd, Cr)
    elif 'chalkt' in n:
        return 1  # ChalkT = Chalk Treated (contaminated)
    
    # CLEAN CLASS (Control samples)
    elif 'clay' in n and 'loam' not in n:
        return 0  # Clay_1 to Clay_30
    elif 'clayloam' in n:
        return 0  # ClayLoam_1 to ClayLoam_30
    elif 'chalk_' in n:
        return 0  # Chalk_1 to Chalk_30 (non-treated)
    elif 'gravel' in n:
        return 0  # Gravel_1 to Gravel_30
    elif 'sandy' in n:
        return 0  # Sandy_1 to Sandy_30
    elif 'garden' in n:
        return 0  # Garden_top_1 to Garden_top_30
    
    # Default:  Assume clean if type unknown
    else:
        return 0


def generate_labels_from_csv(
    filepath: str,
    output_csv: str = "irish_soil_labels.csv"
) -> pd.DataFrame:
    """
    Read Irish Soil CSV and generate labels for both objectives. 
    
    Parameters
    ----------
    filepath : str
        Path to Irish_Soil_DataSet.csv
    output_csv : str
        Path to save labeled dataset
        
    Returns
    -------
    df_labels : pd.DataFrame
        DataFrame with columns:  Sample_Name, Nutrient_Label, Contaminant_Label
    """
    print("="*70)
    print("IRISH SOIL DATASET LABEL GENERATION")
    print("="*70)
    print(f"\nReading CSV: {filepath}")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Read CSV (first column = wavelengths, first row = sample names)
    df = pd.read_csv(filepath, index_col=0)
    
    # Extract sample names from column headers
    sample_names = df. columns. astype(str).tolist()
    n_samples = len(sample_names)
    
    print(f"Found {n_samples} samples")
    print(f"First 5 samples: {sample_names[:5]}")
    
    # Generate labels
    print("\nGenerating labels...")
    nutrient_labels = [get_nutrient_label(name) for name in sample_names]
    contaminant_labels = [get_contaminant_label(name) for name in sample_names]
    
    # Create DataFrame
    df_labels = pd.DataFrame({
        'Sample_Name': sample_names,
        'Nutrient_Label': nutrient_labels,
        'Contaminant_Label': contaminant_labels
    })
    
    # Summary statistics
    print("\n" + "="*70)
    print("OBJECTIVE 1: NUTRIENT LEVEL CLASSIFICATION")
    print("="*70)
    print("Mapping:  Soil Type → Nutrient Retention Capacity (CEC)")
    
    nutrient_counts = df_labels['Nutrient_Label'].value_counts().sort_index()
    label_map = {2: 'High', 1: 'Medium', 0: 'Low', -1: 'Excluded'}
    
    for label_val in [2, 1, 0, -1]:
        if label_val in nutrient_counts.index:
            count = nutrient_counts[label_val]
            label_name = label_map[label_val]
            print(f"  {label_name:10s} ({label_val:2d}): {count:3d} samples")
            
            # Show example samples
            examples = df_labels[df_labels['Nutrient_Label'] == label_val]['Sample_Name'].head(3).tolist()
            print(f"              Examples: {', '.join(examples)}")
    
    n_usable_nutrient = len(df_labels[df_labels['Nutrient_Label'] != -1])
    print(f"\nTotal usable for nutrient classification: {n_usable_nutrient} samples")
    
    print("\n" + "="*70)
    print("OBJECTIVE 2: CONTAMINANT DETECTION")
    print("="*70)
    print("Binary Classification: Clean (0) vs Contaminated (1)")
    
    contaminant_counts = df_labels['Contaminant_Label'].value_counts().sort_index()
    contaminant_map = {0: 'Clean', 1: 'Contaminated'}
    
    for label_val in [0, 1]:
        if label_val in contaminant_counts.index:
            count = contaminant_counts[label_val]
            label_name = contaminant_map[label_val]
            percentage = count / n_samples * 100
            print(f"  {label_name:15s} ({label_val}): {count:3d} samples ({percentage:.1f}%)")
            
            # Show example samples
            examples = df_labels[df_labels['Contaminant_Label'] == label_val]['Sample_Name'].head(3).tolist()
            print(f"                  Examples: {', '.join(examples)}")
    
    # Save to CSV
    df_labels.to_csv(output_csv, index=False)
    print("\n" + "="*70)
    print("LABELS SAVED")
    print("="*70)
    print(f"Output file: {output_csv}")
    print(f"Columns: Sample_Name, Nutrient_Label, Contaminant_Label")
    print("\nUsage in other scripts:")
    print("  df_labels = pd.read_csv('irish_soil_labels.csv')")
    print("  nutrient_labels = df_labels['Nutrient_Label'].values")
    print("  contaminant_labels = df_labels['Contaminant_Label'].values")
    print("="*70)
    
    return df_labels


def validate_labels(df_labels: pd.DataFrame) -> None:
    """
    Validate label distribution and check for issues.
    
    Parameters
    ----------
    df_labels : pd.DataFrame
        Labeled dataset
    """
    print("\n" + "="*70)
    print("LABEL VALIDATION")
    print("="*70)
    
    # Check for balanced classes (Objective 1)
    nutrient_valid = df_labels[df_labels['Nutrient_Label'] != -1]
    nutrient_counts = nutrient_valid['Nutrient_Label'].value_counts()
    
    print("\nObjective 1 (Nutrient) - Class Balance:")
    if len(nutrient_counts) > 0:
        max_count = nutrient_counts.max()
        min_count = nutrient_counts. min()
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        for label in [2, 1, 0]: 
            count = nutrient_counts. get(label, 0)
            print(f"  Label {label}: {count} samples")
        
        if imbalance_ratio > 2.0:
            print(f"\n⚠️ WARNING: Class imbalance detected (ratio: {imbalance_ratio:.2f})")
            print("   Consider using stratified sampling or class weights in training.")
        else:
            print(f"\n✓ Good class balance (ratio: {imbalance_ratio:.2f} < 2.0)")
    
    # Check for balanced classes (Objective 2)
    contaminant_counts = df_labels['Contaminant_Label'].value_counts()
    
    print("\nObjective 2 (Contaminant) - Class Balance:")
    for label in [0, 1]: 
        count = contaminant_counts.get(label, 0)
        percentage = count / len(df_labels) * 100
        print(f"  Label {label}: {count} samples ({percentage:.1f}%)")
    
    if len(contaminant_counts) == 2:
        clean_count = contaminant_counts. get(0, 0)
        contam_count = contaminant_counts.get(1, 0)
        if contam_count > 0:
            imbalance_ratio = clean_count / contam_count
            
            if imbalance_ratio > 3.0 or imbalance_ratio < 0.33:
                print(f"\n⚠️ WARNING:  Significant class imbalance (ratio: {imbalance_ratio:.2f})")
                print("   Consider using class weights or SMOTE for training.")
            else:
                print(f"\n✓ Acceptable class balance (ratio: {imbalance_ratio:.2f})")
    
    # Check for missing labels
    missing_nutrient = len(df_labels[df_labels['Nutrient_Label'] == -1])
    if missing_nutrient > 0:
        print(f"\n✓ {missing_nutrient} samples excluded from nutrient classification (contaminated)")
    
    print("="*70)


def create_stratified_splits(
    df_labels: pd.DataFrame,
    test_size: float = 0.30,
    random_state: int = 42
) -> dict:
    """
    Create stratified train-test splits for both objectives.
    
    Parameters
    ----------
    df_labels : pd.DataFrame
        Labeled dataset
    test_size : float
        Proportion for test set (default 0.30 = 30%)
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    splits : dict
        Dictionary containing train/test indices for both objectives
    """
    from sklearn.model_selection import StratifiedShuffleSplit
    
    print("\n" + "="*70)
    print("STRATIFIED TRAIN-TEST SPLIT")
    print("="*70)
    
    splits = {}
    
    # Objective 1: Nutrient Classification (exclude -1 labels)
    df_nutrient = df_labels[df_labels['Nutrient_Label'] != -1]. copy()
    
    print("\nObjective 1: Nutrient Classification")
    print(f"Total samples: {len(df_nutrient)} (excluded {len(df_labels) - len(df_nutrient)} contaminated)")
    
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for train_idx, test_idx in splitter.split(df_nutrient, df_nutrient['Nutrient_Label']):
        splits['nutrient_train'] = df_nutrient.iloc[train_idx]
        splits['nutrient_test'] = df_nutrient.iloc[test_idx]
    
    print(f"  Training:  {len(splits['nutrient_train'])} samples ({100*(1-test_size):.0f}%)")
    print(f"  Testing:   {len(splits['nutrient_test'])} samples ({100*test_size:.0f}%)")
    
    # Class distribution
    train_dist = splits['nutrient_train']['Nutrient_Label'].value_counts().sort_index()
    test_dist = splits['nutrient_test']['Nutrient_Label'].value_counts().sort_index()
    
    print("\n  Train distribution:", dict(train_dist))
    print("  Test distribution:  ", dict(test_dist))
    
    # Objective 2: Contaminant Detection (use all samples)
    print("\nObjective 2: Contaminant Detection")
    print(f"Total samples: {len(df_labels)}")
    
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for train_idx, test_idx in splitter.split(df_labels, df_labels['Contaminant_Label']):
        splits['contaminant_train'] = df_labels.iloc[train_idx]
        splits['contaminant_test'] = df_labels.iloc[test_idx]
    
    print(f"  Training: {len(splits['contaminant_train'])} samples ({100*(1-test_size):.0f}%)")
    print(f"  Testing:  {len(splits['contaminant_test'])} samples ({100*test_size:.0f}%)")
    
    # Class distribution
    train_dist = splits['contaminant_train']['Contaminant_Label'].value_counts().sort_index()
    test_dist = splits['contaminant_test']['Contaminant_Label'].value_counts().sort_index()
    
    print("\n  Train distribution:", dict(train_dist))
    print("  Test distribution: ", dict(test_dist))
    
    print("="*70)
    
    return splits


if __name__ == "__main__":
    # Configuration
    INPUT_CSV = r"D:\Ryan\UM\Academic\Y4S1\KIE4002 FINAL YEAR PROJECT\Irish_Soil_DataSet\Irish_Soil_DataSet2.csv"
    OUTPUT_CSV = "irish_soil_labels.csv"
    
    print("\n" + "="*70)
    print("IRISH SOIL DATASET LABEL GENERATION SCRIPT")
    print("Context Feeder Methodology:  Phase 1 - Data Loading & Stratified Split")
    print("="*70)
    
    try:
        # Generate labels
        df_labels = generate_labels_from_csv(INPUT_CSV, OUTPUT_CSV)
        
        # Validate labels
        validate_labels(df_labels)
        
        # Demonstrate stratified splitting
        splits = create_stratified_splits(df_labels, test_size=0.30, random_state=42)
        
        # Save split indices (optional)
        print("\n✓ Label generation complete!")
        print(f"\nGenerated files:")
        print(f"  1. {OUTPUT_CSV} - Main label file")
        print(f"\nNext steps:")
        print(f"  1. Load labels:  df_labels = pd.read_csv('{OUTPUT_CSV}')")
        print(f"  2. Merge with spectral data in preprocessing pipeline")
        print(f"  3. Use stratified splits for training/testing")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nTroubleshooting:")
        print("  1. Verify INPUT_CSV path is correct")
        print("  2. Check CSV has sample names in first row (column headers)")
        print("  3. Ensure CSV is accessible (not open in Excel)")