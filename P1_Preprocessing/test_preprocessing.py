"""
Unit Tests for LIBS Preprocessing Pipeline
==========================================
Run with:  pytest test_preprocessing.py -v
"""

import numpy as np
import pytest
from libs_preprocessing import (
    asls_baseline,
    apply_baseline_correction,
    apply_savgol_filter,
    apply_gaussian_filter,
    apply_l2_normalization,
    preprocess_libs_spectra,
    stratified_split
)


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def synthetic_spectrum():
    """Create a single synthetic LIBS spectrum for testing."""
    np.random.seed(42)
    n_features = 12981
    wavelengths = np.linspace(200, 950, n_features)
    
    # Baseline + peaks + noise
    baseline = 1000 * np.exp(-0.005 * (wavelengths - 200))
    peaks = 1500 * np.exp(-0.5 * ((wavelengths - 500) / 1)**2)
    noise = np.random.normal(0, 20, n_features)
    
    return baseline + peaks + noise


@pytest. fixture
def synthetic_dataset():
    """Create synthetic dataset for testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 12981
    
    spectra = np.random.rand(n_samples, n_features) * 1000 + 500
    labels = np.repeat(['High', 'Medium', 'Low'], 34)[: n_samples]
    
    return spectra, labels


# ============================================================================
# BASELINE CORRECTION TESTS
# ============================================================================

class TestBaselineCorrection:
    
    def test_asls_output_shape(self, synthetic_spectrum):
        """Baseline should have same shape as input."""
        baseline = asls_baseline(synthetic_spectrum)
        assert baseline.shape == synthetic_spectrum.shape
    
    def test_asls_baseline_below_spectrum(self, synthetic_spectrum):
        """Baseline should be mostly below the spectrum (for positive data)."""
        baseline = asls_baseline(synthetic_spectrum)
        # At least 90% of points should have baseline <= spectrum
        ratio = np.mean(baseline <= synthetic_spectrum)
        assert ratio > 0.9, f"Baseline above spectrum at {(1-ratio)*100:.1f}% of points"
    
    def test_baseline_correction_removes_trend(self, synthetic_spectrum):
        """Corrected spectrum should have flatter baseline region."""
        baseline = asls_baseline(synthetic_spectrum)
        corrected = synthetic_spectrum - baseline
        
        # Compare standard deviation in first 100 points (baseline region)
        std_before = np.std(synthetic_spectrum[: 100])
        std_after = np.std(corrected[:100])
        
        assert std_after < std_before, "Baseline correction should reduce trend"
    
    def test_apply_baseline_batch(self, synthetic_dataset):
        """Test batch baseline correction."""
        spectra, _ = synthetic_dataset
        corrected = apply_baseline_correction(spectra[: 10], verbose=False)
        
        assert corrected.shape == spectra[: 10].shape
        assert np. all(corrected >= 0), "Corrected spectra should be non-negative"


# ============================================================================
# NOISE REDUCTION TESTS
# ============================================================================

class TestNoiseReduction:
    
    def test_savgol_output_shape(self, synthetic_dataset):
        """Savitzky-Golay should preserve shape."""
        spectra, _ = synthetic_dataset
        smoothed = apply_savgol_filter(spectra[:10])
        assert smoothed.shape == spectra[: 10].shape
    
    def test_savgol_reduces_noise(self, synthetic_dataset):
        """Smoothing should reduce high-frequency noise."""
        spectra, _ = synthetic_dataset
        smoothed = apply_savgol_filter(spectra[:10])
        
        # Calculate noise as std of differences between adjacent points
        noise_before = np.mean(np.std(np.diff(spectra[:10], axis=1), axis=1))
        noise_after = np.mean(np.std(np.diff(smoothed, axis=1), axis=1))
        
        assert noise_after < noise_before, "Smoothing should reduce noise"
    
    def test_savgol_preserves_peaks(self, synthetic_spectrum):
        """Peak maximum should be mostly preserved."""
        from scipy.signal import savgol_filter
        
        smoothed = savgol_filter(synthetic_spectrum, 11, 3)
        
        peak_before = np.max(synthetic_spectrum)
        peak_after = np.max(smoothed)
        
        preservation = peak_after / peak_before
        assert preservation > 0.9, f"Peak preservation {preservation:.2%} < 90%"
    
    def test_gaussian_output_shape(self, synthetic_dataset):
        """Gaussian filter should preserve shape."""
        spectra, _ = synthetic_dataset
        smoothed = apply_gaussian_filter(spectra[:10])
        assert smoothed.shape == spectra[: 10].shape


# ============================================================================
# NORMALIZATION TESTS
# ============================================================================

class TestNormalization: 
    
    def test_l2_norm_is_unity(self, synthetic_dataset):
        """Each normalized spectrum should have L2 norm = 1."""
        spectra, _ = synthetic_dataset
        normalized = apply_l2_normalization(spectra[:10])
        
        norms = np.linalg. norm(normalized, axis=1)
        assert np.allclose(norms, 1.0), f"Norms should be 1, got {norms}"
    
    def test_l2_preserves_relative_intensities(self, synthetic_dataset):
        """Relative intensities should be preserved."""
        spectra, _ = synthetic_dataset
        normalized = apply_l2_normalization(spectra[:10])
        
        # Check that argmax is preserved
        for i in range(10):
            assert np.argmax(spectra[i]) == np.argmax(normalized[i])
    
    def test_l2_output_shape(self, synthetic_dataset):
        """Normalization should preserve shape."""
        spectra, _ = synthetic_dataset
        normalized = apply_l2_normalization(spectra[:10])
        assert normalized.shape == spectra[:10].shape


# ============================================================================
# STRATIFIED SPLIT TESTS
# ============================================================================

class TestStratifiedSplit:
    
    def test_split_sizes(self, synthetic_dataset):
        """Split should produce correct proportions."""
        spectra, labels = synthetic_dataset
        X_train, X_test, y_train, y_test = stratified_split(spectra, labels)
        
        total = len(labels)
        assert len(y_train) == pytest.approx(total * 0.7, abs=2)
        assert len(y_test) == pytest.approx(total * 0.3, abs=2)
    
    def test_stratification(self, synthetic_dataset):
        """Class proportions should be similar in train and test."""
        spectra, labels = synthetic_dataset
        X_train, X_test, y_train, y_test = stratified_split(spectra, labels)
        
        # Calculate class proportions
        train_props = {c: np.mean(y_train == c) for c in np.unique(labels)}
        test_props = {c: np.mean(y_test == c) for c in np.unique(labels)}
        
        for cls in np.unique(labels):
            diff = abs(train_props[cls] - test_props[cls])
            assert diff < 0.1, f"Class {cls} imbalance: {diff:.2%}"


# ============================================================================
# FULL PIPELINE TESTS
# ============================================================================

class TestFullPipeline: 
    
    def test_pipeline_output_keys(self, synthetic_dataset):
        """Pipeline should return all intermediate results."""
        spectra, _ = synthetic_dataset
        results = preprocess_libs_spectra(spectra[: 10], verbose=False)
        
        expected_keys = ['raw', 'baseline_corrected', 'smoothed', 'normalized']
        assert all(k in results for k in expected_keys)
    
    def test_pipeline_output_shapes(self, synthetic_dataset):
        """All outputs should have same shape as input."""
        spectra, _ = synthetic_dataset
        results = preprocess_libs_spectra(spectra[:10], verbose=False)
        
        for key, arr in results.items():
            assert arr.shape == spectra[:10].shape, f"{key} shape mismatch"
    
    def test_pipeline_final_normalized(self, synthetic_dataset):
        """Final output should be L2-normalized."""
        spectra, _ = synthetic_dataset
        results = preprocess_libs_spectra(spectra[:10], verbose=False)
        
        norms = np.linalg. norm(results['normalized'], axis=1)
        assert np.allclose(norms, 1.0)
    
    def test_pipeline_with_gaussian(self, synthetic_dataset):
        """Pipeline should work with Gaussian filter option."""
        spectra, _ = synthetic_dataset
        results = preprocess_libs_spectra(
            spectra[:10], 
            use_gaussian=True, 
            gaussian_sigma=1.5,
            verbose=False
        )
        
        assert 'smoothed' in results
        assert results['smoothed'].shape == spectra[:10].shape


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    
    def test_single_sample(self):
        """Pipeline should work with single sample."""
        single = np.random.rand(1, 1000) * 1000
        results = preprocess_libs_spectra(single, verbose=False)
        
        assert results['normalized'].shape == (1, 1000)
    
    def test_zeros_handling(self):
        """Pipeline should handle zero-valued spectra."""
        zeros = np.zeros((5, 1000))
        zeros[: , 500] = 1  # Add tiny signal to avoid div by zero
        
        results = preprocess_libs_spectra(zeros, verbose=False)
        assert not np.any(np.isnan(results['normalized']))
    
    def test_large_values(self):
        """Pipeline should handle large intensity values."""
        large = np. random.rand(5, 1000) * 1e6
        results = preprocess_libs_spectra(large, verbose=False)
        
        # Check normalized
        norms = np.linalg. norm(results['normalized'], axis=1)
        assert np.allclose(norms, 1.0)


if __name__ == "__main__": 
    pytest.main([__file__, "-v"])