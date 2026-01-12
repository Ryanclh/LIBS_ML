"""
Noise Reduction Filter Comparison for LIBS Spectral Data
Customized for:  320 samples × 12,981 features (197.001-973.5891 nm)
Purpose: Test filters on YOUR actual LIBS dataset before pipeline integration
Date: 2026-01-12

CSV Format Expected:
- First row: Sample names (320 columns)
- First column: Wavelength values (12,981 rows, 197.001-973.5891 nm)
- Data: Spectral intensity values
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import savgol_filter, wiener
import pywt
import time
from sklearn.metrics import mean_squared_error
import seaborn as sns
from pathlib import Path

# ============================================================================
# FILTER IMPLEMENTATIONS (Same as before)
# ============================================================================

class NoiseFilterComparator:
    def __init__(self, spectrum, wavelengths=None):
        """
        Initialize comparator with a single LIBS spectrum
        
        Parameters:  
        -----------
        spectrum : array-like, shape (n_features,)
            Raw LIBS spectrum (12,981 points)
        wavelengths : array-like, optional
            Wavelength values corresponding to spectrum points
        """
        self.raw_spectrum = np.array(spectrum)
        self.wavelengths = wavelengths if wavelengths is not None else np.arange(len(spectrum))
        self.filtered_spectra = {}
        self.metrics = {}
        self.computation_times = {}
        
    def apply_savitzky_golay(self, window_length=11, polyorder=3):
        """Standard Savitzky-Golay filter (LIBS baseline)"""
        start_time = time.time()
        filtered = savgol_filter(self.raw_spectrum, window_length, polyorder)
        elapsed = time.time() - start_time
        
        self.filtered_spectra['Savitzky-Golay'] = filtered
        self.computation_times['Savitzky-Golay'] = elapsed
        return filtered
    
    def apply_modified_sinc(self, cutoff_freq=0.1, num_taps=51):
        """Modified Sinc Filter (Windowed Sinc FIR filter)"""
        start_time = time.time()
        fir_coeff = signal.firwin(num_taps, cutoff_freq, window='hamming')
        filtered = signal.filtfilt(fir_coeff, 1.0, self.raw_spectrum)
        elapsed = time.time() - start_time
        
        self.filtered_spectra['Modified Sinc'] = filtered
        self.computation_times['Modified Sinc'] = elapsed
        return filtered
    
    def apply_gaussian(self, sigma=2.0):
        """Gaussian smoothing filter"""
        start_time = time.time()
        filtered = gaussian_filter1d(self.raw_spectrum, sigma)
        elapsed = time.time() - start_time
        
        self.filtered_spectra['Gaussian'] = filtered
        self.computation_times['Gaussian'] = elapsed
        return filtered
    
    def apply_median(self, kernel_size=5):
        """Median filter (edge-preserving, non-linear)"""
        start_time = time.time()
        filtered = median_filter(self.raw_spectrum, size=kernel_size)
        elapsed = time.time() - start_time
        
        self.filtered_spectra['Median'] = filtered
        self.computation_times['Median'] = elapsed
        return filtered
    
    def apply_wiener(self, noise_variance=None):
        """Wiener filter (adaptive noise reduction)"""
        start_time = time.time()
        filtered = wiener(self.raw_spectrum, mysize=9, noise=noise_variance)
        elapsed = time.time() - start_time
        
        self.filtered_spectra['Wiener'] = filtered
        self.computation_times['Wiener'] = elapsed
        return filtered
    
    def apply_moving_average(self, window_size=11):
        """Simple moving average (baseline comparison)"""
        start_time = time.time()
        kernel = np.ones(window_size) / window_size
        filtered = np.convolve(self. raw_spectrum, kernel, mode='same')
        elapsed = time. time() - start_time
        
        self.filtered_spectra['Moving Average'] = filtered
        self.computation_times['Moving Average'] = elapsed
        return filtered
    
    def apply_wavelet_denoising(self, wavelet='db4', level=3):
        """Wavelet denoising (multi-resolution)"""
        start_time = time.time()
        
        coeffs = pywt.wavedec(self.raw_spectrum, wavelet, level=level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np. log(len(self.raw_spectrum)))
        coeffs_thresh = [coeffs[0]] + [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
        filtered = pywt.waverec(coeffs_thresh, wavelet)
        
        if len(filtered) > len(self.raw_spectrum):
            filtered = filtered[:len(self.raw_spectrum)]
        elif len(filtered) < len(self.raw_spectrum):
            filtered = np.pad(filtered, (0, len(self.raw_spectrum) - len(filtered)), mode='edge')
        
        elapsed = time.time() - start_time
        
        self.filtered_spectra['Wavelet'] = filtered
        self.computation_times['Wavelet'] = elapsed
        return filtered
    
    # ========================================================================
    # EVALUATION METRICS
    # ========================================================================
    
    def calculate_snr(self, filtered_spectrum):
        """Calculate Signal-to-Noise Ratio"""
        noise = self.raw_spectrum - filtered_spectrum
        signal_power = np.mean(filtered_spectrum ** 2)
        noise_power = np.mean(noise ** 2)
        
        if noise_power == 0:
            return np.inf
        
        snr = 10 * np.log10(signal_power / noise_power)
        return snr
    
    def calculate_peak_preservation(self, filtered_spectrum, prominence=0.05):
        """Evaluate how well spectral peaks are preserved"""
        raw_peaks, raw_properties = signal.find_peaks(self.raw_spectrum, prominence=prominence)
        filt_peaks, filt_properties = signal.find_peaks(filtered_spectrum, prominence=prominence)
        
        peak_count_ratio = len(filt_peaks) / len(raw_peaks) if len(raw_peaks) > 0 else 0
        
        if len(raw_peaks) > 0 and len(filt_peaks) > 0:
            raw_heights = self.raw_spectrum[raw_peaks]
            filt_heights = filtered_spectrum[filt_peaks]
            
            height_correlations = []
            position_deviations = []
            
            for i, raw_pos in enumerate(raw_peaks):
                distances = np.abs(filt_peaks - raw_pos)
                if len(distances) > 0:
                    nearest_idx = np.argmin(distances)
                    position_deviations.append(distances[nearest_idx])
                    height_correlations.append(filt_heights[nearest_idx] / raw_heights[i])
            
            avg_height_preservation = np.mean(height_correlations) if height_correlations else 0
            avg_position_deviation = np.mean(position_deviations) if position_deviations else 0
        else:
            avg_height_preservation = 0
            avg_position_deviation = 0
        
        return {
            'peak_count_ratio': peak_count_ratio,
            'height_preservation': avg_height_preservation,
            'position_deviation':  avg_position_deviation
        }
    
    def calculate_rmse(self, filtered_spectrum):
        """Root Mean Square Error vs raw spectrum"""
        return np.sqrt(mean_squared_error(self.raw_spectrum, filtered_spectrum))
    
    def calculate_smoothness(self, filtered_spectrum):
        """Measure smoothness using second derivative"""
        second_derivative = np.diff(filtered_spectrum, n=2)
        smoothness = -np.mean(np.abs(second_derivative))
        return smoothness
    
    def evaluate_all_filters(self):
        """Apply all filters and calculate comprehensive metrics"""
        self.apply_savitzky_golay(window_length=11, polyorder=3)
        self.apply_modified_sinc(cutoff_freq=0.1, num_taps=51)
        self.apply_gaussian(sigma=2.0)
        self.apply_median(kernel_size=5)
        self.apply_wiener(noise_variance=None)
        self.apply_moving_average(window_size=11)
        self.apply_wavelet_denoising(wavelet='db4', level=3)
        
        results = []
        
        for filter_name, filtered_spectrum in self.filtered_spectra.items():
            snr = self.calculate_snr(filtered_spectrum)
            peak_metrics = self.calculate_peak_preservation(filtered_spectrum)
            rmse = self.calculate_rmse(filtered_spectrum)
            smoothness = self.calculate_smoothness(filtered_spectrum)
            comp_time = self.computation_times[filter_name]
            
            results.append({
                'Filter': filter_name,
                'SNR (dB)': snr,
                'Peak Count Ratio': peak_metrics['peak_count_ratio'],
                'Height Preservation': peak_metrics['height_preservation'],
                'Position Deviation': peak_metrics['position_deviation'],
                'RMSE': rmse,
                'Smoothness': smoothness,
                'Time (ms)': comp_time * 1000
            })
            
            self.metrics[filter_name] = {
                'snr': snr,
                'peak_metrics': peak_metrics,
                'rmse': rmse,
                'smoothness': smoothness,
                'time_ms': comp_time * 1000
            }
        
        self.results_df = pd.DataFrame(results)
        return self.results_df
    
    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    
    def plot_comparison(self, zoom_range=None, save_path=None):
        """Visualize all filtered spectra vs raw"""
        num_filters = len(self.filtered_spectra)
        fig, axes = plt.subplots(num_filters + 1, 1, figsize=(14, 3 * (num_filters + 1)))
        
        if zoom_range:
            start, end = zoom_range
            x_data = self.wavelengths[start:end]
            y_raw = self.raw_spectrum[start:end]
        else:
            x_data = self.wavelengths
            y_raw = self.raw_spectrum
        
        axes[0].plot(x_data, y_raw, 'k-', alpha=0.7, linewidth=0.8, label='Raw Spectrum')
        axes[0].set_title('Raw LIBS Spectrum', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Intensity (a.u.)')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        colors = plt.cm.Set2(np.linspace(0, 1, num_filters))
        
        for idx, (filter_name, filtered) in enumerate(self.filtered_spectra.items()):
            if zoom_range:
                y_filt = filtered[start:end]
            else:
                y_filt = filtered
            
            axes[idx + 1].plot(x_data, y_raw, 'k-', alpha=0.3, linewidth=0.5, label='Raw')
            axes[idx + 1].plot(x_data, y_filt, color=colors[idx], linewidth=1.2, label=filter_name)
            
            metrics = self.metrics[filter_name]
            text_str = (f"SNR: {metrics['snr']:.2f} dB | "
                       f"Peak Ratio: {metrics['peak_metrics']['peak_count_ratio']:.3f} | "
                       f"Time: {metrics['time_ms']:.3f} ms")
            
            axes[idx + 1].set_title(f"{filter_name} Filter", fontsize=11, fontweight='bold')
            axes[idx + 1].text(0.02, 0.95, text_str, transform=axes[idx + 1].transAxes,
                              fontsize=9, verticalalignment='top',
                              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            axes[idx + 1].set_ylabel('Intensity (a.u.)')
            axes[idx + 1].legend(loc='upper right')
            axes[idx + 1].grid(alpha=0.3)
        
        axes[-1].set_xlabel('Wavelength (nm)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Comparison plot saved to:  {save_path}")
        
        plt.show()
    
    def plot_metrics_comparison(self, save_path=None):
        """Bar chart comparison of key metrics"""
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()
        
        filters = list(self.filtered_spectra.keys())
        snr_values = [self.metrics[f]['snr'] for f in filters]
        peak_ratios = [self.metrics[f]['peak_metrics']['peak_count_ratio'] for f in filters]
        height_pres = [self.metrics[f]['peak_metrics']['height_preservation'] for f in filters]
        pos_dev = [self.metrics[f]['peak_metrics']['position_deviation'] for f in filters]
        rmse_values = [self.metrics[f]['rmse'] for f in filters]
        time_values = [self.metrics[f]['time_ms'] for f in filters]
        
        colors = plt.cm.Set2(np. linspace(0, 1, len(filters)))
        
        axes[0].bar(filters, snr_values, color=colors, edgecolor='black', linewidth=1.2)
        axes[0].set_title('Signal-to-Noise Ratio (Higher is Better)', fontweight='bold')
        axes[0].set_ylabel('SNR (dB)')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(axis='y', alpha=0.3)
        
        axes[1].bar(filters, peak_ratios, color=colors, edgecolor='black', linewidth=1.2)
        axes[1].axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Perfect Preservation')
        axes[1].set_title('Peak Count Preservation (Closer to 1.0 is Better)', fontweight='bold')
        axes[1].set_ylabel('Peak Count Ratio')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)
        
        axes[2].bar(filters, height_pres, color=colors, edgecolor='black', linewidth=1.2)
        axes[2].axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Perfect Preservation')
        axes[2].set_title('Peak Height Preservation (Closer to 1.0 is Better)', fontweight='bold')
        axes[2].set_ylabel('Avg Height Ratio')
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].legend()
        axes[2].grid(axis='y', alpha=0.3)
        
        axes[3].bar(filters, pos_dev, color=colors, edgecolor='black', linewidth=1.2)
        axes[3].set_title('Peak Position Deviation (Lower is Better)', fontweight='bold')
        axes[3].set_ylabel('Avg Position Shift (points)')
        axes[3].tick_params(axis='x', rotation=45)
        axes[3].grid(axis='y', alpha=0.3)
        
        axes[4].bar(filters, rmse_values, color=colors, edgecolor='black', linewidth=1.2)
        axes[4].set_title('RMSE vs Raw (Balance Needed)', fontweight='bold')
        axes[4].set_ylabel('RMSE')
        axes[4].tick_params(axis='x', rotation=45)
        axes[4].grid(axis='y', alpha=0.3)
        
        axes[5].bar(filters, time_values, color=colors, edgecolor='black', linewidth=1.2)
        axes[5].set_title('Computation Time (Lower is Better for MCU)', fontweight='bold')
        axes[5].set_ylabel('Time (ms)')
        axes[5].tick_params(axis='x', rotation=45)
        axes[5].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt. savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Metrics comparison saved to: {save_path}")
        
        plt.show()
    
    def generate_recommendation(self):
        """Generate automated recommendation based on metrics"""
        print("\n" + "=" * 70)
        print("RECOMMENDATION BASED ON COMPREHENSIVE ANALYSIS")
        print("=" * 70)
        
        filters = list(self.filtered_spectra.keys())
        
        snr_norm = np.array([self.metrics[f]['snr'] for f in filters])
        snr_norm = (snr_norm - snr_norm.min()) / (snr_norm.max() - snr_norm.min())
        
        peak_ratio_norm = np.array([self.metrics[f]['peak_metrics']['peak_count_ratio'] for f in filters])
        peak_ratio_norm = 1 - np.abs(peak_ratio_norm - 1)
        
        height_norm = np.array([self.metrics[f]['peak_metrics']['height_preservation'] for f in filters])
        height_norm = 1 - np.abs(height_norm - 1)
        
        pos_dev_norm = np.array([self.metrics[f]['peak_metrics']['position_deviation'] for f in filters])
        pos_dev_norm = 1 - (pos_dev_norm / pos_dev_norm.max() if pos_dev_norm.max() > 0 else pos_dev_norm)
        
        time_norm = np.array([self.metrics[f]['time_ms'] for f in filters])
        time_norm = 1 - (time_norm / time_norm.max() if time_norm.max() > 0 else time_norm)
        
        weights = {
            'snr': 0.25,
            'peak_ratio':  0.20,
            'height':  0.20,
            'position': 0.25,
            'time': 0.10
        }
        
        composite_scores = (
            weights['snr'] * snr_norm +
            weights['peak_ratio'] * peak_ratio_norm +
            weights['height'] * height_norm +
            weights['position'] * pos_dev_norm +
            weights['time'] * time_norm
        )
        
        ranking = np.argsort(composite_scores)[::-1]
        
        print("\nRANKING (Top to Bottom):")
        print("-" * 70)
        for rank, idx in enumerate(ranking, 1):
            filter_name = filters[idx]
            score = composite_scores[idx]
            print(f"{rank}.{filter_name: 20s} | Composite Score: {score:.4f}")
        
        best_filter = filters[ranking[0]]
        print("\n" + "=" * 70)
        print(f"🏆 RECOMMENDED FILTER: {best_filter}")
        print("=" * 70)
        
        print(f"\nJustification:")
        print(f"  • SNR: {self.metrics[best_filter]['snr']:.2f} dB")
        print(f"  • Peak Preservation: {self.metrics[best_filter]['peak_metrics']['peak_count_ratio']:.3f}")
        print(f"  • Position Deviation: {self.metrics[best_filter]['peak_metrics']['position_deviation']:.3f} points")
        print(f"  • Computation Time: {self.metrics[best_filter]['time_ms']:.3f} ms")
        
        return best_filter, composite_scores


# ============================================================================
# MULTI-SAMPLE AGGREGATOR
# ============================================================================

class MultiSampleFilterComparison:
    """Aggregate filter comparison across multiple LIBS spectra"""
    
    def __init__(self, spectra_df, wavelengths):
        """
        Parameters:
        -----------
        spectra_df : pd.DataFrame
            Shape (12981 features × 320 samples)
        wavelengths : array-like
            Wavelength values (12,981 points)
        """
        self. spectra_df = spectra_df
        self.wavelengths = wavelengths
        self. sample_results = []
        self.aggregated_results = None
        
    def run_comparison(self, num_samples=20, random_seed=42):
        """
        Run filter comparison on multiple samples
        
        Parameters: 
        -----------
        num_samples : int
            Number of random samples to test (default 20)
        random_seed :  int
            Random seed for reproducibility
        """
        np.random.seed(random_seed)
        
        # Randomly select samples
        all_sample_names = self.spectra_df. columns.tolist()
        selected_samples = np.random.choice(all_sample_names, 
                                           size=min(num_samples, len(all_sample_names)), 
                                           replace=False)
        
        print("=" * 70)
        print(f"MULTI-SAMPLE FILTER COMPARISON ({len(selected_samples)} samples)")
        print("=" * 70)
        
        for i, sample_name in enumerate(selected_samples, 1):
            print(f"\n[{i}/{len(selected_samples)}] Processing: {sample_name}")
            
            spectrum = self.spectra_df[sample_name].values
            
            comparator = NoiseFilterComparator(spectrum, self.wavelengths)
            results_df = comparator.evaluate_all_filters()
            
            # Add sample identifier
            results_df['Sample'] = sample_name
            self.sample_results.append(results_df)
            
            print(f"  ✓ Completed: Best filter = {results_df. iloc[results_df['SNR (dB)'].idxmax()]['Filter']}")
        
        # Aggregate results
        all_results = pd.concat(self.sample_results, ignore_index=True)
        
        # Calculate mean and std for each filter
        agg_metrics = all_results.groupby('Filter').agg({
            'SNR (dB)': ['mean', 'std'],
            'Peak Count Ratio': ['mean', 'std'],
            'Height Preservation': ['mean', 'std'],
            'Position Deviation': ['mean', 'std'],
            'RMSE':  ['mean', 'std'],
            'Smoothness': ['mean', 'std'],
            'Time (ms)': ['mean', 'std']
        }).round(4)
        
        self.aggregated_results = agg_metrics
        
        return self.aggregated_results
    
    def plot_aggregated_results(self, save_path=None):
        """Plot aggregated comparison across all samples"""
        if self. aggregated_results is None: 
            print("❌ Run comparison first using run_comparison()")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        metrics = ['SNR (dB)', 'Peak Count Ratio', 'Height Preservation', 
                   'Position Deviation', 'RMSE', 'Time (ms)']
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(self.aggregated_results)))
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            filters = self.aggregated_results.index. tolist()
            means = self.aggregated_results[(metric, 'mean')].values
            stds = self.aggregated_results[(metric, 'std')].values
            
            bars = ax.bar(filters, means, yerr=stds, color=colors, 
                         edgecolor='black', linewidth=1.5, capsize=5, alpha=0.8)
            
            # Highlight best performer
            if metric in ['SNR (dB)', 'Peak Count Ratio', 'Height Preservation']:
                best_idx = np.argmax(means)
            else:
                best_idx = np.argmin(means)
            
            bars[best_idx].set_edgecolor('red')
            bars[best_idx].set_linewidth(3)
            
            ax.set_title(f'{metric} (Mean ± Std)', fontweight='bold', fontsize=12)
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)
            
            if metric == 'Peak Count Ratio':
                ax. axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Aggregated comparison saved to: {save_path}")
        
        plt.show()
    
    def export_results(self, output_dir='filter_comparison_results'):
        """Export all results to CSV and summary report"""
        Path(output_dir).mkdir(exist_ok=True)
        
        # Individual sample results
        all_results = pd.concat(self.sample_results, ignore_index=True)
        all_results.to_csv(f'{output_dir}/individual_sample_results.csv', index=False)
        
        # Aggregated results
        self.aggregated_results.to_csv(f'{output_dir}/aggregated_results.csv')
        
        # Summary report
        with open(f'{output_dir}/summary_report.txt', 'w') as f:
            f. write("=" * 70 + "\n")
            f.write("NOISE FILTER COMPARISON SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Total samples tested: {len(self.sample_results)}\n")
            f.write(f"Wavelength range: {self.wavelengths[0]:.3f} - {self.wavelengths[-1]:.3f} nm\n")
            f.write(f"Features per spectrum: {len(self.wavelengths)}\n\n")
            f.write("AGGREGATED RESULTS:\n")
            f.write("-" * 70 + "\n")
            f.write(self.aggregated_results.to_string())
            f.write("\n\n")
            
            # Determine winner
            snr_means = self.aggregated_results[('SNR (dB)', 'mean')]
            best_filter = snr_means.idxmax()
            
            f.write("=" * 70 + "\n")
            f.write(f"🏆 RECOMMENDED FILTER:  {best_filter}\n")
            f.write("=" * 70 + "\n")
            f.write(f"Average SNR: {snr_means[best_filter]:.2f} ± {self.aggregated_results. loc[best_filter, ('SNR (dB)', 'std')]:.2f} dB\n")
            f.write(f"Average Time: {self.aggregated_results.loc[best_filter, ('Time (ms)', 'mean')]:.3f} ± {self.aggregated_results.loc[best_filter, ('Time (ms)', 'std')]:.3f} ms\n")
        
        print(f"\n✓ All results exported to: {output_dir}/")
        print(f"  • individual_sample_results.csv")
        print(f"  • aggregated_results.csv")
        print(f"  • summary_report.txt")


# ============================================================================
# MAIN EXECUTION:  LOAD YOUR CSV AND RUN COMPARISON
# ============================================================================

if __name__ == "__main__": 
    print("=" * 70)
    print("LIBS NOISE FILTER COMPARISON - LOADING YOUR DATASET")
    print("=" * 70)
    
    # ========================================================================
    # STEP 1: LOAD YOUR CSV FILE
    # ========================================================================
    
    # TODO: Update this path to your CSV file location
    csv_file_path = r"D:\Ryan\UM\Academic\Y4S1\KIE4002 FINAL YEAR PROJECT\Irish_Soil_DataSet\Irish_Soil_DataSet2.csv"  # ⚠️ CHANGE THIS TO YOUR FILE PATH
    
    print(f"\n📂 Loading CSV from: {csv_file_path}")
    
    try:
        # Read CSV with first column as wavelengths, first row as sample names
        df_raw = pd.read_csv(csv_file_path, index_col=0)
        
        # Extract wavelengths (first column values)
        wavelengths = df_raw.index.values. astype(float)
        
        # Extract sample names (column headers)
        sample_names = df_raw.columns.tolist()
        
        print(f"✓ Successfully loaded:")
        print(f"  • Wavelength range: {wavelengths[0]:.3f} - {wavelengths[-1]:.3f} nm")
        print(f"  • Number of features: {len(wavelengths)}")
        print(f"  • Number of samples: {len(sample_names)}")
        print(f"  • Expected:  12,981 features × 320 samples")
        
        # Validation check
        if len(wavelengths) != 12981:
            print(f"⚠️  WARNING: Expected 12,981 features but got {len(wavelengths)}")
        if len(sample_names) != 320:
            print(f"⚠️  WARNING: Expected 320 samples but got {len(sample_names)}")
        
        # Display first few samples
        print(f"\n📋 Sample names (first 5): {sample_names[:5]}")
        
    except FileNotFoundError:
        print(f"\n❌ ERROR: File not found at '{csv_file_path}'")
        print("Please update the 'csv_file_path' variable with your actual file path.")
        exit(1)
    except Exception as e:
        print(f"\n❌ ERROR loading CSV: {str(e)}")
        exit(1)
    
    # ========================================================================
    # STEP 2: RUN MULTI-SAMPLE COMPARISON
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("RUNNING FILTER COMPARISON ON MULTIPLE SAMPLES")
    print("=" * 70)
    
    # Initialize multi-sample comparator
    multi_comparator = MultiSampleFilterComparison(df_raw, wavelengths)
    
    # Run on 20 random samples (adjust num_samples as needed)
    aggregated_results = multi_comparator.run_comparison(num_samples=20, random_seed=42)
    
    # Display aggregated results
    print("\n" + "=" * 70)
    print("AGGREGATED RESULTS (Mean ± Std across all samples)")
    print("=" * 70)
    print(aggregated_results)
    
    # ========================================================================
    # STEP 3: VISUALIZE RESULTS
    # ========================================================================
    
    multi_comparator.plot_aggregated_results(save_path='aggregated_filter_comparison.png')
    
    # ========================================================================
    # STEP 4: DETAILED SINGLE-SAMPLE VISUALIZATION
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("GENERATING DETAILED PLOTS FOR REPRESENTATIVE SAMPLE")
    print("=" * 70)
    
    # Select one representative sample for detailed visualization
    representative_sample = sample_names[0]
    print(f"Using sample: {representative_sample}")
    
    spectrum = df_raw[representative_sample].values
    single_comparator = NoiseFilterComparator(spectrum, wavelengths)
    single_comparator.evaluate_all_filters()
    
    # Full spectrum comparison
    single_comparator.plot_comparison(save_path='filter_comparison_full_spectrum.png')
    
    # Zoomed view:  Phosphorus peaks (253-256 nm)
    zoom_start = np.argmin(np.abs(wavelengths - 250))
    zoom_end = np.argmin(np.abs(wavelengths - 260))
    single_comparator.plot_comparison(zoom_range=(zoom_start, zoom_end), 
                                     save_path='filter_comparison_P_peaks.png')
    
    # Zoomed view:  Potassium peaks (766-770 nm)
    zoom_start_K = np.argmin(np.abs(wavelengths - 765))
    zoom_end_K = np.argmin(np. abs(wavelengths - 772))
    single_comparator. plot_comparison(zoom_range=(zoom_start_K, zoom_end_K), 
                                     save_path='filter_comparison_K_peaks.png')
    
    # Metrics comparison
    single_comparator. plot_metrics_comparison(save_path='filter_metrics_barplot.png')
    
    # Generate recommendation
    best_filter, scores = single_comparator.generate_recommendation()
    
    # ========================================================================
    # STEP 5: EXPORT ALL RESULTS
    # ========================================================================
    
    multi_comparator.export_results(output_dir='filter_comparison_results')
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE - NEXT STEPS FOR YOUR FYP")
    print("=" * 70)
    print("\n📊 Generated files:")
    print("  1. aggregated_filter_comparison.png - Overall performance across all samples")
    print("  2. filter_comparison_full_spectrum.png - Detailed view of one sample")
    print("  3. filter_comparison_P_peaks.png - Zoomed P I 253.5nm region")
    print("  4. filter_comparison_K_peaks.png - Zoomed K I 766.5nm region")
    print("  5. filter_metrics_barplot.png - Quantitative metrics comparison")
    print("  6. filter_comparison_results/ - CSV exports and summary report")
    
    print("\n🎯 DECISION CRITERIA:")
    print("  1. If Modified Sinc SNR > SG SNR by >5%: Use Modified Sinc, cite Schmid 2022")
    print("  2. If SG and Modified Sinc similar (<2% diff): Use SG for MCU efficiency")
    print("  3. Check 'Position Deviation' - CRITICAL for peak identification accuracy")
    print("  4. Validate with end-to-end classification (Phase 3 of your pipeline)")
    
    print("\n⚠️  IMPORTANT:")
    print("  • Review the zoomed P and K peak plots - visual inspection matters!")
    print("  • Check if any filter introduces peak shifts (Position Deviation)")
    print("  • Consider computation time for MCU deployment (<5ms target)")
    print("  • Document your decision with quantitative justification in thesis")
    
    print("\n" + "=" * 70)