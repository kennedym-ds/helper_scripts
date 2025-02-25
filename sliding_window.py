import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
import timeit
import matplotlib.pyplot as plt

def calculate_window_stats_optimized(window, previous_stats=None, leaving_value=None, entering_value=None):
    """
    Calculates summary statistics for a window, optimized with incremental updates,
    epsilon for stability, sample skewness/kurtosis, and skipping redundant calculations.
    """
    stats = {}
    window_size = len(window)
    epsilon = 1e-9  # Small epsilon to avoid division by zero

    if previous_stats is not None and leaving_value is not None and entering_value is not None:
        # Optimization: If leaving and entering values are the same, stats don't change
        if leaving_value == entering_value:
            return previous_stats  # Return previous stats directly - no change

        # Incremental updates for S1, S2, S3, S4
        prev_S1 = previous_stats.get('S1', np.sum(window))
        prev_S2 = previous_stats.get('S2', np.sum(window**2))
        prev_S3 = previous_stats.get('S3', np.sum(window**3))
        prev_S4 = previous_stats.get('S4', np.sum(window**4))

        S1 = prev_S1 - leaving_value + entering_value
        S2 = prev_S2 - (leaving_value**2) + (entering_value**2)
        S3 = prev_S3 - (leaving_value**3) + (entering_value**3)
        S4 = prev_S4 - (leaving_value**4) + (entering_value**4)

        stats['S1'] = S1
        stats['S2'] = S2
        stats['S3'] = S3
        stats['S4'] = S4

        mean = S1 / window_size
        stats['mean'] = mean

        if window_size > 1:
            variance = (S2 - (S1**2) / window_size) / (window_size - 1) # Sample variance
            stats['stdev'] = np.sqrt(variance)
            m2 = variance # Using sample variance directly as m2 for sample skew/kurtosis

            m3 = (S3 - 3*mean*S2 + 2*(mean**3)*window_size) / window_size # Adjusted m3 calculation (central moment approx)
            m4 = (S4 - 4*mean*S3 + 6*(mean**2)*S2 - 3*(mean**4)*window_size) / window_size # Adjusted m4

            if m2 > epsilon: # Using epsilon here for variance check
                # Sample Skewness
                if window_size >= 3:
                    sample_skewness = (np.sqrt(window_size * (window_size - 1)) / (window_size - 2)) * (m3 / (m2**(1.5) + epsilon)) # Epsilon in denominator
                    stats['skew'] = sample_skewness
                else:
                    stats['skew'] = 0.0 # Undefined for window_size < 3, set to 0

                # Sample Kurtosis (Excess)
                if window_size >= 4:
                    sample_kurtosis =  ((window_size*(window_size+1)) / ((window_size-1)*(window_size-2)*(window_size-3))) * (m4 / (m2**2 + epsilon)) # Epsilon in denominator
                    sample_kurtosis -= (3*(window_size-1)**2) / ((window_size-2)*(window_size-3))
                    stats['kurtosis'] = sample_kurtosis
                else:
                    stats['kurtosis'] = 0.0 # Undefined for window_size < 4, set to 0
            else:
                stats['skew'] = 0.0
                stats['kurtosis'] = 0.0
        else: # Window size 1 case
            stats['stdev'] = 0.0
            stats['skew'] = 0.0
            stats['kurtosis'] = 0.0

        stats['median'] = np.median(window) # Recalculate median and percentiles
        stats['p25'] = np.percentile(window, 25)
        stats['p75'] = np.percentile(window, 75)


    else: # First window or fallback
        stats['S1'] = np.sum(window)
        stats['S2'] = np.sum(window**2)
        stats['S3'] = np.sum(window**3)
        stats['S4'] = np.sum(window**4)
        stats['mean'] = np.mean(window)
        stats['stdev'] = np.std(window)
        stats['skew'] = skew(window)
        stats['kurtosis'] = kurtosis(window)
        stats['median'] = np.median(window)
        stats['p25'] = np.percentile(window, 25)
        stats['p75'] = np.percentile(window, 75)

    return stats

def compare_windows(stats1, stats2, thresholds):
    """Compares summary statistics of two windows and checks against thresholds."""
    anomalous = False
    reasons = []
    for stat_name, threshold in thresholds.items():
        diff = abs(stats2[stat_name] - stats1[stat_name]) # Absolute difference
        # Or use relative difference: diff = abs((stats2[stat_name] - stats1[stat_name]) / (stats1[stat_name] + 1e-9)) # Avoid div by zero

        if diff > threshold:
            anomalous = True
            reasons.append(f"{stat_name} difference ({diff:.2f}) exceeds threshold ({threshold})")
    return anomalous, reasons

def sliding_window_anomaly_detection_optimized(time_series, window_size, step_size, thresholds):
    """
    Detects anomalies using sliding windows with optimized statistic calculation.
    """
    anomalies = []
    stats_history = []
    previous_stats = None # Initialize for the first window

    for i in range(0, len(time_series) - window_size + 1, step_size):
        window = time_series[i : i + window_size]

        if i > 0 and step_size == 1: # Incremental update is most effective for step_size=1 (overlapping windows)
            leaving_value = time_series[i - 1] # Value leaving is the point before the start of current window
            entering_value = time_series[i + window_size - 1] # Value entering is the last point of current window
            current_stats = calculate_window_stats_optimized(window, previous_stats, leaving_value, entering_value)
        else: # First window or if step_size > 1 (recalculate for simplicity)
            current_stats = calculate_window_stats_optimized(window)

        if current_stats is not previous_stats: # Only append if stats were actually recalculated (optimization in calculate_window_stats_optimized)
            stats_history.append(current_stats)
            previous_stats = current_stats # Update previous_stats for next iteration

        if i > 0 and current_stats is not previous_stats: # Compare only if stats were recalculated and not the first window
            previous_stats_for_comparison = stats_history[-2] # Access the stats of the *previous* window
            is_anomalous, reasons = compare_windows(previous_stats_for_comparison, current_stats, thresholds)

            if is_anomalous:
                anomaly_index = i + window_size - 1
                anomalies.append({
                    'index': anomaly_index,
                    'value': time_series[anomaly_index],
                    'reasons': ", ".join(reasons)
                })

    return pd.DataFrame(anomalies)


def sliding_window_anomaly_detection_baseline(time_series, window_size, step_size, thresholds):
    """
    Baseline anomaly detection function without optimized stats calculation (for performance comparison).
    Uses the simpler calculate_window_stats function (using scipy.stats directly each time).
    """
    anomalies = []
    stats_history = []

    for i in range(0, len(time_series) - window_size + 1, step_size):
        window = time_series[i : i + window_size]
        current_stats = calculate_window_stats_baseline(window) # Using baseline stats calculation
        stats_history.append(current_stats)

        if i > 0:
            previous_stats = stats_history[-2]
            is_anomalous, reasons = compare_windows(previous_stats, current_stats, thresholds)

            if is_anomalous:
                anomaly_index = i + window_size - 1
                anomalies.append({
                    'index': anomaly_index,
                    'value': time_series[anomaly_index],
                    'reasons': ", ".join(reasons)
                })

    return pd.DataFrame(anomalies)


def calculate_window_stats_baseline(window):
    """Baseline function to calculate window statistics directly using scipy.stats (no optimization)."""
    stats = {
        'mean': np.mean(window),
        'stdev': np.std(window),
        'skew': skew(window),
        'kurtosis': kurtosis(window),
        'median': np.median(window),
        'p25': np.percentile(window, 25),
        'p75': np.percentile(window, 75)
    }
    return stats


if __name__ == "__main__":
    # 1. Generate sample time series data
    np.random.seed(42)
    data = np.sin(np.linspace(0, 20*np.pi, 10000)) + np.random.normal(0, 0.1, 10000)
    data[2000:2050] += 1.5  # Mean shift anomaly
    data[6000:6100] *= 2    # Variance increase anomaly
    time_series_data = pd.Series(data)

    # 2. Define parameters
    window_size = 50
    step_size = 1
    thresholds = {
        'mean': 0.5,
        'stdev': 0.3,
        'skew': 0.2,
        'kurtosis': 1.0
    }

    # 3. Run anomaly detection - Optimized version
    anomaly_results_optimized = sliding_window_anomaly_detection_optimized(time_series_data, window_size, step_size, thresholds)
    print("Optimized Anomaly Results:")
    print(anomaly_results_optimized)

    # 4. Run anomaly detection - Baseline version (for performance comparison)
    anomaly_results_baseline = sliding_window_anomaly_detection_baseline(time_series_data, window_size, step_size, thresholds)


    # 5. Performance comparison using timeit
    n_iterations = 10
    optimized_time = timeit.timeit(lambda: sliding_window_anomaly_detection_optimized(time_series_data, window_size, step_size, thresholds), number=n_iterations)
    baseline_time = timeit.timeit(lambda: sliding_window_anomaly_detection_baseline(time_series_data, window_size, step_size, thresholds), number=n_iterations)

    print(f"\nPerformance Comparison (averaged over {n_iterations} iterations):")
    print(f"Baseline function time: {baseline_time/n_iterations:.4f} seconds per iteration")
    print(f"Optimized function time: {optimized_time/n_iterations:.4f} seconds per iteration")
    speedup = baseline_time / optimized_time
    print(f"Speedup: {speedup:.2f}x")


    # 6. Visualization of results
    plt.figure(figsize=(14, 7))
    plt.plot(time_series_data, label='Time Series Data')

    if not anomaly_results_optimized.empty:
        anomaly_indices_optimized = anomaly_results_optimized['index'].tolist()
        anomaly_values_optimized = time_series_data[anomaly_indices_optimized].tolist()
        plt.scatter(anomaly_indices_optimized, anomaly_values_optimized, color='red', label='Optimized Anomalies', s=100)

    if not anomaly_results_baseline.empty: # Optionally plot baseline anomalies too for comparison
        anomaly_indices_baseline = anomaly_results_baseline['index'].tolist()
        anomaly_values_baseline = time_series_data[anomaly_indices_baseline].tolist()
        plt.scatter(anomaly_indices_baseline, anomaly_values_baseline, color='orange', label='Baseline Anomalies', s=60, marker='x')


    plt.title('Time Series Anomaly Detection with Optimized Sliding Windows')
    plt.xlabel('Time Point')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()