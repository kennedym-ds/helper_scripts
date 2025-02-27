import stumpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, SecondLocator, MinuteLocator
from scipy.stats import mode
from statsmodels.tsa.stattools import acf


def generate_etch_chamber_data(num_wafers=25, wafer_process_time_mean=100,
                              wafer_process_time_std=5,
                              step_up_time_mean=20, step_up_time_std=2,
                              step_down_time_mean=15, step_down_time_std=1.5,
                              load_position=10, process_position=50,
                              noise_level=0.5, disturbance_wafer_1=None,
                              disturbance_start_offset_1=0, disturbance_magnitude_1=0,
                              disturbance_wafer_2=None, disturbance_plateau=50,
                              disturbance_duration_2=0):
    """Generates etch chamber data with variable wafer processing times."""

    timestamps = []
    values = []
    wafer_labels = []
    current_time = 0

    for wafer in range(1, num_wafers + 1):
        # Sample processing times
        wafer_process_time = int(np.round(np.random.normal(wafer_process_time_mean, wafer_process_time_std)))
        step_up_time = int(np.round(np.random.normal(step_up_time_mean, step_up_time_std)))
        step_down_time = int(np.round(np.random.normal(step_down_time_mean, step_down_time_std)))

        wafer_process_time = max(1, wafer_process_time)
        step_up_time = max(1, step_up_time)
        step_down_time = max(1, step_down_time)

        # Step Up
        for i in range(step_up_time):
            values.append(load_position + (process_position - load_position) * (i / step_up_time))
            if disturbance_wafer_1 is not None and wafer - 1 == disturbance_wafer_1 and i >= disturbance_start_offset_1:
                values[-1] += disturbance_magnitude_1 * np.sin(2 * np.pi * (i - disturbance_start_offset_1) / (step_up_time - disturbance_start_offset_1))
            wafer_labels.append(wafer)
            timestamps.append(current_time)
            current_time += 1

        # Process Wafer
        for i in range(wafer_process_time):
            if disturbance_wafer_2 is not None and wafer - 1 == disturbance_wafer_2 and i < disturbance_duration_2:
                values.append(disturbance_plateau)
            else:
                values.append(process_position)
            wafer_labels.append(wafer)
            timestamps.append(current_time)
            current_time += 1

        # Step Down
        for i in range(step_down_time):
            values.append(process_position - (process_position - load_position) * (i / step_down_time))
            wafer_labels.append(wafer)
            timestamps.append(current_time)
            current_time += 1

    values = np.array(values) + np.random.normal(0, noise_level, len(values))
    df = pd.DataFrame({'seconds': timestamps, 'value': values, 'wafer': wafer_labels})
    df['timestamp'] = pd.to_datetime('2024-01-01') + pd.to_timedelta(df['seconds'], unit='s')
    return df

def plot_time_series_and_distances(df, distances, m, threshold, indices, ax=None):
    """Plots time series and matrix profile, highlighting anomalies."""
    if ax is None:
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        ax = axes.flatten()

    # --- Time Series Plot ---
    ax[0].plot(df['timestamp'], df['value'], linewidth=0.8, color='blue', label='Normal')

    # Find contiguous segments
    anomalous_segments = []
    if len(indices) > 0:
        start = indices[0]
        for i in range(len(indices) - 1):
            if indices[i+1] - indices[i] != 1:
                anomalous_segments.append((start, indices[i] + m))
                start = indices[i+1]
        anomalous_segments.append((start, indices[-1] + m))

    for start, end in anomalous_segments:
        end = min(end, len(df))
        ax[0].plot(df['timestamp'].iloc[start:end], df['value'].iloc[start:end], color='red', linewidth=1.2, label='Anomaly' if start == anomalous_segments[0][0] else "")

    ax[0].set_ylabel('Value')
    ax[0].set_xlabel('Time (Seconds)')
    ax[0].set_title('Etch Chamber Load Position - Time Series with Detected Anomalies')
    ax[0].xaxis.set_major_formatter(DateFormatter("%H:%M:%S"))
    ax[0].xaxis.set_major_locator(MinuteLocator(interval=10))
    ax[0].xaxis.set_minor_locator(MinuteLocator(interval=1))
    ax[0].set_xlim(df['timestamp'].iloc[0], df['timestamp'].iloc[-1])
    ax[0].set_ylim(9, 60)
    ax[0].legend()

    # --- Matrix Profile Plot ---
    ax[1].plot(df['timestamp'][m-1:], distances, linewidth=0.8, label='Matrix Profile (Long)')
    ax[1].axhline(y=threshold, color='r', linestyle='--', label=f'Threshold = {threshold:.2f}')
    anomaly_timestamps = df['timestamp'][m-1:].iloc[indices]
    ax[1].scatter(anomaly_timestamps, distances[indices], color='red', s=60, zorder=5, label='Detected Anomaly Start')

    ax[1].set_ylabel('Distance')
    ax[1].set_xlabel('Time (Seconds)')
    ax[1].set_title('Matrix Profile (Long Subsequences)')
    ax[1].xaxis.set_major_formatter(DateFormatter("%H:%M:%S"))
    ax[1].xaxis.set_major_locator(MinuteLocator(interval=10))
    ax[1].xaxis.set_minor_locator(MinuteLocator(interval=1))
    ax[1].set_xlim(df['timestamp'].iloc[0], df['timestamp'].iloc[-1])
    ax[1].legend()

    plt.tight_layout()
    return ax

def smooth_distances(distances, window_size=5):
    """Applies a moving average filter to smooth the distances."""
    if window_size % 2 == 0:
        window_size += 1
    if window_size > len(distances):
        print("Warning: Smoothing window size is larger than distances array. Reducing window size.")
        window_size = len(distances)
        if window_size % 2 == 0:
            window_size -= 1
    if window_size <= 1:
        return distances

    weights = np.ones(window_size) / window_size
    smoothed_distances = np.convolve(distances, weights, mode='valid')
    pad_size = (len(distances) - len(smoothed_distances)) // 2
    smoothed_distances = np.pad(smoothed_distances, (pad_size, len(distances) - len(smoothed_distances) - pad_size), mode='edge')
    return smoothed_distances


def detect_anomalies_with_threshold(df, m, num_std_devs=3, smoothing_window=5):
    """Detects anomalies with thresholding, after smoothing."""
    distances = stumpy.stump(df['value'], m=m)
    distances_subset = distances[:, 0]

    if distances_subset.size <= 1 or np.all(distances_subset == distances_subset[0]):
        print("Warning: Insufficient data or all distances identical. Returning no anomalies.")
        return distances_subset, np.array([], dtype=int), 0

    distances_subset = distances_subset.astype(np.float64)
    smoothed_distances = smooth_distances(distances_subset, window_size=smoothing_window)

    distances_mean = np.mean(smoothed_distances)
    distances_std = np.std(smoothed_distances, ddof=1)
    threshold = distances_mean + num_std_devs * distances_std

    print(f"distances std: {distances_std}")
    print(f"distances mean: {distances_mean}")
    print(f"Threshold: {threshold}")

    indices = np.where(smoothed_distances > threshold)[0]
    indices = np.sort(indices)
    return smoothed_distances, indices, threshold

def get_anomalous_wafers(df, indices, m):
    """Identifies the primary wafer for each anomaly using the mode."""
    anomalous_wafers = set()
    if len(indices) == 0:
        return []

    subsequences = []
    current_subsequence = [indices[0]]
    for i in range(1, len(indices)):
        if indices[i] == indices[i-1] + 1:
            current_subsequence.append(indices[i])
        else:
            subsequences.append(current_subsequence)
            current_subsequence = [indices[i]]
    subsequences.append(current_subsequence)

    for subsequence in subsequences:
        start_index = subsequence[0]
        end_index = min(subsequence[-1] + m, len(df))
        wafer_labels_in_subsequence = df['wafer'].iloc[start_index:end_index].to_numpy()
        most_frequent_wafer = mode(wafer_labels_in_subsequence, keepdims=False)[0]
        anomalous_wafers.add(most_frequent_wafer)

    return sorted(list(anomalous_wafers))

def infer_wafer_period(df):
    """Infers the wafer processing period (m) using autocorrelation and wafer labels."""
    autocorr = acf(df['value'], nlags=len(df) - 1, fft=True)
    threshold = 2 / np.sqrt(len(df))
    peaks = [i for i in range(1, len(autocorr)) if autocorr[i] > threshold]

    if not peaks:
        print("Warning: No significant autocorrelation peak. Using median wafer count.")
        initial_period = int(np.median(df['wafer'].value_counts().to_numpy()))
    else:
        initial_period = peaks[0]

    wafer_transitions = np.where(df['wafer'].diff() != 0)[0]
    if len(wafer_transitions) < 2:
        print("Warning: Not enough wafers to refine. Using autocorrelation method.")
        return initial_period

    periods = np.diff(wafer_transitions)
    period = int(np.median(periods))

    return period if period > 0 else initial_period

def main():
    # Parameters
    num_wafers = 25
    wafer_process_time_mean = 100
    wafer_process_time_std = 5
    step_up_time_mean = 20
    step_up_time_std = 2
    step_down_time_mean = 15
    step_down_time_std = 1.5
    noise_level = 0.5
    disturbance_wafer_1 = 12
    disturbance_start_offset_1 = 0
    disturbance_magnitude_1 = 10
    disturbance_wafer_2 = 18
    disturbance_plateau = 55
    disturbance_duration_2 = 30

    # Generate data
    df = generate_etch_chamber_data(num_wafers=num_wafers,
                                    wafer_process_time_mean=wafer_process_time_mean,
                                    wafer_process_time_std=wafer_process_time_std,
                                    step_up_time_mean=step_up_time_mean,
                                    step_up_time_std=step_up_time_std,
                                    step_down_time_mean=step_down_time_mean,
                                    step_down_time_std=step_down_time_std,
                                    noise_level=noise_level,
                                    disturbance_wafer_1=disturbance_wafer_1,
                                    disturbance_start_offset_1=disturbance_start_offset_1,
                                    disturbance_magnitude_1=disturbance_magnitude_1,
                                    disturbance_wafer_2=disturbance_wafer_2,
                                    disturbance_plateau=disturbance_plateau,
                                    disturbance_duration_2=disturbance_duration_2)

    # Infer period
    m = infer_wafer_period(df)
    print(f"Inferred Wafer Period (m): {m}")

    # Anomaly Detection
    num_std_devs = 2
    smoothing_window = 25
    distances, indices, threshold = detect_anomalies_with_threshold(df, m, num_std_devs, smoothing_window)
    anomalous_wafers = get_anomalous_wafers(df, indices, m)

    # Visualization
    plot_time_series_and_distances(df, distances, m, threshold, indices)
    plt.show()

    # Output
    if anomalous_wafers:
        print(f"Anomalous wafers detected: {', '.join(map(str, anomalous_wafers))}")
        for wafer in anomalous_wafers:
            print(f"Data for Wafer {wafer}:")
            print(df[df['wafer'] == wafer])
    else:
        print("No anomalies detected.")

if __name__ == "__main__":
    main()