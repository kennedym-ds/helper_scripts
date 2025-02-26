import stumpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from scipy.signal import find_peaks

def generate_complex_time_series_v2(length=5000, base_period=50, pattern_periods=None, noise_level=0.2, anomaly_start=None, anomaly_length=None, anomaly_type="spike"):
    """Generates a complex, multi-pattern time series."""
    timestamps = pd.to_datetime(np.arange(length), unit='D', origin=pd.Timestamp('2024-01-01'))
    values = np.zeros(length)

    if pattern_periods is None:
      pattern_periods = [(0, base_period, 5, 'sine')]

    for start, period, amplitude, func_type in pattern_periods:
        for i in range(start, length):
            if i >= start + period * (length // period):
                break;
            if func_type == 'sine':
                values[i] += amplitude * np.sin(2 * np.pi * (i - start) / period)
            elif func_type == 'sawtooth':
                values[i] += amplitude * ((i - start) % period / period)
            elif func_type == 'square':
                values[i] += amplitude * (1 if (i - start) % period < period / 2 else -1)
            else:
                raise ValueError("Invalid function_type.")

    if anomaly_start is not None and anomaly_length is not None:
        if not (0 <= anomaly_start < length and 0 <= anomaly_start + anomaly_length <= length):
            raise ValueError("Anomaly indices out of bounds.")

        if anomaly_type == 'spike':
            values[anomaly_start] += 15
        elif anomaly_type == 'level_shift':
            values[anomaly_start:anomaly_start + anomaly_length] += 7
        elif anomaly_type == 'pattern_change':
            for i in range(anomaly_start, anomaly_start + anomaly_length):
                values[i] = 3 * np.sin(2 * np.pi * i / (base_period / 2))
        elif anomaly_type == 'custom':
            for i in range(anomaly_start, anomaly_start + anomaly_length):
                values[i] += 0.1 * (i - anomaly_start)
        else:
            raise ValueError("Invalid anomaly_type.")

    values += np.random.normal(0, noise_level, length)
    df = pd.DataFrame({'timestamp': timestamps, 'value': values})
    return df

def plot_anomalies(df, distances, m, indices, motif_indices, ax=None):  # Added motif_indices
    """Plots the time series, matrix profile, anomalies, and motifs."""
    if ax is None:
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(14, 6))
        ax = axes.flatten()

    # Plot Time Series
    ax[0].plot(df['timestamp'], df['value'], linewidth=0.8)
    ax[0].set_ylabel('Value')
    ax[0].set_title('Time Series Data')
    ax[0].xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))

    # Plot Matrix Profile
    ax[1].plot(df['timestamp'][m - 1:], distances, linewidth=0.8)
    ax[1].set_ylabel('Distance')
    ax[1].set_title('Matrix Profile')
    ax[1].xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))

    # Highlight Anomalies
    anomaly_timestamps = df['timestamp'][m - 1:][indices]
    ax[1].scatter(anomaly_timestamps, distances[indices], c='red', s=60, label='Anomalies', zorder=5)
    ax[0].scatter(df['timestamp'][m - 1:][indices], df['value'][m - 1:][indices], c='red', s=60, label='Anomalies', zorder=5)

    # Highlight Motifs
    motif_timestamps = df['timestamp'][m - 1:][motif_indices]  # Use motif_indices
    ax[1].scatter(motif_timestamps, distances[motif_indices], c='green', s=60, marker='x', label="Motifs", zorder=5)
    ax[0].scatter(df['timestamp'][m - 1:][motif_indices], df['value'][m - 1:][motif_indices], c='green', s=60, marker='x', label="Motifs", zorder=5)

    ax[1].legend()
    ax[0].legend()

    plt.tight_layout()
    return ax

def detect_anomalies_with_threshold(df, m, threshold):
    """Detects anomalies using a threshold."""
    distances = stumpy.stump(df['value'], m=m)
    indices = np.where(distances[:, 0] > threshold)[0]
    indices = np.sort(indices)
    return distances[:, 0], indices

def find_motifs(distances, num_motifs=5): # Added function to find motifs
    """
    Finds the indices of the top 'num_motifs' motifs in the matrix profile.
    """
    sorted_indices = np.argsort(distances)  # Sort distances in ascending order
    return sorted_indices[:num_motifs]


def main():
    # Time series parameters
    length = 5000
    base_period = 50
    pattern_periods = [
        (0, 50, 5, 'sine'),
        (500, 100, 3, 'sawtooth'),
        (1500, 75, 8, 'square'),
        (2500, 40, 6, 'sine'),
        (3500, 60, 4, 'sawtooth')
    ]
    anomaly_start = 4000
    anomaly_length = int(0.02 * length)
    anomaly_type = 'pattern_change'

    df = generate_complex_time_series_v2(length, base_period, pattern_periods,
                                        noise_level=0.2, anomaly_start=anomaly_start,
                                        anomaly_length=anomaly_length, anomaly_type=anomaly_type)

    # Anomaly Detection
    m = 50
    threshold = 1.5
    distances, indices = detect_anomalies_with_threshold(df, m, threshold)

    # Motif Finding
    num_motifs = 20  # Find the top 10 motifs
    motif_indices = find_motifs(distances, num_motifs)

    # Visualize
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(14, 6))
    plot_anomalies(df, distances, m, indices, motif_indices, axes.flatten())  # Pass motif_indices
    plt.show()

    print("\nTop Anomaly Subsequences:")
    for i, index in enumerate(indices):
        start_idx = index
        end_idx = index + m
        print(f"Anomaly {i+1}:")
        print(df.iloc[start_idx:end_idx])
        print("-" * 20)

    print("\nTop Motif Subsequences:")  # Print motif subsequences
    for i, index in enumerate(motif_indices):
        start_idx = index
        end_idx = index + m
        print(f"Motif {i+1}:")
        print(df.iloc[start_idx:end_idx])
        print("-" * 20)

if __name__ == "__main__":
    main()