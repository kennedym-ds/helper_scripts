import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import ruptures as rpt
from bocd import BOCD
import stumpy
from scipy.stats import gaussian_kde, zscore
from dtaidistance import dtw  # For DTW distance
from typing import List, Tuple, Dict, Union, Optional


def z_normalize(data: np.ndarray) -> np.ndarray:
    """
    Z-normalizes the input data.

    This function standardizes the data by subtracting the mean and dividing by the standard deviation.
    If the standard deviation is zero, it returns an array of zeros to avoid division by zero.

    Args:
        data: NumPy array of data to normalize.

    Returns:
        Z-normalized NumPy array.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a NumPy array.")
    std_dev = np.std(data)
    if std_dev == 0:
        return np.zeros_like(data)  # Return zeros if standard deviation is zero to avoid division by zero
    return (data - np.mean(data)) / std_dev


class TimeSeriesAnomalyDetector:
    """
    A comprehensive class for detecting anomalies in time series data using
    various unsupervised methods.

    It includes methods based on statistical techniques, forecasting, clustering,
    density estimation, change point detection, matrix profile, template matching,
    and one-class SVM.
    """

    def __init__(self, data: Union[np.ndarray, pd.Series], timestamps: Optional[np.ndarray] = None):
        """
        Initializes the TimeSeriesAnomalyDetector with time series data and optional timestamps.

        Args:
            data: The time series data as a NumPy array or Pandas Series. Must be 1-dimensional and not contain NaN or inf values.
            timestamps: Optional timestamps corresponding to the data. If None and data is a Pandas Series, index is used as timestamps.
                        If None and data is a NumPy array, a range of integers is used as timestamps.

        Raises:
            TypeError: if data is not a NumPy array or Pandas Series.
            ValueError: if data contains NaN or inf values, or if data is not 1-dimensional.
        """
        if not isinstance(data, (np.ndarray, pd.Series)):
            raise TypeError("Data must be a NumPy array or Pandas Series.")

        if isinstance(data, pd.Series):
            self.data: np.ndarray = data.values
            if timestamps is None:
                self.timestamps: np.ndarray = data.index.values
            else:
                self.timestamps: np.ndarray = timestamps
        else:  # isinstance(data, np.ndarray)
            self.data: np.ndarray = data
            if timestamps is None:
                self.timestamps: np.ndarray = np.arange(len(data))
            else:
                self.timestamps: np.ndarray = timestamps

        if np.isnan(self.data).any() or np.isinf(self.data).any():
            raise ValueError("Input data contains NaN or inf values. Please preprocess the data to handle missing or infinite values.")
        if self.data.ndim != 1:
            raise ValueError("Data must be a 1-dimensional array.")

        self.results: Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]] = {}  # Store results for each method

    def _rolling_window_features(self, window_size: int) -> np.ndarray:
        """Calculates rolling window features (mean, std, skew, kurtosis) for the time series data.

        Args:
            window_size: The size of the rolling window. Must be a positive integer less than or equal to the data length.

        Returns:
            NumPy array of features, where each row corresponds to a time point and columns are [rolling_mean, rolling_std, rolling_skew, rolling_kurt].

        Raises:
            ValueError: if window_size is not a positive integer or exceeds data length.
        """
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("window_size must be a positive integer.")
        if window_size > len(self.data):
            raise ValueError("window_size cannot be larger than the data length.")

        df = pd.DataFrame(self.data)
        rolling_mean = df.rolling(window_size).mean().fillna(method='bfill').values.flatten()
        rolling_std = df.rolling(window_size).std().fillna(method='bfill').values.flatten()
        rolling_skew = df.rolling(window_size).skew().fillna(method='bfill').values.flatten()
        rolling_kurt = df.rolling(window_size).kurt().fillna(method='bfill').values.flatten()

        features = np.column_stack([rolling_mean, rolling_std, rolling_skew, rolling_kurt])
        return features

    def _modified_zscore(self, x: np.ndarray) -> np.ndarray:
        """Calculates the Modified Z-Score for a given array.

        The Modified Z-Score is more robust to outliers than the standard Z-score.

        Args:
            x: NumPy array of values.

        Returns:
            NumPy array of Modified Z-Scores.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("Input must be a NumPy array.")
        median = np.median(x)
        mad = np.median(np.abs(x - median))
        modified_z = 0.6745 * (x - median) / mad if mad != 0 else np.zeros(len(x))
        return modified_z

    def rolling_window_zscore(self, window_size: int, threshold: float = 3.5) -> np.ndarray:
        """Detects anomalies using rolling window statistics and Modified Z-Score.

        For each rolling window, features (mean, std, skew, kurtosis) are calculated.
        Anomalies are detected in each feature series based on the Modified Z-Score.

        Args:
            window_size: Size of the rolling window (must be a positive integer).
            threshold: Threshold for Modified Z-Score to identify anomalies (must be a number).

        Returns:
            NumPy array of anomaly indices.
        """
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("window_size must be a positive integer.")
        if not isinstance(threshold, (int, float)):
            raise TypeError("threshold must be a number.")
        if window_size > len(self.data):
            raise ValueError("window_size cannot be larger than the data length.")

        features = self._rolling_window_features(window_size)
        anomalies: List[int] = []
        for i in range(features.shape[1]):  # Iterate over each feature column
            modified_z = self._modified_zscore(features[:, i])
            anomaly_indices = np.where(np.abs(modified_z) > threshold)[0]
            anomalies.extend(anomaly_indices) # Extend the list with new anomaly indices
        anomalies_array = np.unique(anomalies) # Remove duplicates and convert to array
        self.results['rolling_window_zscore'] = anomalies_array
        return anomalies_array

    def stl_decomposition(self, seasonal: int, threshold: float = 3.5) -> np.ndarray:
        """Detects anomalies using STL decomposition and Modified Z-Score on residuals.

        STL (Seasonal-Trend decomposition using Loess) decomposes the time series into seasonal, trend, and residual components.
        Anomalies are detected in the residual component using the Modified Z-Score.

        Args:
            seasonal: Seasonal period for STL decomposition (must be a positive integer).
            threshold: Threshold for Modified Z-Score on residuals (must be a number).

        Returns:
            NumPy array of anomaly indices.
        """
        if not isinstance(seasonal, int) or seasonal <= 0:
            raise ValueError("seasonal must be a positive integer.")
        if not isinstance(threshold, (int, float)):
            raise TypeError("threshold must be a number.")

        df = pd.DataFrame({'value': self.data}, index=self.timestamps)  # Using timestamps for time series index
        try:
            stl = STL(df['value'], seasonal=seasonal, robust=True)
            res = stl.fit()
            residuals = res.resid.values
            modified_z = self._modified_zscore(residuals)
            anomalies = np.where(np.abs(modified_z) > threshold)[0]
            self.results['stl_decomposition'] = anomalies
            return anomalies
        except Exception as e:
            print(f"Error during STL decomposition: {e}. Returning empty anomaly array for STL.")
            return np.array([])

    def forecasting_arima(self, train_size: float = 0.8, order: Tuple[int, int, int] = (5, 1, 0), threshold: float = 3.5) -> np.ndarray:
        """Detects anomalies using ARIMA forecasting and Modified Z-Score on errors.

        An ARIMA model is trained on the training portion of the data, and then used to forecast the test portion.
        Anomalies are detected in the test set where the Modified Z-Score of the forecasting errors exceeds the threshold.

        Args:
            train_size: Proportion of data to use for training ARIMA model (must be between 0 and 1 exclusive).
            order: Order of the ARIMA model (p, d, q) as a tuple of three integers.
            threshold: Threshold for Modified Z-Score on forecasting errors (must be a number).

        Returns:
            NumPy array of anomaly indices.
        """
        if not 0 < train_size < 1:
            raise ValueError("train_size must be between 0 and 1 (exclusive).")
        if not isinstance(order, tuple) or len(order) != 3 or not all(isinstance(o, int) for o in order):
            raise ValueError("ARIMA order must be a tuple of 3 integers (p, d, q).")
        if not isinstance(threshold, (int, float)):
            raise TypeError("threshold must be a number.")

        train_data = self.data[:int(len(self.data) * train_size)]
        test_data = self.data[int(len(self.data) * train_size):]

        try:
            model = ARIMA(train_data, order=order)
            model_fit = model.fit()
            predictions = model_fit.predict(start=len(train_data), end=len(self.data) - 1)
            errors = test_data - predictions
            modified_z = self._modified_zscore(errors)
            anomalies = np.where(np.abs(modified_z) > threshold)[0] + int(len(self.data) * train_size)
            self.results['forecasting_arima'] = anomalies
            return anomalies
        except Exception as e:
            print(f"Error during ARIMA forecasting: {e}. Returning empty anomaly array for ARIMA.")
            return np.array([])

    def forecasting_exp_smoothing(self, train_size: float = 0.8, seasonal_periods: int = 7, threshold: float = 3.5) -> np.ndarray:
        """Detects anomalies using Exponential Smoothing and Modified Z-Score on errors.

        Exponential Smoothing is used for forecasting, considering seasonal components.
        Anomalies are detected based on the Modified Z-Score of the forecasting errors in the test set.

        Args:
            train_size: Proportion of data to use for training Exponential Smoothing model (must be between 0 and 1 exclusive).
            seasonal_periods: Seasonal period for Exponential Smoothing (must be a positive integer).
            threshold: Threshold for Modified Z-Score on forecasting errors (must be a number).

        Returns:
            NumPy array of anomaly indices.
        """
        if not 0 < train_size < 1:
            raise ValueError("train_size must be between 0 and 1 (exclusive).")
        if not isinstance(seasonal_periods, int) or seasonal_periods <= 0:
            raise ValueError("seasonal_periods must be a positive integer.")
        if not isinstance(threshold, (int, float)):
            raise TypeError("threshold must be a number.")

        train_data = self.data[:int(len(self.data) * train_size)]
        test_data = self.data[int(len(self.data) * train_size):]

        try:
            model = ExponentialSmoothing(train_data, seasonal='add', seasonal_periods=seasonal_periods)
            model_fit = model.fit()
            predictions = model_fit.predict(start=len(train_data), end=len(self.data) - 1)
            errors = test_data - predictions
            modified_z = self._modified_zscore(errors)
            anomalies = np.where(np.abs(modified_z) > threshold)[0] + int(len(self.data) * train_size)
            self.results['forecasting_exp_smoothing'] = anomalies
            return anomalies
        except Exception as e:
            print(f"Error during Exponential Smoothing forecasting: {e}. Returning empty anomaly array for Exponential Smoothing.")
            return np.array([])

    def clustering_dbscan(self, window_size: int, eps: float = 0.5, min_samples: int = 5) -> np.ndarray:
        """Detects anomalies using DBSCAN clustering on rolling window features.

        Rolling window features are extracted and scaled. DBSCAN is used to cluster these features.
        Points in sparse regions (noise points labeled as -1 by DBSCAN) are considered anomalies.

        Args:
            window_size: Size of the rolling window for feature extraction (must be a positive integer).
            eps: The maximum distance between two samples for DBSCAN (must be a positive number).
            min_samples: The minimum number of samples in a neighborhood for DBSCAN (must be a positive integer).

        Returns:
            NumPy array of anomaly indices.
        """
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("window_size must be a positive integer.")
        if not isinstance(eps, (int, float)) or eps <= 0:
            raise ValueError("eps must be a positive number.")
        if not isinstance(min_samples, int) or min_samples <= 0:
            raise ValueError("min_samples must be a positive integer.")
        if window_size > len(self.data):
            raise ValueError("window_size cannot be larger than the data length.")

        features = self._rolling_window_features(window_size)
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)  # Scale features for DBSCAN

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(scaled_features)
        anomalies = np.where(clusters == -1)[0] + window_size - 1
        self.results['clustering_dbscan'] = anomalies
        return anomalies

    def clustering_kmeans(self, window_size: int, n_clusters: int = 5, threshold_factor: float = 3.0) -> np.ndarray:
        """Detects anomalies using K-Means clustering on rolling window features.

        Rolling window features are extracted and scaled. K-Means clusters these features.
        Anomalies are points that are far from their cluster centroids, based on a distance threshold.

        Args:
            window_size: Size of the rolling window for feature extraction (must be a positive integer).
            n_clusters: Number of clusters for K-Means (must be a positive integer).
            threshold_factor: Factor to multiply standard deviation of distances to determine anomaly threshold (must be a number).

        Returns:
            NumPy array of anomaly indices.
        """
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("window_size must be a positive integer.")
        if not isinstance(n_clusters, int) or n_clusters <= 0:
            raise ValueError("n_clusters must be a positive integer.")
        if not isinstance(threshold_factor, (int, float)):
            raise TypeError("threshold_factor must be a number.")
        if window_size > len(self.data):
            raise ValueError("window_size cannot be larger than the data length.")

        features = self._rolling_window_features(window_size)
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        kmeans.fit(scaled_features)
        distances = kmeans.transform(scaled_features)
        min_distances = np.min(distances, axis=1)
        threshold = np.mean(min_distances) + threshold_factor * np.std(min_distances)
        anomalies = np.where(min_distances > threshold)[0] + window_size - 1
        self.results['clustering_kmeans'] = anomalies
        return anomalies

    def isolation_forest(self, window_size: int, contamination: float = 0.05) -> np.ndarray:
        """Detects anomalies using Isolation Forest on rolling window features.

        Isolation Forest isolates anomalies by randomly partitioning the data. Anomalies are expected to be isolated in fewer partitions.
        Rolling window features are used as input to the Isolation Forest model.

        Args:
            window_size: Size of the rolling window for feature extraction (must be a positive integer).
            contamination: The proportion of outliers in the data set (must be between 0 and 0.5, typically small).

        Returns:
            NumPy array of anomaly indices.
        """
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("window_size must be a positive integer.")
        if not isinstance(contamination, (int, float)) or not 0.0 < contamination < 0.5: # Relaxed upper bound to 0.5 for practical use
            raise ValueError("contamination must be a float between 0.0 and 0.5.")
        if window_size > len(self.data):
            raise ValueError("window_size cannot be larger than the data length.")

        features = self._rolling_window_features(window_size)
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        model = IsolationForest(contamination=contamination, random_state=42)
        model.fit(scaled_features)
        predictions = model.predict(scaled_features)
        anomalies = np.where(predictions == -1)[0] + window_size - 1
        self.results['isolation_forest'] = anomalies
        return anomalies

    def kde_residuals(self, seasonal: int, bandwidth: float = 0.1, threshold_percentile: float = 1.0) -> np.ndarray:
        """Detects anomalies using KDE on residuals after STL decomposition.

        After STL decomposition, Kernel Density Estimation (KDE) is applied to the residuals.
        Anomalies are points in the residuals distribution that have very low probability density, below a percentile threshold.

        Args:
            seasonal: Seasonal period for STL decomposition (must be a positive integer).
            bandwidth: Bandwidth for Gaussian KDE (must be a positive number).
            threshold_percentile: Percentile of densities below which points are considered anomalies (must be between 0 and 100).

        Returns:
            NumPy array of anomaly indices.
        """
        if not isinstance(seasonal, int) or seasonal <= 0:
            raise ValueError("seasonal must be a positive integer.")
        if not isinstance(bandwidth, (int, float)) or bandwidth <= 0:
            raise ValueError("bandwidth must be a positive number.")
        if not isinstance(threshold_percentile, (int, float)) or not 0.0 <= threshold_percentile <= 100.0:
            raise ValueError("threshold_percentile must be a float between 0.0 and 100.0.")

        df = pd.DataFrame({'value': self.data}, index=self.timestamps)
        try:
            stl = STL(df['value'], seasonal=seasonal, robust=True)
            res = stl.fit()
            residuals = res.resid.values

            kde = gaussian_kde(residuals, bw_method=bandwidth)
            densities = kde.evaluate(residuals)
            threshold = np.percentile(densities, threshold_percentile)
            anomalies = np.where(densities < threshold)[0]
            self.results['kde_residuals'] = anomalies
            return anomalies
        except Exception as e:
            print(f"Error during KDE residuals analysis: {e}. Returning empty anomaly array for KDE Residuals.")
            return np.array([])

    def change_point_detection_ruptures(self, model: str = "l2", pen: float = 10) -> np.ndarray:
        """Detects anomalies using change point detection (ruptures library).

        Change point detection algorithms from the `ruptures` library are used to find change points in the time series.
        Points around the detected change points are considered anomalies.

        Args:
            model: Model type for ruptures library ('l1', 'l2', 'rbf', 'linear').
            pen: Penalty value for change point detection (must be a positive number).

        Returns:
            NumPy array of anomaly indices.
        """
        if not isinstance(model, str) or model not in ['l1', 'l2', 'rbf', 'linear']:
            raise ValueError("model must be a string and one of ['l1', 'l2', 'rbf', 'linear'].")
        if not isinstance(pen, (int, float)) or pen <= 0:
            raise ValueError("pen must be a positive number.")

        try:
            algo = rpt.Pelt(model=model, min_size=2, jump=1).fit(self.data)
            change_points = algo.predict(pen=pen)
            # Convert change points to anomaly indices (consider points around change points as anomalous)
            anomalies: List[int] = []
            for cp in change_points:
                anomalies.extend(range(max(0, cp - 2), min(len(self.data), cp + 2)))  # Add points around each change point
            anomalies_array = np.unique(anomalies)
            self.results['change_point_detection_ruptures'] = anomalies_array
            return anomalies_array
        except Exception as e:
            print(f"Error during ruptures change point detection: {e}. Returning empty anomaly array for ruptures.")
            return np.array([])

    def change_point_detection_bocd(self, lambda_: int = 20, alpha: float = 0.1, beta: float = 0.01, kappa: float = 0.1, mu: float = 0, threshold: float = 0.5) -> np.ndarray:
        """Detects anomalies using Bayesian Online Change Point Detection (BOCD).

        BOCD algorithm detects change points online as the data stream comes in.
        Anomalies are detected based on the growth probability calculated by BOCD exceeding a threshold.

        Args:
            lambda_: Prior parameter for BOCD (must be a positive integer).
            alpha: Prior parameter for BOCD (must be a number, typically small positive).
            beta: Prior parameter for BOCD (must be a number, typically small positive).
            kappa: Prior parameter for BOCD (must be a number, typically small positive).
            mu: Prior parameter for BOCD (must be a number).
            threshold: Threshold for growth probability to identify anomalies (must be a number between 0 and 1).

        Returns:
            NumPy array of anomaly indices.
        """
        if not isinstance(lambda_, int) or lambda_ <= 0:
            raise ValueError("lambda_ must be a positive integer.")
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise ValueError("alpha must be a non-negative number.")
        if not isinstance(beta, (int, float)) or beta < 0:
            raise ValueError("beta must be a non-negative number.")
        if not isinstance(kappa, (int, float)) or kappa < 0:
            raise ValueError("kappa must be a non-negative number.")
        if not isinstance(mu, (int, float)):
            raise TypeError("mu must be a number.")
        if not isinstance(threshold, (int, float)) or not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be a float between 0.0 and 1.0.")

        bc = BOCD(lambda_=lambda_, alpha=alpha, beta=beta, kappa=kappa, mu=mu)
        growth_probs: List[float] = []
        try:
            for x in self.data:
                bc.update(x)
                growth_probs.append(bc.growth_probs[-1])  # Use the last growth probability

            anomalies = np.array([i for i, prob in enumerate(growth_probs) if prob > threshold])  # Indices where growth prob exceeds threshold
            self.results['change_point_detection_bocd'] = anomalies
            return anomalies
        except Exception as e:
            print(f"Error during BOCD change point detection: {e}. Returning empty anomaly array for BOCD.")
            return np.array([])

    def matrix_profile(self, window_size: int, threshold_factor: float = 3.0) -> np.ndarray:
        """Detects anomalies using the Matrix Profile (stumpy library).

        Matrix Profile algorithm identifies discords (anomalies) in a time series by comparing each subsequence to its nearest neighbors.
        Anomalies are points where the Matrix Profile value (distance to nearest neighbor) is significantly higher than average.

        Args:
            window_size: Size of the sliding window for Matrix Profile (must be an integer greater than 1).
            threshold_factor: Factor to multiply standard deviation of Matrix Profile to determine anomaly threshold (must be a number).

        Returns:
            NumPy array of anomaly indices.
        """
        if not isinstance(window_size, int) or window_size <= 1:
            raise ValueError("window_size must be an integer greater than 1.")
        if not isinstance(threshold_factor, (int, float)):
            raise TypeError("threshold_factor must be a number.")
        if window_size > len(self.data):
            raise ValueError("window_size cannot be larger than the data length.")

        try:
            mp = stumpy.stump(self.data, m=window_size)
            matrix_profile_values = mp[:, 0] # Matrix profile is the first column
            threshold = np.mean(matrix_profile_values) + threshold_factor * np.std(matrix_profile_values)
            anomalies = np.where(matrix_profile_values > threshold)[0]
            self.results['matrix_profile'] = anomalies
            return anomalies
        except Exception as e:
            print(f"Error during Matrix Profile calculation: {e}. Returning empty anomaly array for Matrix Profile.")
            return np.array([])

    def one_class_svm(self, window_size: int, nu: float = 0.05, kernel: str = "rbf", gamma: Union[str, float] = 'scale') -> np.ndarray:
        """Detects anomalies using One-Class SVM on rolling window features.

        One-Class SVM is an unsupervised algorithm for anomaly detection, effective in high-dimensional spaces.
        It learns a boundary around normal data and identifies anomalies as points outside this boundary.
        Rolling window features are used as input for One-Class SVM.

        Args:
            window_size: Size of the rolling window for feature extraction (must be a positive integer).
            nu: An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors (must be between 0 and 1).
            kernel: Kernel type to be used in the algorithm ('linear', 'poly', 'rbf', 'sigmoid', 'precomputed').
            gamma: Kernel coefficient for 'rbf', 'poly' and 'sigmoid'. 'scale' (default) is pass gamma as 1 / (n_features * X.var()) or a float number.

        Returns:
            NumPy array of anomaly indices.
        """
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("window_size must be a positive integer.")
        if not isinstance(nu, (int, float)) or not 0.0 < nu <= 1.0:
            raise ValueError("nu must be a float between 0.0 and 1.0 (exclusive at 0.0 and inclusive at 1.0).")
        if not isinstance(kernel, str) or kernel not in ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']:
            raise ValueError("kernel must be a string and one of ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'].")
        if not isinstance(gamma, (str, float)):
            raise TypeError("gamma must be a string 'scale' or a float number.")
        if isinstance(gamma, float) and gamma < 0:
            raise ValueError("gamma must be a non-negative float.")
        if window_size > len(self.data):
            raise ValueError("window_size cannot be larger than the data length.")


        features = self._rolling_window_features(window_size)
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        try:
            oc_svm = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
            oc_svm.fit(scaled_features)
            predictions = oc_svm.predict(scaled_features)
            anomalies = np.where(predictions == -1)[0] + window_size - 1
            self.results['one_class_svm'] = anomalies
            return anomalies
        except Exception as e:
            print(f"Error during One-Class SVM: {e}. Returning empty anomaly array for One-Class SVM.")
            return np.array([])

    def template_matching(self, templates: Dict[str, np.ndarray], threshold: float = 0.8) -> Dict[str, np.ndarray]:
        """Detects anomalies using template matching with correlation.

        This method compares sliding windows of the time series data with predefined templates using normalized cross-correlation.
        If the correlation exceeds a threshold for a given template, the corresponding window is marked as an anomaly for that template type.

        Args:
            templates: Dictionary of templates. Keys are template names (e.g., "step_up", "spike"),
                        values are NumPy arrays representing the templates (must be 1-dimensional NumPy arrays).
            threshold: Correlation threshold for anomaly detection (must be a number between -1 and 1).

        Returns:
            Dictionary of anomaly indices for each template. Keys are template names and values are NumPy arrays of anomaly indices.
        """
        if not isinstance(templates, dict) or not templates:
            raise ValueError("templates must be a non-empty dictionary.")
        if not isinstance(threshold, (int, float)) or not -1.0 <= threshold <= 1.0:
            raise ValueError("threshold must be a number between -1 and 1.")

        window_size = 0 # Initialize window_size
        results: Dict[str, np.ndarray] = {} # Initialize results dictionary

        for template_name, template in templates.items():
            if not isinstance(template, np.ndarray) or template.ndim != 1:
                raise ValueError(f"Template '{template_name}' must be a 1-dimensional NumPy array.")
            if not window_size: # Set window size from the first template length
                window_size = len(template)
            elif len(template) != window_size:
                raise ValueError(f"Template '{template_name}' length must be consistent with other templates.")
            if window_size > len(self.data):
                raise ValueError(f"Template length ({window_size}) cannot be larger than the data length.")


            correlations: List[float] = []
            template_norm = z_normalize(template)

            for i in range(len(self.data) - window_size + 1):
                window = z_normalize(self.data[i: i + window_size])
                correlation = np.dot(window, template_norm) if np.linalg.norm(window) > 0 and np.linalg.norm(template_norm) > 0 else 0 # handle zero norm vectors
                correlations.append(correlation)

            anomalies = np.where(np.array(correlations) > threshold)[0]
            results[template_name] = anomalies

        self.results['template_matching'] = results
        return results

    def plot_anomalies(self, method_name: str, ax: Optional[plt.Axes] = None, **kwargs) -> plt.Axes:
        """Plots the time series data and highlights anomalies detected by the specified method.

        Args:
            method_name: The name of the method whose results should be plotted (must be a valid method name in self.results).
            ax: Pre-existing axes for the plot. If None, a new figure and axes are created.
            **kwargs: Additional keyword arguments to pass to `plt.plot` for the data line style and properties.

        Returns:
            matplotlib.axes.Axes: The axes object containing the plot.

        Raises:
            ValueError: if method_name is not found in self.results.
        """
        if method_name not in self.results:
            raise ValueError(f"Method '{method_name}' not found in results. Run the method first.")
        anomalies = self.results[method_name]

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(self.timestamps, self.data, label="Data", **kwargs)

        if anomalies is None or (isinstance(anomalies, np.ndarray) and anomalies.size == 0) or (isinstance(anomalies, dict) and all(not indices.size for indices in anomalies.values())):
            print(f"No anomalies found by {method_name} to plot.") # Or use logging, or just don't print.
            # Still plot the data even if no anomalies to highlight the timeseries
        elif isinstance(anomalies, dict):  # Handle template matching results
            colors = plt.cm.viridis(np.linspace(0, 1, len(anomalies))) # Distinct colors for templates
            for idx, (template_name, anomaly_indices) in enumerate(anomalies.items()):
                ax.scatter(self.timestamps[anomaly_indices], self.data[anomaly_indices], label=f"Anomalies ({template_name})", marker="x", s=60, color=colors[idx], zorder=5) # Use different colors, zorder to put anomalies on top
        elif isinstance(anomalies, np.ndarray) and anomalies.size > 0:  # Handle regular anomaly arrays
            ax.scatter(self.timestamps[anomalies], self.data[anomalies], color='r', label=f"Anomalies ({method_name})", marker="x", s=60, zorder=5) # zorder to put anomalies on top

        ax.set_title(f"Time Series with Anomalies ({method_name})")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend()
        return ax

    def get_results(self) -> Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]:
        """Returns the anomaly detection results for all run methods.

        Returns:
            Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]: A dictionary where keys are method names and values are
            numpy arrays of anomaly indices or dictionaries for template matching results.
        """
        return self.results

    def reset_results(self) -> None:
        """Clears the stored anomaly detection results, allowing for fresh detections."""
        self.results = {}

    def detect_all(self, params: Optional[Dict[str, Dict]] = None) -> Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]:
        """Runs all anomaly detection methods with default or provided parameters.

        Args:
            params: Optional dictionary of parameters to override default parameters for each method.
                    The structure should be a nested dictionary where the key is the method name and the value
                    is a dictionary of parameters for that method. Example:
                    `{'rolling_window_zscore': {'window_size': 30, 'threshold': 4.0}, 'isolation_forest': {'contamination': 0.1}}`

        Returns:
            Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]: A dictionary of anomaly indices for each method run.
        """
        if params is None:
            params = {}

        # Default parameters for each method
        default_params: Dict[str, Dict] = {
            'rolling_window_zscore': {'window_size': 20, 'threshold': 3.5},
            'stl_decomposition': {'seasonal': 7, 'threshold': 3.5},
            'forecasting_arima': {'train_size': 0.8, 'order': (5, 1, 0), 'threshold': 3.5},
            'forecasting_exp_smoothing': {'train_size': 0.8, 'seasonal_periods': 7, 'threshold': 3.5},
            'clustering_dbscan': {'window_size': 20, 'eps': 0.3, 'min_samples': 5},
            'clustering_kmeans': {'window_size': 20, 'n_clusters': 5, 'threshold_factor': 3.0},
            'isolation_forest': {'window_size': 20, 'contamination': 0.05},
            'kde_residuals': {'seasonal': 7, 'bandwidth': 0.1, 'threshold_percentile': 1.0},
            'change_point_detection_ruptures': {'model': 'l2', 'pen': 10},
            'change_point_detection_bocd': {'lambda_': 20, 'alpha': 0.1, 'beta': 0.01, 'kappa': 0.1, 'mu': 0, 'threshold': 0.5},
            'matrix_profile': {'window_size': 20, 'threshold_factor': 3.0},
            'one_class_svm': {'window_size': 20, 'nu': 0.05, 'kernel': 'rbf', 'gamma': 'scale'}
        }

        methods = [
            'rolling_window_zscore',
            'stl_decomposition',
            'forecasting_arima',
            'forecasting_exp_smoothing',
            'clustering_dbscan',
            'clustering_kmeans',
            'isolation_forest',
            'kde_residuals',
            'change_point_detection_ruptures',
            'change_point_detection_bocd',
            'matrix_profile',
            'one_class_svm'
        ]

        for method_name in methods:
            method_params = default_params.get(method_name, {}).copy() # Use default params as base
            method_params.update(params.get(method_name, {})) # Override with user-provided params if available
            detection_method = getattr(self, method_name, None) # Dynamically get the detection method
            if detection_method:
                try:
                    detection_method(**method_params) # Call the method with combined parameters
                except Exception as e:
                    print(f"Error running method {method_name}: {e}") # Report any errors during method execution

        return self.results


if __name__ == '__main__':
    # Generate synthetic time series data with anomalies
    np.random.seed(42)
    data = np.sin(np.linspace(0, 10 * np.pi, 200)) + np.random.normal(0, 0.1, 200)
    data[30:40] += 1.5  # Spike anomaly, increased magnitude for better visualization
    data[150:170] -= 1.0  # Dip anomaly, increased magnitude for better visualization
    timestamps = np.arange(len(data))

    # Example Usage
    detector = TimeSeriesAnomalyDetector(data, timestamps)

    # Run all detection methods with default parameters
    print("Running all detection methods with default parameters...")
    results_all_default = detector.detect_all()
    print("Default parameter detection completed.")

    # Plot anomalies for each method
    print("Plotting anomalies for default parameter results...")
    for method_name in results_all_default:
        if method_name != 'template_matching': # template matching needs templates to be run.
            ax = detector.plot_anomalies(method_name, linewidth=0.7) # Added linewidth for better data line visibility
            plt.title(f"Anomaly Detection - {method_name} (Default Params)") # More informative plot titles
            plt.show()
    print("Default parameter anomaly plotting completed.")


    # Example of template matching
    print("Running Template Matching...")
    templates = {
        "spike": np.array([0, 0.7, 1.2, 0.7, 0]), # Adjusted template values to better match synthetic spike
        "dip": np.array([0, -0.7, -1.2, -0.7, 0])  # Adjusted template values to better match synthetic dip
    }
    template_results = detector.template_matching(templates, threshold=0.75) # Slightly adjusted threshold
    ax = detector.plot_anomalies('template_matching', linewidth=0.7) # Plot template matching results
    plt.title("Anomaly Detection - Template Matching") # More informative plot titles
    plt.show()
    print("Template Matching completed and plotted.")


    # Example of running specific methods with custom parameters
    print("Running specific methods with custom parameters...")
    custom_params = {
        'rolling_window_zscore': {'window_size': 30, 'threshold': 4.0}, # Adjusted params for rolling_window_zscore
        'isolation_forest': {'contamination': 0.1} # Adjusted params for isolation_forest
    }
    detector.reset_results() # Clear previous results before running with custom params
    results_custom_params = detector.detect_all(params=custom_params)
    print("Custom parameter detection completed.")


    # Plot anomalies for methods run with custom parameters
    print("Plotting anomalies for custom parameter results...")
    for method_name in results_custom_params:
         if method_name != 'template_matching': # template matching results are already plotted.
            ax = detector.plot_anomalies(method_name, linewidth=0.7) # Plotting with linewidth
            plt.title(f"Anomaly Detection - {method_name} (Custom Params)") # More informative plot titles
            plt.show()
    print("Custom parameter anomaly plotting completed.")


    print("\nAll anomaly detection methods completed. Results are stored in detector.results attribute.")
    print("Example results (Rolling Window Z-score Anomalies using Custom Parameters):", detector.get_results().get('rolling_window_zscore'))