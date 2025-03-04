library(ggplot2)
library(zoo)
library(changepoint)
library(imputeTS)
library(tseries)
library(ecp)  # Using ecp as the primary non-parametric alternative

# 1. Generate Synthetic Soch Chart Data
generate_synthetic_soch_data <- function(n = 500,
                                         base_mean = 100,
                                         base_sd = 5,
                                         ucl = 115,
                                         lcl = 85,
                                         seed = 42,
                                         shift_points = NULL,
                                         trend_points = NULL,
                                         sd_shift_points = NULL,
                                         oscillation_points = NULL,
                                         outlier_points = NULL,
                                         missing_points = NULL) {
  set.seed(seed)
  data <- data.frame(Time = 1:n,
                     Value = rnorm(n, mean = base_mean, sd = base_sd))

  data$UCL <- ucl
  data$LCL <- lcl
  data$CenterLine <- base_mean

  # Introduce Mean Shifts
  if (!is.null(shift_points)) {
    for (shift in shift_points) {
      data$Value[shift$start:shift$end] <- rnorm(shift$end - shift$start + 1, shift$mean, base_sd)
    }
  }

  # Introduce Trends
  if (!is.null(trend_points)) {
    for (trend in trend_points) {
      trend_length = trend$end - trend$start + 1
      trend_values <- seq(0, trend$slope * (trend_length - 1), length.out = trend_length)
      data$Value[trend$start:trend$end] <- data$Value[trend$start:trend$end] + trend_values
    }
  }

  # Introduce Standard Deviation shifts
  if (!is.null(sd_shift_points)) {
    for (sd_shift in sd_shift_points) {
      data$Value[sd_shift$start:sd_shift$end] <- rnorm(sd_shift$end - sd_shift$start + 1, mean = base_mean, sd = base_sd * sd_shift$multiplier)
    }
  }

  # Introduce Oscillations
  if (!is.null(oscillation_points)) {
    for (oscillation in oscillation_points) {
      oscillation_length = oscillation$end - oscillation$start + 1
      oscillation_values <- oscillation$amplitude * sin(2 * pi * (1:oscillation_length) / oscillation$period)
      data$Value[oscillation$start:oscillation$end] <- data$Value[oscillation$start:oscillation$end] + oscillation_values
    }
  }

  # Introduce Outliers
  if (!is.null(outlier_points)) {
    for (outlier in outlier_points) {
      data$Value[outlier] <- data$Value[outlier] + rnorm(1, mean = 5 * base_sd, sd = base_sd)  # Add large deviation
    }
  }

  # Introduce Missing Values
  if (!is.null(missing_points)) {
    data$Value[missing_points] <- NA
  }

  return(data)
}

# 2. Preprocess Soch Data
preprocess_soch_data <- function(data) {
  # Interpolate missing values using linear interpolation
  data$Value <- na_interpolation(data$Value, option = "linear")
  return(data)
}

# 3. Detect Anomalies
detect_anomalies <- function(data, window_size = 30, k = 3, time_window = 100, acf_lag_max = 50, value_col = "Value") {

  # Use the specified column for calculations
  value <- data[[value_col]]

  # Out-of-Control Points (still use original Value for UCL/LCL)
  data$Out_of_Control <- data$Value > data$UCL | data$Value < data$LCL

  # Step Changes
  data$Step_Change <- c(NA, abs(diff(value))) > (k * sd(value, na.rm = TRUE))

  # Rolling Statistics
  data$Rolling_Mean <- rollapply(value, width = window_size, FUN = mean, align = "right", fill = NA)
  data$Rolling_SD <- rollapply(value, width = window_size, FUN = sd, align = "right", fill = NA)
  data$Rolling_Q25 <- rollapply(value, width = window_size, FUN = quantile, probs = 0.25, align = "right", fill = NA, na.rm = TRUE)
  data$Rolling_Q75 <- rollapply(value, width = window_size, FUN = quantile, probs = 0.75, align = "right", fill = NA, na.rm = TRUE)

  data$Mean_Shift_Rolling <- abs(data$Value - data$Rolling_Mean) > (2 * data$Rolling_SD)
  data$SD_Shift_Rolling <- (data$Rolling_SD > 1.5 * mean(data$Rolling_SD, na.rm = TRUE)) | (data$Rolling_SD < 0.5 * mean(data$Rolling_SD, na.rm = TRUE))

  # Change Point Detection (Parametric)
  mean_changes <- cpt.mean(value, method = "PELT", penalty = "BIC")
  var_changes <- cpt.var(value, method = "PELT", penalty = "BIC")
  meanvar_changes <- cpt.meanvar(value, method = "PELT", penalty = "BIC")

  # Change Point Detection (Non-Parametric - using ecp)
  ecp_results <- e.divisive(as.matrix(value), sig.lvl = 0.05)  # Convert to matrix
  ecp_change_points <- ecp_results$estimates[c(-1, -length(ecp_results$estimates))] #Remove start and end

  change_points <- cpts(meanvar_changes)

  data$changepoint_mean <- FALSE
  data$changepoint_var <- FALSE
  data$changepoint_meanvar <- FALSE
  data$changepoint_ecp <- FALSE  # Flag for ecp change points

  if (length(change_points) > 0) {
    data$changepoint_meanvar <- any((max(data$Time) - data$Time[change_points]) <= time_window)
  }
  if (length(cpts(mean_changes)) > 0) {
    data$changepoint_mean <- any((max(data$Time) - data$Time[cpts(mean_changes)]) <= time_window)
  }
  if (length(cpts(var_changes)) > 0) {
    data$changepoint_var <- any((max(data$Time) - data$Time[cpts(var_changes)]) <= time_window)
  }
  if (length(ecp_change_points) > 0) {
    data$changepoint_ecp <- any((max(data$Time) - data$Time[ecp_change_points]) <= time_window)
  }

  # Oscillations (Autocorrelation)
  acf_result <- acf(value, lag.max = acf_lag_max, plot = FALSE, na.action = na.pass)
  significant_lags <- which(abs(acf_result$acf) > 2 / sqrt(length(value))) - 1  # -1 for R indexing
  data$Oscillation <- FALSE
  if (length(significant_lags) > 0) {
    data$Oscillation <- TRUE
  }

  # Combine Anomaly Flags
  data$Anomaly <- data$Out_of_Control |
    !is.na(data$Step_Change) & data$Step_Change |
    !is.na(data$Mean_Shift_Rolling) & data$Mean_Shift_Rolling |
    !is.na(data$SD_Shift_Rolling) & data$SD_Shift_Rolling |
    data$Oscillation |
    !is.na(data$changepoint_mean) & data$changepoint_mean |
    !is.na(data$changepoint_var) & data$changepoint_var |
    !is.na(data$changepoint_meanvar) & data$changepoint_meanvar |
    !is.na(data$changepoint_ecp) & data$changepoint_ecp  # Include ecp flag

  return(data)
}

# 4. Plot Soch Chart with Anomalies
plot_soch_with_anomalies <- function(data, value_col, title = "Soch Chart with Anomaly Detection") {
  p <- ggplot(data, aes(x = Time, y = .data[[value_col]])) +
    geom_line() +
    geom_point(aes(color = Anomaly)) +
    geom_ribbon(aes(ymin = LCL, ymax = UCL), fill = "grey", alpha = 0.3) +
    geom_hline(aes(yintercept = CenterLine), linetype = "dashed") +
    labs(title = title, x = "Time", y = value_col, color = "Anomaly Detected") +
    theme_bw()

  print(p)
  return(p)
}

# 5. Stationarity Check
is_stationary <- function(data, value_col = "Value", sig_level = 0.05) {
  # Perform ADF test
  adf_result <- adf.test(data[[value_col]], alternative = "stationary")

  # Check p-value against significance level
  is_stationary <- adf_result$p.value < sig_level

  return(is_stationary)
}

# 6. Main Function
main <- function() {
  # --- Generate Synthetic Data ---
  synthetic_data <- generate_synthetic_soch_data(
    n = 600,
    base_mean = 50,
    base_sd = 4,
    ucl = 62,
    lcl = 38,
    shift_points = list(list(start = 200, end = 250, mean = 58)),
    trend_points = list(list(start = 300, end = 380, slope = -0.2)),
    sd_shift_points = list(list(start = 400, end = 450, multiplier = 2.5)),
    oscillation_points = list(list(start = 500, end = 580, period = 20, amplitude = 10)),
    outlier_points = c(50, 150, 480),
    missing_points = c(100, 101, 350, 351, 352)
  )

  # --- Preprocess Data ---
  preprocessed_data <- preprocess_soch_data(synthetic_data)

  # --- Check for Stationarity ---
  if (is_stationary(preprocessed_data)) {
    cat("Data is likely stationary.\n")

    # --- Detect Anomalies (Stationary Case) ---
    analyzed_data <- detect_anomalies(preprocessed_data)

    # --- Plot Results (Stationary Case)---
    plot_soch_with_anomalies(analyzed_data, value_col = "Value")
    print(paste("Number of anomalies detected:", sum(analyzed_data$Anomaly)))

  } else {
    cat("Data is likely non-stationary.\n")

    # --- Handle Non-Stationarity ---
    # Example: Detrending and Seasonal Differencing (adjust as needed)
    preprocessed_data$Detrended_Value <- residuals(lm(Value ~ Time, data = preprocessed_data))
    preprocessed_data$Differenced_Seasonal_Value <- c(rep(NA, 7), diff(preprocessed_data$Detrended_Value, lag = 7))  # Assuming weekly

    # --- Detect Anomalies (Non-Stationary Case) ---
    analyzed_data <- detect_anomalies(preprocessed_data, value_col = "Differenced_Seasonal_Value")

    # --- Plot Results (Non-Stationary Case) ---
    plot_soch_with_anomalies(analyzed_data, value_col = "Differenced_Seasonal_Value")
    print(paste("Number of anomalies detected:", sum(analyzed_data$Anomaly)))
  }
}

# Run the analysis
main()