library(ggplot2)
library(zoo)
library(changepoint)
library(imputeTS)
library(tseries)
library(ecp)

# 1. Generate Synthetic Soch Chart Data (No changes here)
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

# 2. Preprocess Soch Data (No changes here)
preprocess_soch_data <- function(data) {
  data$Value <- na_interpolation(data$Value, option = "linear")
  return(data)
}

# 3. Detect and Classify Anomalies (Major Changes)
detect_anomalies <- function(data, window_size = 30, k_step = 3, k_roll = 2.5, time_window = 100, acf_lag_max = 50, value_col = "Value") {

  value <- data[[value_col]]

  # Initialize Classification as "Normal"
  data$Classification <- "Normal"

  # 1. Out-of-Control Points
  out_of_control_indices <- which(data$Value > data$UCL | data$Value < data$LCL)
  data$Classification[out_of_control_indices] <- "Outlier"

  # 2. Step Changes
  step_change_indices <- which(c(NA, abs(diff(value))) > (k_step * sd(value, na.rm = TRUE)))
  data$Classification[step_change_indices] <- "Step Change" #Higher priority

  # 3. Rolling Statistics
  data$Rolling_Mean <- rollapply(value, width = window_size, FUN = mean, align = "right", fill = NA)
  data$Rolling_SD <- rollapply(value, width = window_size, FUN = sd, align = "right", fill = NA)
  data$Rolling_Q25 <- rollapply(value, width = window_size, FUN = quantile, probs = 0.25, align = "right", fill = NA, na.rm = TRUE)
  data$Rolling_Q75 <- rollapply(value, width = window_size, FUN = quantile, probs = 0.75, align = "right", fill = NA, na.rm = TRUE)

  mean_shift_indices <- which(abs(data$Value - data$Rolling_Mean) > (k_roll * data$Rolling_SD))
  data$Classification[mean_shift_indices] <- "Mean Shift (Rolling)"

  sd_shift_indices <- which((data$Rolling_SD > 1.5 * mean(data$Rolling_SD, na.rm = TRUE)) | (data$Rolling_SD < 0.5 * mean(data$Rolling_SD, na.rm = TRUE)))
   data$Classification[sd_shift_indices] <- "SD Shift (Rolling)"

  # 4. Change Point Detection (Parametric)
    mean_changes <- cpt.mean(value, method = "PELT", penalty = "BIC",minseglen=2)
    var_changes <- cpt.var(value, method = "PELT", penalty = "BIC", minseglen=2)
    meanvar_changes <- cpt.meanvar(value, method = "PELT", penalty = "BIC",minseglen=2)

  # 5. Change Point Detection (Non-Parametric - using ecp)
  ecp_results <- e.divisive(as.matrix(value), sig.lvl = 0.05)
  ecp_change_points <- ecp_results$estimates[c(-1, -length(ecp_results$estimates))]


  # Classify based on change points (prioritize mean/variance changes)
  if (length(cpts(meanvar_changes)) > 0) {
    for (cp in cpts(meanvar_changes)) {
      if ((max(data$Time) - data$Time[cp]) <= time_window) {
        data$Classification[cp] <- "Mean/Var Change (CP)"  #Most descriptive
      }
    }
  }
    if (length(cpts(mean_changes)) > 0) {
        for (cp in cpts(mean_changes)) {
            if ((max(data$Time) - data$Time[cp]) <= time_window) {
                data$Classification[cp] <- "Mean Change (CP)"
            }
        }
    }
    if (length(cpts(var_changes)) > 0) {
        for (cp in cpts(var_changes)) {
            if ((max(data$Time) - data$Time[cp]) <= time_window) {
                 data$Classification[cp] <- "Var Change (CP)"
            }
        }
    }
  if (length(ecp_change_points) > 0) {
    for (cp in ecp_change_points) {
      if ((max(data$Time) - data$Time[cp]) <= time_window) {
        data$Classification[cp] <- "Distribution Change (ECP)" #If its not picked up by the others
      }
    }
  }

  # 6. Oscillations (Autocorrelation)
  acf_result <- acf(value, lag.max = acf_lag_max, plot = FALSE, na.action = na.pass)
  significant_lags <- which(abs(acf_result$acf) > 2 / sqrt(length(value))) - 1
  if (length(significant_lags) > 0) {
     #Check to see if oscillation flag should be applied
      data$Classification[length(significant_lags)] <- "Oscillation" #Just classify one point.
  }

  return(data)
}

# 4. Plot Soch Chart with Anomalies (Modified for Classification)
plot_soch_with_anomalies <- function(data, value_col, title = "Soch Chart with Anomaly Detection") {
  p <- ggplot(data, aes(x = Time, y = .data[[value_col]])) +
    geom_line() +
    geom_point(aes(color = Classification)) +  # Color by Classification
    geom_ribbon(aes(ymin = LCL, ymax = UCL), fill = "grey", alpha = 0.3) +
    geom_hline(aes(yintercept = CenterLine), linetype = "dashed") +
    labs(title = title, x = "Time", y = value_col, color = "Classification") + # Legend title
    theme_bw() +
        scale_color_manual(values = c("Normal" = "black",
                                    "Outlier" = "red",
                                    "Step Change" = "blue",
                                    "Mean Shift (Rolling)" = "orange",
                                    "SD Shift (Rolling)" = "purple",
                                    "Mean/Var Change (CP)" = "darkgreen",
                                    "Mean Change (CP)" = "green",
                                    "Var Change (CP)" = "lightgreen",
                                    "Distribution Change (ECP)" = "brown",
                                    "Oscillation" = "pink"))  # Explicit color mapping

  print(p)
  return(p)
}

# 5. Stationarity Check (No changes here)
is_stationary <- function(data, value_col = "Value", sig_level = 0.05) {
  adf_result <- adf.test(data[[value_col]], alternative = "stationary")
  is_stationary <- adf_result$p.value < sig_level
  return(is_stationary)
}

# 6. Main Function (Slightly modified)
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
        oscillation_points = list(list(start = 500, end=580, period = 20, amplitude = 10)),
        outlier_points = c(50, 150, 480),
        missing_points = c(100, 101, 350,351,352)
    )

  # --- Preprocess Data ---
  preprocessed_data <- preprocess_soch_data(synthetic_data)

  # --- Check for Stationarity ---
  if (is_stationary(preprocessed_data)) {
    cat("Data is likely stationary.\n")
    analyzed_data <- detect_anomalies(preprocessed_data)
    plot_soch_with_anomalies(analyzed_data, value_col = "Value")
  } else {
    cat("Data is likely non-stationary.\n")
    preprocessed_data$Detrended_Value <- residuals(lm(Value ~ Time, data = preprocessed_data))
        preprocessed_data$Differenced_Seasonal_Value <- c(rep(NA, 7), diff(preprocessed_data$Detrended_Value, lag = 7))
    analyzed_data <- detect_anomalies(preprocessed_data, value_col = "Differenced_Seasonal_Value")
    plot_soch_with_anomalies(analyzed_data, value_col = "Differenced_Seasonal_Value")
  }
    # --- Print Summary ---
    print(table(analyzed_data$Classification)) # Show counts of each classification

}

# Run the analysis
main()