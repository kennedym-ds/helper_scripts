library(ggplot2)
library(zoo)
library(changepoint)
library(imputeTS)
library(tseries)
library(ecp)

# 1. Generate Synthetic Soch Chart Data (Enhanced)
generate_synthetic_soch_data <- function(n = 500,
                                         base_mean = 100,
                                         base_sd = 5,
                                         ucl = NULL,  # Can be NULL for no limits
                                         lcl = NULL,  # Can be NULL for no limits
                                         seed = 42,
                                         shift_points = NULL,
                                         trend_params = NULL, # list(slope = , start =, end=)
                                         sd_shift_points = NULL,
                                         oscillation_params = NULL,  #list(amplitude =, period =, start=, end=)
                                         outlier_points = NULL,
                                         missing_points = NULL,
                                         stationary = TRUE) { # Added stationary flag
    set.seed(seed)
    data <- data.frame(Time = 1:n,
                     Value = rnorm(n, mean = base_mean, sd = base_sd))

    # Handle No Limits Case
    if (is.null(ucl)) {
        data$UCL <- NA_real_  # Use NA for missing limits
    } else {
        data$UCL <- ucl
    }
    if (is.null(lcl)) {
        data$LCL <- NA_real_
    } else {
        data$LCL <- lcl
    }
    data$CenterLine <- base_mean


    # Introduce Mean Shifts (even if non-stationary)
    if (!is.null(shift_points)) {
        for (shift in shift_points) {
            data$Value[shift$start:shift$end] <- rnorm(shift$end - shift$start + 1, shift$mean, base_sd)
        }
    }


    # Introduce Trend (if non-stationary)
    if (!stationary && !is.null(trend_params)) {
        trend_length = trend_params$end - trend_params$start + 1
        trend_values <- seq(0, trend_params$slope * (trend_length - 1), length.out = trend_length)
        data$Value[trend_params$start:trend_params$end] <- data$Value[trend_params$start:trend_params$end] + trend_values
    }

    #Introduce Standard Deviation shifts
    if(!is.null(sd_shift_points)){
        for(sd_shift in sd_shift_points){
            data$Value[sd_shift$start:sd_shift$end] <- rnorm(sd_shift$end-sd_shift$start+1, mean=base_mean, sd = base_sd*sd_shift$multiplier)
        }
    }

    # Introduce Oscillations
    if (!stationary && !is.null(oscillation_params)) {
        oscillation_length = oscillation_params$end - oscillation_params$start + 1
        oscillation_values <- oscillation_params$amplitude * sin(2*pi* (1:oscillation_length) / oscillation_params$period)
        data$Value[oscillation_params$start:oscillation_params$end] <- data$Value[oscillation_params$start:oscillation_params$end] + oscillation_values

    }

    # Introduce Outliers
    if (!is.null(outlier_points)) {
        for (outlier in outlier_points) {
            data$Value[outlier] <- data$Value[outlier] + rnorm(1, mean = 5 * base_sd, sd = base_sd)
        }
    }

    # Introduce Missing Values
    if (!is.null(missing_points)) {
        data$Value[missing_points] <- NA
    }

    return(data)
}

# 2. Preprocess Soch Data (Handles NA limits)
preprocess_soch_data <- function(data) {
    data$Value <- na_interpolation(data$Value, option = "linear")
    return(data)
}

# 3. Detect and Classify Anomalies (Handles NA limits)
detect_anomalies <- function(data, window_size = 30, k_step = 3, k_roll = 2.5, time_window = 100, acf_lag_max = 50, value_col = "Value") {

    value <- data[[value_col]]
    data$Classification <- "Normal"

    # 1. Out-of-Control Points (Handle NA limits)
    if (!all(is.na(data$UCL))) {  # Check if UCL exists
        outlier_indices_ucl <- which(data$Value > data$UCL)
        data$Classification[outlier_indices_ucl] <- "Outlier"
    }
    if (!all(is.na(data$LCL))) {  # Check if LCL exists
        outlier_indices_lcl <- which(data$Value < data$LCL)
        data$Classification[outlier_indices_lcl] <- "Outlier"
    }


    # 2. Step Changes
    step_change_indices <- which(c(NA, abs(diff(value))) > (k_step * sd(value, na.rm = TRUE)))
    data$Classification[step_change_indices] <- "Step Change"

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
    mean_changes <- cpt.mean(value, method = "PELT", penalty = "BIC", minseglen = 2)
    var_changes <- cpt.var(value, method = "PELT", penalty = "BIC", minseglen = 2)
    meanvar_changes <- cpt.meanvar(value, method = "PELT", penalty = "BIC", minseglen = 2)

    # 5. Change Point Detection (Non-Parametric - using ecp)
    ecp_results <- e.divisive(as.matrix(value), sig.lvl = 0.05)
    ecp_change_points <- ecp_results$estimates[c(-1, -length(ecp_results$estimates))]

    # Classify based on change points
    if (length(cpts(meanvar_changes)) > 0) {
        for (cp in cpts(meanvar_changes)) {
            if ((max(data$Time) - data$Time[cp]) <= time_window) {
                data$Classification[cp] <- "Mean/Var Change (CP)"
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
                data$Classification[cp] <- "Distribution Change (ECP)"
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

# 4. Plot Soch Chart with Anomalies (No changes here)
plot_soch_with_anomalies <- function(data, value_col, title = "Soch Chart with Anomaly Detection") {
    p <- ggplot(data, aes(x = Time, y = .data[[value_col]])) +
        geom_line() +
        geom_point(aes(color = Classification)) +
        geom_ribbon(aes(ymin = LCL, ymax = UCL), fill = "grey", alpha = 0.3) +  # Will be empty if no limits
        geom_hline(aes(yintercept = CenterLine), linetype = "dashed") +
        labs(title = title, x = "Time", y = value_col, color = "Classification") +
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
                                      "Oscillation" = "pink"))

    print(p)
    return(p)
}

# 5. Stationarity Check (No changes here)
is_stationary <- function(data, value_col = "Value", sig_level = 0.05) {
    adf_result <- adf.test(data[[value_col]], alternative = "stationary")
    is_stationary <- adf_result$p.value < sig_level
    return(is_stationary)
}

# 6. Main Function (Demonstrates different scenarios)
main <- function() {
    # --- Scenario 1: Stationary Data with Limits ---
    cat("\n--- Scenario 1: Stationary Data with Limits ---\n")
    synthetic_data_s_l <- generate_synthetic_soch_data(
        n = 200, base_mean = 50, base_sd = 4, ucl = 62, lcl = 38,
        shift_points = list(list(start = 100, end = 120, mean = 58)),
        outlier_points = c(50, 150)
    )
    preprocessed_data_s_l <- preprocess_soch_data(synthetic_data_s_l)
    if (is_stationary(preprocessed_data_s_l)) {
        analyzed_data_s_l <- detect_anomalies(preprocessed_data_s_l)
        plot_soch_with_anomalies(analyzed_data_s_l, value_col = "Value", title = "Scenario 1: Stationary with Limits")
        print(table(analyzed_data_s_l$Classification))
    }


    # --- Scenario 2: Non-Stationary Data with Limits ---
    cat("\n--- Scenario 2: Non-Stationary Data with Limits ---\n")
    synthetic_data_ns_l <- generate_synthetic_soch_data(
        n = 200, base_mean = 50, base_sd = 4, ucl = 65, lcl = 35,
        stationary = FALSE,
        trend_params = list(slope = 0.1, start = 1, end = 200),
        oscillation_params = list(amplitude = 8, period = 20, start = 1, end = 200),
        outlier_points = c(50, 150)
    )
    preprocessed_data_ns_l <- preprocess_soch_data(synthetic_data_ns_l)

    if (!is_stationary(preprocessed_data_ns_l)) {
        preprocessed_data_ns_l$Detrended_Value <- residuals(lm(Value ~ Time, data = preprocessed_data_ns_l))
        preprocessed_data_ns_l$Differenced_Seasonal_Value <- c(rep(NA, 20), diff(preprocessed_data_ns_l$Detrended_Value, lag = 20))  # Match oscillation period
        analyzed_data_ns_l <- detect_anomalies(preprocessed_data_ns_l, value_col = "Differenced_Seasonal_Value")
        plot_soch_with_anomalies(analyzed_data_ns_l, value_col = "Differenced_Seasonal_Value", title = "Scenario 2: Non-Stationary with Limits")
        print(table(analyzed_data_ns_l$Classification))
    }

    # --- Scenario 3: Stationary Data without Limits ---
    cat("\n--- Scenario 3: Stationary Data without Limits ---\n")
    synthetic_data_s_nl <- generate_synthetic_soch_data(
        n = 200, base_mean = 50, base_sd = 4, ucl = NULL, lcl = NULL,  # No limits
        shift_points = list(list(start = 100, end = 120, mean = 58))
    )
    preprocessed_data_s_nl <- preprocess_soch_data(synthetic_data_s_nl)
    if (is_stationary(preprocessed_data_s_nl)) {
        analyzed_data_s_nl <- detect_anomalies(preprocessed_data_s_nl)
        plot_soch_with_anomalies(analyzed_data_s_nl, value_col = "Value", title = "Scenario 3: Stationary without Limits")
        print(table(analyzed_data_s_nl$Classification))
    }

    # --- Scenario 4: Non-Stationary Data without Limits ---
    cat("\n--- Scenario 4: Non-Stationary Data without Limits ---\n")
    synthetic_data_ns_nl <- generate_synthetic_soch_data(
        n = 200,
        base_mean = 50,
        base_sd = 4,
        ucl = NULL,  # No limits
        lcl = NULL,  # No limits
        stationary = FALSE,  # Explicitly non-stationary
        trend_params = list(slope = -0.15, start = 50, end = 150) #Add a trend
    )
    preprocessed_data_ns_nl <- preprocess_soch_data(synthetic_data_ns_nl)

    if (!is_stationary(preprocessed_data_ns_nl)) {
        preprocessed_data_ns_nl$Detrended_Value <- residuals(lm(Value ~ Time, data = preprocessed_data_ns_nl))
        analyzed_data_ns_nl <- detect_anomalies(preprocessed_data_ns_nl, value_col = "Detrended_Value")  # Analyze detrended
        plot_soch_with_anomalies(analyzed_data_ns_nl, value_col = "Detrended_Value", title = "Scenario 4: Non-Stationary without Limits") #Plot detrended
        print(table(analyzed_data_ns_nl$Classification))
    }
}

# Run the analysis
main()