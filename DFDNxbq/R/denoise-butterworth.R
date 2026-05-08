#' Butterworth low-pass filter
#'
#' 3rd-order Butterworth low-pass filter for suppressing high-frequency
#' motion artifacts. Applied per channel.
#'
#' @param data Multi-channel time series matrix (samples x channels)
#' @param sampling_rate Sampling frequency in Hz (default 238)
#' @param cutoff Cutoff frequency in Hz (default 15)
#' @param order Filter order (default 3)
#' @return Filtered matrix (samples x channels)
#' @export
butterworth_filter <- function(data, sampling_rate = 238, cutoff = 15, order = 3) {
  data <- as.matrix(data)
  n_samples <- nrow(data)
  n_channels <- ncol(data)

  nyquist <- sampling_rate / 2
  Wn <- cutoff / nyquist
  bf <- butter(order, Wn, type = "low")

  result <- matrix(NA, nrow = n_samples, ncol = n_channels)
  for (ch in seq_len(n_channels)) {
    x <- data[, ch]
    if (sum(!is.na(x)) > 3 * order) {
      result[, ch] <- filtfilt(bf, x)
    } else {
      result[, ch] <- x
    }
  }
  result
}
