#' Savitzky–Golay filter
#'
#' Smooths while preserving signal shape (peaks, valleys).
#' Suitable for secondary fine smoothing.
#'
#' @param data Multi-channel time series matrix (samples x channels)
#' @param p Polynomial order (default 3)
#' @param n Window width (must be odd; default 5\% of samples, min 11)
#' @return Filtered matrix (samples x channels)
#' @export
savgol_filter <- function(data, p = 3, n = NULL) {
  data <- as.matrix(data)
  n_samples <- nrow(data)
  n_channels <- ncol(data)

  if (is.null(n)) {
    n <- max(11, round(n_samples * 0.05))
    if (n %% 2 == 0) n <- n + 1
    n <- min(n, n_samples - 1)
  }

  if (n %% 2 == 0) n <- n + 1
  if (n <= p) n <- p + 2
  if (n %% 2 == 0) n <- n + 1

  result <- matrix(NA, nrow = n_samples, ncol = n_channels)

  for (ch in seq_len(n_channels)) {
    x <- data[, ch]
    valid <- !is.na(x)

    if (sum(valid) < n) {
      result[, ch] <- x
      next
    }

    x_valid <- x[valid]

    if (length(x_valid) >= n + 5) {
      filtered <- sgolayfilt(x_valid, p = p, n = n)
    } else {
      n_actual <- min(n, length(x_valid) - 1)
      if (n_actual %% 2 == 0) n_actual <- n_actual - 1
      if (n_actual > p) {
        filtered <- sgolayfilt(x_valid, p = min(p, n_actual - 2), n = n_actual)
      } else {
        filtered <- x_valid
      }
    }

    result[valid, ch] <- filtered
    result[!valid, ch] <- if (any(!valid)) x[!valid] else NA
  }

  result
}
