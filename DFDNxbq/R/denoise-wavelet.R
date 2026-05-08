#' Wavelet soft/hard threshold denoising
#'
#' Multi-resolution decomposition using Daubechies or Symlet wavelets.
#' Robust to non-stationary noise while preserving transient fall impacts.
#'
#' @param data Multi-channel time series matrix (samples x channels)
#' @param wavelet Wavelet basis: "d4", "d6", "d8", "s4", "s8" (default "d4")
#' @param n_level Decomposition level (default 4)
#' @param threshold_type "soft" or "hard" thresholding (default "soft")
#' @param universal Use universal threshold? (default TRUE)
#' @return Denoised matrix (samples x channels)
#' @export
wavelet_denoise <- function(data, wavelet = "d4", n_level = 4,
                            threshold_type = "soft", universal = TRUE) {
  data <- as.matrix(data)
  n_samples <- nrow(data)
  n_channels <- ncol(data)

  target_len <- 2^ceiling(log2(n_samples))

  result <- matrix(NA, nrow = n_samples, ncol = n_channels)

  for (ch in seq_len(n_channels)) {
    x <- data[, ch]
    valid <- !is.na(x)

    if (sum(valid) < 2^n_level) {
      result[, ch] <- x
      next
    }

    x_filled <- x
    x_filled[!valid] <- mean(x, na.rm = TRUE)

    if (n_samples < target_len) {
      x_filled <- c(x_filled, rep(0, target_len - n_samples))
    }

    wt <- dwt(x_filled, filter = wavelet, n.levels = n_level)

    for (level in seq_len(n_level)) {
      details <- wt@W[[level]]
      sigma <- mad(details, center = 0)
      if (universal) {
        thr <- sigma * sqrt(2 * log(length(details)))
      } else {
        thr <- sigma * sqrt(2 * log(length(details))) * 0.8
      }
      if (threshold_type == "soft") {
        details <- sign(details) * pmax(abs(details) - thr, 0)
      } else {
        details <- details * (abs(details) > thr)
      }
      wt@W[[level]] <- details
    }

    denoised <- idwt(wt)
    if (length(denoised) > n_samples) {
      denoised <- denoised[seq_len(n_samples)]
    }

    result[valid, ch] <- denoised[valid]
    result[!valid, ch] <- x[!valid]
  }

  result
}
