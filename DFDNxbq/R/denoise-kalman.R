#' Kalman filter for posture-residual extraction
#'
#' Implements a local-level Kalman filter via the \pkg{dlm} package.
#' Returns both the filtered signal and the residual (innovation) sequence.
#' The residual component can enhance fall discriminability.
#'
#' @param data Multi-channel time series matrix (samples x channels)
#' @param dW State transition noise variance (controls smoothness).
#'   If NULL, estimated automatically as \code{dV * 0.01}.
#' @param dV Observation noise variance.
#'   If NULL, estimated as \code{var(data) * 0.1}.
#' @return List with components:
#'   \item{filtered}{Filtered signal matrix (samples x channels)}
#'   \item{residual}{Innovation sequence matrix (samples x channels)}
#' @export
kalman_filter <- function(data, dW = NULL, dV = NULL) {
  data <- as.matrix(data)
  n_samples <- nrow(data)
  n_channels <- ncol(data)

  if (is.null(dV)) {
    dV <- var(as.vector(data), na.rm = TRUE) * 0.1
  }
  if (is.null(dW)) {
    dW <- dV * 0.01
  }

  build_dlm <- function(par) {
    dlmModPoly(order = 1, dV = exp(par[1]), dW = exp(par[2]))
  }

  result_filtered <- matrix(NA, nrow = n_samples, ncol = n_channels)
  result_residual <- matrix(NA, nrow = n_samples, ncol = n_channels)

  for (ch in seq_len(n_channels)) {
    x <- data[, ch]
    valid <- !is.na(x)

    if (sum(valid) < 10) {
      result_filtered[, ch] <- x
      result_residual[, ch] <- rep(0, n_samples)
      next
    }

    x_valid <- x[valid]

    fit <- tryCatch({
      mle <- dlmMLE(x_valid, par = log(c(dV, dW)), build = build_dlm)
      mod <- build_dlm(mle$par)
      dlmFilter(x_valid, mod)
    }, error = function(e) {
      mod <- dlmModPoly(order = 1, dV = dV, dW = dW)
      dlmFilter(x_valid, mod)
    })

    filtered <- drop(fit$m[-1])
    residual <- drop(fit$f) - x_valid

    result_filtered[valid, ch] <- filtered
    result_residual[valid, ch] <- residual
  }

  list(filtered = result_filtered, residual = result_residual)
}
