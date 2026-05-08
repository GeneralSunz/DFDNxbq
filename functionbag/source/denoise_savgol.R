# Savitzky–Golay滤波器
# 在平滑的同时保持信号形状（峰值、谷值等）
# 适用于二次精细平滑
#
# 输入: data - 多通道时间序列矩阵 (样本数 x 通道数)
#       p - 多项式阶数，默认3
#       n - 窗口宽度（必须为奇数），默认11
# 输出: 滤波后的矩阵 (样本数 x 通道数)

savgol_filter <- function(data, p = 3, n = NULL) {
  require(signal)

  data <- as.matrix(data)
  n_samples <- nrow(data)
  n_channels <- ncol(data)

  # 默认窗口宽度：取采样点的5%或11，取较大奇数
  if (is.null(n)) {
    n <- max(11, round(n_samples * 0.05))
    if (n %% 2 == 0) n <- n + 1
    n <- min(n, n_samples - 1)
  }

  # n必须是奇数且大于p
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

    # 对有效数据进行SG滤波
    x_valid <- x[valid]

    # 使用signal::sgolayfilt
    if (length(x_valid) >= n + 5) {
      filtered <- sgolayfilt(x_valid, p = p, n = n)
    } else {
      # 数据太短，降阶处理
      n_actual <- min(n, length(x_valid) - 1)
      if (n_actual %% 2 == 0) n_actual <- n_actual - 1
      if (n_actual > p) {
        filtered <- sgolayfilt(x_valid, p = min(p, n_actual - 2), n = n_actual)
      } else {
        filtered <- x_valid
      }
    }

    result[valid, ch] <- filtered
    # 保留无效值位置
    result[!valid, ch] <- if (any(!valid)) x[!valid] else NA
  }

  return(result)
}
