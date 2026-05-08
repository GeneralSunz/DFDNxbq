# 小波软阈值去噪
# 使用db4或sym8小波进行多分辨率分解，对非平稳噪声鲁棒
# 能保留跌倒冲击的瞬态峰值
#
# 输入: data - 多通道时间序列矩阵 (样本数 x 通道数)
#       wavelet - 小波基，默认"d4" (Daubechies 4)
#                可选 "d4", "d6", "d8", "s4", "s8" (symlet)
#       n_level - 分解层数，默认4
#       threshold_type - 阈值类型: "soft" (软阈值) 或 "hard" (硬阈值)
#       universal - 是否使用通用阈值，默认TRUE
# 输出: 去噪后的矩阵 (样本数 x 通道数)

wavelet_denoise <- function(data, wavelet = "d4", n_level = 4,
                            threshold_type = "soft", universal = TRUE) {
  require(wavelets)

  data <- as.matrix(data)
  n_samples <- nrow(data)
  n_channels <- ncol(data)

  # 确保样本长度为2的幂，不足则补零
  target_len <- 2^ceiling(log2(n_samples))

  result <- matrix(NA, nrow = n_samples, ncol = n_channels)

  for (ch in seq_len(n_channels)) {
    x <- data[, ch]
    valid <- !is.na(x)

    if (sum(valid) < 2^n_level) {
      result[, ch] <- x
      next
    }

    # 用均值填充NA以便小波变换
    x_filled <- x
    x_filled[!valid] <- mean(x, na.rm = TRUE)

    # 若长度不是2的幂，补零
    if (n_samples < target_len) {
      x_filled <- c(x_filled, rep(0, target_len - n_samples))
    }

    # 离散小波变换
    wt <- dwt(x_filled, filter = wavelet, n.levels = n_level)

    # 提取各层细节系数 (W是列表, W[[level]]对应第level层)
    for (level in seq_len(n_level)) {
      details <- wt@W[[level]]

      # 估计噪声标准差 (使用MAD估计)
      sigma <- mad(details, center = 0)

      # 计算阈值
      if (universal) {
        thr <- sigma * sqrt(2 * log(length(details)))
      } else {
        thr <- sigma * sqrt(2 * log(length(details))) * 0.8
      }

      # 应用阈值
      if (threshold_type == "soft") {
        details <- sign(details) * pmax(abs(details) - thr, 0)
      } else {
        details <- details * (abs(details) > thr)
      }

      wt@W[[level]] <- details
    }

    # 重构信号
    denoised <- idwt(wt)

    # 裁剪回原始长度
    if (length(denoised) > n_samples) {
      denoised <- denoised[seq_len(n_samples)]
    }

    result[valid, ch] <- denoised[valid]
    result[!valid, ch] <- x[!valid]
  }

  return(result)
}
