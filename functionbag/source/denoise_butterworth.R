# Butterworth低通滤波器
# 3阶巴特沃斯低通滤波器，用于抑制高频运动伪影
# 输入: data - 多通道时间序列矩阵 (样本数 x 通道数)
#       sampling_rate - 采样率 (Hz)，默认238
#       cutoff - 截止频率 (Hz)，默认15
#       order - 滤波器阶数，默认3
# 输出: 滤波后的矩阵 (样本数 x 通道数)

butterworth_filter <- function(data, sampling_rate = 238, cutoff = 15, order = 3) {
  require(signal)

  # 将输入转为矩阵
  data <- as.matrix(data)
  n_samples <- nrow(data)
  n_channels <- ncol(data)

  # 设计巴特沃斯低通滤波器
  nyquist <- sampling_rate / 2
  Wn <- cutoff / nyquist
  bf <- butter(order, Wn, type = "low")

  # 对每个通道进行零相位滤波
  result <- matrix(NA, nrow = n_samples, ncol = n_channels)
  for (ch in seq_len(n_channels)) {
    x <- data[, ch]
    # 检查是否有足够的有效数据
    if (sum(!is.na(x)) > 3 * order) {
      result[, ch] <- filtfilt(bf, x)
    } else {
      result[, ch] <- x
    }
  }

  return(result)
}
