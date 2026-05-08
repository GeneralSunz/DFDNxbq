# 卡尔曼滤波器（提取"姿态残差"分量）
# 基于dlm包实现，对每个通道独立进行卡尔曼滤波
# 输出包含: 滤波后信号 + 残差（创新序列）
# 残差分量可提升跌倒可区分性 (Liu & Lin, 2026)
#
# 输入: data - 多通道时间序列矩阵 (样本数 x 通道数)
#       dW - 状态转移噪声方差 (控制平滑程度)
#       dV - 观测噪声方差
# 输出: list with components:
#       $filtered - 滤波后信号 (样本数 x 通道数)
#       $residual - 残差/创新序列 (样本数 x 通道数)

kalman_filter <- function(data, dW = NULL, dV = NULL) {
  require(dlm)

  data <- as.matrix(data)
  n_samples <- nrow(data)
  n_channels <- ncol(data)

  # 自动估计噪声方差
  if (is.null(dV)) {
    dV <- var(as.vector(data), na.rm = TRUE) * 0.1
  }
  if (is.null(dW)) {
    dW <- dV * 0.01
  }

  # 构建局部水平模型: y_t = mu_t + v_t, mu_t = mu_{t-1} + w_t
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

    # 尝试MLE估计参数，若失败则使用默认值
    fit <- tryCatch({
      mle <- dlmMLE(x_valid, par = log(c(dV, dW)), build = build_dlm)
      mod <- build_dlm(mle$par)
      dlmFilter(x_valid, mod)
    }, error = function(e) {
      mod <- dlmModPoly(order = 1, dV = dV, dW = dW)
      dlmFilter(x_valid, mod)
    })

    # 提取滤波后的状态 (mu_t)
    filtered <- drop(fit$m[-1])  # 去掉初始状态

    # 计算残差 (创新 / 新息)
    # 残差 = 观测值 - 一步预测
    residual <- drop(fit$f) - x_valid

    # 将结果填回原位置
    result_filtered[valid, ch] <- filtered
    result_residual[valid, ch] <- residual
  }

  return(list(filtered = result_filtered, residual = result_residual))
}
