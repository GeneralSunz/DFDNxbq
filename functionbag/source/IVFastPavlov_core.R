# ============================================================
# IVFastPavlov_core.R – 核心函数定义
# 从 IVFastPavlov_Amendment.r 提取 (BASE_CONFIG + 函数定义)
# 移除: rm(list=ls()), library(), set.seed(), OUTPUT_DIR
# ============================================================

Sys.setenv(TORCH_VERIFY_LOAD = "FALSE")

# ============================================================
# 0. 全局基础配置 (各实验共享的参数)
# ============================================================
BASE_CONFIG <- list(
  fs = 238,
  data_path = "output/fallalld_denoised.rds",
  sp_ratio = 0.01,
  fast_subjects = c(1,2,3,5,6,7,8,9,10,11,12,14,15),
  myseed = 124514,
  L_pre_ms = 800,
  delta_ms = 50,
  # 动态分块
  enable_dynamic_patching = TRUE,
  patch_short_len = 8,
  patch_long_len = 24,
  patch_overlap = 1,
  patch_min = 4,
  patch_max = 16,
  patch_min_len = 4,
  patch_complexity_smooth = 5,
  # 编码器 & HIBA
  d_model = 16,
  encoder_type = "conv_lstm",  # "cnn", "conv_lstm", "gru_tcn_se"
  use_hiba = TRUE,
  hiba_group_size = 3,
  hiba_n_heads = 2,
  hiba_dropout = 0.4,
  # 网络架构
  n_filters = 16,
  n_filters2 = 32,
  kernel_size = 3,
  dropout = 0.3,
  gru_hidden = 16,
  n_tcn_blocks = 4,
  # 训练
  n_epochs = 30,
  batch_size = 32,
  dl_patience = 7,
  lr = 3e-4,
  val_ratio = 0.3,
  min_deep_samples = 10,
  weight_decay = 5e-3,
  lr_scheduler_patience = 8,
  lr_scheduler_factor = 0.7,
  grad_clip_norm = 1.0,
  focal_alpha = 0.5,
  focal_gamma = 1.0,
  focal_gamma_warmup = 8,  # 前 N 个 epoch 用 gamma=0 (BCE) 预热
 
  denoise_method = "none",  # "none", "butterworth", "kalman", "wavelet", "savgol"
  denoise_bw_cutoff = 15,
  denoise_bw_order = 3,
  denoise_kalman_dV = NULL,
  denoise_kalman_dW = NULL,
  denoise_wavelet_name = "d4",
  denoise_wavelet_level = 4,
  denoise_wavelet_threshold = "soft",
  denoise_sg_p = 3,
  denoise_sg_n = NULL,
  # 分类器
  do_oversample = TRUE,
  use_global_features = FALSE,
  to_use_deep = TRUE,
  rf_ntree = 100,
  rf_classwt_adl = 1,
  rf_classwt_fall = 3,
  svm_cost = 1,
  svm_classwt_fall = 0,  # 0 = auto (n_neg/n_pos)
  svm_kernel = "linear",
  svm_tune = FALSE,
  tune_classifiers = TRUE,   # TRUE 时对所有分类器做验证集网格调参
  svm_tune_grid = list(kernel = c("linear", "radial"), cost = c(0.5, 1, 5)),
  cb_tune_grid = list(iterations = c(100, 300), depth = c(4, 6), lr = c(0.1, 0.05)),
  rf_tune_grid = list(ntree = c(100, 200), mtry_ratio = c(0.33, 0.5)),
  cb_iterations = 500,
  cb_depth = 6,
  cb_learning_rate = 0.1,
  cb_subsample = 0.8,
  cb_colsample_bylevel = 0.8,
  cb_l2_leaf_reg = 3,
  stacking_grid_coarse = 0.2,
  stacking_grid_fine = 0.05,
  # PCA (全局特征)
  pca_var_ratio = 0.95,
  pca_max_dim = 20,
  # 因果手工特征
  use_causal_features = FALSE,
  causal_window_ms = c(50, 100, 200, 400)
)

# ============================================================
# 1. 核心函数 (源自 IIFastPavlov_Amendment.r)
# ============================================================

# ---- 1a. 数据加载 ----
load_data <- function(rds_path) {
  if (!file.exists(rds_path)) stop(sprintf("RDS not found: %s", rds_path))
  data_list <- readRDS(rds_path)
  cat(sprintf("  加载完成: %d 个窗口样本\n", length(data_list)))
  data_list
}

build_labels <- function(data_list) {
  sapply(data_list, function(e) isTRUE(e$IsFall))
}

build_subjects <- function(data_list) {
  sapply(data_list, function(e) e$SubjectID)
}

# ---- 1b. 降噪 ----
denoise_r_dir <- if (dir.exists("../R")) "../R" else "R"

if (file.exists(file.path(denoise_r_dir, "denoise_butterworth.R"))) {
  source(file.path(denoise_r_dir, "denoise_butterworth.R"), local = TRUE)
} else {
  butterworth_filter <- function(data, sampling_rate = 238, cutoff = 15, order = 3) {
    if (!requireNamespace("signal", quietly = TRUE)) return(data)
    data <- as.matrix(data); nyq <- sampling_rate / 2
    bf <- signal::butter(order, cutoff / nyq, type = "low")
    result <- data
    for (ch in seq_len(ncol(data))) result[, ch] <- signal::filtfilt(bf, data[, ch])
    result
  }
}
if (file.exists(file.path(denoise_r_dir, "denoise_kalman.R"))) {
  source(file.path(denoise_r_dir, "denoise_kalman.R"), local = TRUE)
} else {
  kalman_filter <- function(data, dW = NULL, dV = NULL) {
    list(filtered = data, residual = data * 0)
  }
}
if (file.exists(file.path(denoise_r_dir, "denoise_wavelet.R"))) {
  source(file.path(denoise_r_dir, "denoise_wavelet.R"), local = TRUE)
} else {
  wavelet_denoise <- function(data, ...) data
}
if (file.exists(file.path(denoise_r_dir, "denoise_savgol.R"))) {
  source(file.path(denoise_r_dir, "denoise_savgol.R"), local = TRUE)
} else {
  savgol_filter <- function(data, p = 3, n = NULL) data
}

apply_denoise <- function(data_list, method = "kalman", fs = 238,
                          bw_cutoff = 15, bw_order = 3,
                          kalman_dV = NULL, kalman_dW = NULL,
                          wavelet_name = "d4", wavelet_level = 4,
                          wavelet_threshold = "soft",
                          sg_p = 3, sg_n = NULL) {
  if (method == "none") return(data_list)
  field_map <- list(butterworth = c("Acc_Butterworth", "Gyr_Butterworth"),
                    kalman = c("Acc_Kalman", "Gyr_Kalman"),
                    wavelet = c("Acc_Wavelet", "Gyr_Wavelet"),
                    savgol = c("Acc_SavGol", "Gyr_SavGol"))
  acc_field <- field_map[[method]][1]; gyr_field <- field_map[[method]][2]
  has_precomputed <- !is.null(data_list[[1]][[acc_field]]) && !is.null(data_list[[1]][[gyr_field]])
  n <- length(data_list)
  cat(sprintf("  应用 %s 降噪 (%d 样本) ...\n", method, n))
  if (has_precomputed) {
    cat(sprintf("  直接读取 %s / %s\n", acc_field, gyr_field))
    for (i in seq_len(n)) {
      data_list[[i]]$Acc_raw <- data_list[[i]][[acc_field]]
      data_list[[i]]$Gyr_raw <- data_list[[i]][[gyr_field]]
    }
  } else {
    pb <- txtProgressBar(min = 0, max = n, style = 3, width = 50)
    for (i in seq_len(n)) {
      if (method == "butterworth") {
        data_list[[i]]$Acc_raw <- butterworth_filter(data_list[[i]]$Acc_raw, fs, bw_cutoff, bw_order)
        data_list[[i]]$Gyr_raw <- butterworth_filter(data_list[[i]]$Gyr_raw, fs, bw_cutoff, bw_order)
      } else if (method == "kalman") {
        ar <- kalman_filter(data_list[[i]]$Acc_raw, kalman_dW, kalman_dV)
        gr <- kalman_filter(data_list[[i]]$Gyr_raw, kalman_dW, kalman_dV)
        data_list[[i]]$Acc_raw <- ar$filtered; data_list[[i]]$Gyr_raw <- gr$filtered
      } else if (method == "wavelet") {
        data_list[[i]]$Acc_raw <- wavelet_denoise(data_list[[i]]$Acc_raw, wavelet = wavelet_name,
                                                    n_level = wavelet_level, threshold_type = wavelet_threshold)
        data_list[[i]]$Gyr_raw <- wavelet_denoise(data_list[[i]]$Gyr_raw, wavelet = wavelet_name,
                                                    n_level = wavelet_level, threshold_type = wavelet_threshold)
      } else if (method == "savgol") {
        data_list[[i]]$Acc_raw <- savgol_filter(data_list[[i]]$Acc_raw, sg_p, sg_n)
        data_list[[i]]$Gyr_raw <- savgol_filter(data_list[[i]]$Gyr_raw, sg_p, sg_n)
      }
      setTxtProgressBar(pb, i)
    }
    close(pb)
  }
  data_list
}

# ---- 1c. 峰值锚点 & 滑动窗口 ----
compute_peak_anchor <- function(acc_mat) {
  if (is.null(acc_mat) || nrow(acc_mat) == 0 || ncol(acc_mat) < 3) return(NA)
  svm <- sqrt(rowSums(acc_mat^2))
  which.max(svm)
}

generate_sliding_windows <- function(data_item, delta_ms, L_pre, fs = 238) {
  peak_anchor <- compute_peak_anchor(data_item$Acc_raw)
  if (is.na(peak_anchor)) return(list(windows = list(), deltas = numeric()))
  n_total <- nrow(data_item$Acc_raw)
  n_delta <- round(delta_ms * fs / 1000)
  start_idx <- peak_anchor - L_pre - n_delta + 1
  end_idx   <- peak_anchor - n_delta
  windows <- list(); valid_deltas <- numeric()
  if (start_idx >= 1 && end_idx <= n_total && (end_idx - start_idx + 1) == L_pre) {
    acc_win <- data_item$Acc_raw[start_idx:end_idx, , drop = FALSE]
    gyr_win <- data_item$Gyr_raw[start_idx:end_idx, , drop = FALSE]
    windows <- list(list(data = cbind(acc_win, gyr_win), end_idx = end_idx,
                         peak_anchor = peak_anchor, win_start = start_idx))
    valid_deltas <- delta_ms
  }
  list(windows = windows, deltas = valid_deltas)
}

# ---- 1d. 复杂度评估 & 动态分块 ----
compute_complexity_profile <- function(signal, smooth_win = 5) {
  if (!is.matrix(signal) || nrow(signal) < 2 || ncol(signal) < 1) return(rep(0.5, NROW(signal)))
  T <- nrow(signal); C <- ncol(signal)
  grad_mag <- matrix(0, nrow = T, ncol = C)
  for (ch in 1:C) {
    grad_mag[, ch] <- c(0, abs(diff(signal[, ch])))
  }
  raw_complexity <- rowSums(grad_mag)
  if (smooth_win > 1 && T > smooth_win) {
    half <- floor(smooth_win / 2); smoothed <- numeric(T)
    for (t in 1:T) {
      left <- max(1, t - half); right <- min(T, t + half)
      smoothed[t] <- mean(raw_complexity[left:right])
    }
    raw_complexity <- smoothed
  }
  c_min <- min(raw_complexity); c_max <- max(raw_complexity)
  if (c_max > c_min) (raw_complexity - c_min) / (c_max - c_min) else rep(0.5, T)
}

adaptive_patching <- function(signal, complexity,
                               short_len = 8, long_len = 24,
                               overlap = 2, min_patches = 4,
                               max_patches = 16, min_patch_len = 4) {
  T <- nrow(signal); C <- ncol(signal)
  if (T < short_len + long_len) return(list(signal))
  threshold <- median(complexity)
  is_high <- complexity > threshold
  patches <- list(); pos <- 1
  while (pos <= T && length(patches) < max_patches) {
    if (pos > length(is_high)) break
    current_len <- if (is_high[pos]) short_len else long_len
    block_end <- min(pos + current_len - 1, T)
    actual_len <- block_end - pos + 1
    if (actual_len >= 2) {
      if (actual_len < min_patch_len && length(patches) >= 1) {
        prev_idx <- length(patches)
        patches[[prev_idx]] <- rbind(patches[[prev_idx]], signal[pos:block_end, , drop = FALSE])
      } else {
        patches[[length(patches) + 1]] <- signal[pos:block_end, , drop = FALSE]
      }
    }
    step <- current_len - overlap; if (step < 1) step <- 1
    pos <- pos + step
  }
  if (length(patches) < min_patches) {
    patches <- list(); uniform_len <- max(2, floor(T / min_patches))
    for (i in 1:min_patches) {
      start <- (i - 1) * uniform_len + 1
      end <- min(i * uniform_len, T)
      if (start <= T) patches[[i]] <- signal[start:end, , drop = FALSE]
    }
  }
  patches[!sapply(patches, is.null)]
}

uniform_patching <- function(signal, n_blocks = 8) {
  T <- nrow(signal); C <- ncol(signal)
  block_len <- max(2, floor(T / n_blocks))
  patches <- list()
  for (i in 1:n_blocks) {
    start <- (i - 1) * block_len + 1; end <- min(i * block_len, T)
    if (start <= T) patches[[i]] <- signal[start:end, , drop = FALSE]
  }
  patches[!sapply(patches, is.null)]
}

# ---- 1e. 神经网络模块 ----
depthwise_sep_conv1d <- nn_module("depthwise_sep_conv1d",
  initialize = function(in_channels, out_channels, kernel_size, padding = "same", bias = TRUE) {
    self$depthwise <- nn_conv1d(in_channels, in_channels, kernel_size, groups = in_channels, padding = padding, bias = bias)
    self$pointwise <- nn_conv1d(in_channels, out_channels, 1, bias = bias)
  },
  forward = function(x) { x <- self$depthwise(x); self$pointwise(x) }
)

se_block <- nn_module("se_block",
  initialize = function(channels, reduction = 4) {
    reduced <- max(1, floor(channels / reduction))
    self$gap <- nn_adaptive_avg_pool1d(1)
    self$fc1 <- nn_linear(channels, reduced)
    self$fc2 <- nn_linear(reduced, channels)
  },
  forward = function(x) {
    b <- x$size(1); c <- x$size(2)
    squeeze <- self$gap(x)$view(c(b, c))
    excite <- torch_relu(self$fc1(squeeze))
    excite <- torch_sigmoid(self$fc2(excite))$view(c(b, c, 1))
    x * excite
  }
)

tcn_block <- nn_module("tcn_block",
  initialize = function(channels, kernel_size = 3, dilation = 1, dropout = 0.5) {
    padding <- dilation * (kernel_size - 1) %/% 2
    self$conv <- nn_conv1d(channels, channels, kernel_size, padding = padding, dilation = dilation)
    self$bn   <- nn_group_norm(channels, channels)  # InstanceNorm equivalent (每组1通道)
    self$relu <- nn_relu()
    self$drop <- nn_dropout(dropout)
  },
  forward = function(x) {
    residual <- x
    out <- self$relu(self$bn(self$conv(x)))
    out <- self$drop(out)
    torch_relu(out + residual)
  }
)

# ---- 补丁编码器 ----
patch_encoder_cnn <- nn_module("patch_encoder_cnn",
  initialize = function(n_channels = 6, d_model = 64, n_filters = 8, n_filters2 = 16, kernel_size = 3) {
    self$conv1 <- nn_conv1d(n_channels, n_filters, kernel_size, padding = "same")
    self$bn1   <- nn_group_norm(n_filters, n_filters)  # InstanceNorm equivalent
    self$conv2 <- nn_conv1d(n_filters, n_filters2, kernel_size, padding = "same")
    self$bn2   <- nn_group_norm(n_filters2, n_filters2)  # InstanceNorm equivalent
    self$gap   <- nn_adaptive_avg_pool1d(1)
    self$dropout <- nn_dropout(0.3)
    self$proj  <- nn_linear(n_filters2, d_model)
  },
  forward = function(x) {
    ndim <- x$dim()
    if (ndim == 2) x <- x$unsqueeze(1)
    x <- x$permute(c(1, 3, 2))
    x <- torch_relu(self$bn1(self$conv1(x)))
    x <- torch_relu(self$bn2(self$conv2(x)))
    x <- self$gap(x)$squeeze(-1)
    x <- self$dropout(x)
    self$proj(x)$squeeze(1)
  }
)

patch_encoder_convlstm <- nn_module("patch_encoder_convlstm",
  initialize = function(n_channels = 6, d_model = 64, n_filters = 8, lstm_hidden = 16, kernel_size = 3) {
    self$conv1 <- nn_conv1d(n_channels, n_filters, kernel_size, padding = "same")
    self$bn1   <- nn_group_norm(n_filters, n_filters)  # InstanceNorm equivalent
    self$pool1 <- nn_max_pool1d(2, stride = 2, ceil_mode = TRUE)
    self$lstm  <- nn_lstm(n_filters, lstm_hidden, bidirectional = TRUE, batch_first = TRUE)
    self$dropout <- nn_dropout(0.3)
    self$proj  <- nn_linear(lstm_hidden * 2, d_model)
  },
  forward = function(x) {
    ndim <- x$dim()
    if (ndim == 2) x <- x$unsqueeze(1)
    x <- x$permute(c(1, 3, 2))
    x <- torch_relu(self$bn1(self$conv1(x)))
    x <- self$pool1(x)
    x <- x$permute(c(1, 3, 2))
    lstm_out <- self$lstm(x)[[1]]
    embedding <- lstm_out[, dim(lstm_out)[2], ]
    embedding <- self$dropout(embedding)
    self$proj(embedding)$squeeze(1)
  }
)

patch_encoder_gru_tcn_se <- nn_module("patch_encoder_gru_tcn_se",
  initialize = function(n_channels = 6, d_model = 64, n_filters = 8, n_filters2 = 16,
                        kernel_size = 3, gru_hidden = 16, n_tcn_blocks = 4, dropout = 0.5) {
    self$dsconv1 <- depthwise_sep_conv1d(n_channels, n_filters, kernel_size, padding = "same")
    self$bn1     <- nn_group_norm(n_filters, n_filters)  # InstanceNorm equivalent
    self$se1     <- se_block(n_filters, reduction = 4)
    self$pool1   <- nn_max_pool1d(2, stride = 2, ceil_mode = TRUE)
    self$dsconv2 <- depthwise_sep_conv1d(n_filters, n_filters2, kernel_size, padding = "same")
    self$bn2     <- nn_group_norm(n_filters2, n_filters2)  # InstanceNorm equivalent
    self$se2     <- se_block(n_filters2, reduction = 4)
    self$pool2   <- nn_max_pool1d(2, stride = 2, ceil_mode = TRUE)
    self$tcn_blocks <- nn_module_list()
    for (d in 2^(0:(n_tcn_blocks - 1))) {
      self$tcn_blocks$append(tcn_block(n_filters2, kernel_size, dilation = d, dropout = dropout))
    }
    self$gap  <- nn_adaptive_avg_pool1d(1)
    self$proj <- nn_linear(n_filters2, d_model)
  },
  forward = function(x) {
    ndim <- x$dim()
    if (ndim == 2) x <- x$unsqueeze(1)
    x <- x$permute(c(1, 3, 2))
    x <- self$dsconv1(x); x <- self$bn1(x); x <- torch_relu(x); x <- self$se1(x); x <- self$pool1(x)
    x <- self$dsconv2(x); x <- self$bn2(x); x <- torch_relu(x); x <- self$se2(x); x <- self$pool2(x)
    for (i in seq_along(self$tcn_blocks)) x <- self$tcn_blocks[[i]](x)
    x <- self$gap(x)$squeeze(-1)
    self$proj(x)$squeeze(1)
  }
)

create_patch_encoder <- function(encoder_type = "cnn", n_channels = 6, d_model = 64) {
  switch(encoder_type,
    cnn = patch_encoder_cnn(n_channels = n_channels, d_model = d_model),
    convlstm = patch_encoder_convlstm(n_channels = n_channels, d_model = d_model),
    gru_tcn_se = patch_encoder_gru_tcn_se(n_channels = n_channels, d_model = d_model),
    patch_encoder_cnn(n_channels = n_channels, d_model = d_model))
}

# ---- 1f. HIBA 注意力 ----
simple_mha <- nn_module("simple_mha",
  initialize = function(d_model, n_heads, dropout = 0.1) {
    self$n_heads <- n_heads; self$d_head <- d_model %/% n_heads
    self$d_model <- d_model; self$scale <- sqrt(self$d_head)
    self$wq <- nn_linear(d_model, d_model); self$wk <- nn_linear(d_model, d_model)
    self$wv <- nn_linear(d_model, d_model); self$wo <- nn_linear(d_model, d_model)
    self$dropout <- nn_dropout(dropout)
  },
  forward = function(query, key, value) {
    batch_size <- query$size(1); seq_len_q <- query$size(2); seq_len_k <- key$size(2)
    Q <- self$wq(query)$view(c(batch_size, seq_len_q, self$n_heads, self$d_head))$transpose(2, 3)
    K <- self$wk(key)$view(c(batch_size, seq_len_k, self$n_heads, self$d_head))$transpose(2, 3)
    V <- self$wv(value)$view(c(batch_size, seq_len_k, self$n_heads, self$d_head))$transpose(2, 3)
    scores <- Q$matmul(K$transpose(-2, -1)) / self$scale
    attn_weights <- nnf_softmax(scores, dim = -1)
    attn_weights <- self$dropout(attn_weights)
    out <- attn_weights$matmul(V)
    out <- out$transpose(2, 3)$reshape(c(batch_size, seq_len_q, self$d_model))
    out <- self$wo(out)
    list(out, attn_weights$mean(dim = 2, keepdim = TRUE))
  }
)

HIBA_attention <- nn_module("HIBA_attention",
  initialize = function(d_model = 64, n_heads = 4, group_size = 3, dropout = 0.1) {
    self$group_size <- group_size; self$d_model <- d_model; dtype <- torch_float32()
    self$intra_attn <- simple_mha(d_model, min(n_heads, max(1, d_model %/% 4)), dropout)
    self$intra_norm <- nn_layer_norm(d_model)
    self$inter_attn <- simple_mha(d_model, min(n_heads, max(1, d_model %/% 4)), dropout)
    self$inter_norm <- nn_layer_norm(d_model)
    self$ffn <- nn_sequential(nn_linear(d_model, d_model * 2), nn_relu(),
                               nn_dropout(dropout), nn_linear(d_model * 2, d_model), nn_dropout(dropout))
    self$ffn_norm <- nn_layer_norm(d_model)
    self$q_global <- nn_parameter(torch_randn(1, 1, d_model, dtype = dtype) * 0.5)
    self$global_attn <- simple_mha(d_model, 1, dropout)
  },
  forward = function(z) {
    device <- z$device; dtype <- torch_float32(); n_blocks <- z$size(1)
    if (n_blocks == 0) return(list(z_global = torch_zeros(self$d_model, device = device, dtype = dtype), alpha = torch_zeros(0, device = device, dtype = dtype)))
    if (n_blocks == 1) return(list(z_global = z$squeeze(1), alpha = torch_ones(1, device = device, dtype = dtype)))
    z <- z$unsqueeze(1)
    n_groups <- ceiling(n_blocks / self$group_size); padded_n <- n_groups * self$group_size; pad_len <- padded_n - n_blocks
    if (pad_len > 0) z_padded <- torch_cat(list(z, torch_zeros(1, pad_len, self$d_model, device = device, dtype = dtype)), dim = 2) else z_padded <- z
    z_groups <- z_padded$reshape(c(n_groups, self$group_size, self$d_model))
    intra_result <- self$intra_attn(z_groups, z_groups, z_groups)
    z_intra <- self$intra_norm(z_groups + intra_result[[1]])
    group_query <- z_intra$mean(dim = 2, keepdim = TRUE)
    inter_result <- self$inter_attn(group_query, z_intra, z_intra)
    z_inter <- self$inter_norm(group_query + inter_result[[1]])
    z_inter_expanded <- z_inter$expand(c(-1, self$group_size, -1))
    z_enhanced <- self$ffn_norm(z_intra + self$ffn(z_intra + z_inter_expanded))
    z_flat <- z_enhanced$reshape(c(1, padded_n, self$d_model)); z_out <- z_flat[, 1:n_blocks, ]
    q_global <- self$q_global$to(device = device, dtype = dtype)
    global_result <- self$global_attn(q_global, z_out, z_out)
    global_vec <- global_result[[1]]; attn_weights <- global_result[[2]]
    z_global <- global_vec$squeeze(1)$squeeze(1)
    alpha <- nnf_softmax(attn_weights, dim = -1)$squeeze(1)$squeeze(1)
    list(z_global = z_global, alpha = alpha)
  }
)

# ---- 1g. Patch-HIBA 完整模型 (带注意力捕获) ----
patch_hiba_net <- nn_module("patch_hiba_net",
  initialize = function(n_channels = 6, d_model = 64, encoder_type = "cnn",
                        use_hiba = TRUE, hiba_n_heads = 4, hiba_group_size = 3, hiba_dropout = 0.1) {
    self$d_model <- d_model; self$use_hiba <- use_hiba
    self$patch_encoder <- create_patch_encoder(encoder_type, n_channels, d_model)
    if (use_hiba) self$hiba <- HIBA_attention(d_model, hiba_n_heads, hiba_group_size, hiba_dropout)
    # 使用 MLP 替代线性分类器, 增加表达能力
    self$classifier <- nn_sequential(
      nn_linear(d_model, d_model * 2),
      nn_relu(),
      nn_dropout(0.4),
      nn_linear(d_model * 2, 1)
    )
  },
  forward = function(patch_list, return_attention = FALSE) {
    batch_size <- length(patch_list); device <- self$classifier[[4]]$weight$device; dtype <- torch_float32()
    if (batch_size == 0) return(list(logits = torch_zeros(0, 1, device = device, dtype = dtype),
                                      embedding = torch_zeros(0, self$d_model, device = device, dtype = dtype),
                                      attention = list()))
    embeddings <- list(); all_attentions <- list()
    for (i in 1:batch_size) {
      patches <- patch_list[[i]]; n_patches <- length(patches)
      if (n_patches == 0) { embeddings[[i]] <- torch_zeros(self$d_model, device = device, dtype = dtype); all_attentions[[i]] <- NULL; next }
      first_tensor <- patches[[1]]; dtype <- first_tensor$dtype
      patch_features <- list()
      for (j in 1:n_patches) patch_features[[j]] <- self$patch_encoder(patches[[j]])
      z <- patch_features[[1]]$unsqueeze(1)
      if (n_patches >= 2) for (j in 2:n_patches) z <- torch_cat(list(z, patch_features[[j]]$unsqueeze(1)), dim = 1)
      if (self$use_hiba && n_patches > 1) {
        hiba_out <- self$hiba(z)
        embeddings[[i]] <- hiba_out$z_global
        all_attentions[[i]] <- hiba_out$alpha
      } else {
        embeddings[[i]] <- z$mean(dim = 1)
        all_attentions[[i]] <- NULL
      }
    }
    embedding_mat <- embeddings[[1]]$unsqueeze(1)
    if (batch_size >= 2) for (i in 2:batch_size) embedding_mat <- torch_cat(list(embedding_mat, embeddings[[i]]$unsqueeze(1)), dim = 1)
    logits <- self$classifier(embedding_mat)
    if (return_attention) list(logits = logits, embedding = embedding_mat, attention = all_attentions)
    else list(logits = logits, embedding = embedding_mat)
  }
)

# ---- 1h. 训练函数 (返回历史) ----
train_patch_hiba_model <- function(train_patches, train_y, val_patches = NULL, val_y = NULL,
                                    n_epochs = 50, batch_size = 8, patience = 10,
                                    d_model = 64, encoder_type = "cnn", use_hiba = TRUE,
                                    hiba_n_heads = 4, hiba_group_size = 3,
                                    lr = 1e-3, weight_decay = 5e-4, focal_alpha = 0.6, focal_gamma = 1.0,
                                    focal_gamma_warmup = 3, do_oversample = TRUE,
                                    hiba_dropout = 0.4,
                                    grad_clip_norm = 1.0, lr_scheduler_patience = 5, lr_scheduler_factor = 0.5,
                                    dl_patience = 3) {
  n_samples <- length(train_patches)
  model <- patch_hiba_net(n_channels = 6, d_model = d_model, encoder_type = encoder_type,
                          use_hiba = use_hiba, hiba_n_heads = hiba_n_heads,
                          hiba_group_size = hiba_group_size, hiba_dropout = hiba_dropout)
  device <- "cpu"
  tryCatch({ if (cuda_is_available()) device <- "cuda" }, error = function(e) NULL)
  model <- model$to(device = device)
  optimizer <- optim_adam(model$parameters, lr = lr, weight_decay = weight_decay)
  focal_loss_fn <- function(pred, target, alpha, gamma) {
    eps <- 1e-4; prob <- torch_clamp(torch_sigmoid(pred), eps, 1 - eps)
    pt <- target * prob + (1 - target) * (1 - prob)
    alpha_t <- target * alpha + (1 - target) * (1 - alpha)
    loss <- -alpha_t * (1 - pt)^gamma * torch_log(pt)
    # 裁剪极端 loss 值防止梯度爆炸
    # loss clamp removed: was causing gradient suppression
    mean(loss)
  }
  best_val_loss <- Inf; best_state <- NULL; patience_counter <- 0; lr_plateau_counter <- 0
  history <- data.frame(epoch = integer(), train_loss = double(), val_loss = double())
  # 准备类别平衡采样
  fall_idx <- which(train_y == 1)
  adl_idx <- which(train_y == 0)
  n_fall <- length(fall_idx); n_adl <- length(adl_idx)
  use_bal <- do_oversample && n_fall > 0 && n_adl > 0
  if (use_bal) {
    cat(sprintf("    使用平衡采样 (Fall=%d, ADL=%d), gamma预热=%d epoch\n", n_fall, n_adl, focal_gamma_warmup))
    n_bal <- max(n_fall, n_adl) * 1.5  # epoch内batch数: 降低重采样强度减少过拟合
    total_batches <- ceiling(n_bal / batch_size)
  } else {
    total_batches <- ceiling(n_samples / batch_size)
  }
  for (epoch in 1:n_epochs) {
    model$train(); total_loss <- 0; n_batches <- 0
    # gamma 预热: 前 warmup 个 epoch 用 0 (即 BCE), 之后线性增加到目标 gamma
    current_gamma <- if (epoch <= focal_gamma_warmup) 0 else {
      min(focal_gamma, (epoch - focal_gamma_warmup) * focal_gamma / max(1, n_epochs - focal_gamma_warmup))
    }
    if (use_bal) {
      # 平衡采样: 每 batch 约 half 来自 Fall, half 来自 ADL
      half_bs <- max(2, batch_size %/% 2)
      epoch_pb <- txtProgressBar(min = 0, max = total_batches, style = 3, width = 40)
      for (step in 1:total_batches) {
        f_idx <- sample(fall_idx, min(half_bs, n_fall), replace = TRUE)
        a_idx <- sample(adl_idx, min(half_bs, n_adl), replace = TRUE)
        idx <- sample(c(f_idx, a_idx))
        optimizer$zero_grad()
        batch_patches <- list()
        for (b in idx) {
          patches <- train_patches[[b]]; tensor_patches <- list()
          for (j in seq_along(patches)) tensor_patches[[j]] <- torch_tensor(patches[[j]], dtype = torch_float())$to(device = device)
          batch_patches[[length(batch_patches) + 1]] <- tensor_patches
        }
        out <- model(batch_patches)
        batch_y_t <- torch_tensor(as.numeric(train_y[idx]), dtype = torch_float())$view(c(-1, 1))$to(device = device)
        loss <- focal_loss_fn(out$logits, batch_y_t, focal_alpha, current_gamma)
        loss$backward(); nn_utils_clip_grad_norm_(model$parameters, grad_clip_norm); optimizer$step()
        total_loss <- total_loss + loss$item(); n_batches <- n_batches + 1
        setTxtProgressBar(epoch_pb, n_batches)
      }
    } else {
      indices <- sample(n_samples)
      epoch_pb <- txtProgressBar(min = 0, max = total_batches, style = 3, width = 40)
      for (i in seq(1, n_samples, batch_size)) {
        idx <- indices[i:min(i + batch_size - 1, n_samples)]
        optimizer$zero_grad()
        batch_patches <- list()
        for (b in idx) {
          patches <- train_patches[[b]]; tensor_patches <- list()
          for (j in seq_along(patches)) tensor_patches[[j]] <- torch_tensor(patches[[j]], dtype = torch_float())$to(device = device)
          batch_patches[[length(batch_patches) + 1]] <- tensor_patches
        }
        out <- model(batch_patches)
        batch_y_t <- torch_tensor(as.numeric(train_y[idx]), dtype = torch_float())$view(c(-1, 1))$to(device = device)
        loss <- focal_loss_fn(out$logits, batch_y_t, focal_alpha, current_gamma)
        loss$backward(); nn_utils_clip_grad_norm_(model$parameters, grad_clip_norm); optimizer$step()
        total_loss <- total_loss + loss$item(); n_batches <- n_batches + 1
        setTxtProgressBar(epoch_pb, n_batches)
      }
    }
    close(epoch_pb)
    avg_loss <- total_loss / n_batches
    # Validation
    vl <- NA
    if (!is.null(val_patches) && !is.null(val_y) && length(val_patches) > 0) {
      model$eval()
      with_no_grad({
        val_batch <- list()
        for (vi in seq_along(val_patches)) {
          vp <- val_patches[[vi]]; vt <- list()
          for (j in seq_along(vp)) vt[[j]] <- torch_tensor(vp[[j]], dtype = torch_float())$to(device = device)
          val_batch[[vi]] <- vt
        }
        val_out <- model(val_batch)
        val_y_t <- torch_tensor(as.numeric(val_y), dtype = torch_float())$view(c(-1, 1))$to(device = device)
        vl <- as.numeric(nnf_binary_cross_entropy_with_logits(val_out$logits, val_y_t))
      })
      if (vl >= best_val_loss * 0.995) {
        lr_plateau_counter <- lr_plateau_counter + 1
        if (lr_plateau_counter >= lr_scheduler_patience) {
          lr_plateau_counter <- 0; current_lr <- optimizer$param_groups[[1]]$lr
          new_lr <- current_lr * lr_scheduler_factor
          if (new_lr >= 1e-6) optimizer$param_groups[[1]]$lr <- new_lr
        }
      } else { lr_plateau_counter <- 0 }
      # gamma预热期内: 仅跳过早停, 但仍追踪最低 val_loss
      if (epoch <= focal_gamma_warmup) {
        if (vl < best_val_loss) { best_val_loss <- vl; best_state <- model$state_dict() }
        if (epoch == focal_gamma_warmup) { patience_counter <- 0 }
      } else {
        if (vl < best_val_loss) { best_val_loss <- vl; best_state <- model$state_dict(); patience_counter <- 0 }
        else { patience_counter <- patience_counter + 1; if (patience_counter >= patience) { if (!is.null(best_state)) model$load_state_dict(best_state); break } }
      }
    }
    history <- rbind(history, data.frame(epoch = epoch, train_loss = avg_loss, val_loss = vl))
    cat(sprintf("  Epoch %d/%d: train_loss = %.4f", epoch, n_epochs, avg_loss))
    if (!is.na(vl)) cat(sprintf(" | val_loss = %.4f", vl))
    cat("\n")
  }
  if (!is.null(best_state)) model$load_state_dict(best_state)
  model$to(device = "cpu"); model$eval()
  list(model = model, history = history)
}

# ---- 1i. 嵌入提取 ----
extract_patch_hiba_embedding <- function(model, patches_list, return_attention = FALSE) {
  model$eval(); n_windows <- length(patches_list)
  if (n_windows == 0) return(list(embedding = matrix(0, nrow = 0, ncol = model$d_model), attention = list()))
  all_embeddings <- list(); all_attentions <- list()
  batch_size <- 32
  for (start in seq(1, n_windows, batch_size)) {
    end <- min(start + batch_size - 1, n_windows)
    batch_patches <- list()
    for (i in start:end) {
      patches <- patches_list[[i]]; tensor_patches <- list()
      if (length(patches) > 0) for (j in seq_along(patches)) tensor_patches[[j]] <- torch_tensor(patches[[j]], dtype = torch_float())
      batch_patches[[length(batch_patches) + 1]] <- tensor_patches
    }
    with_no_grad({
      out <- if (return_attention) model(batch_patches, return_attention = TRUE) else model(batch_patches, return_attention = FALSE)
      emb <- as.matrix(out$embedding$cpu())
      for (k in 1:nrow(emb)) all_embeddings[[length(all_embeddings) + 1]] <- emb[k, ]
      if (return_attention) all_attentions <- c(all_attentions, out$attention)
    })
  }
  list(embedding = do.call(rbind, all_embeddings), attention = all_attentions)
}

# ---- 1j. 分类器 ----
to_factor <- function(labels, pos_class = "Fall") factor(ifelse(labels, pos_class, "ADL"), levels = c("ADL", pos_class))

train_rf <- function(train_features, train_labels, ntree = 100, mtry = NULL, classwt_adl = 1, classwt_fall = 3) {
  library(randomForest)
  y <- to_factor(train_labels); p <- ncol(train_features)
  if (is.null(mtry)) mtry <- max(1, floor(sqrt(p)))
  randomForest(x = train_features, y = y, ntree = ntree, mtry = mtry, importance = TRUE,
               classwt = c(ADL = classwt_adl, Fall = classwt_fall))
}
predict_rf <- function(model, test_features) {
  pred_prob <- predict(model, test_features, type = "prob")
  list(class = predict(model, test_features), prob = pred_prob[, "Fall"])
}

train_svm <- function(train_features, train_labels, cost = 1, gamma = NULL,
                       classwt_fall = NULL, kernel = "linear") {
  library(e1071); y <- to_factor(train_labels)
  if (is.null(gamma)) gamma <- 1 / ncol(train_features)
  # 自动计算类别权重: fall 权重 = n_adl / n_fall
  if (is.null(classwt_fall) || length(classwt_fall) == 0 || is.na(classwt_fall) || classwt_fall <= 0) {
    n_pos <- sum(train_labels); n_neg <- sum(!train_labels)
    classwt_fall <- if (n_pos > 0 && n_neg > 0) n_neg / n_pos else 1
  }
  e1071::svm(x = train_features, y = y, kernel = kernel, cost = cost, gamma = gamma,
             class.weights = c(ADL = 1, Fall = classwt_fall), probability = TRUE)
}
predict_svm <- function(model, test_features) {
  pred_prob <- attr(predict(model, test_features, probability = TRUE), "probabilities")
  list(class = predict(model, test_features), prob = pred_prob[, "Fall"])
}

load_catboost <- function() {
  if (require(catboost, quietly = TRUE)) return(TRUE)
  local_pkg <- "../catboost"
  if (file.exists(local_pkg) && file.exists(file.path(local_pkg, "DESCRIPTION"))) {
    tryCatch({ install.packages(local_pkg, repos = NULL, type = "source", quiet = TRUE); if (require(catboost, quietly = TRUE)) return(TRUE) }, error = function(e) NULL)
  }
  dll_path <- file.path(local_pkg, "inst", "libs", "x64", "libcatboostr.dll")
  if (!file.exists(dll_path)) dll_path <- file.path(local_pkg, "libs", "x64", "libcatboostr.dll")
  if (file.exists(dll_path)) {
    tryCatch({ dyn.load(dll_path); r_file <- file.path(local_pkg, "R", "catboost.R"); if (!file.exists(r_file)) r_file <- file.path(local_pkg, "catboost.R"); if (file.exists(r_file)) source(r_file, local = FALSE); return(TRUE) }, error = function(e) NULL)
  }
  FALSE
}

train_catboost <- function(train_features, train_labels, iterations = 100, depth = 4, learning_rate = 0.1,
                           subsample = 0.8, colsample_bylevel = 0.8, l2_leaf_reg = 3) {
  if (!load_catboost()) stop("catboost 无法加载")
  n_pos <- sum(train_labels); n_neg <- sum(!train_labels)
  scale_pos_weight <- if (n_pos > 0 && n_neg > 0) n_neg / n_pos else 1.0
  train_pool <- catboost.load_pool(data = as.matrix(train_features), label = as.numeric(train_labels))
  params <- list(loss_function = "Logloss", eval_metric = "AUC", iterations = iterations, depth = depth,
                 learning_rate = learning_rate, subsample = subsample, colsample_bylevel = colsample_bylevel,
                 l2_leaf_reg = l2_leaf_reg, class_weights = c(1.0, scale_pos_weight), thread_count = 1, verbose = 0)
  catboost.train(train_pool, params = params)
}
predict_catboost <- function(model, test_features) {
  test_pool <- catboost.load_pool(data = as.matrix(test_features))
  prob <- catboost.predict(model, test_pool, prediction_type = "Probability")
  list(prob = prob, class = prob >= 0.5)
}

# ---- 1l. 调参（验证集召回率为目标）----
tune_rf_cv <- function(train_x, train_y, val_x, val_y, grid) {
  library(randomForest); yf <- to_factor(train_y)
  best_model <- NULL; best_recall <- -Inf; best_params <- NULL
  p <- ncol(train_x)
  for (ntree in grid$ntree) {
    for (mtry_ratio in grid$mtry_ratio) {
      mtry <- max(1, floor(p * mtry_ratio))
      m <- tryCatch(randomForest(x = train_x, y = yf, ntree = ntree, mtry = mtry), error = function(e) NULL)
      if (is.null(m)) next
      prob <- tryCatch(predict(m, val_x, type = "prob")[, "Fall"], error = function(e) NULL)
      if (is.null(prob)) next
      met <- calc_metrics(val_y, prob)
      if (!is.na(met["recall"]) && met["recall"] > best_recall) {
        best_recall <- met["recall"]; best_model <- m; best_params <- list(ntree = ntree, mtry = mtry)
      }
    }
  }
  list(model = best_model, recall = best_recall, params = best_params)
}

tune_svm_cv <- function(train_x, train_y, val_x, val_y, grid, classwt_fall) {
  # 自动计算类别权重 (与 train_svm 保持一致)
  if (is.null(classwt_fall) || length(classwt_fall) == 0 || is.na(classwt_fall) || classwt_fall <= 0) {
    n_pos <- sum(train_y); n_neg <- sum(!train_y)
    classwt_fall <- if (n_pos > 0 && n_neg > 0) n_neg / n_pos else 1
  }
  best_model <- NULL; best_recall <- -Inf; best_params <- NULL
  for (kernel in grid$kernel) {
    for (cost in grid$cost) {
      gamma <- 1 / ncol(train_x)
      m <- tryCatch(
        e1071::svm(x = train_x, y = to_factor(train_y), kernel = kernel, cost = cost, gamma = gamma,
                   class.weights = c(ADL = 1, Fall = classwt_fall), probability = TRUE),
        error = function(e) NULL)
      if (is.null(m)) next
      prob <- tryCatch(attr(predict(m, val_x, probability = TRUE), "probabilities")[, "Fall"], error = function(e) NULL)
      if (is.null(prob)) next
      met <- calc_metrics(val_y, prob)
      if (!is.na(met["recall"]) && met["recall"] > best_recall) {
        best_recall <- met["recall"]; best_model <- m; best_params <- list(kernel = kernel, cost = cost)
      }
    }
  }
  list(model = best_model, recall = best_recall, params = best_params)
}

tune_catboost_cv <- function(train_x, train_y, val_x, val_y, grid) {
  if (!load_catboost()) stop("catboost 无法加载")
  n_pos <- sum(train_y); n_neg <- sum(!train_y); sw <- if (n_pos > 0 && n_neg > 0) n_neg / n_pos else 1.0
  train_pool <- catboost.load_pool(data = as.matrix(train_x), label = as.numeric(train_y))
  val_pool <- catboost.load_pool(data = as.matrix(val_x))
  best_model <- NULL; best_recall <- -Inf; best_params <- NULL
  for (iter in grid$iterations) {
    for (depth in grid$depth) {
      for (lr in grid$lr) {
        params <- list(loss_function = "Logloss", iterations = iter, depth = depth, learning_rate = lr,
                       class_weights = c(1.0, sw), thread_count = 1, verbose = 0)
        m <- tryCatch(catboost.train(train_pool, params = params), error = function(e) NULL)
        if (is.null(m)) next
        prob <- tryCatch(catboost.predict(m, val_pool, prediction_type = "Probability"), error = function(e) NULL)
        if (is.null(prob)) next
        met <- calc_metrics(val_y, prob)
        if (!is.na(met["recall"]) && met["recall"] > best_recall) {
          best_recall <- met["recall"]; best_model <- m; best_params <- list(iter = iter, depth = depth, lr = lr)
        }
      }
    }
  }
  list(model = best_model, recall = best_recall, params = best_params)
}

# ---- 1k. 评估 ----
calc_metrics <- function(true_labels, pred_prob, threshold = 0.5) {
  pred_class <- pred_prob >= threshold
  tp <- sum(pred_class & true_labels); tn <- sum(!pred_class & !true_labels)
  fp <- sum(pred_class & !true_labels); fn <- sum(!pred_class & true_labels)
  recall <- if ((tp + fn) > 0) tp / (tp + fn) else NA
  specificity <- if ((tn + fp) > 0) tn / (tn + fp) else NA
  precision <- if ((tp + fp) > 0) tp / (tp + fp) else NA
  f1 <- if (!is.na(precision) && !is.na(recall) && (precision + recall) > 0) 2 * precision * recall / (precision + recall) else NA
  auc <- NA
  tryCatch({ if (length(unique(true_labels)) > 1) auc <- as.numeric(pROC::auc(pROC::roc(true_labels, pred_prob, quiet = TRUE))) }, error = function(e) {})
  c(recall = recall, specificity = specificity, precision = precision, f1 = f1, auc = auc, tp = tp, tn = tn, fp = fp, fn = fn)
}

find_best_threshold <- function(val_true, val_prob, step = 0.01) {
  thresholds <- seq(0.05, 0.95, by = step); best_t <- 0.5; best_f1 <- -Inf
  for (t in thresholds) {
    m <- calc_metrics(val_true, val_prob, threshold = t)
    if (!is.na(m["f1"]) && m["f1"] > best_f1) { best_t <- t; best_f1 <- m["f1"] }
  }
  list(threshold = best_t, f1 = best_f1)
}

aggregate_predictions <- function(window_preds, window_orig_idx, aggregator = "mean") {
  unique_idx <- unique(window_orig_idx); agg_preds <- numeric(length(unique_idx))
  for (i in seq_along(unique_idx)) {
    mask <- window_orig_idx == unique_idx[i]
    agg_preds[i] <- if (aggregator == "mean") mean(window_preds[mask], na.rm = TRUE) else max(window_preds[mask], na.rm = TRUE)
  }
  list(predictions = agg_preds, original_indices = unique_idx)
}

# ---- 1l. 混淆矩阵 ----
calc_confusion <- function(true_labels, pred_prob, threshold = 0.5) {
  pred_class <- pred_prob >= threshold
  cm <- table(Actual = ifelse(true_labels, "Fall", "ADL"),
              Predicted = ifelse(pred_class, "Fall", "ADL"))
  cm
}

# ---- 1m. 全局特征提取 (基线) ----
extract_global_features <- function(data_list, peak_anchors, fs = 238) {
  feature_mean <- function(x) mean(x, na.rm = TRUE)
  feature_var <- function(x) var(x, na.rm = TRUE)
  feature_rms <- function(x) sqrt(mean(x^2, na.rm = TRUE))
  feature_peak <- function(x) max(abs(x), na.rm = TRUE)
  feature_kurtosis <- function(x) { n <- length(x); if (n < 4) return(0); mu <- mean(x, na.rm = TRUE); s <- sd(x, na.rm = TRUE); if (s == 0) return(0); sum((x - mu)^4, na.rm = TRUE) / (n * s^4) - 3 }
  feature_skewness <- function(x) { n <- length(x); if (n < 3) return(0); mu <- mean(x, na.rm = TRUE); s <- sd(x, na.rm = TRUE); if (s == 0) return(0); sum((x - mu)^3, na.rm = TRUE) / (n * s^3) }
  feature_iqr <- function(x) IQR(x, na.rm = TRUE)
  feature_sma <- function(mat) sum(abs(mat[, 1])) + sum(abs(mat[, 2])) + sum(abs(mat[, 3]))
  feature_spectral_energy <- function(x) { N <- length(x); X <- stats::fft(x); sum(Mod(X[1:floor(N/2)])^2) }
  feature_dominant_freq <- function(x, fs) { N <- length(x); X <- stats::fft(x); mag <- Mod(X[1:floor(N/2)]); idx <- which.max(mag); (idx - 1) * fs / N }
  feature_spectral_entropy <- function(x) { N <- length(x); X <- stats::fft(x); mag <- Mod(X[1:floor(N/2)]); p <- mag / sum(mag); p <- p[p > 0]; -sum(p * log(p)) }
  feature_dc <- function(x) Re(stats::fft(x)[1]) / length(x)
  feature_hjorth <- function(x) {
    var0 <- var(x, na.rm = TRUE); dx <- diff(x); var1 <- var(dx, na.rm = TRUE)
    ddx <- diff(dx); var2 <- var(ddx, na.rm = TRUE)
    activity <- var0; mobility <- if (var0 > 0) sqrt(var1 / var0) else 0
    complexity <- if (var1 > 0 && mobility > 0) sqrt(var2 / var1) / mobility else 0
    c(activity = ifelse(is.finite(activity), activity, 0), mobility = ifelse(is.finite(mobility), mobility, 0), complexity = ifelse(is.finite(complexity), complexity, 0))
  }
  feature_ar <- function(x, order = 4) {
    x_centered <- x - mean(x, na.rm = TRUE); coefs <- rep(0, order)
    tryCatch({ ar_fit <- stats::ar(x_centered, aic = FALSE, order.max = order, method = "burg"); coefs <- ar_fit$ar; if (length(coefs) < order) coefs <- c(coefs, rep(0, order - length(coefs))) }, error = function(e) {})
    coefs
  }
  extract_channel_features <- function(x, fs) {
    hj <- feature_hjorth(x); ar <- feature_ar(x, 4)
    c(mean = feature_mean(x), var = feature_var(x), rms = feature_rms(x), peak = feature_peak(x),
      kurtosis = feature_kurtosis(x), skewness = feature_skewness(x), iqr = feature_iqr(x),
      spectral_energy = feature_spectral_energy(x), dominant_freq = feature_dominant_freq(x, fs),
      spectral_entropy = feature_spectral_entropy(x), dc = feature_dc(x),
      hjorth_activity = unname(hj["activity"]), hjorth_mobility = unname(hj["mobility"]),
      hjorth_complexity = unname(hj["complexity"]),
      ar1 = unname(ar[1]), ar2 = unname(ar[2]), ar3 = unname(ar[3]), ar4 = unname(ar[4]))
  }
  n <- length(data_list); all_features <- vector("list", n)
  for (i in seq_len(n)) {
    acc <- data_list[[i]]$Acc_raw; gyr <- data_list[[i]]$Gyr_raw
    ti <- peak_anchors[i]; if (is.na(ti)) ti <- floor(nrow(acc) / 2)
    pre_end <- min(ti, nrow(acc)); if (pre_end < 10) pre_end <- min(nrow(acc), 100)
    acc_pre <- acc[1:pre_end, , drop = FALSE]; gyr_pre <- gyr[1:pre_end, , drop = FALSE]
    ch_names <- c("acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z")
    features <- c()
    for (ch in 1:3) { ch_feat <- extract_channel_features(acc_pre[, ch], fs); names(ch_feat) <- sprintf("%s_%s", ch_names[ch], names(ch_feat)); features <- c(features, ch_feat) }
    for (ch in 1:3) { ch_feat <- extract_channel_features(gyr_pre[, ch], fs); names(ch_feat) <- sprintf("%s_%s", ch_names[3 + ch], names(ch_feat)); features <- c(features, ch_feat) }
    features <- c(features, acc_sma = feature_sma(acc_pre), gyr_sma = feature_sma(gyr_pre))
    features[!is.finite(features)] <- 0; all_features[[i]] <- features
  }
  as.data.frame(do.call(rbind, all_features))
}

# ---- 1n. 分层抽样 ----
stratified_sample <- function(data_list, frac = 0.005, seed = 114514) {
  set.seed(seed); subjects <- build_subjects(data_list); falls <- build_labels(data_list)
  df <- data.frame(idx = seq_along(data_list), subject = subjects, fall = falls)
  df$group_id <- paste(df$subject, df$fall, sep = "_"); unique_groups <- unique(df$group_id); sampled_idx <- c()
  for (gid in unique_groups) {
    grp_idx <- which(df$group_id == gid); n_grp <- length(grp_idx)
    n_take <- min(max(1, ceiling(n_grp * frac)), n_grp)
    sampled_idx <- c(sampled_idx, sample(grp_idx, size = n_take, replace = FALSE))
  }
  data_list[sort(sampled_idx)]
}

# ---- 1o. 统计显著性: McNemar 检验 ----
mcnemar_test <- function(true_labels, pred1, pred2) {
  # pred1, pred2: binary predictions
  b <- sum(pred1 == true_labels & pred2 != true_labels)  # 1 correct, 2 wrong
  c <- sum(pred1 != true_labels & pred2 == true_labels)  # 1 wrong, 2 correct
  n_discordant <- b + c
  if (n_discordant < 1) return(list(statistic = 0, p_value = 1.0, b = b, c = c))
  chi2 <- (abs(b - c) - 1)^2 / n_discordant  # continuity correction
  p_value <- pchisq(chi2, df = 1, lower.tail = FALSE)
  list(statistic = chi2, p_value = p_value, b = b, c = c)
}

# ---- 1p. Markdown 表格输出 ----
write_md_table <- function(df, file_path, caption = "", digits = 4) {
  if (is.null(df) || nrow(df) == 0) return(invisible(NULL))
  df_out <- as.data.frame(df)
  for (j in seq_len(ncol(df_out))) {
    if (is.numeric(df_out[, j])) df_out[, j] <- round(df_out[, j], digits)
  }
  lines <- c()
  if (nchar(caption) > 0) lines <- c(lines, paste("##", caption), "")
  header <- paste("|", paste(colnames(df_out), collapse = " | "), "|")
  separator <- paste("|", paste(rep("---", ncol(df_out)), collapse = " | "), "|")
  body <- apply(df_out, 1, function(row) {
    paste("|", paste(as.character(row), collapse = " | "), "|")
  })
  writeLines(c(lines, header, separator, body), file_path)
  cat(sprintf("  MD 已保存: %s\n", file_path))
  invisible(NULL)
}

# ============================================================
# 2. 单次实验运行函数
# ============================================================
run_single_experiment <- function(config, data_list, labels, subjects,
                                   exp_name = "default", output_subdir = NULL) {
  cfg <- config
  exp_dir <- if (is.null(output_subdir)) file.path(OUTPUT_DIR, exp_name) else output_subdir
  dir.create(exp_dir, showWarnings = FALSE, recursive = TRUE)
  saveRDS(cfg, file.path(exp_dir, "config.rds"))

  cat(sprintf("\n%s\n", paste(rep("=", 70), collapse = "")))
  cat(sprintf("  实验: %s\n", exp_name))
  cat(sprintf("  编码器: %s | 分块: %s | HIBA: %s | 降噪: %s\n",
              cfg$encoder_type, if (cfg$enable_dynamic_patching) "动态" else "均匀",
              if (cfg$use_hiba) "ON" else "OFF", cfg$denoise_method))
  cat(sprintf("%s\n\n", paste(rep("=", 70), collapse = "")))

  N_sample <- length(data_list)
  n_pre <- round(cfg$L_pre_ms * cfg$fs / 1000)
  unique_subjs <- sort(unique(subjects))
  test_subjs <- intersect(cfg$fast_subjects, unique_subjs)
  n_folds <- length(test_subjs)
  if (n_folds == 0) stop("fast_subjects 与数据受试者无交集, 无法进行 LOSO 验证")

  # 结果存储
  results <- list(
    rf = list(all_pred = numeric(N_sample), all_true = labels, best_thresholds = numeric()),
    svm = list(all_pred = numeric(N_sample), all_true = labels, best_thresholds = numeric()),
    catboost = list(all_pred = numeric(N_sample), all_true = labels, best_thresholds = numeric()),
    ensemble = list(all_pred = numeric(N_sample), all_true = labels, best_thresholds = numeric())
  )
  fold_tested <- logical(N_sample)

  # 全局基线
  if (cfg$use_global_features) {
    results$rf_global <- list(all_pred = numeric(N_sample), all_true = labels, best_thresholds = numeric())
    results$svm_global <- list(all_pred = numeric(N_sample), all_true = labels, best_thresholds = numeric())
    results$catboost_global <- list(all_pred = numeric(N_sample), all_true = labels, best_thresholds = numeric())
    results$ensemble_global <- list(all_pred = numeric(N_sample), all_true = labels, best_thresholds = numeric())
  }
  fold_tested_global <- logical(N_sample)

  # 收集所有折的训练历史和注意力权重
  all_history <- list()
  all_attention <- list()
  all_patch_info <- list()  # 时间位置信息

  # Fold 进度条
  fold_pb <- txtProgressBar(min = 0, max = n_folds, style = 3, width = 50)
  for (fold_idx in seq_len(n_folds)) {
    fold_result <- tryCatch({
      test_subj <- test_subjs[fold_idx]
      train_orig_idx <- which(subjects != test_subj)
      test_orig_idx  <- which(subjects == test_subj)

      if (sum(labels[train_orig_idx]) == 0) { cat(sprintf("  折 %d: 训练集无跌倒, 跳过\n", fold_idx)); next }

      cat(sprintf("\n  --- Fold %d/%d ---\n", fold_idx, n_folds))

      # ---- 窗口生成 ----
      train_windows <- list(); train_window_orig_idx <- integer()
      for (idx in train_orig_idx) {
        result <- generate_sliding_windows(data_list[[idx]], cfg$delta_ms, n_pre, cfg$fs)
        if (length(result$windows) > 0) {
          train_windows <- c(train_windows, result$windows)
          train_window_orig_idx <- c(train_window_orig_idx, rep(idx, length(result$windows)))
        }
      }
      test_windows <- list(); test_window_orig_idx <- integer()
      for (idx in test_orig_idx) {
        result <- generate_sliding_windows(data_list[[idx]], cfg$delta_ms, n_pre, cfg$fs)
        if (length(result$windows) > 0) {
          test_windows <- c(test_windows, result$windows)
          test_window_orig_idx <- c(test_window_orig_idx, rep(idx, length(result$windows)))
        }
      }
      n_train_windows <- length(train_windows); n_test_windows <- length(test_windows)
      if (n_train_windows == 0 || n_test_windows == 0) { cat("  窗口为空, 跳过\n"); next }

      train_window_data <- lapply(train_windows, function(w) w$data)
      test_window_data <- lapply(test_windows, function(w) w$data)
      train_y <- labels[train_window_orig_idx]; test_y <- labels[test_window_orig_idx]

      # ---- 标准化 (带防御检查) ----
      # 确保所有窗口数据都是矩阵
      for (j in seq_len(n_train_windows)) {
        if (!is.matrix(train_window_data[[j]])) train_window_data[[j]] <- as.matrix(train_window_data[[j]])
      }
      for (j in seq_len(n_test_windows)) {
        if (!is.matrix(test_window_data[[j]])) test_window_data[[j]] <- as.matrix(test_window_data[[j]])
      }
      train_local <- array(0, dim = c(n_train_windows, n_pre, 6))
      for (j in seq_len(n_train_windows)) train_local[j, , ] <- train_window_data[[j]]
      test_local <- array(0, dim = c(n_test_windows, n_pre, 6))
      for (j in seq_len(n_test_windows)) test_local[j, , ] <- test_window_data[[j]]
      for (ch in 1:dim(train_local)[3]) {
        ch_mean <- mean(train_local[, , ch], na.rm = TRUE)
        ch_sd <- sd(train_local[, , ch], na.rm = TRUE)
        if (is.na(ch_sd) || ch_sd == 0) ch_sd <- 1
        train_local[, , ch] <- (train_local[, , ch] - ch_mean) / ch_sd
        test_local[, , ch]  <- (test_local[, , ch] - ch_mean) / ch_sd
      }
      for (j in seq_len(n_train_windows)) train_window_data[[j]] <- train_local[j, , ]
      for (j in seq_len(n_test_windows)) test_window_data[[j]] <- test_local[j, , ]

      # ---- 验证集划分 ----
      set.seed(cfg$myseed + fold_idx)
      unique_train_orig <- unique(train_window_orig_idx)
      n_unique <- length(unique_train_orig)
      n_val_unique <- max(1, round(cfg$val_ratio * n_unique))
      val_unique_idx <- sample(unique_train_orig, size = n_val_unique)
      val_mask <- train_window_orig_idx %in% val_unique_idx
      train_mask <- !val_mask

      # ---- 分块 (防御: sig 降维时确保仍为矩阵) ----
      do_patch <- function(wdata, ep) {
        n <- length(wdata); pl <- vector("list", n); nv <- integer(n)
        for (i in seq_len(n)) {
          sig <- wdata[[i]]
          if (!is.matrix(sig)) sig <- as.matrix(sig)
          if (cfg$enable_dynamic_patching && nrow(sig) >= 16) {
            complexity <- compute_complexity_profile(sig, cfg$patch_complexity_smooth)
            pl[[i]] <- adaptive_patching(sig, complexity, cfg$patch_short_len, cfg$patch_long_len,
                                          cfg$patch_overlap, cfg$patch_min, cfg$patch_max, cfg$patch_min_len)
          } else {
            pl[[i]] <- uniform_patching(sig, 8)
          }
          nv[i] <- length(pl[[i]])
        }
        patch_positions <- vector("list", n)
        if (ep) {
          for (i in seq_len(n)) {
            patches <- pl[[i]]
            if (length(patches) == 0) { patch_positions[[i]] <- NULL; next }
            pos_list <- list(); sig_len <- NROW(wdata[[i]])
            pos <- 1; pidx <- 1
            while (pos <= sig_len && pidx <= length(patches)) {
              patch_rows <- NROW(patches[[pidx]])
              pos_list[[pidx]] <- c(start = pos, end = pos + patch_rows - 1)
              pos <- pos + patch_rows
              pidx <- pidx + 1
            }
            patch_positions[[i]] <- pos_list
          }
        }
        list(patches = pl, n_blocks = nv, positions = patch_positions)
      }

      train_patch <- do_patch(train_window_data[train_mask], TRUE)
      val_patch <- do_patch(train_window_data[val_mask], FALSE)
      test_patch <- do_patch(test_window_data, TRUE)

      train_patches <- train_patch$patches; val_patches <- val_patch$patches; test_patches <- test_patch$patches
      train_y_fold <- train_y[train_mask]; val_y_local <- train_y[val_mask]

      cat(sprintf("    Train:%d Val:%d Test:%d windows\n",
                  length(train_patches), length(val_patches), length(test_patches)))
      cat(sprintf("    [分布] Train: Fall=%d ADL=%d | Val: Fall=%d ADL=%d | Test: Fall=%d ADL=%d\n",
                  sum(train_y_fold), sum(!train_y_fold),
                  sum(val_y_local), sum(!val_y_local),
                  sum(test_y), sum(!test_y)))

      # ---- 训练深度模型 ----
      deep_model <- NULL; fold_history <- NULL
      use_deep <- (length(train_y_fold) >= cfg$min_deep_samples) &&
                  (length(unique(train_y_fold)) > 1) && cfg$to_use_deep

      if (use_deep) {
        tryCatch({
          train_result <- train_patch_hiba_model(
            train_patches, train_y_fold, val_patches, val_y_local,
            n_epochs = cfg$n_epochs, batch_size = cfg$batch_size,
            patience = cfg$dl_patience, d_model = cfg$d_model,
            encoder_type = cfg$encoder_type, use_hiba = cfg$use_hiba,
            hiba_n_heads = cfg$hiba_n_heads, hiba_group_size = cfg$hiba_group_size,
            hiba_dropout = cfg$hiba_dropout,
            lr = cfg$lr, weight_decay = cfg$weight_decay,
            focal_alpha = cfg$focal_alpha, focal_gamma = cfg$focal_gamma,
            focal_gamma_warmup = cfg$focal_gamma_warmup, do_oversample = cfg$do_oversample,
            grad_clip_norm = cfg$grad_clip_norm,
            lr_scheduler_patience = cfg$lr_scheduler_patience,
            lr_scheduler_factor = cfg$lr_scheduler_factor,
            dl_patience = cfg$dl_patience
          )
          deep_model <- train_result$model
          fold_history <- train_result$history
          all_history[[length(all_history) + 1]] <- cbind(fold = fold_idx, fold_history)
        }, error = function(e) {
          cat(sprintf("    模型训练失败: %s\n", conditionMessage(e)))
          deep_model <<- NULL
        })
      }

      # ---- 提取特征 ----
      train_feat <- NULL; val_feat <- NULL; test_feat <- NULL
      if (!is.null(deep_model)) {
        tryCatch({
          train_embed <- extract_patch_hiba_embedding(deep_model, train_patches, return_attention = FALSE)$embedding
          val_embed <- extract_patch_hiba_embedding(deep_model, val_patches, return_attention = FALSE)$embedding
          test_embed_with_attn <- extract_patch_hiba_embedding(deep_model, test_patches, return_attention = TRUE)
          test_embed <- test_embed_with_attn$embedding
          # 确保特征为矩阵
          if (is.null(dim(train_embed))) train_embed <- as.matrix(train_embed)
          if (is.null(dim(val_embed))) val_embed <- as.matrix(val_embed)
          if (is.null(dim(test_embed))) test_embed <- as.matrix(test_embed)

          # ---- 诊断: 嵌入向量统计 ----
          diag_mean <- mean(train_embed, na.rm = TRUE)
          diag_sd <- sd(as.numeric(train_embed), na.rm = TRUE)
          diag_min <- min(train_embed, na.rm = TRUE)
          diag_max <- max(train_embed, na.rm = TRUE)
          cat(sprintf("    [嵌入] train: %dx%d, mean=%.4f, sd=%.4f, [%.4f, %.4f], NaN=%d, Inf=%d\n",
                      nrow(train_embed), ncol(train_embed),
                      diag_mean, diag_sd, diag_min, diag_max,
                      sum(is.na(train_embed)), sum(is.infinite(train_embed))))
          cat(sprintf("    [嵌入] val: %dx%d, mean=%.4f, sd=%.4f\n",
                      nrow(val_embed), ncol(val_embed),
                      mean(val_embed, na.rm = TRUE), sd(as.numeric(val_embed), na.rm = TRUE)))
          cat(sprintf("    [嵌入] test: %dx%d, mean=%.4f, sd=%.4f\n",
                      nrow(test_embed), ncol(test_embed),
                      mean(test_embed, na.rm = TRUE), sd(as.numeric(test_embed), na.rm = TRUE)))
          # 检查是否全是常数/零输出
          zero_cols <- which(apply(train_embed, 2, function(col) all(abs(col) < 1e-8)))
          const_cols <- which(apply(train_embed, 2, function(col) length(unique(round(col, 6))) == 1))
          if (length(zero_cols) > 0) cat(sprintf("    [嵌入] 警告: train_embed 有 %d 个零列\n", length(zero_cols)))
          if (length(const_cols) > 0) cat(sprintf("    [嵌入] 警告: train_embed 有 %d 个常数列\n", length(const_cols)))
          if (diag_sd < 1e-6) cat(sprintf("    [嵌入] 严重: sd≈0, 嵌入已坍塌!\n"))
          # HIBA 注意力诊断
          if (cfg$use_hiba && length(test_embed_with_attn$attention) > 0) {
            attn_vals <- unlist(lapply(test_embed_with_attn$attention, function(a) {
              if (!is.null(a)) as.numeric(a$cpu()) else NA
            }))
            if (length(attn_vals) > 0 && !all(is.na(attn_vals))) {
              cat(sprintf("    [HIBA] 注意力: mean=%.4f, sd=%.4f, range=[%.4f, %.4f]\n",
                          mean(attn_vals, na.rm = TRUE), sd(attn_vals, na.rm = TRUE),
                          min(attn_vals, na.rm = TRUE), max(attn_vals, na.rm = TRUE)))
            }
          }

          if (cfg$use_hiba && length(test_embed_with_attn$attention) > 0) {
            for (ai in seq_along(test_embed_with_attn$attention)) {
              attn_w <- test_embed_with_attn$attention[[ai]]
              if (!is.null(attn_w)) {
                n_patches_test <- length(test_patches[[ai]])
                all_attention[[length(all_attention) + 1]] <- list(
                  fold = fold_idx, test_window_idx = ai,
                  orig_idx = test_window_orig_idx[ai], n_patches = n_patches_test,
                  attention_weights = as.numeric(attn_w$cpu()),
                  patch_positions = test_patch$positions[[ai]]
                )
              }
            }
          }
          train_feat <- train_embed; val_feat <- val_embed; test_feat <- test_embed
        }, error = function(e) {
          cat(sprintf("    特征提取失败: %s\n", conditionMessage(e)))
          cat(sprintf("    位置: %s\n", deparse(conditionCall(e), nlines = 1)[1]))
          deep_model <<- NULL
          train_feat <- NULL; val_feat <- NULL; test_feat <- NULL
        })
      } else {
        train_feat <- NULL; val_feat <- NULL; test_feat <- NULL
      }

      # ---- 深度特征分类器 ----
      if (!is.null(train_feat) && ncol(train_feat) >= 1 && nrow(train_feat) >= 2) {
        train_y_bal <- train_y_fold
        feat_mu <- colMeans(train_feat, na.rm = TRUE)
        feat_sd <- apply(train_feat, 2, sd, na.rm = TRUE)
        feat_sd[is.na(feat_sd) | feat_sd == 0] <- 1
        train_feat_s <- sweep(sweep(train_feat, 2, feat_mu, "-"), 2, feat_sd, "/")
        val_feat_s <- sweep(sweep(val_feat, 2, feat_mu, "-"), 2, feat_sd, "/")
        test_feat_s <- sweep(sweep(test_feat, 2, feat_mu, "-"), 2, feat_sd, "/")
        train_feat_s[!is.finite(train_feat_s)] <- 0
        val_feat_s[!is.finite(val_feat_s)] <- 0; test_feat_s[!is.finite(test_feat_s)] <- 0

        fold_probs <- list(); val_probs <- list()
        do_tune <- isTRUE(cfg$tune_classifiers)

        # ---- RF ----
        tryCatch({
          rf_model <- if (do_tune) {
            tuned_rf <- tune_rf_cv(as.data.frame(train_feat_s), train_y_bal,
                                   as.data.frame(val_feat_s), val_y_local, cfg$rf_tune_grid)
            tuned_rf$model
          } else {
            train_rf(as.data.frame(train_feat_s), train_y_bal, cfg$rf_ntree,
                     classwt_adl = cfg$rf_classwt_adl, classwt_fall = cfg$rf_classwt_fall)
          }
          rf_val <- predict_rf(rf_model, as.data.frame(val_feat_s))
          rf_test <- predict_rf(rf_model, as.data.frame(test_feat_s))
          fold_probs$rf <- rf_test$prob; val_probs$rf <- rf_val$prob
        }, error = function(e) cat(sprintf("    RF error: %s\n", e$message)))

        # ---- SVM ----
        tryCatch({
          svm_model <- if (do_tune) {
            tuned_svm <- tune_svm_cv(as.data.frame(train_feat_s), train_y_bal,
                                     as.data.frame(val_feat_s), val_y_local,
                                     cfg$svm_tune_grid, cfg$svm_classwt_fall)
            tuned_svm$model
          } else {
            train_svm(as.data.frame(train_feat_s), train_y_bal, cfg$svm_cost,
                      classwt_fall = cfg$svm_classwt_fall, kernel = cfg$svm_kernel)
          }
          svm_val <- predict_svm(svm_model, as.data.frame(val_feat_s))
          svm_test <- predict_svm(svm_model, as.data.frame(test_feat_s))
          fold_probs$svm <- svm_test$prob; val_probs$svm <- svm_val$prob
        }, error = function(e) cat(sprintf("    SVM error: %s\n", e$message)))

        # ---- CatBoost ----
        tryCatch({
          cb_model <- if (do_tune) {
            tuned_cb <- tune_catboost_cv(as.data.frame(train_feat_s), train_y_bal,
                                         as.data.frame(val_feat_s), val_y_local, cfg$cb_tune_grid)
            tuned_cb$model
          } else {
            train_catboost(as.data.frame(train_feat_s), train_y_bal,
                           cfg$cb_iterations, cfg$cb_depth, cfg$cb_learning_rate,
                           cfg$cb_subsample, cfg$cb_colsample_bylevel, cfg$cb_l2_leaf_reg)
          }
          cb_val <- predict_catboost(cb_model, as.data.frame(val_feat_s))
          cb_test <- predict_catboost(cb_model, as.data.frame(test_feat_s))
          fold_probs$catboost <- cb_test$prob; val_probs$catboost <- cb_val$prob
        }, error = function(e) cat(sprintf("    CatBoost error: %s\n", e$message)))

        avail_names <- intersect(names(val_probs), names(fold_probs))
        for (name in avail_names) {
          val_agg <- aggregate_predictions(val_probs[[name]], train_window_orig_idx[val_mask])
          bt <- find_best_threshold(labels[val_agg$original_indices], val_agg$predictions)
          test_agg <- aggregate_predictions(fold_probs[[name]], test_window_orig_idx)
          # 存储概率和最优阈值 (阈值将在最终评估时使用)
          results[[name]]$best_thresholds <- c(results[[name]]$best_thresholds, bt$threshold)
          for (i in seq_along(test_agg$original_indices)) {
            orig_i <- test_agg$original_indices[i]
            results[[name]]$all_pred[orig_i] <- test_agg$predictions[i]
          }
        }
        fold_tested[unique(test_window_orig_idx)] <- TRUE

        if (length(avail_names) >= 2) {
          val_prob_mat <- do.call(cbind, val_probs[avail_names])
          n_models <- length(avail_names); best_w <- rep(1 / n_models, n_models); best_val_f1 <- 0
          for (step in list(seq(0, 1, by = cfg$stacking_grid_coarse), seq(0, 1, by = cfg$stacking_grid_fine))) {
            for (w1 in step) for (w2 in step) {
              w_sum <- w1 + w2
              if (n_models == 3) { w3 <- 1 - w_sum; if (w3 < 0 || w3 > 1) next; w <- c(w1, w2, w3) }
              else w <- c(w1, 1 - w1)
              val_p <- val_prob_mat %*% w
              agg <- aggregate_predictions(as.numeric(val_p), train_window_orig_idx[val_mask])
              m <- calc_metrics(labels[agg$original_indices], agg$predictions)
              if (!is.na(m["f1"]) && m["f1"] > best_val_f1) { best_val_f1 <- m["f1"]; best_w <- w }
            }
          }
          # 权重正则化防止单一模型过拟合
          best_w <- pmax(0.2, pmin(0.6, best_w))
          best_w <- best_w / sum(best_w)
          test_prob_mat <- do.call(cbind, fold_probs[avail_names])
          test_ens <- as.numeric(test_prob_mat %*% best_w)
          test_agg <- aggregate_predictions(test_ens, test_window_orig_idx)
          for (i in seq_along(test_agg$original_indices)) {
            orig_i <- test_agg$original_indices[i]
            results$ensemble$all_pred[orig_i] <- test_agg$predictions[i]
          }
        }
      } else {
        if (!is.null(train_feat)) cat(sprintf("    特征维度不足 (nrow=%d, ncol=%d), 跳过分类器\n", nrow(train_feat), ncol(train_feat)))
      }

      # ---- 全局手工特征基线 ----
      if (cfg$use_global_features) {
    cat("\n  --- 全局手工特征基线 ---\n")
    train_peak_anchors <- sapply(data_list[train_orig_idx], function(item) compute_peak_anchor(item$Acc_raw))
    test_peak_anchors <- sapply(data_list[test_orig_idx], function(item) compute_peak_anchor(item$Acc_raw))
    train_gf_df <- extract_global_features(data_list[train_orig_idx], train_peak_anchors, cfg$fs)
    test_gf_df <- extract_global_features(data_list[test_orig_idx], test_peak_anchors, cfg$fs)
    train_gf_mat <- as.matrix(train_gf_df); test_gf_mat <- as.matrix(test_gf_df)
    train_gf_mat[!is.finite(train_gf_mat)] <- 0; test_gf_mat[!is.finite(test_gf_mat)] <- 0
    gf_mu <- colMeans(train_gf_mat, na.rm = TRUE); gf_sd <- apply(train_gf_mat, 2, sd, na.rm = TRUE)
    gf_sd[gf_sd == 0 | is.na(gf_sd)] <- 1
    train_gf_s <- sweep(sweep(train_gf_mat, 2, gf_mu, "-"), 2, gf_sd, "/")
    test_gf_s <- sweep(sweep(test_gf_mat, 2, gf_mu, "-"), 2, gf_sd, "/")
    train_gf_s[!is.finite(train_gf_s)] <- 0; test_gf_s[!is.finite(test_gf_s)] <- 0
    pca_gf <- prcomp(train_gf_s, center = FALSE, scale. = FALSE)
    cumvar <- cumsum(pca_gf$sdev^2 / sum(pca_gf$sdev^2))
    n_pca <- which(cumvar >= cfg$pca_var_ratio)[1]
    if (is.na(n_pca) || length(n_pca) == 0) n_pca <- min(cfg$pca_max_dim, ncol(train_gf_s))
    n_pca <- min(n_pca, ncol(train_gf_s))
    train_global_orig <- pca_gf$x[, 1:n_pca, drop = FALSE]
    test_global_orig <- predict(pca_gf, test_gf_s)[, 1:n_pca, drop = FALSE]
    train_global <- train_global_orig[match(train_window_orig_idx, train_orig_idx), , drop = FALSE]
    test_global <- test_global_orig[match(test_window_orig_idx, test_orig_idx), , drop = FALSE]
    train_g_fold <- train_global[train_mask, , drop = FALSE]
    val_g_fold <- train_global[val_mask, , drop = FALSE]
    test_g_fold <- test_global

    gfold_probs <- list(); gval_probs <- list()
    tryCatch({
      rf_g <- train_rf(as.data.frame(train_g_fold), train_y_fold, cfg$rf_ntree)
      rfv_g <- predict_rf(rf_g, as.data.frame(val_g_fold)); rft_g <- predict_rf(rf_g, as.data.frame(test_g_fold))
      gfold_probs$rf <- rft_g$prob; gval_probs$rf <- rfv_g$prob
    }, error = function(e) cat(sprintf("    RF(基线) error: %s\n", e$message)))
    tryCatch({
      svm_g <- train_svm(as.data.frame(train_g_fold), train_y_fold, cfg$svm_cost,
                          classwt_fall = cfg$svm_classwt_fall,
                          kernel = cfg$svm_kernel)
      svmv_g <- predict_svm(svm_g, as.data.frame(val_g_fold)); svmt_g <- predict_svm(svm_g, as.data.frame(test_g_fold))
      gfold_probs$svm <- svmt_g$prob; gval_probs$svm <- svmv_g$prob
    }, error = function(e) cat(sprintf("    SVM(基线) error: %s\n", e$message)))
    tryCatch({
      cb_g <- train_catboost(as.data.frame(train_g_fold), train_y_fold, cfg$cb_iterations, cfg$cb_depth, cfg$cb_learning_rate)
      cbv_g <- predict_catboost(cb_g, as.data.frame(val_g_fold)); cbt_g <- predict_catboost(cb_g, as.data.frame(test_g_fold))
      gfold_probs$catboost <- cbt_g$prob; gval_probs$catboost <- cbv_g$prob
    }, error = function(e) cat(sprintf("    CatBoost(基线) error: %s\n", e$message)))
    g_avail <- intersect(names(gval_probs), names(gfold_probs))
    for (name in g_avail) {
      val_agg <- aggregate_predictions(gval_probs[[name]], train_window_orig_idx[val_mask])
      bt <- find_best_threshold(labels[val_agg$original_indices], val_agg$predictions)
      test_agg <- aggregate_predictions(gfold_probs[[name]], test_window_orig_idx)
      g_name <- paste0(name, "_global")
      results[[g_name]]$best_thresholds <- c(results[[g_name]]$best_thresholds, bt$threshold)
      for (i in seq_along(test_agg$original_indices)) {
        orig_i <- test_agg$original_indices[i]
        results[[g_name]]$all_pred[orig_i] <- test_agg$predictions[i]
      }
    }
    fold_tested_global[unique(test_window_orig_idx)] <- TRUE
    if (length(g_avail) >= 2) {
      gval_mat <- do.call(cbind, gval_probs[g_avail]); n_gm <- length(g_avail); best_gw <- rep(1/n_gm, n_gm); best_gf1 <- 0
      for (step in list(seq(0,1,by=cfg$stacking_grid_coarse), seq(0,1,by=cfg$stacking_grid_fine))) {
        for (w1 in step) for (w2 in step) {
          w_sum <- w1 + w2
          if (n_gm == 3) { w3 <- 1 - w_sum; if (w3 < 0 || w3 > 1) next; w <- c(w1,w2,w3) } else w <- c(w1,1-w1)
          gval_p <- gval_mat %*% w
          agg <- aggregate_predictions(as.numeric(gval_p), train_window_orig_idx[val_mask])
          m <- calc_metrics(labels[agg$original_indices], agg$predictions)
          if (!is.na(m["f1"]) && m["f1"] > best_gf1) { best_gf1 <- m["f1"]; best_gw <- w }
        }
      }
      # 权重正则化防止单一模型过拟合
      best_gw <- pmax(0.2, pmin(0.6, best_gw))
      best_gw <- best_gw / sum(best_gw)
      gtest_mat <- do.call(cbind, gfold_probs[g_avail]); gtest_ens <- as.numeric(gtest_mat %*% best_gw)
      gagg_test <- aggregate_predictions(gtest_ens, test_window_orig_idx)
      for (i in seq_along(gagg_test$original_indices)) {
        orig_i <- gagg_test$original_indices[i]
        results$ensemble_global$all_pred[orig_i] <- gagg_test$predictions[i]
      }
    }
    }
    }, error = function(e) {
      cat(sprintf("    Fold %d 内部错误: %s\n", fold_idx, conditionMessage(e)))
      cat(sprintf("    位置: %s\n", deparse(conditionCall(e), nlines = 1)[1]))
      NULL
    })  # 关闭 fold-level tryCatch

    # ---- 清理 ----
    rm(train_windows, test_windows, train_local, test_local, train_window_data, test_window_data,
       train_patches, val_patches, test_patches, deep_model); gc()
    tryCatch({ torch::cuda_empty_cache() }, error = function(e) NULL)
    setTxtProgressBar(fold_pb, fold_idx)
  }
  close(fold_pb)

  # ---- 计算最终指标 ----
  final_metrics <- list()
  for (nm in names(results)) {
    r <- results[[nm]]
    tested <- if (grepl("_global$", nm)) fold_tested_global else fold_tested
    if (sum(tested) > 0) {
      # 使用每个分类器在验证集上的最优阈值; 若无则用 0.5
      thresh <- if (length(r$best_thresholds) > 0) median(r$best_thresholds) else 0.5
      m <- calc_metrics(r$all_true[tested], r$all_pred[tested], threshold = thresh)
      cm <- calc_confusion(r$all_true[tested], r$all_pred[tested], threshold = thresh)
      final_metrics[[nm]] <- list(metrics = m, confusion = cm, n_tested = sum(tested))
    }
  }

  list(
    exp_name = exp_name,
    config = cfg,
    final_metrics = final_metrics,
    fold_history = do.call(rbind, all_history),
    attention_data = all_attention,
    patch_info = all_patch_info,
    raw_results = results,
    fold_tested = fold_tested,
    fold_tested_global = fold_tested_global,
    output_dir = exp_dir
  )
}
