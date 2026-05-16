# ============================================================
# online_inference.R — DFDNxbq 实时在线推理
#
# 提供两个核心函数：
#   1. create_streaming_predictor() — 创建流式预测器（闭包）
#   2. evaluate_online_simulation()  — 离线模拟评估
#
# 注意：训练阶段使用全局峰值锚定窗口，而推理阶段使用滑动窗口内
# 的局部最大峰值作为代理锚点。索引逻辑与训练时完全一致。
# ============================================================

# ---- 辅助：将 patch_hiba_net 模型包装为标准接口 ----

#' Wrap a patch_hiba_net model for streaming prediction
#'
#' patch_hiba_net 接受 patch 列表，而实时推理需要 [window_len, 6] 输入。
#' 此包装器在内部完成分块，对外暴露统一接口。
#'
#' @param model A trained patch_hiba_net model
#' @param enable_dynamic_patching Use adaptive patching (TRUE) or uniform (FALSE)
#' @param patch_short_len Short patch length for adaptive patching
#' @param patch_long_len Long patch length for adaptive patching
#' @param patch_overlap Overlap between adjacent patches
#' @param patch_min Minimum number of patches
#' @param patch_max Maximum number of patches
#' @param patch_min_len Minimum patch length
#' @return A function: f(window_matrix) -> probability scalar
#' @export
wrap_patch_hiba_model <- function(model,
                                   enable_dynamic_patching = TRUE,
                                   patch_short_len = 8,
                                   patch_long_len = 24,
                                   patch_overlap = 1,
                                   patch_min = 4,
                                   patch_max = 16,
                                   patch_min_len = 4) {
  force(model)
  model$eval()

  function(window_mat) {
    # window_mat: [window_len, 6] numeric matrix
    if (!is.matrix(window_mat)) window_mat <- as.matrix(window_mat)

    # 分块（与训练时一致）
    if (enable_dynamic_patching && nrow(window_mat) >= 16) {
      complexity <- compute_complexity_profile(window_mat, 5)
      patches <- adaptive_patching(window_mat, complexity,
                                    patch_short_len, patch_long_len,
                                    patch_overlap, patch_min,
                                    patch_max, patch_min_len)
    } else {
      patches <- uniform_patching(window_mat, 8)
    }

    if (length(patches) == 0) return(0)

    # 转为 torch tensors
    tensor_patches <- lapply(patches, function(p) {
      torch_tensor(p, dtype = torch_float())
    })

    # 推理
    out <- model(list(tensor_patches))
    prob <- as.numeric(torch_sigmoid(out$logits)$cpu())
    prob
  }
}

# ---- 1. 流式预测器工厂 ----

#' Create a real-time streaming fall warning predictor
#'
#' 返回一个闭包函数，每调用一次处理一个采样点，内部维护环形缓冲区状态。
#'
#' @param model A trained torch model, or a wrapped prediction function.
#'   If a torch nn_module, it should accept input [1, window_len, 6] and
#'   return logits (will apply sigmoid internally).
#'   If a function, it should accept a [window_len, 6] matrix and return
#'   a probability scalar in [0, 1].
#' @param scaler_mean Numeric vector of length 6 (channel-wise mean from training)
#' @param scaler_sd   Numeric vector of length 6 (channel-wise SD from training)
#' @param window_len  Window length in samples (default 190 = 800ms at 238Hz)
#' @param n_delta     Samples between window end and peak (default 12 = 50ms at 238Hz).
#'                    训练时 delta_ms=50 对应 50*238/1000≈12。
#' @param fs          Sampling frequency in Hz (default 238)
#' @param peak_thresh_g SVM (合加速度) threshold for local peak detection.
#'   Default 2.0g. 过低会增加误报，过高会漏检——建议通过验证集调优。
#' @param prob_threshold Probability threshold for triggering warning (default 0.5)
#' @param buffer_sec  Buffer duration in seconds (default 1.5)
#' @param local_peak_sec Look-back window in seconds for local peak search (default 1.0)
#' @param cooldown_ms Minimum interval between consecutive warnings in ms (default 500)
#' @param verbose     Print warning messages to console (default TRUE)
#'
#' @return A function `f(new_sample)` where `new_sample` is a numeric vector
#'   of length 6: c(acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z).
#'   Returns a list(warning = logical, prob = numeric, peak_value = numeric).
#'   `prob` is NA when no peak-based inference was performed.
#' @export
#'
#' @examples
#' \dontrun{
#'   # 加载训练好的模型和标准化参数
#'   model <- torch_load("model.pt")
#'   scaler_mean <- c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
#'   scaler_sd   <- c(1.0, 1.1, 1.2, 0.8, 0.9, 1.0)
#'
#'   # 创建预测器
#'   predict_fn <- create_streaming_predictor(model, scaler_mean, scaler_sd)
#'
#'   # 模拟实时数据流
#'   for (i in 1:n_samples) {
#'     result <- predict_fn(c(acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z))
#'     if (result$warning) cat("Warning triggered!\n")
#'   }
#' }
create_streaming_predictor <- function(model,
                                        scaler_mean,
                                        scaler_sd,
                                        window_len = 190,
                                        n_delta = 12,
                                        fs = 238,
                                        peak_thresh_g = 2.0,
                                        prob_threshold = 0.5,
                                        buffer_sec = 1.5,
                                        local_peak_sec = 1.0,
                                        cooldown_ms = 500,
                                        verbose = TRUE) {
  # ---- 参数校验 ----
  if (length(scaler_mean) != 6 || length(scaler_sd) != 6) {
    stop("scaler_mean 和 scaler_sd 必须为长度为 6 的数值向量")
  }
  if (any(scaler_sd <= 0)) {
    stop("scaler_sd 中所有值必须 > 0")
  }

  # ---- 派生参数 ----
  buf_size     <- round(buffer_sec * fs)          # 环形缓冲区大小 (样本数)
  local_peak_n <- round(local_peak_sec * fs)       # 局部峰值搜索窗口 (样本数)
  cooldown_n   <- round(cooldown_ms * fs / 1000)   # 冷却时间 (样本数)
  min_buf_len  <- window_len + n_delta             # 最小需要样本数: 190 + 12 = 202

  if (buf_size < min_buf_len) {
    stop(sprintf("buffer_sec=%.1f 产生的 buf_size=%d 小于最小需求 %d",
                 buffer_sec, buf_size, min_buf_len))
  }

  # ---- 判断 model 类型 ----
  if (is.function(model)) {
    predict_fn <- model
  } else {
    # 假设是 torch nn_module，接受 [batch, window_len, 6]
    model_obj <- model
    model_obj$eval()
    predict_fn <- function(window_mat) {
      x <- torch_tensor(window_mat, dtype = torch_float())$unsqueeze(1)
      x <- x$to(device = model_obj$classifier[[4]]$weight$device)
      with_no_grad({
        logit <- model_obj(x)
        as.numeric(torch_sigmoid(logit)$cpu())
      })
    }
  }

  # ---- 环形缓冲区状态（environment 避免 <<- 子赋值问题）----
  state <- new.env(parent = emptyenv())
  state$buf   <- matrix(0, nrow = buf_size, ncol = 6)  # 预分配
  state$head  <- 1L                                      # 下次写入位置 (1-based)
  state$count <- 0L                                      # 累计写入样本数
  state$last_warning_sample <- -Inf                      # 上次预警时的累计样本序号

  # ---- 标准化函数（捕获 scaler 参数）----
  standardize <- function(mat) {
    # mat: [window_len, 6]
    out <- mat
    for (ch in 1:6) {
      out[, ch] <- (out[, ch] - scaler_mean[ch]) / scaler_sd[ch]
    }
    out
  }

  # ---- 返回的预测函数 ----
  function(new_sample) {
    # ① 写入环形缓冲区
    state$buf[state$head, ] <- as.numeric(new_sample[1:6])
    state$head  <- (state$head %% buf_size) + 1L
    state$count <- state$count + 1L

    n_valid <- min(state$count, buf_size)

    # ② 检查缓冲区是否足够
    if (n_valid < min_buf_len) {
      return(list(warning = FALSE, prob = NA_real_,
                  peak_value = NA_real_, peak_idx = NA_integer_))
    }

    # ③ 冷却检查
    if (state$count - state$last_warning_sample < cooldown_n) {
      return(list(warning = FALSE, prob = NA_real_,
                  peak_value = NA_real_, peak_idx = NA_integer_))
    }

    # ④ 提取按时间排序的有效数据
    if (state$count <= buf_size) {
      chron_data <- state$buf[1:n_valid, , drop = FALSE]
    } else {
      # 环形: 从 head 到末尾，再从 1 到 head-1
      chron_data <- rbind(
        state$buf[state$head:buf_size, , drop = FALSE],
        state$buf[1:(state$head - 1), , drop = FALSE]
      )
    }

    # ⑤ 计算合加速度 SVM
    svm <- sqrt(rowSums(chron_data[, 1:3, drop = FALSE]^2))

    # ⑥ 在最近 local_peak_n 个样本中搜索局部峰值
    search_start <- max(1, n_valid - local_peak_n + 1)
    search_svm   <- svm[search_start:n_valid]
    rel_peak     <- which.max(search_svm)                        # 搜索窗口内的相对位置
    peak_value   <- search_svm[rel_peak]
    peak_idx     <- search_start + rel_peak - 1                   # 时间序绝对位置

    # ⑦ 峰值未达阈值，跳过
    if (peak_value < peak_thresh_g) {
      return(list(warning = FALSE, prob = NA_real_,
                  peak_value = peak_value, peak_idx = peak_idx))
    }

    # ⑧ 按训练时规则提取窗口: [peak_idx - window_len - n_delta + 1, peak_idx - n_delta]
    win_start <- peak_idx - window_len - n_delta + 1
    win_end   <- peak_idx - n_delta

    if (win_start < 1 || win_end > n_valid) {
      return(list(warning = FALSE, prob = NA_real_,
                  peak_value = peak_value, peak_idx = peak_idx))
    }

    window_data <- chron_data[win_start:win_end, , drop = FALSE]

    # ⑨ 标准化 + 预测
    window_std <- standardize(window_data)
    prob <- tryCatch(
      predict_fn(window_std),
      error = function(e) {
        if (verbose) message("预测出错: ", e$message)
        NA_real_
      }
    )

    if (is.na(prob)) {
      return(list(warning = FALSE, prob = NA_real_,
                  peak_value = peak_value, peak_idx = peak_idx))
    }

    # ⑩ 阈值判断
    warning_triggered <- (prob >= prob_threshold)

    if (warning_triggered) {
      state$last_warning_sample <- state$count
      if (verbose) {
        cat(sprintf("[WARNING] 样本#%d | 峰值=%.2fg | 概率=%.4f | 峰值位置=%d\n",
                    state$count, peak_value, prob, peak_idx))
      }
    }

    list(warning    = warning_triggered,
         prob       = prob,
         peak_value = peak_value,
         peak_idx   = peak_idx)
  }
}

# ---- 2. 离线模拟评估 ----

#' Evaluate streaming predictor on a test set (offline simulation)
#'
#' 对每条测试记录逐采样点模拟实时数据流，统计预警情况并计算混淆矩阵。
#'
#' @param data_list List of data items, each with Acc_raw (matrix), Gyr_raw (matrix),
#'   IsFall (logical), SubjectID, etc.
#' @param predictor_factory A function that returns a streaming predictor.
#'   Typically \code{create_streaming_predictor()} with fixed parameters.
#' @param warning_margin_ms Minimum ms before true peak for a warning to count
#'   as "timely" for fall samples. Default 100ms. Warning must occur at least
#'   this many ms before the true impact peak to be a TP.
#' @param fs Sampling frequency (Hz), default 238
#' @param verbose No longer used (kept for backward compatibility)
#' @param pb Show progress bar (default TRUE)
#'
#' @return A list with:
#'   \item{metrics}{Named vector: recall, specificity, precision, f1, auc, tp, tn, fp, fn}
#'   \item{trial_results}{Data frame with per-trial results}
#'   \item{config}{List of evaluation parameters}
#' @export
#'
#' @examples
#' \dontrun{
#'   result <- evaluate_online_simulation(
#'     data_list = test_data,
#'     predictor_factory = function() {
#'       create_streaming_predictor(model, scaler_mean, scaler_sd,
#'         peak_thresh_g = 2.0, prob_threshold = 0.5)
#'     }
#'   )
#'   print(result$metrics)
#' }
evaluate_online_simulation <- function(data_list,
                                        predictor_factory,
                                        warning_margin_ms = 100,
                                        fs = 238,
                                        verbose = TRUE,
                                        pb = TRUE) {
  n_trials <- length(data_list)
  warning_margin_n <- round(warning_margin_ms * fs / 1000)  # 样本数

  # 结果容器
  trial_results <- data.frame(
    trial_id    = seq_len(n_trials),
    is_fall     = logical(n_trials),
    n_samples   = integer(n_trials),
    true_peak_idx    = integer(n_trials),
    true_peak_value  = numeric(n_trials),
    first_warning_sample = integer(n_trials),  # 0 = no warning
    warning_before_peak = logical(n_trials),
    pred_prob_at_warning = numeric(n_trials),
    stringsAsFactors = FALSE
  )

  if (pb) pb_bar <- txtProgressBar(min = 0, max = n_trials, style = 3, width = 50)

  for (trial_idx in seq_len(n_trials)) {
    item <- data_list[[trial_idx]]
    acc  <- item$Acc_raw
    gyr  <- item$Gyr_raw

    if (is.null(acc) || is.null(gyr) || nrow(acc) == 0) {
      trial_results$is_fall[trial_idx]     <- isTRUE(item$IsFall)
      trial_results$n_samples[trial_idx]   <- 0
      trial_results$true_peak_idx[trial_idx]   <- NA_integer_
      trial_results$true_peak_value[trial_idx] <- NA_real_
      trial_results$first_warning_sample[trial_idx] <- 0L
      trial_results$warning_before_peak[trial_idx]  <- FALSE
      trial_results$pred_prob_at_warning[trial_idx] <- NA_real_
      if (pb) setTxtProgressBar(pb_bar, trial_idx)
      next
    }

    n_samp <- nrow(acc)
    is_fall <- isTRUE(item$IsFall)

    # 计算真实冲击峰值（全局最大 SVM）
    true_svm <- sqrt(rowSums(acc[, 1:3, drop = FALSE]^2))
    true_peak_idx   <- which.max(true_svm)
    true_peak_value <- true_svm[true_peak_idx]

    # 创建新的预测器实例（每条记录独立状态）
    predictor <- predictor_factory()

    # 逐采样点模拟
    first_warning_sample <- 0L
    warning_prob         <- NA_real_

    for (s in seq_len(n_samp)) {
      sample_vec <- c(acc[s, 1], acc[s, 2], acc[s, 3],
                      gyr[s, 1], gyr[s, 2], gyr[s, 3])
      result <- predictor(sample_vec)

      if (isTRUE(result$warning)) {
        first_warning_sample <- s
        warning_prob <- result$prob
        break  # 只记录首次预警
      }
    }

    # 判断是否在峰值前发出预警
    warning_before_peak <- FALSE
    if (first_warning_sample > 0 && is_fall) {
      # 预警需在真实冲击峰值前至少 warning_margin_n 个样本
      warning_before_peak <- (first_warning_sample <= true_peak_idx - warning_margin_n)
    }

    trial_results$is_fall[trial_idx]            <- is_fall
    trial_results$n_samples[trial_idx]          <- n_samp
    trial_results$true_peak_idx[trial_idx]      <- true_peak_idx
    trial_results$true_peak_value[trial_idx]    <- true_peak_value
    trial_results$first_warning_sample[trial_idx]  <- first_warning_sample
    trial_results$warning_before_peak[trial_idx]   <- warning_before_peak
    trial_results$pred_prob_at_warning[trial_idx]  <- warning_prob

    if (pb) setTxtProgressBar(pb_bar, trial_idx)
  }

  if (pb) close(pb_bar)

  # ---- 计算混淆矩阵 ----
  fall_mask <- trial_results$is_fall
  adl_mask  <- !trial_results$is_fall

  tp <- sum(fall_mask & trial_results$warning_before_peak, na.rm = TRUE)
  fn <- sum(fall_mask & !trial_results$warning_before_peak, na.rm = TRUE)
  fp <- sum(adl_mask & (trial_results$first_warning_sample > 0), na.rm = TRUE)
  tn <- sum(adl_mask & (trial_results$first_warning_sample == 0), na.rm = TRUE)

  recall      <- if ((tp + fn) > 0) tp / (tp + fn) else NA_real_
  specificity <- if ((tn + fp) > 0) tn / (tn + fp) else NA_real_
  precision   <- if ((tp + fp) > 0) tp / (tp + fp) else NA_real_
  f1 <- if (!is.na(precision) && !is.na(recall) && (precision + recall) > 0) {
    2 * precision * recall / (precision + recall)
  } else NA_real_

  scores <- ifelse(trial_results$first_warning_sample > 0,
                   trial_results$pred_prob_at_warning, 0)
  scores[is.na(scores)] <- 0
  auc <- NA_real_
  tryCatch({
    if (length(unique(trial_results$is_fall)) > 1) {
      auc <- as.numeric(pROC::auc(pROC::roc(trial_results$is_fall, scores, quiet = TRUE)))
    }
  }, error = function(e) {})

  metrics <- c(recall = recall, specificity = specificity, precision = precision,
               f1 = f1, auc = auc, tp = tp, tn = tn, fp = fp, fn = fn)

  cat(sprintf("  [在线评估] Recall=%.4f  Spec=%.4f  Prec=%.4f  F1=%.4f  AUC=%.4f  |  TP=%d  FN=%d  FP=%d  TN=%d\n",
              recall, specificity, precision, f1, auc, tp, fn, fp, tn))

  list(
    metrics       = metrics,
    trial_results = trial_results,
    config = list(
      warning_margin_ms = warning_margin_ms,
      fs                = fs,
      n_trials          = n_trials
    )
  )
}

# ---- 3. 使用示例 ----

#' Example: Run online simulation evaluation
#'
#' 完整示例：加载模型、标准化参数，在测试集上运行在线模拟评估。
#'
#' @param model_path Path to saved torch model (.pt file)
#' @param scaler_path Path to saved scaler parameters (.rds file with list(mean, sd))
#' @param test_data_path Path to test data .rds file
#' @param peak_thresh_g SVM threshold for peak detection
#' @param prob_threshold Decision threshold
#' @param model_is_patch_hiba Whether the model is a patch_hiba_net (needs wrapping)
#' @return Evaluation result list
#' @export
run_online_evaluation <- function(model_path,
                                   scaler_path = NULL,
                                   test_data_path,
                                   peak_thresh_g = 2.0,
                                   prob_threshold = 0.5,
                                   model_is_patch_hiba = TRUE) {
  # 加载模型
  cat("加载模型:", model_path, "\n")
  model <- torch_load(model_path)
  model$eval()

  # 加载或构造标准化参数
  if (!is.null(scaler_path) && file.exists(scaler_path)) {
    cat("加载标准化参数:", scaler_path, "\n")
    scaler <- readRDS(scaler_path)
    scaler_mean <- scaler$mean
    scaler_sd   <- scaler$sd
  } else {
    # 若未提供，使用占位值（实际使用时必须提供训练时的值）
    warning("未提供 scaler 参数文件，使用单位标准化（可能不准确）")
    scaler_mean <- rep(0, 6)
    scaler_sd   <- rep(1, 6)
  }

  # 加载测试数据
  cat("加载测试数据:", test_data_path, "\n")
  test_data <- readRDS(test_data_path)

  # 包装模型
  if (model_is_patch_hiba) {
    cat("包装 patch_hiba_net 模型...\n")
    predict_fn <- wrap_patch_hiba_model(model)
  } else {
    predict_fn <- model  # 直接使用
  }

  # 创建预测器工厂
  predictor_factory <- function() {
    create_streaming_predictor(
      model           = predict_fn,
      scaler_mean     = scaler_mean,
      scaler_sd       = scaler_sd,
      peak_thresh_g   = peak_thresh_g,
      prob_threshold  = prob_threshold,
      verbose         = FALSE   # 评估时关闭逐条输出
    )
  }

  # 运行评估
  evaluate_online_simulation(
    data_list          = test_data,
    predictor_factory  = predictor_factory,
    verbose            = TRUE
  )
}
