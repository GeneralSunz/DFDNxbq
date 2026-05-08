#!/usr/bin/env Rscript
# ============================================================
# pipeline.R – IVFastPavlov 函数包主入口
# 完全自包含于 functionbag/ 目录
#
# 数据流:
#   PKL → convert_pkl_to_rdata.py → NPZ+CSV → pipeline.R → RDS结果
#
# 使用方式:
#   cd functionbag
#   Rscript pipeline.R <npz_path> <csv_path> [output_dir] [sample_ratio]
# ============================================================

# ---- 0. 参数解析 ----
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  cat("Usage:\n")
  cat("  cd functionbag\n")
  cat("  Rscript pipeline.R <npz_path> <csv_path> [output_dir] [sample_ratio]\n")
  quit(status = 1)
}
npz_path <- args[1]
csv_path <- args[2]
out_dir  <- if (length(args) >= 3) args[3] else "output"
sp_ratio <- if (length(args) >= 4) as.numeric(args[4]) else NULL

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# ---- 1. 工作目录验证 ----
if (!file.exists("source/IVFastPavlov_core.R")) {
  stop("请在 functionbag/ 目录下运行:\n  cd functionbag && Rscript pipeline.R ...")
}

# ---- 2. 依赖包 ----
required <- c("torch", "abind", "randomForest", "e1071", "pROC", "reticulate")
for (pkg in required) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    cat(sprintf("[pipeline] 安装 %s ...\n", pkg))
    install.packages(pkg, repos = "https://cloud.r-project.org")
    library(pkg, character.only = TRUE)
  }
}

# ---- 3. 载入降噪函数 (从 source/) ----
denoise_files <- c("denoise_butterworth.R", "denoise_kalman.R",
                   "denoise_wavelet.R", "denoise_savgol.R")
for (f in denoise_files) {
  fp <- file.path("source", f)
  if (file.exists(fp)) source(fp, local = TRUE)
}

# ---- 4. 载入 IVFastPavlov 核心函数 (从 source/) ----
# 不含 setup 副作用: rm(list=ls()), library(), set.seed(), OUTPUT_DIR
source("source/IVFastPavlov_core.R", local = TRUE)

# ---- 5. 数据载入 ----
load_data_from_npz <- function(npz_path, csv_path) {
  np <- import("numpy")
  signals <- np$load(npz_path)
  acc_all <- signals$get("acc_all")
  gyr_all <- signals$get("gyr_all")
  signals$close()

  meta <- read.csv(csv_path, stringsAsFactors = FALSE)
  n <- nrow(meta)
  cat(sprintf("  [load] %d 样本, 信号维度: %s\n", n,
              paste(dim(acc_all), collapse = " x ")))

  data_list <- vector("list", n)
  pb <- txtProgressBar(min = 0, max = n, style = 3, width = 40)
  for (i in seq_len(n)) {
    acc_mat <- acc_all[i, , ]
    gyr_mat <- gyr_all[i, , ]
    if (is.null(dim(acc_mat))) acc_mat <- matrix(acc_mat, nrow = 1)
    if (is.null(dim(gyr_mat))) gyr_mat <- matrix(gyr_mat, nrow = 1)
    data_list[[i]] <- list(
      Acc_raw    = acc_mat,
      Gyr_raw    = gyr_mat,
      IsFall     = as.logical(meta$IsFall[i]),
      SubjectID  = meta$SubjectID[i],
      ActivityID = meta$ActivityID[i],
      TrialNo    = meta$TrialNo[i]
    )
    setTxtProgressBar(pb, i)
  }
  close(pb)
  cat(sprintf("  [load] data_list 构建完成 (%d 个元素)\n", length(data_list)))
  data_list
}

# ---- 6. 数据验证 ----
validate_data <- function(data_list, labels, subjects, fast_subjects, verbose = TRUE) {
  n  <- length(data_list)
  nf <- sum(labels)
  na <- n - nf
  ns <- length(unique(subjects))
  ts <- intersect(fast_subjects, unique(subjects))
  warnings <- c()

  if (verbose) {
    cat(sprintf("[check] %d 样本: %d 跌倒, %d ADL (%.1f%%)\n",
                n, nf, na, 100 * mean(labels)))
    cat(sprintf("[check] %d 受试者, %d 在 fast_subjects 中\n", ns, length(ts)))
  }

  if (n < 10) {
    warnings <- c(warnings, sprintf("样本太少 (%d), LOSO 将无有效结果", n))
  }
  if (nf < 2) {
    warnings <- c(warnings, sprintf("跌倒样本不足 (%d), 分类器无法训练", nf))
  }
  if (na < 2) {
    warnings <- c(warnings, sprintf("ADL 样本不足 (%d), 分类器无法训练", na))
  }
  if (ns < 2) {
    warnings <- c(warnings, sprintf("受试者 < 2, LOSO 交叉验证无法进行"))
  }
  if (length(ts) < 1) {
    warnings <- c(warnings,
      sprintf("fast_subjects 与数据受试者无交集, 使用全部受试者"))
  }

  if (length(warnings) > 0) {
    cat("  [check] ⚠ 数据警告:\n")
    for (w in warnings) cat(sprintf("         - %s\n", w))
    if (n < 5 || nf < 1 || na < 1) {
      cat("  [check] ✗ 数据严重不足, 跳过实验\n")
      return(FALSE)
    }
    cat("  [check] → 继续执行, 但结果可能不可靠\n")
  } else {
    if (verbose) cat("  [check] ✓ 数据通过验证\n")
  }
  TRUE
}

# ---- 7. 主管道 ----
run_pipeline <- function(data_list, output_dir = "output",
                          sample_ratio = NULL,
                          config_overrides = list()) {
  labels   <- sapply(data_list, function(e) isTRUE(e$IsFall))
  subjects <- sapply(data_list, function(e) {
    if (is.null(e$SubjectID)) 0 else e$SubjectID
  })

  cat(sprintf("[pipeline] 加载 %d 样本 (%.1f%% 跌倒, %d 受试者)\n",
              length(data_list), 100 * mean(labels), length(unique(subjects))))

  # 过滤无正样本受试者
  subj_has_pos <- tapply(labels, subjects, any)
  valid_subjs  <- names(subj_has_pos)[subj_has_pos]
  keep <- subjects %in% valid_subjs
  data_list <- data_list[keep]
  labels    <- labels[keep]
  subjects  <- subjects[keep]
  cat(sprintf("[pipeline] 过滤无跌倒受试者后: %d 受试者\n",
              length(unique(subjects))))

  # 数据验证
  validated <- validate_data(data_list, labels, subjects,
                             BASE_CONFIG$fast_subjects)
  if (!validated) {
    cat("[pipeline] 数据验证未通过, 跳过实验\n")
    return(list(
      exp_name = "pipeline_run",
      final_metrics = list(),
      note = "数据不足, 跳过实验",
      output_dir = output_dir
    ))
  }

  # 分层抽样
  if (!is.null(sample_ratio) && sample_ratio < 1) {
    cat(sprintf("[pipeline] 分层抽样 ratio=%.4f ...\n", sample_ratio))
    data_list <- stratified_sample(data_list, sample_ratio, seed = 124514)
    labels    <- sapply(data_list, function(e) isTRUE(e$IsFall))
    subjects  <- sapply(data_list, function(e) {
      if (is.null(e$SubjectID)) 0 else e$SubjectID
    })
    cat(sprintf("[pipeline] 抽样后: %d 样本 (%.1f%% 跌倒)\n",
                length(data_list), 100 * mean(labels)))

    # 抽样后再次验证
    if (!validate_data(data_list, labels, subjects,
                       BASE_CONFIG$fast_subjects, verbose = FALSE)) {
      cat("[pipeline] 抽样后数据不足, 跳过实验\n")
      return(list(exp_name = "pipeline_run", final_metrics = list(),
                  note = "抽样后数据不足", output_dir = output_dir))
    }
  }

  # 配置
  cfg <- BASE_CONFIG
  for (n in names(config_overrides)) cfg[[n]] <- config_overrides[[n]]
  cfg$data_path <- NULL

  # 降噪
  cat(sprintf("[pipeline] 降噪: %s ...\n", cfg$denoise_method))
  data_list <- apply_denoise(data_list, cfg$denoise_method,
    fs = cfg$fs,
    bw_cutoff = cfg$denoise_bw_cutoff, bw_order = cfg$denoise_bw_order,
    kalman_dV = cfg$denoise_kalman_dV, kalman_dW = cfg$denoise_kalman_dW,
    wavelet_name = cfg$denoise_wavelet_name,
    wavelet_level = cfg$denoise_wavelet_level,
    wavelet_threshold = cfg$denoise_wavelet_threshold,
    sg_p = cfg$denoise_sg_p, sg_n = cfg$denoise_sg_n)

  # 运行实验
  cat(sprintf("[pipeline] 运行实验 (编码器=%s, HIBA=%s, 降噪=%s)\n",
              cfg$encoder_type, cfg$use_hiba, cfg$denoise_method))
  result <- run_single_experiment(cfg, data_list, labels, subjects,
                                   exp_name = "pipeline_run",
                                   output_subdir = output_dir)

  # 保存
  result_path <- file.path(output_dir, "pipeline_results.rds")
  saveRDS(result, result_path)
  cat(sprintf("[pipeline] 结果已保存: %s\n", result_path))

  # 摘要
  metrics_found <- FALSE
  cat("\n[pipeline] ===== 结果摘要 =====\n")
  for (nm in names(result$final_metrics)) {
    m <- result$final_metrics[[nm]]$metrics
    if (!is.null(m) && !is.na(m["auc"])) {
      metrics_found <- TRUE
      cat(sprintf("  %-20s AUC=%.3f F1=%.3f Recall=%.3f Spec=%.3f (n=%d)\n",
                  nm, m["auc"], m["f1"], m["recall"], m["specificity"],
                  result$final_metrics[[nm]]$n_tested))
    }
  }
  if (!metrics_found) {
    cat("  (无有效指标 – 可能数据不足以训练分类器)\n")
    cat("  建议: 增加样本数或检查 class balance\n")
  }

  result
}

# ---- 8. 主入口 ----
if (!interactive()) {
  cat("========================================\n")
  cat("IVFastPavlov Pipeline - 函数包\n")
  cat("========================================\n\n")

  cat(sprintf("  NPZ: %s\n  CSV: %s\n  Out: %s\n\n",
              npz_path, csv_path, out_dir))

  data_list <- load_data_from_npz(npz_path, csv_path)
  result <- run_pipeline(data_list, output_dir = out_dir,
                          sample_ratio = sp_ratio)
  cat("\n[pipeline] 管道执行完毕!\n")
}
