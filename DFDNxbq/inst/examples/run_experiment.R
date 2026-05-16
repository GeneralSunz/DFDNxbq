#' ============================================================
#' run_experiment.R — DFDNxbq 完整实验示例
#' ============================================================
#'
#' 前提:
#'   1. 已安装 DFDNxbq 包
#'   2. FallAllD.pkl 已通过 convert_pkl_to_rdata.py 转换为 NPZ+CSV
#'   3. reticulate 可访问 Python numpy
#'   4. torch 后端已安装 (运行 install_dfdnxbq_deps())
#'
#' 首次使用前请确保 torch 后端已安装:
#'   library(DFDNxbq)
#'   install_dfdnxbq_deps()
#'
#' 转换命令 (在 functionbag/ 目录下):
#'   python convert_pkl ../FallAllD.pkl ./output
#'
#' 使用方式:
#'   library(DFDNxbq)
#'   source("run_experiment.R")
#' ============================================================
Sys.setenv(TORCH_VERIFY_LOAD = "FALSE")
library(DFDNxbq)

# ---- 1. 数据加载 ----
npz_path <- "output/fallalld_signals.npz"
csv_path <- "output/fallalld_metadata.csv"

if (!file.exists(npz_path)) {
  stop("请先运行: python convert_pkl_to_rdata.py ../FallAllD.pkl ./output")
}

data_list <- load_data_from_npz(npz_path, csv_path)

## ---- 2. 管道运行 (抽样 0.1%) ----
# result <- run_pipeline(
#   data_list,
#   output_dir   = "output",
#   sample_ratio = 0.001
# )

# # ---- 3. 查看结果 ----
# cat("\n===== 关键指标 =====\n")
# for (nm in names(result$final_metrics)) {
#   m <- result$final_metrics[[nm]]$metrics
#   if (!is.null(m) && !is.na(m["auc"])) {
#     cat(sprintf("%-20s AUC=%.3f  F1=%.3f  Recall=%.3f  Spec=%.3f\n",
#                 nm, m["auc"], m["f1"], m["recall"], m["specificity"]))
#   }
# }

# ---- 4. 直接使用单实验函数 ----
# library(DFDNxbq)
sample_ratio <- 0.03  # 抽样比例
data_subset  <- stratified_sample(data_list, frac = sample_ratio)

labels   <- build_labels(data_subset)
subjects <- build_subjects(data_subset)
cat(sprintf("抽样后: %d 样本 (%.1f%% 跌倒)\n", length(data_subset), mean(labels) * 100))

cfg <- BASE_CONFIG
cfg$encoder_type    <- "cnn" # 编码器选择
cfg$denoise_method  <- "butterworth"  # 降噪方式选择
cfg$use_hiba        <- TRUE # 是否使用 HIBA 模块
cfg$n_epochs        <- 50   # 训练轮数

result2 <- run_single_experiment(
  cfg, data_subset, labels, subjects, # 实验参数
  exp_name      = "cnn_butterworth_test", # 实验名称
  output_subdir = "output/cnn_test" # 输出子目录
)

cat("\n===== 单实验指标 =====\n")
for (nm in names(result2$final_metrics)) {
  m <- result2$final_metrics[[nm]]$metrics
  if (!is.null(m) && !is.na(m["auc"])) {
    cat(sprintf("%-20s AUC=%.3f  F1=%.3f\n", nm, m["auc"], m["f1"]))
  }
}

# ============================================================
# ---- 5. 在线实时推理 ----
# ============================================================
cat("\n\n========== 在线实时推理 ==========\n")

# 准备标准化参数 + 10-epoch 演示模型
windows <- do.call(c, lapply(data_subset, \(item) {
  r <- generate_sliding_windows(item, cfg$delta_ms, 190, cfg$fs)
  lapply(r$windows, `[[`, "data")
}))
arr <- array(unlist(windows), dim = c(190, 6, length(windows)))
scaler <- list(mean = apply(arr, 2, mean), sd = {s <- apply(arr, 2, sd); s[s==0]<-1; s})

demo_data <- lapply(windows, \(w) as.matrix(scale(w, scaler$mean, scaler$sd)))
demo_patches <- lapply(demo_data, \(sig) uniform_patching(sig, 8))
n_win <- sapply(data_subset, \(x) length(generate_sliding_windows(x, cfg$delta_ms, 190, cfg$fs)$windows))
demo_labels <- build_labels(data_subset)[rep(seq_along(data_subset), n_win)]

model <- train_patch_hiba_model(demo_patches, demo_labels,
  n_epochs = 10, batch_size = 32, patience = 5, d_model = 16,
  encoder_type = "cnn", use_hiba = TRUE, do_oversample = TRUE)$model

# 推理管线
predict_fn <- wrap_patch_hiba_model(model)
streaming <- create_streaming_predictor(predict_fn, scaler$mean, scaler$sd, peak_thresh_g = 2.0)

# 单条流式测试
item <- data_subset[[which(labels)[1]]]
for (s in seq_len(nrow(item$Acc_raw))) {
  r <- streaming(c(item$Acc_raw[s, ], item$Gyr_raw[s, ]))
  if (r$warning) { cat(sprintf("  预警 @%d/%d | prob=%.4f\n", s, nrow(item$Acc_raw), r$prob)); break }
}

# 全量在线评估
evaluate_online_simulation(data_subset, \()
  create_streaming_predictor(predict_fn, scaler$mean, scaler$sd, peak_thresh_g = 2.0))
