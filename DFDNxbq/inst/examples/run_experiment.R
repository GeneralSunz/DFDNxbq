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
sample_ratio <- 0.001  # 抽样比例
data_subset  <- stratified_sample(data_list, frac = sample_ratio)

labels   <- build_labels(data_subset)
subjects <- build_subjects(data_subset)
cat(sprintf("抽样后: %d 样本 (%.1f%% 跌倒)\n", length(data_subset), mean(labels) * 100))

cfg <- BASE_CONFIG
cfg$encoder_type    <- "cnn"
cfg$denoise_method  <- "butterworth"
cfg$use_hiba        <- TRUE
cfg$n_epochs        <- 10  

result2 <- run_single_experiment(
  cfg, data_subset, labels, subjects,
  exp_name      = "cnn_butterworth_test",
  output_subdir = "output/cnn_test"
)

cat("\n===== 单实验指标 =====\n")
for (nm in names(result2$final_metrics)) {
  m <- result2$final_metrics[[nm]]$metrics
  if (!is.null(m) && !is.na(m["auc"])) {
    cat(sprintf("%-20s AUC=%.3f  F1=%.3f\n", nm, m["auc"], m["f1"]))
  }
}
