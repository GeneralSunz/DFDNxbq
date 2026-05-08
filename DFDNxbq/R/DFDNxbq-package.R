#' DFDNxbq: Deep Fall Detect Net by xbq
#'
#' Implements the DFDNxbq fall detection pipeline:
#' \itemize{
#'   \item Deep encoders: ConvLSTM, CNN, GRU-TCN-SE
#'   \item HIBA hierarchical attention mechanism
#'   \item Ensemble classifiers: Random Forest, SVM, CatBoost
#'   \item LOSO (Leave-One-Subject-Out) cross-validation
#'   \item Signal denoising: Butterworth, Kalman, Wavelet, Savitzky-Golay
#' }
#'
#' @section Main entry points:
#' \code{\link{run_pipeline}} — high-level pipeline
#' \code{\link{run_single_experiment}} — single experiment execution
#' \code{\link{load_data_from_npz}} — load NPZ+CSV data
#'
#' @docType package
#' @name DFDNxbq
NULL

#' @import torch
#' @import signal
#' @importFrom randomForest randomForest
#' @importFrom e1071 svm
#' @importFrom dlm dlmModPoly dlmMLE dlmFilter
#' @importFrom wavelets dwt idwt
#' @importFrom reticulate import
NULL

.onLoad <- function(libname, pkgname) {
  # 抑制 torch 自带的加载验证（解决 reticulate conda 环境不匹配等问题）
  Sys.setenv(TORCH_VERIFY_LOAD = "FALSE")

  # 温和地检查 torch 是否可正常加载
  if (requireNamespace("torch", quietly = TRUE)) {
    tryCatch({
      if (!torch::torch_is_installed()) {
        packageStartupMessage(
          "=================================================================\n",
          "  警告: torch 的 Lantern 后端尚未安装。\n",
          "  请运行以下命令安装 PyTorch 后端依赖:\n",
          "    library(DFDNxbq)\n",
          "    install_dfdnxbq_deps()\n",
          "  或手动运行:\n",
          "    torch::install_torch()\n",
          "================================================================="
        )
      }
    }, error = function(e) {
      packageStartupMessage(
        "  注意: torch 环境检查异常 (", conditionMessage(e), ")\n",
        "  如需修复可尝试: install_dfdnxbq_deps(reinstall = TRUE)\n"
      )
    })
  }
}

#' Install torch and additional dependencies for DFDNxbq
#'
#' Installs the torch Lantern (libtorch) backend required for
#' deep neural network training. Handles common environment issues
#' such as reticulate / conda misconfiguration.
#'
#' @param reinstall Whether to reinstall torch from scratch (default FALSE)
#' @param ... Additional arguments passed to \code{torch::install_torch()}
#' @return Invisibly returns TRUE if installation succeeded, FALSE otherwise
#' @export
install_dfdnxbq_deps <- function(reinstall = FALSE, ...) {
  if (!requireNamespace("torch", quietly = TRUE)) {
    install.packages("torch")
  }

  # 重置 reticulate 配置，避免 conda 环境不匹配问题
  tryCatch({
    reticulate::py_config()
  }, error = function(e) {
    cat("注意: reticulate Python 配置异常 (", conditionMessage(e), ")\n")
    cat("尝试重置 reticulate 配置...\n")
    tryCatch({
      reticulate::py_available(initialize = TRUE)
    }, error = function(e2) NULL)
  })

  cat("正在安装 torch Lantern 后端 (libtorch) ...\n")
  cat("这可能需要几分钟时间，请耐心等待。\n\n")

  install_args <- list(...)
  if (reinstall) install_args$reinstall <- TRUE
  if (is.null(install_args$type)) install_args$type <- "binary"

  do.call(torch::install_torch, install_args)

  # 验证安装
  Sys.setenv(TORCH_VERIFY_LOAD = "FALSE")
  installed <- tryCatch(torch::torch_is_installed(), error = function(e) FALSE)
  if (installed) {
    cat("\n✓ torch 后端安装成功!\n")
    invisible(TRUE)
  } else {
    cat("\n✗ torch 后端安装似乎未完成。\n")
    cat("请尝试手动运行: torch::install_torch(reinstall = TRUE)\n")
    cat("或设置镜像: options(torch_repo = 'https://torch-cdn.mlverse.org')\n")
    invisible(FALSE)
  }
}

# ============================================================
# 全局基础配置 (各实验共享的参数)
# ============================================================

#' Default configuration for DFDNxbq experiments
#'
#' A list of hyperparameters controlling the entire pipeline:
#' signal processing, deep encoder architecture, HIBA attention,
#' training, denoising, and classifier settings.
#'
#' @format A list with the following components:
#' \describe{
#'   \item{fs}{Sampling frequency (Hz), default 238}
#'   \item{sp_ratio}{Stratified sampling ratio, default 0.01}
#'   \item{fast_subjects}{Subject IDs for LOSO test set}
#'   \item{L_pre_ms}{Pre-peak window length (ms), default 800}
#'   \item{delta_ms}{Sliding window delta (ms), default 50}
#'   \item{enable_dynamic_patching}{Whether to use adaptive patching}
#'   \item{encoder_type}{One of "cnn", "conv_lstm", "gru_tcn_se"}
#'   \item{use_hiba}{Whether to use HIBA attention}
#'   \item{d_model}{Embedding dimension}
#'   \item{n_epochs}{Training epochs}
#'   \item{batch_size}{Training batch size}
#'   \item{denoise_method}{Denoising method: "none", "butterworth", "kalman", "wavelet", "savgol"}
#'   ... (see source for full list)
#' }
#' @export
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
  encoder_type = "conv_lstm",
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
  focal_gamma_warmup = 8,
  # 降噪
  denoise_method = "none",
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
  svm_classwt_fall = 0,
  svm_kernel = "linear",
  svm_tune = FALSE,
  tune_classifiers = TRUE,
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
