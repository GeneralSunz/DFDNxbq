# DFDNxbq: Deep Fall Detect Net by xbq


## 项目概述

DFDNxbq (Deep Fall Detect Net by xbq) 是一个基于深度学习的跌倒检测 R 包，专门用于处理可穿戴传感器（加速度计和陀螺仪）信号数据。项目实现了完整的跌倒检测流水线，包括：

- **信号降噪**：支持 Butterworth、Kalman、Wavelet、Savitzky-Golay 四种降噪方法
- **深度编码器**：提供 CNN、ConvLSTM、GRU-TCN-SE 三种深度神经网络架构，用于从信号片段中提取特征
- **HIBA 注意力机制**：分层层次间注意力（Hierarchical Inter-Block Attention），对编码后的信号片段进行自适应加权聚合
- **集成分类器**：使用 Random Forest、SVM、CatBoost 三种分类器进行最终分类，并通过加权集成提升性能
- **LOSO 交叉验证**：采用 Leave-One-Subject-Out 验证方案评估模型泛化能力

该流水线基于 FallAllD 数据集开发，支持从 NPZ 格式加载数据，也可通过 RDS 格式保存和复用预处理结果。

## 数据来源：
前往**https://www.kaggle.com/datasets/sankalpsinghvishen/derived-fallalld-dataset**
下载FallAllD.pkl原始数据。

## 环境要求

| 组件 | 版本 |
|------|------|
| R | 4.4.3 |
| Python | 3.7+ |
| 操作系统 | Windows 10/11 |

### R 包依赖

| 包名 | 类型 | 说明 |
|------|------|------|
| randomForest | Imports | 随机森林分类器 |
| e1071 | Imports | SVM 分类器 |
| pROC | Imports | ROC 曲线评估 |
| signal | Imports | 信号处理 |
| dlm | Imports | 卡尔曼滤波 |
| wavelets | Imports | 小波降噪 |
| reticulate | Imports | Python 接口 |
| torch | Suggests | 深度学习后端 |
| catboost | Suggests | CatBoost 分类器（本地加载） |

## 安装步骤

### 1. 安装 R 4.4.3

从 CRAN 下载并安装 R 4.4.3：

<https://cran.r-project.org/bin/windows/base/>

> **注意**：R 4.4.2 与 torch 包存在 DLL 兼容性问题（访问 `nn_module$parameters` 时段错误），必须使用 R 4.4.3。

### 2. 安装基础 R 包

启动 R，运行以下命令安装基础依赖：

```r
install.packages(c("randomForest", "e1071", "pROC", "signal", "dlm", "wavelets", "reticulate"),
                 type = "binary")
```

> **提示**：安装过程中如询问是否从源码编译，请选 **no**（选 **yes** 需要 RTools，且无必要）。

### 3. 检查 reticulate 配置

包加载时需要 `reticulate` 访问 Python 以加载 NPZ 数据。如果提示 conda 环境问题，检查 `~/.Rprofile` 中是否包含以下内容：

```r
library(reticulate)
use_condaenv("directml", required = TRUE)  # 建议将 required 改为 FALSE
```

将 `.Rprofile` 中的 `required = TRUE` 改为 `required = FALSE`，避免找不到 conda 环境时中断运行。

也可直接通过环境变量指定 Python 路径：

```r
Sys.setenv(RETICULATE_PYTHON = "path/to/your/python.exe")
```

### 4. 安装 DFDNxbq 包
注意替换实际路径
```r
install.packages("path\\DFDNxbq", repos = NULL, type = "source")
```

如果安装时报错，检查是否缺少依赖包，按第 2 步补装。

### 5. 安装 torch 深度学习后端

```r
install.packages("torch", type = "binary")
library(torch)
install_torch()
```

验证安装：

```r
library(torch)
x <- torch_tensor(c(1, 2, 3))
print(x)
```

> **注意**：如果 `install_torch()` 返回空，说明 Lantern 后端已安装。可通过创建 `nn_linear` 并访问 `$parameters` 验证兼容性：
> ```r
> m <- nn_linear(10, 5)
> length(m$parameters)  # 不应段错误
> ```

### 6. CatBoost（可选）

CatBoost 已在包内 `inst/extdata/catboost/` 目录下附带完整 R 包源码和 DLL，**无需单独安装**。运行时由 `load_catboost()` 函数自动从本地加载。

如需独立安装可直接运行：

```r
install.packages(
  "path\\DFDNxbq\\inst\\extdata\\catboost",
  repos = NULL, type = "source"
)
```

## 数据准备

FallAllD 数据集需要先转换为 R 可读格式。在 `functionbag/` 目录下运行：

```bash
python convert_pkl_to_rdata.py ../FallAllD.pkl ./output
```

生成两个文件：
- `output/fallalld_signals.npz` — 信号数据
- `output/fallalld_metadata.csv` — 元数据

## 运行实验

### 完整示例

```r
library(DFDNxbq)

# 加载数据
data_list <- load_data_from_npz("output/fallalld_signals.npz", "output/fallalld_metadata.csv")

# 构建标签和受试者信息
labels <- build_labels(data_list)
subjects <- build_subjects(data_list)

# 分层抽样（千分之一）
data_subset <- stratified_sample(data_list, frac = 0.001)
labels_sub <- build_labels(data_subset)
subjects_sub <- build_subjects(data_subset)

# 配置并运行实验
cfg <- BASE_CONFIG
cfg$encoder_type <- "cnn"
cfg$denoise_method <- "butterworth"
cfg$use_hiba <- TRUE
cfg$n_epochs <- 10

result <- run_single_experiment(
  cfg, data_subset, labels_sub, subjects_sub,
  exp_name = "cnn_butterworth_test",
  output_subdir = "output/cnn_test"
)

# 查看结果
for (nm in names(result$final_metrics)) {
  m <- result$final_metrics[[nm]]$metrics
  if (!is.null(m) && !is.na(m["auc"])) {
    cat(sprintf("%-20s AUC=%.3f  F1=%.3f\n", nm, m["auc"], m["f1"]))
  }
}
```

也可直接运行示例脚本：

```r
source("path\\DFDNxbq\\inst\\examples\\run_experiment.R", encoding = "UTF-8")
```

## 在线实时推理

训练阶段采用"峰值锚定"方法：先计算整条动作记录的合加速度全局最大值（冲击峰值），然后截取该峰值前 800ms 的窗口作为训练样本。**部署时无法提前知道未来峰值**，因此推理阶段使用滑动窗口内的**局部最大峰值**作为代理锚点，实现真正的在线预测。

### 架构说明

| 阶段 | 锚点来源 | 窗口提取方式 |
|------|---------|-------------|
| 训练 | 全局合加速度最大值 | `[peak - 201, peak - 12]`，不含峰值点 |
| 推理 | 缓冲区最近 1 秒内的局部最大值 | 同上，索引逻辑完全一致 |

推理流程：

1. 环形缓冲区（默认 1.5 秒）接收实时六轴数据流
2. 每当缓冲区满 202 个样本后，在最近 1 秒内搜索 SVM（合加速度）局部最大值
3. 若局部峰值 ≥ `peak_thresh_g`，提取其前 800ms 窗口（不含峰值点）
4. 用训练时的标准化参数（`scaler_mean` / `scaler_sd`）对窗口做 z-score 标准化
5. 送入模型预测，若概率 ≥ `prob_threshold` 则触发预警
6. 冷却机制（默认 500ms）防止同一事件重复预警

### 新增函数

| 函数 | 说明 |
|------|------|
| `wrap_patch_hiba_model()` | 将 `patch_hiba_net` 模型（接受 patch 列表）包装为标准窗口输入接口 |
| `create_streaming_predictor()` | 创建流式预测器闭包，维护环形缓冲区状态，每调用一次处理一个采样点 |
| `evaluate_online_simulation()` | 逐 trial 模拟实时数据流，计算 trial 级别的混淆矩阵和指标 |
| `run_online_evaluation()` | 一键函数：加载模型 → 加载 scaler → 加载测试数据 → 运行在线评估 |

### 使用示例

```r
library(DFDNxbq)

# ---- 方式一：离线模拟评估 ----
# 前提：已有训练好的模型文件和标准化参数文件

model <- torch_load("output/model.pt")
scaler <- readRDS("output/scaler.rds")  # list(mean = c(...), sd = c(...))
test_data <- readRDS("output/test_data.rds")

# 包装 patch_hiba_net 模型（如果模型直接接受 [window_len, 6] 则跳过此步）
predict_fn <- wrap_patch_hiba_model(model)

# 创建预测器工厂
factory <- function() {
  create_streaming_predictor(
    model          = predict_fn,
    scaler_mean    = scaler$mean,
    scaler_sd      = scaler$sd,
    peak_thresh_g  = 2.0,        # SVM 局部峰值阈值 (g)
    prob_threshold = 0.5,        # 模型概率阈值
    verbose        = FALSE
  )
}

# 运行评估
result <- evaluate_online_simulation(test_data, factory)
print(result$metrics)
# 输出: recall, specificity, precision, f1, auc, tp, tn, fp, fn

# ---- 方式二：真实实时数据流 ----
streaming <- create_streaming_predictor(
  model = predict_fn,
  scaler_mean = scaler$mean,
  scaler_sd   = scaler$sd,
  peak_thresh_g = 2.0,
  prob_threshold = 0.5
)

# 模拟逐个采样点到达
for (i in 1:n_samples) {
  sample <- c(acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z)
  result <- streaming(sample)
  if (result$warning) {
    cat(sprintf("跌倒预警! 概率=%.4f\n", result$prob))
  }
}

# ---- 方式三：一键评估 ----
result <- run_online_evaluation(
  model_path     = "output/model.pt",
  scaler_path    = "output/scaler.rds",
  test_data_path = "output/test_data.rds",
  peak_thresh_g  = 2.0,
  prob_threshold = 0.5,
  model_is_patch_hiba = TRUE
)
```

### 参数调优建议

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `peak_thresh_g` | 2.0 | SVM 局部峰值阈值 (g)。**过低**（< 1.5g）→ 日常活动频繁触发检测 → 误报增加；**过高**（> 3.0g）→ 漏检跌倒前的预警窗口 → 召回率下降。建议在验证集上网格搜索 1.5/2.0/2.5/3.0 |
| `prob_threshold` | 0.5 | 模型输出概率阈值。可用 `find_best_threshold()` 在验证集上确定最优值 |
| `buffer_sec` | 1.5 | 缓冲区时长 (秒)。需 ≥ (window_len + n_delta) / fs ≈ 0.85s，推荐 1.5s 留有余量 |
| `cooldown_ms` | 500 | 预警冷却时间 (ms)。防止同一事件触发多次预警 |

### 评估指标说明

离线模拟评估按 **trial 级别**（非窗口级别）计算混淆矩阵：

| 情况 | 判定 |
|------|------|
| 跌倒记录 + 在真实冲击峰值前 ≥ 100ms 发出预警 | TP |
| 跌倒记录 + 未预警或预警太晚 | FN |
| 日常活动 + 触发了任意预警 | FP |
| 日常活动 + 未触发预警 | TN |

## 实验结果

以下结果基于 FallAllD 数据集 **千分之一（0.1%）分层抽样**，10 个训练周期。

### 实验配置

| 参数 | 值 |
|------|-----|
| 编码器 | CNN |
| 降噪 | Butterworth |
| HIBA 注意力 | 启用 |
| 训练周期 | 10 |
| 验证方案 | LOSO（留一受试者交叉验证） |
| 分块策略 | 动态自适应分块 |
| 抽样比例 | 0.001 |

### 运行日志摘要

```
[load] 6605 样本, 信号维度: 6605 x 4760 x 3
[load] data_list 构建完成 (6605 个元素)
Sampled: 28 samples (13 fall, 46.4%)
encoder = cnn  |  denoise = butterworth  |  HIBA = ON  |  epochs = 10
======================================================================
  实验: cnn_butterworth_test
  编码器: cnn | 分块: 动态 | HIBA: ON | 降噪: butterworth
======================================================================

  |                                                  |   0%
  --- Fold 1/13 ---
    Train:17 Val:7 Test:2 windows
    [分布] Train: Fall=9 ADL=8 | Val: Fall=3 ADL=4 | Test: Fall=1 ADL=1
    使用平衡采样 (Fall=9, ADL=8), gamma预热=8 epoch
  |========================================| 100%
  Epoch 1/10: train_loss = 0.3406 | val_loss = 0.6922
  |========================================| 100%
  Epoch 2/10: train_loss = 0.3459 | val_loss = 0.6925
  |========================================| 100%
  Epoch 3/10: train_loss = 0.3505 | val_loss = 0.6927
  |========================================| 100%
  Epoch 4/10: train_loss = 0.3480 | val_loss = 0.6929
  |========================================| 100%
  Epoch 5/10: train_loss = 0.3441 | val_loss = 0.6930
  |========================================| 100%
  Epoch 6/10: train_loss = 0.3486 | val_loss = 0.6931
  |========================================| 100%
  Epoch 7/10: train_loss = 0.3456 | val_loss = 0.6932
  |========================================| 100%
  Epoch 8/10: train_loss = 0.3459 | val_loss = 0.6933
  |========================================| 100%
  Epoch 9/10: train_loss = 0.2503 | val_loss = 0.6934
  |========================================| 100%
  Epoch 10/10: train_loss = 0.1768 | val_loss = 0.6935
    [嵌入] train: 17x16, mean=-0.0266, sd=0.3537, NaN=0, Inf=0
    [嵌入] val: 7x16, mean=-0.0260, sd=0.3548
    [嵌入] test: 2x16, mean=-0.0264, sd=0.3592
    [HIBA] 注意力: mean=0.0645, sd=0.0021, range=[0.0624, 0.0667]

 |==============================================    |  92%
  --- Fold 13/13 ---
    Train:17 Val:7 Test:2 windows
    [分布] Train: Fall=6 ADL=11 | Val: Fall=6 ADL=1 | Test: Fall=1 ADL=1
    使用平衡采样 (Fall=6, ADL=11), gamma预热=8 epoch
  |========================================| 100%
  Epoch 1/10: train_loss = 0.3386 | val_loss = 0.7672
  |========================================| 100%
  Epoch 2/10: train_loss = 0.3315 | val_loss = 0.7706
  |========================================| 100%
  Epoch 3/10: train_loss = 0.3390 | val_loss = 0.7736
  |========================================| 100%
  Epoch 4/10: train_loss = 0.3312 | val_loss = 0.7764
  |========================================| 100%
  Epoch 5/10: train_loss = 0.3334 | val_loss = 0.7794
  |========================================| 100%
  Epoch 6/10: train_loss = 0.3308 | val_loss = 0.7823
  |========================================| 100%
  Epoch 7/10: train_loss = 0.3298 | val_loss = 0.7851
  |========================================| 100%
  Epoch 8/10: train_loss = 0.3323 | val_loss = 0.7877
  |========================================| 100%
  Epoch 9/10: train_loss = 0.2356 | val_loss = 0.7900
  |========================================| 100%
  Epoch 10/10: train_loss = 0.1609 | val_loss = 0.7916
    [嵌入] train: 17x16, mean=-0.0114, sd=0.3511, NaN=0, Inf=0
    [嵌入] val: 7x16, mean=-0.0099, sd=0.3530
    [嵌入] test: 2x16, mean=-0.0102, sd=0.3544
    [HIBA] 注意力: mean=0.0625, sd=0.0000, range=[0.0624, 0.0626]
  |==================================================| 100%
13-fold LOSO completed
Total epochs trained: 130
Best val_loss: 0.6425
```

### 最终指标

| Classifier | AUC | F1 | Recall | Spec. |
|------------|:---:|:--:|:------:|:-----:|
| rf | 0.807 | 0.706 | 0.833 | 0.545 |
| svm | 0.659| 0.690 | 0.500 | 0.273 |
| catboost | 0.818 | 0.759 | 0.750 | 0.545 |
| ensemble | 0.818 | 0.720 | 0.750 | 0.636 |

### 结果说明

当前结果**不代表模型真实性能**，仅验证流水线可正常运行。由于抽样比例极低（0.1%），每个 LOSO Fold 仅包含约 17 个训练样本和 2 个测试样本，模型难以充分收敛。要获得有意义的评估指标，需要：

1. **增加抽样比例**：建议至少 10-20%
2. **延长训练周期**：推荐 50-100 个 epoch，配合早停机制
3. **复用论文配置**：在30%的抽样比例，50个epoch下可以得到与论文相近的结果
## 可能遇到的问题与解决

### 1. torch 报错 "Lantern is not loaded"

**原因**：R 版本与 torch 包编译版本不匹配（如 R 4.4.2 + torch built for 4.4.3），访问 `model$parameters` 时触发段错误。

**解决**：升级 R 到 4.4.3，或重新安装匹配的 torch 版本。

### 2. 安装时提示 "Rcpp 编译失败"

**原因**：Windows 上缺少 RTools，无法从源码编译 R 包。

**解决**：安装时选择二进制包（回答 `no`），或显式指定：
```r
install.packages("Rcpp", type = "binary")
```

### 3. reticulate conda 环境找不到

**原因**：`.Rprofile` 中设置了 `use_condaenv("directml", required = TRUE)`，但 conda 环境路径不对。

**解决**：
```r
# 临时方案：跳过 .Rprofile
Sys.setenv(R_PROFILE_USER = "")

# 长期方案：修改 .Rprofile，将 required 改为 FALSE
# use_condaenv("directml", required = FALSE)
```

### 4. CatBoost 报错 "could not find function catboost.load_pool"

**原因**：包内 catboost 模块未正确加载。

**解决**：确保 DFDNxbq 包已正确安装，`load_catboost()` 会自动从 `inst/extdata/catboost/` 加载本地 DLL 和 R 代码。如仍有问题，可手动安装：
```r
install.packages(
  "path\\DFDNxbq\\inst\\extdata\\catboost",
  repos = NULL, type = "source"
)
```

### 5. 包安装时中断（退出状态非0）

**原因**：缺少依赖包，或 reticulate 初始化失败。

**解决**：按顺序检查：
1. 先安装所有 Imports 依赖（第 2 步）
2. 临时设置 `Sys.setenv(R_PROFILE_USER = "")` 绕过 .Rprofile
3. 重新运行 `install.packages("...DFDNxbq", repos = NULL, type = "source")`

### 6. 结果全为 NA 或 AUC = 0.5

**原因**：抽样比例过低 + 训练 epoch 太少，导致模型未收敛。

**解决**：增大 `sample_ratio` 和 `n_epochs`，参考上文"结果说明"中的建议。


```
DFDNxbq: Deep Fall Detect Net by xbq.
R package version 1.1.0.
```

