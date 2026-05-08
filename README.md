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
R package version 1.0.0.
```

