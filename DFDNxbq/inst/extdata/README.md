# DFDNxbq 外部依赖文件说明

## 必要依赖（运行时用户提供）

| 文件 | 说明 | 来源 |
|------|------|------|
| `*.npz` | 加速度+陀螺仪信号数组 (acc_all, gyr_all) | FallAllD.pkl → convert_pkl_to_rdata.py |
| `*.csv` | 样本元数据 (SubjectID, ActivityID, TrialNo, IsFall) | 同上 |

转换脚本位于 `functionbag/convert_pkl_to_rdata.py`。

## torch 后端安装

DFDNxbq 使用 `torch` R 包训练深度神经网络，需要额外安装 Lantern (libtorch) 后端：

```r
# 安装 torch R 包（如尚未安装）
install.packages("torch")

# 安装 torch 后端依赖
library(DFDNxbq)
install_dfdnxbq_deps()
```

如遇网络问题可手动设置镜像：
```r
options(repos = c(CRAN = "https://cloud.r-project.org"))
torch::install_torch()
```

## 可选依赖

### catboost 本地源码包

如果无法通过 `install.packages("catboost")` 在线安装，可将 catboost
源码包放入 `inst/extdata/catboost/` 目录下，包会自动检测加载。

需要的文件结构：
```
inst/extdata/catboost/
├── DESCRIPTION          # 包描述
├── R/catboost.R         # R 接口函数
├── inst/libs/x64/
│   └── libcatboostr.dll # Windows 64-bit 动态库
└── (其他 catboost 源文件)
```

## 数据说明

`BASE_CONFIG$fast_subjects` 中硬编码的受试者 ID (1-15)
来自 FallAllD 数据集，如需用于其他数据集需修改此配置。
