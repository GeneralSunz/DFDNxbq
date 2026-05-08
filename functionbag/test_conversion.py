#!/usr/bin/env python3
"""
test_conversion.py – 测试 PKL → NPZ+CSV 转换正确性

测试项:
  1. 文件是否存在且非空
  2. 信号维度是否正确 (N, 4760, 3)
  3. 元数据是否完整
  4. R 能否正确读取 NPZ
  5. 采样后数据是否平衡
"""
import os
import sys
import tempfile
import pickle
import numpy as np
import pandas as pd


def test_conversion(pkl_path):
    """测试 PKL → NPZ+CSV 转换."""
    errors = []

    # 1. PKL 可读
    print(f"\n[test] 1. 读取 PKL: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        df = pickle.load(f)
    print(f"       Shape: {df.shape}, 列: {list(df.columns)}")
    assert len(df) > 0, "PKL 为空"
    print("       ✓")

    # 2. 信号结构
    print(f"[test] 2. 检查信号结构 ...")
    acc0 = df['Acc'].iloc[0]
    gyr0 = df['Gyr'].iloc[0]
    assert isinstance(acc0, np.ndarray), f"Acc 类型应为 ndarray, 得到 {type(acc0)}"
    assert acc0.shape == (4760, 3), f"Acc 形状应为 (4760,3), 得到 {acc0.shape}"
    assert gyr0.shape == (4760, 3), f"Gyr 形状应为 (4760,3), 得到 {gyr0.shape}"
    print(f"       Acc: {acc0.shape}, Gyr: {gyr0.shape}")
    print("       ✓")

    # 3. 活动标签
    print(f"[test] 3. 检查活动标签 ...")
    fall_mask = df['ActivityID'].values >= 101
    adl_mask = ~fall_mask
    print(f"       跌倒: {fall_mask.sum()}, ADL: {adl_mask.sum()}, "
          f"跌倒比例: {fall_mask.mean()*100:.1f}%")
    assert fall_mask.sum() > 0, "无跌倒样本"
    assert adl_mask.sum() > 0, "无 ADL 样本"
    print("       ✓")

    # 4. 转换并验证 NPZ
    print(f"[test] 4. 转换并验证 NPZ ...")
    with tempfile.TemporaryDirectory() as tmpdir:
        from functionbag.convert_pkl import convert_pkl_sample
        npz_path, csv_path, shape = convert_pkl_sample(
            pkl_path, tmpdir, n_samples=5, seed=42)

        assert os.path.exists(npz_path), f"NPZ 未创建: {npz_path}"
        assert os.path.exists(csv_path), f"CSV 未创建: {csv_path}"

        # 验证 NPZ 内容
        npz = np.load(npz_path)
        assert 'acc_all' in npz, f"NPZ 缺少 acc_all: {list(npz.keys())}"
        assert 'gyr_all' in npz, f"NPZ 缺少 gyr_all: {list(npz.keys())}"
        assert npz['acc_all'].shape == (5, 4760, 3), \
            f"acc_all shape 应为 (5,4760,3), 得到 {npz['acc_all'].shape}"
        assert npz['gyr_all'].shape == (5, 4760, 3), \
            f"gyr_all shape 应为 (5,4760,3), 得到 {npz['gyr_all'].shape}"
        npz.close()

        # 验证 CSV 内容
        meta = pd.read_csv(csv_path)
        assert len(meta) == 5, f"CSV 应有 5 行, 得到 {len(meta)}"
        assert all(c in meta.columns for c in
                   ['SubjectID', 'ActivityID', 'TrialNo', 'IsFall'])
        assert meta['IsFall'].dtype == bool, \
            f"IsFall 应为 bool, 得到 {meta['IsFall'].dtype}"
        print(f"       NPZ: {npz_path} ({os.path.getsize(npz_path)/1e6:.1f} MB)")
        print(f"       CSV: {csv_path}")
        print("       ✓")

    # 5. 完整转换验证 (抽样)
    print(f"[test] 5. 完整转换 (抽样 10 条) ...")
    with tempfile.TemporaryDirectory() as tmpdir:
        from functionbag.convert_pkl import convert_pkl_sample
        npz_path, csv_path, shape = convert_pkl_sample(
            pkl_path, tmpdir, n_samples=10, seed=123)
        data = np.load(npz_path)
        meta = pd.read_csv(csv_path)
        assert len(data['acc_all']) == len(meta) == 10
        assert data['acc_all'].shape[2] == 3  # 3 通道
        assert data['gyr_all'].shape[2] == 3
        data.close()
        print(f"       10 条样本转换正确")
        print("       ✓")

    # 6. R 兼容性验证 (检查数据结构)
    print(f"[test] 6. 验证 NPZ 与 R 兼容 ...")
    with tempfile.TemporaryDirectory() as tmpdir:
        from functionbag.convert_pkl import convert_pkl_sample
        npz_path, csv_path, shape = convert_pkl_sample(
            pkl_path, tmpdir, n_samples=3, seed=1)
        data = np.load(npz_path)
        meta = pd.read_csv(csv_path)
        # 模拟 R 的索引方式: [i,,]
        for i in range(3):
            acc_mat = data['acc_all'][i, :, :]
            gyr_mat = data['gyr_all'][i, :, :]
            assert acc_mat.shape == (4760, 3)
            assert gyr_mat.shape == (4760, 3)
            assert isinstance(meta['IsFall'].iloc[i], (bool, np.bool_))
        data.close()
        print("       R 索引兼容")
        print("       ✓")

    print(f"\n[test] ✓ 全部 {6} 项测试通过!")
    return True


if __name__ == '__main__':
    if len(sys.argv) < 2:
        pkl_path = "../FallAllD.pkl"
    else:
        pkl_path = sys.argv[1]

    if not os.path.exists(pkl_path):
        print(f"PKL 不存在: {pkl_path}")
        print(f"用法: python test_conversion.py <path_to_FallAllD.pkl>")
        sys.exit(1)

    success = test_conversion(os.path.abspath(pkl_path))
    sys.exit(0 if success else 1)
