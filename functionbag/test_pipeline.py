#!/usr/bin/env python3
"""
test_pipeline.py – 在抽样数据上验证完整管道

流程:
  1. 从 PKL 抽取千分之一样本 (~6-7 条)
  2. 转换为 NPZ+CSV
  3. 调用 R pipeline 执行实验
  4. 验证结果文件是否存在

Usage:
  python test_pipeline.py                    # 千分之一抽样
  python test_pipeline.py --sample 10        # 固定 10 条
  python test_pipeline.py --full             # 全量(慎用)
"""
import argparse
import os
import sys
import time
import tempfile
import glob
import subprocess

# 添加 functionbag 到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="测试 IVFastPavlov 管道")
    parser.add_argument("--pkl", default="../FallAllD.pkl",
                        help="PKL 文件路径")
    parser.add_argument("--sample", type=int, default=None,
                        help="抽样条数")
    parser.add_argument("--ratio", type=float, default=0.001,
                        help="抽样比例 (默认 0.001 = 千分之一)")
    parser.add_argument("--full", action="store_true",
                        help="全量运行")
    parser.add_argument("--out", default="test_output",
                        help="输出目录")
    args = parser.parse_args()

    pkl_path = os.path.abspath(args.pkl)
    out_dir  = os.path.abspath(args.out)

    if not os.path.exists(pkl_path):
        print(f"[test] ✗ PKL 不存在: {pkl_path}")
        sys.exit(1)

    # 确定抽样
    if args.full:
        print("[test] 全量运行")
        sample_n, sample_r = None, None
    elif args.sample is not None:
        print(f"[test] 抽样 {args.sample} 条")
        sample_n, sample_r = args.sample, None
    else:
        print(f"[test] 抽样比例 {args.ratio} (千分之一)")
        sample_n, sample_r = None, args.ratio

    # 步骤 1: 转换
    print("\n[test] === 步骤 1: PKL → NPZ+CSV ===")
    t0 = time.time()
    from functionbag.convert_pkl import convert_pkl_sample, convert_pkl

    if sample_n is not None:
        npz_path, csv_path, shape = convert_pkl_sample(
            pkl_path, out_dir, n_samples=sample_n, seed=42)
    elif sample_r is not None:
        # 计算抽样条数
        import pickle
        with open(pkl_path, 'rb') as f:
            df = pickle.load(f)
        n_samples = max(3, int(len(df) * sample_r))
        npz_path, csv_path, shape = convert_pkl_sample(
            pkl_path, out_dir, n_samples=n_samples, seed=42)
    else:
        npz_path, csv_path, shape = convert_pkl(pkl_path, out_dir)

    t1 = time.time()
    print(f"[test] 转换耗时: {t1-t0:.1f}s")
    print(f"[test] 信号维度: {shape}")

    # 步骤 2: R 管道
    print("\n[test] === 步骤 2: R 管道 ===")
    r_script = os.path.join(os.path.dirname(__file__), "pipeline.R")
    cmd = ["Rscript", r_script, npz_path, csv_path, out_dir]
    if sample_r is not None:
        cmd.append(str(sample_r))

    print(f"[test] 执行: {' '.join(cmd)}")
    t2 = time.time()
    env = os.environ.copy()
    env["RETICULATE_PYTHON"] = sys.executable

    result = subprocess.run(cmd, env=env, capture_output=False, text=True)
    t3 = time.time()

    if result.returncode != 0:
        print(f"\n[test] ✗ R 管道失败 (code={result.returncode})")
        print(f"STDERR:\n{result.stderr}")
        sys.exit(1)

    print(f"\n[test] R 管道耗时: {t3-t2:.1f}s")

    # 步骤 3: 验证输出
    print("\n[test] === 步骤 3: 结果验证 ===")
    results_rds = os.path.join(out_dir, "pipeline_results.rds")
    metrics_md  = os.path.join(out_dir, "pipeline_run", "metrics.md")

    if os.path.exists(results_rds):
        size_mb = os.path.getsize(results_rds) / 1e6
        print(f"[test] ✓ pipeline_results.rds ({size_mb:.2f} MB)")
    else:
        print(f"[test] ? pipeline_results.rds 未找到, 检查子目录...")
        # 尝试搜索
        for root, dirs, files in os.walk(out_dir):
            for f in files:
                if 'results' in f or 'metrics' in f:
                    fpath = os.path.join(root, f)
                    print(f"       找到: {fpath} ({os.path.getsize(fpath)/1e3:.0f} KB)")

    # 打印关键指标文件
    for root, dirs, files in os.walk(os.path.join(out_dir, "pipeline_run")):
        for f in sorted(files):
            fpath = os.path.join(root, f)
            if f.endswith(".md"):
                with open(fpath) as fh:
                    content = fh.read()
                print(f"\n--- {f} ---")
                print(content[:500])
            elif f.endswith(".png"):
                print(f"  [图] {f} ({os.path.getsize(fpath)/1e3:.0f} KB)")

    print("\n[test] ✓ 管道测试完成!")
    print(f"[test] 结果目录: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
