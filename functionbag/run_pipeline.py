#!/usr/bin/env python3
"""
run_pipeline.py – IVFastPavlov 函数包统一入口

一键式流程:
  1. PKL → NPZ+CSV (convert_pkl_to_rdata.py)
  2. R pipeline 执行 (pipeline.R)
  3. 结果汇总

Usage:
  python run_pipeline.py <pkl_path>                    # 全量运行
  python run_pipeline.py <pkl_path> --sample N         # 抽样 N 条测试
  python run_pipeline.py <pkl_path> --ratio 0.001      # 千分之一抽样
  python run_pipeline.py <pkl_path> --out_dir ./results
"""
import argparse
import os
import subprocess
import sys
import pickle
import numpy as np


def get_pkl_info(pkl_path):
    """Get basic info about the PKL file."""
    with open(pkl_path, 'rb') as f:
        df = pickle.load(f)
    n = len(df)
    n_fall = sum(df['ActivityID'].values >= 101)
    subjects = sorted(df['SubjectID'].unique())
    return dict(rows=n, falls=n_fall, adl=n - n_fall,
                subjects=len(subjects), subject_list=subjects)


def run_convert(pkl_path, out_dir, sample_n=None, sample_ratio=None):
    """Run the PKL → NPZ+CSV conversion step."""
    from functionbag.convert_pkl import convert_pkl, convert_pkl_sample

    if sample_n is not None:
        return convert_pkl_sample(pkl_path, out_dir, n_samples=sample_n)
    elif sample_ratio is not None and sample_ratio < 1:
        info = get_pkl_info(pkl_path)
        n_samples = max(3, int(info['rows'] * sample_ratio))
        return convert_pkl_sample(pkl_path, out_dir, n_samples=n_samples)
    else:
        return convert_pkl(pkl_path, out_dir)


def run_r_pipeline(npz_path, csv_path, out_dir, sample_ratio=None):
    """Run the R pipeline."""
    r_script = os.path.join(os.path.dirname(__file__), "pipeline.R")
    cmd = ["Rscript", r_script, npz_path, csv_path, out_dir]
    if sample_ratio is not None:
        cmd.append(str(sample_ratio))

    print(f"[run] 执行: {' '.join(cmd)}")
    env = os.environ.copy()
    # Ensure R can find Python for reticulate
    # (use the same Python that ran this script)
    env["RETICULATE_PYTHON"] = sys.executable

    result = subprocess.run(cmd, env=env, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"[run] R 管道失败 (code={result.returncode})")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="IVFastPavlov 函数包 — 一键运行管道")
    parser.add_argument("pkl_path", help="FallAllD.pkl 路径")
    parser.add_argument("--out_dir", default="output",
                        help="输出目录 (默认: output)")
    parser.add_argument("--sample", type=int, default=None,
                        help="抽样 N 条记录用于测试")
    parser.add_argument("--ratio", type=float, default=None,
                        help="抽样比例 (例如 0.001 = 千分之一)")
    parser.add_argument("--skip_r", action="store_true",
                        help="仅转换, 不运行 R 管道")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 显示数据概况
    if os.path.exists(args.pkl_path):
        info = get_pkl_info(args.pkl_path)
        print(f"[run] PKL: {args.pkl_path}")
        print(f"[run]      {info['rows']} 行, "
              f"{info['falls']} 跌倒, {info['adl']} ADL, "
              f"{info['subjects']} 受试者")

    # 步骤 1: 转换
    print("\n[run] === 步骤 1/2: PKL → NPZ+CSV ===")
    npz_path, csv_path, shape = run_convert(
        args.pkl_path, args.out_dir,
        sample_n=args.sample, sample_ratio=args.ratio)
    print(f"[run] NPZ: {npz_path}")
    print(f"[run] CSV: {csv_path}")
    print(f"[run] 信号维度: {shape}")

    if args.skip_r:
        print("[run] 跳过 R 管道 (--skip_r)")
        return

    # 步骤 2: R 管道
    print("\n[run] === 步骤 2/2: R 管道 ===")
    success = run_r_pipeline(npz_path, csv_path, args.out_dir,
                              sample_ratio=args.ratio)
    if success:
        results_file = os.path.join(args.out_dir, "pipeline_results.rds")
        if os.path.exists(results_file):
            size_mb = os.path.getsize(results_file) / 1e6
            print(f"\n[run] ✓ 管道完成! 结果: {results_file} ({size_mb:.1f} MB)")
        else:
            print(f"\n[run] ✓ 管道完成! 结果目录: {args.out_dir}")
    else:
        print(f"\n[run] ✗ 管道失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
