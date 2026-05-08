"""
PKL → NPZ+CSV conversion for IVFastPavlov pipeline.

Converts FallAllD.pkl into:
  - {basename}_signals.npz  (stacked Acc/Gyr arrays, float64)
  - {basename}_metadata.csv (SubjectID, ActivityID, TrialNo, IsFall)

Usage:
  python convert_pkl_to_rdata.py <pkl_path> [output_dir]
"""
import pickle
import numpy as np
import pandas as pd
import os
import sys
from tqdm import tqdm


def convert_pkl(pkl_path, output_dir="."):
    basename = os.path.splitext(os.path.basename(pkl_path))[0]

    print(f"[convert] Loading {pkl_path} ...")
    with open(pkl_path, 'rb') as f:
        df = pickle.load(f)

    n = len(df)
    print(f"[convert] {n} samples, stacking signals ...")

    acc_list = []
    gyr_list = []
    for i in tqdm(range(n), desc="Stacking Acc/Gyr", unit="sample"):
        acc_list.append(df['Acc'].iloc[i])
        gyr_list.append(df['Gyr'].iloc[i])
    acc_all = np.stack(acc_list).astype(np.float64)  # (N, T, 3)
    gyr_all = np.stack(gyr_list).astype(np.float64)  # (N, T, 3)
    is_fall = df['ActivityID'].values >= 101

    npz_path = os.path.join(output_dir, f"{basename}_signals.npz")
    np.savez(npz_path, acc_all=acc_all, gyr_all=gyr_all)
    print(f"[convert] Signals -> {npz_path}  ({acc_all.nbytes/1e9:.2f} GB)")

    csv_path = os.path.join(output_dir, f"{basename}_metadata.csv")
    meta = pd.DataFrame({
        'SubjectID':   df['SubjectID'].values.astype(int),
        'ActivityID':  df['ActivityID'].values.astype(int),
        'TrialNo':     df['TrialNo'].values.astype(int),
        'IsFall':      is_fall,
    })
    meta.to_csv(csv_path, index=False)
    print(f"[convert] Metadata -> {csv_path}  ({n} rows)")

    return npz_path, csv_path, acc_all.shape


def convert_pkl_sample(pkl_path, output_dir=".", n_samples=10, seed=42):
    """Convert a random sample from PKL (for testing)."""
    basename = os.path.splitext(os.path.basename(pkl_path))[0]

    print(f"[convert] Loading {pkl_path} ...")
    with open(pkl_path, 'rb') as f:
        df = pickle.load(f)

    rng = np.random.RandomState(seed)
    idx = rng.choice(len(df), min(n_samples, len(df)), replace=False)

    sample_df = df.iloc[idx]
    n = len(sample_df)
    print(f"[convert] Sampling {n} rows ...")

    acc_list = []
    gyr_list = []
    for i in tqdm(range(n), desc="Stacking Acc/Gyr", unit="sample"):
        acc_list.append(sample_df['Acc'].iloc[i])
        gyr_list.append(sample_df['Gyr'].iloc[i])
    acc_all = np.stack(acc_list).astype(np.float64)
    gyr_all = np.stack(gyr_list).astype(np.float64)
    is_fall = sample_df['ActivityID'].values >= 101

    tag = f"{basename}_sample{n}"
    npz_path = os.path.join(output_dir, f"{tag}_signals.npz")
    np.savez(npz_path, acc_all=acc_all, gyr_all=gyr_all)
    print(f"[convert] Signals -> {npz_path}")

    csv_path = os.path.join(output_dir, f"{tag}_metadata.csv")
    meta = pd.DataFrame({
        'SubjectID':   sample_df['SubjectID'].values.astype(int),
        'ActivityID':  sample_df['ActivityID'].values.astype(int),
        'TrialNo':     sample_df['TrialNo'].values.astype(int),
        'IsFall':      is_fall,
    })
    meta.to_csv(csv_path, index=False)
    print(f"[convert] Metadata -> {csv_path}")

    return npz_path, csv_path, acc_all.shape


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    pkl_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    os.makedirs(out_dir, exist_ok=True)
    convert_pkl(pkl_path, out_dir)
    print("[convert] Done!")
