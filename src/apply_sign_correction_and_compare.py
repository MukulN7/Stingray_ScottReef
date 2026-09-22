import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def main():
    script_dir = Path(__file__).resolve().parent
    repo_dir = script_dir.parent
    
    transformed_csv_path = repo_dir / "data" / "estimated" / "sp_lp_nav.csv"
    corrected_csv_path = repo_dir / "data" / "estimated" / "sp_lp_nav_corrected.csv"
    pose_data_path = repo_dir / "data" / "ground_truth" / "1785329318720_stereo_pose_est.data"
    plots_dir = repo_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load transformed nav velocity CSV
    nav_df = pd.read_csv(transformed_csv_path)
    
    # 2. Apply sign correction (-Vx, +Vy, +Vz)
    corrected_df = nav_df.copy()
    col_x = [c for c in nav_df.columns if "vx" in c.lower()][0]
    col_y = [c for c in nav_df.columns if "vy" in c.lower()][0]
    col_z = [c for c in nav_df.columns if "vz" in c.lower()][0]

    corrected_df["vx_sp_lp_nav_corrected"] = -1.0 * nav_df[col_x]
    corrected_df["vy_sp_lp_nav_corrected"] = 1.0 * nav_df[col_y]
    corrected_df["vz_sp_lp_nav_corrected"] = 1.0 * nav_df[col_z]
    
    # Preserve original columns if needed, but ensure corrected velocity columns exist
    out_df = pd.DataFrame({
        "frame": nav_df["frame"],
        "timestamp": nav_df["timestamp"],
        "vx_sp_lp_nav_corrected": corrected_df["vx_sp_lp_nav_corrected"],
        "vy_sp_lp_nav_corrected": corrected_df["vy_sp_lp_nav_corrected"],
        "vz_sp_lp_nav_corrected": corrected_df["vz_sp_lp_nav_corrected"]
    })

    out_df.to_csv(corrected_csv_path, index=False)
    print(f"Saved sign-corrected CSV to: {corrected_csv_path}")
    
    # 3. Parse stereo pose estimation data for GT positions and timestamps
    ts_list, x_list, y_list, z_list = [], [], [], []
    with open(pose_data_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%") or line.startswith("ORIGIN"):
                continue
            parts = line.split()
            ts_list.append(float(parts[1]))
            x_list.append(float(parts[4]))  # North
            y_list.append(float(parts[5]))  # East
            z_list.append(float(parts[6]))  # Depth
            
    ts = np.array(ts_list)
    x = np.array(x_list)
    y = np.array(y_list)
    z = np.array(z_list)
    
    # 4. Calculate GT velocity from consecutive positions in nav frame
    dt = np.diff(ts)
    gt_vx = np.diff(x) / dt
    gt_vy = np.diff(y) / dt
    gt_vz = np.diff(z) / dt
    
    # Align frame count n
    n = min(len(out_df), len(gt_vx))
    
    gt_vx = gt_vx[:n]
    gt_vy = gt_vy[:n]
    gt_vz = gt_vz[:n]
    
    est_vx = out_df["vx_sp_lp_nav_corrected"].values[:n]
    est_vy = out_df["vy_sp_lp_nav_corrected"].values[:n]
    est_vz = out_df["vz_sp_lp_nav_corrected"].values[:n]
    
    # 5. Calculate metrics (MAE, RMSE, Pearson correlation)
    axes = [
        ("Vx", gt_vx, est_vx),
        ("Vy", gt_vy, est_vy),
        ("Vz", gt_vz, est_vz)
    ]
    
    print("=" * 65)
    print("CORRECTED NAVIGATION FRAME VELOCITY COMPARISON (-Vx, +Vy, +Vz)")
    print("=" * 65)
    print(f"{'Axis':<6} | {'MAE (m/s)':<14} | {'RMSE (m/s)':<14} | {'Pearson Correlation (r)':<22}")
    print("-" * 65)
    
    for name, gt_val, est_val in axes:
        mae = np.mean(np.abs(est_val - gt_val))
        rmse = np.sqrt(np.mean((est_val - gt_val) ** 2))
        corr, _ = pearsonr(gt_val, est_val)
        print(f"{name:<6} | {mae:<14.6f} | {rmse:<14.6f} | {corr:<22.6f}")
    print("=" * 65)

if __name__ == "__main__":
    main()
