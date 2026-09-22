import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    script_dir = Path(__file__).resolve().parent
    repo_dir = script_dir.parent
    
    pose_data_path = repo_dir / "data" / "ground_truth" / "1785329318720_stereo_pose_est.data"
    corrected_csv_path = repo_dir / "data" / "estimated" / "sp_lp_nav_corrected.csv"
    estimated_poses_path = repo_dir / "data" / "estimated" / "estimated_poses.csv"
    plots_dir = repo_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Read transformed corrected velocity CSV
    vel_df = pd.read_csv(corrected_csv_path)
    
    # Identify velocity column names
    col_x = [c for c in vel_df.columns if "vx" in c.lower()][0]
    col_y = [c for c in vel_df.columns if "vy" in c.lower()][0]
    col_z = [c for c in vel_df.columns if "vz" in c.lower()][0]
    
    vx_val = vel_df[col_x].values
    vy_val = vel_df[col_y].values
    vz_val = vel_df[col_z].values
    
    # 2. Parse GT pose file
    gt_rows = []
    with open(pose_data_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%") or line.startswith("ORIGIN"):
                continue
            parts = line.split()
            gt_rows.append({
                "ts": float(parts[1]),
                "x_gt": float(parts[4]),  # North
                "y_gt": float(parts[5]),  # East
                "z_gt": float(parts[6]),  # Depth
            })
    gt_df = pd.DataFrame(gt_rows)
    
    n_vel = len(vel_df)
    n_poses = min(n_vel + 1, len(gt_df))
    n = n_poses - 1  # number of velocity steps
    
    ts = gt_df["ts"].values[:n_poses]
    gt_x = gt_df["x_gt"].values[:n_poses]
    gt_y = gt_df["y_gt"].values[:n_poses]
    gt_z = gt_df["z_gt"].values[:n_poses]
    
    dt = np.diff(ts)
    
    # 3. Integrate position from initial GT pose
    calc_x = np.zeros(n_poses)
    calc_y = np.zeros(n_poses)
    calc_z = np.zeros(n_poses)
    
    calc_x[0] = gt_x[0]
    calc_y[0] = gt_y[0]
    calc_z[0] = gt_z[0]
    
    for i in range(n):
        calc_x[i + 1] = calc_x[i] + vx_val[i] * dt[i]
        calc_y[i + 1] = calc_y[i] + vy_val[i] * dt[i]
        calc_z[i + 1] = calc_z[i] + vz_val[i] * dt[i]
        
    # Save estimated poses to CSV
    frames_poses = np.arange(1, n_poses + 1)
    poses_df = pd.DataFrame({
        "frame": frames_poses,
        "timestamp": ts,
        "north": calc_x,
        "east": calc_y,
        "depth": calc_z
    })
    poses_df.to_csv(estimated_poses_path, index=False)
    print(f"Saved estimated poses to: {estimated_poses_path}")

    # 4. Calculate Errors (MAE, RMSE)
    err_x = calc_x - gt_x
    err_y = calc_y - gt_y
    err_z = calc_z - gt_z
    
    mae_x = np.mean(np.abs(err_x))
    rmse_x = np.sqrt(np.mean(err_x ** 2))
    
    mae_y = np.mean(np.abs(err_y))
    rmse_y = np.sqrt(np.mean(err_y ** 2))
    
    mae_z = np.mean(np.abs(err_z))
    rmse_z = np.sqrt(np.mean(err_z ** 2))
    
    err_2d = np.sqrt(err_x ** 2 + err_y ** 2)
    err_3d = np.sqrt(err_x ** 2 + err_y ** 2 + err_z ** 2)
    
    mae_2d = np.mean(err_2d)
    rmse_2d = np.sqrt(np.mean(err_2d ** 2))
    
    mae_3d = np.mean(err_3d)
    rmse_3d = np.sqrt(np.mean(err_3d ** 2))
    
    print("=" * 70)
    print("INTEGRATED POSITION EVALUATION (Corrected sp_lp vs GT)")
    print("=" * 70)
    print(f"{'Axis / Metric':<24} | {'MAE (meters)':<18} | {'RMSE (meters)':<18}")
    print("-" * 70)
    print(f"{'North (X)':<24} | {mae_x:<18.4f} | {rmse_x:<18.4f}")
    print(f"{'East (Y)':<24} | {mae_y:<18.4f} | {rmse_y:<18.4f}")
    print(f"{'Depth (Z)':<24} | {mae_z:<18.4f} | {rmse_z:<18.4f}")
    print("-" * 70)
    print(f"{'2D Horizontal Error':<24} | {mae_2d:<18.4f} | {rmse_2d:<18.4f}")
    print(f"{'3D Overall Error':<24} | {mae_3d:<18.4f} | {rmse_3d:<18.4f}")
    print("=" * 70)
    
    # 5. Generate Top-Down Trajectory Plot (East vs North)
    plt.figure(figsize=(9, 9), dpi=150)
    plt.plot(gt_y - gt_y[0], gt_x - gt_x[0], label="Ground Truth (GT)", color="black", linewidth=2.0)
    plt.plot(calc_y - calc_y[0], calc_x - calc_x[0], label="Integrated VO (Corrected sp_lp)", color="tab:blue", linestyle="--", linewidth=1.8)
    plt.scatter([0], [0], color="green", s=80, zorder=5, label="Start Point")
    plt.scatter([gt_y[-1] - gt_y[0]], [gt_x[-1] - gt_x[0]], color="black", marker="x", s=80, zorder=5, label="GT End")
    plt.scatter([calc_y[-1] - calc_y[0]], [calc_x[-1] - calc_x[0]], color="red", marker="o", s=80, zorder=5, label="VO End")
    
    plt.xlabel("East Relative Position (m)")
    plt.ylabel("North Relative Position (m)")
    plt.title(f"Top-Down 2D Trajectory Comparison\n(2D MAE: {mae_2d:.2f} m, 2D End Drift: {err_2d[-1]:.2f} m)")
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.axis("equal")
    plt.tight_layout()
    
    plot_2d_path = plots_dir / "trajectory_top_down_2d.png"
    plt.savefig(plot_2d_path)
    plt.close()
    print(f"Saved 2D trajectory plot: {plot_2d_path}")

if __name__ == "__main__":
    main()
