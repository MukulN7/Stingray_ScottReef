import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

def main():
    script_dir = Path(__file__).resolve().parent
    repo_dir = script_dir.parent
    
    baseline_csv_path = repo_dir / "data" / "velocity" / "velocity_baseline.csv"
    pose_data_path = repo_dir / "data" / "ground_truth" / "1785329318720_stereo_pose_est.data"
    output_dir = repo_dir / "data" / "estimated"
    output_csv_path = output_dir / "sp_lp_nav.csv"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load baseline velocity CSV
    vel_df = pd.read_csv(baseline_csv_path)
    
    # 2. Parse stereo pose estimation data (timestamp, position, Euler angles)
    # Field 1: Timestamp (s)
    # Field 4: X (North), Field 5: Y (East), Field 6: Z (Depth)
    # Field 7: X-axis Euler angle (rad), Field 8: Y-axis Euler angle (rad), Field 9: Z-axis Euler angle (rad)
    rows = []
    with open(pose_data_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%") or line.startswith("ORIGIN"):
                continue
            parts = line.split()
            rows.append({
                "ts": float(parts[1]),
                "x_gt": float(parts[4]),
                "y_gt": float(parts[5]),
                "z_gt": float(parts[6]),
                "ex": float(parts[7]),
                "ey": float(parts[8]),
                "ez": float(parts[9]),
            })
            
    pose_df = pd.DataFrame(rows)
    
    # Number of frames to process
    n = min(len(vel_df), len(pose_df) - 1)
    
    frames = vel_df["frame"].values[:n]
    timestamps = pose_df["ts"].values[:n]
    
    # Extract camera-frame sp_lp velocities
    vx_cam = vel_df["vx_sp_lp"].values[:n]
    vy_cam = vel_df["vy_sp_lp"].values[:n]
    vz_cam = vel_df["vz_sp_lp"].values[:n]
    v_cam_all = np.column_stack([vx_cam, vy_cam, vz_cam])
    
    # Euler angles for frame transitions (midpoint between pose i and i+1)
    ex_mid = (pose_df["ex"].values[:n] + pose_df["ex"].values[1:n+1]) / 2.0
    ey_mid = (pose_df["ey"].values[:n] + pose_df["ey"].values[1:n+1]) / 2.0
    ez_mid = (pose_df["ez"].values[:n] + pose_df["ez"].values[1:n+1]) / 2.0
    
    # Transform camera-frame velocity into navigation frame
    v_nav_all = np.zeros_like(v_cam_all)
    for i in range(n):
        rot_nav_to_cam = R.from_euler('xyz', [ex_mid[i], ey_mid[i], ez_mid[i]])
        rot_cam_to_nav = rot_nav_to_cam.inv()
        v_nav_all[i] = rot_cam_to_nav.apply(v_cam_all[i])
        
    out_df = pd.DataFrame({
        "frame": frames,
        "timestamp": timestamps,
        "vx_sp_lp_nav": v_nav_all[:, 0],
        "vy_sp_lp_nav": v_nav_all[:, 1],
        "vz_sp_lp_nav": v_nav_all[:, 2]
    })
    
    out_df.to_csv(output_csv_path, index=False)
    print(f"Transformed velocity successfully saved to: {output_csv_path}")

if __name__ == "__main__":
    main()
