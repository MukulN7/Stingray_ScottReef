# Stingray ScottReef Stereo Visual Odometry

This repository contains the processing pipeline and numerical datasets for the ScottReef Stereo Visual Odometry (VO) research project.

## Overview

The pipeline processes camera-frame velocity estimates from stereo visual odometry, transforms them into the navigation frame using reference pose orientation, applies a finalized axis/sign correction matrix ($S = \text{diag}(-1, 1, 1)$), and integrates the corrected velocities over time to compute estimated 3D position trajectories.

## Processing Pipeline

1. **Camera to Navigation Transformation (`src/transform_velocity_nav.py`)**
   Transforms VO camera-frame velocity $V_{\text{cam}}$ into the navigation frame $V_{\text{nav}}$ using intrinsic XYZ Euler orientation angles:
   - $R_{\text{nav}\to\text{cam}} = R_z(\psi) R_y(\theta) R_x(\phi)$
   - $R_{\text{cam}\to\text{nav}} = R_{\text{nav}\to\text{cam}}^T$
   - $V_{\text{nav}} = R_{\text{cam}\to\text{nav}} V_{\text{cam}}$

2. **Axis/Sign Correction (`src/apply_sign_correction_and_compare.py`)**
   Applies the finalized sign convention to transformed navigation velocities:
   - $S = \text{diag}(-1, 1, 1)$
   - $V_{\text{final}} = S \, V_{\text{nav}}$

3. **Position Integration (`src/integrate_position.py`)**
   Integrates corrected navigation velocities over time step $\Delta t$ starting from the initial reference ground-truth pose:
   - $\mathbf{p}_{k+1} = \mathbf{p}_k + \mathbf{v}_k \Delta t_k$

## Repository Structure

```
Stingray_ScottReef/
│
├── src/
│   ├── transform_velocity_nav.py             # Transforms VO camera velocity to nav frame
│   ├── apply_sign_correction_and_compare.py  # Applies sign matrix correction & computes metrics
│   └── integrate_position.py                 # Integrates corrected velocity to estimated 3D position
│
├── data/
│   ├── ground_truth/
│   │   └── 1785329318720_stereo_pose_est.data# Reference GT pose dataset
│   │
│   ├── velocity/
│   │   └── velocity_baseline.csv            # Original VO camera-frame velocity
│   │
│   └── estimated/
│       ├── sp_lp_nav.csv                    # Transformed navigation-frame velocity
│       ├── sp_lp_nav_corrected.csv          # Final corrected navigation-frame velocity
│       └── estimated_poses.csv              # Integrated estimated 3D position trajectory
│
├── plots/
│   └── trajectory_top_down_2d.png           # Top-down 2D Ground Truth vs VO trajectory plot
│
├── docs/
│   └── .gitkeep                             # Documentation directory
│
├── .gitignore
└── README.md
```

## Running the Pipeline

To execute the full processing pipeline:

```bash
python src/transform_velocity_nav.py
python src/apply_sign_correction_and_compare.py
python src/integrate_position.py
```
