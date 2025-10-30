#!/usr/bin/env python3
"""
scripts/verify_workspace_variation.py

Generate (or load) workspace samples via ClimbBotEnv and verify
that the sampled goal positions show reasonable variation.

Usage examples:
  # generate (or load cache) and print stats, no plots:
  python scripts/verify_workspace_variation.py

  # generate lots of samples and show plots:
  python scripts/verify_workspace_variation.py --max-samples 30000 --angle-samples 24 --linear-samples 24 --plot

  # change expected x-range if you want different thresholds:
  python scripts/verify_workspace_variation.py --min-x 0.41 --max-x 0.815
"""

import argparse
import numpy as np
import os
import sys
import time

# change this import path if your env module name differs
from envs.varied_poses_env import ClimbBotEnv

HAS_MPL = True
try:
    import matplotlib.pyplot as plt  # optional plotting
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
except Exception:
    HAS_MPL = False

def describe_cloud(pts):
    if pts is None or pts.size == 0:
        return None
    stats = {}
    stats['n'] = int(pts.shape[0])
    stats['min'] = pts.min(axis=0)
    stats['median'] = np.median(pts, axis=0)
    stats['mean'] = pts.mean(axis=0)
    stats['max'] = pts.max(axis=0)
    stats['std'] = pts.std(axis=0)
    return stats

def print_stats(name, stats):
    if stats is None:
        print(f"{name}: no samples")
        return
    print(f"{name}: n={stats['n']}")
    for i, ax in enumerate(['x','y','z']):
        print(f"  {ax}: min={stats['min'][i]:.4f}, median={stats['median'][i]:.4f}, mean={stats['mean'][i]:.4f}, max={stats['max'][i]:.4f}, std={stats['std'][i]:.4f}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xml", type=str, default="assets/robot_mjcf.xml", help="Path to MJCF")
    p.add_argument("--max-samples", type=int, default=15000, help="max_samples for generator")
    p.add_argument("--angle-samples", type=int, default=20)
    p.add_argument("--linear-samples", type=int, default=20)
    p.add_argument("--deterministic", action="store_true", help="Use deterministic RNG for generation")
    p.add_argument("--no-save-cache", dest="save_cache", action="store_false", help="Don't save workspace cache")
    p.add_argument("--plot", action="store_true", help="Show plots (requires matplotlib)")
    p.add_argument("--min-x", type=float, default=0.414, help="Expected min X (world frame) threshold")
    p.add_argument("--max-x", type=float, default=0.815, help="Expected max X (world frame) threshold")
    p.add_argument("--save-out", type=str, default="", help="Optional .npz file to save sampled clouds")
    args = p.parse_args()

    env = ClimbBotEnv(xml_path="assets/robot_mjcf.xml", debug=True)
    env.generate_workspace_samples(max_samples=50000, angle_samples=24, linear_samples=24, deterministic=False, save_cache=True, save_qvecs=True)
    print("right world x min/max:", env._workspace_samples["r_world"][:,0].min(), env._workspace_samples["r_world"][:,0].max())
    print("right world y min/max:", env._workspace_samples["r_world"][:,1].min(), env._workspace_samples["r_world"][:,1].max())
    print("right world z min/max:", env._workspace_samples["r_world"][:,2].min(), env._workspace_samples["r_world"][:,2].max())
    print("left world x min/max:", env._workspace_samples["l_world"][:,0].min(), env._workspace_samples["l_world"][:,0].max())
    print("left world y min/max:", env._workspace_samples["l_world"][:,1].min(), env._workspace_samples["l_world"][:,1].max())
    print("left world z min/max:", env._workspace_samples["l_world"][:,2].min(), env._workspace_samples["l_world"][:,2].max())
    

if __name__ == "__main__":
    main()