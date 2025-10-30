#!/usr/bin/env python3
"""
Sample many goal poses from ClimbBotEnv and plot them in 3D.

Usage:
    python sample_and_plot_goals.py --xml /path/to/climbbot3.xml --n 500 --seed 0 --out sampled_goals.png

This assumes your ClimbBotEnv class is importable from the PYTHONPATH.
If it is defined in a file named `climb_env.py`, adjust the import accordingly:
    from climb_env import ClimbBotEnv
"""
import argparse
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ---- Adjust this import to match where your class lives ----
# Example: if your env is defined in climb_env.py in the same folder:
# from climb_env import ClimbBotEnv
#
# If it's inside a package, import accordingly.
from envs.varied_poses_env import ClimbBotEnv  # adjust as needed  

def ensure_axes_equal(ax):
    """Make 3D axes have equal scale."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    plot_radius = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d(x_middle - plot_radius, x_middle + plot_radius)
    ax.set_ylim3d(y_middle - plot_radius, y_middle + plot_radius)
    ax.set_zlim3d(z_middle - plot_radius, z_middle + plot_radius)

def draw_box(ax, center, half_extents, edge_color="k", alpha=0.05, label=None):
    """Draw a transparent cube (workspace box) given center and half-extents."""
    cx, cy, cz = center
    hx, hy, hz = half_extents
    # 8 corners
    corners = np.array([
        [cx - hx, cy - hy, cz - hz],
        [cx + hx, cy - hy, cz - hz],
        [cx + hx, cy + hy, cz - hz],
        [cx - hx, cy + hy, cz - hz],
        [cx - hx, cy - hy, cz + hz],
        [cx + hx, cy - hy, cz + hz],
        [cx + hx, cy + hy, cz + hz],
        [cx - hx, cy + hy, cz + hz],
    ])
    # 12 edges (pairs of corner indices)
    edges = [
        (0,1),(1,2),(2,3),(3,0),  # bottom
        (4,5),(5,6),(6,7),(7,4),  # top
        (0,4),(1,5),(2,6),(3,7)   # verticals
    ]
    for (i,j) in edges:
        xs = [corners[i,0], corners[j,0]]
        ys = [corners[i,1], corners[j,1]]
        zs = [corners[i,2], corners[j,2]]
        ax.plot(xs, ys, zs, color=edge_color, linewidth=0.8, alpha=max(0.3, alpha))
    if label:
        ax.text(cx, cy, cz + half_extents[2] + 0.02, label, color=edge_color)

def sample_goals(env, n, arm="r", seed=None):
    rng = env.np_random
    if rng is None:
        # seed environment RNG so sampling is reproducible
        env.reset(seed=seed)
        rng = env.np_random
    pts = []
    for i in range(n):
        p = env.sample_target(rng=rng, arm=arm)
        pts.append(np.asarray(p, dtype=np.float32))
    return np.vstack(pts)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=str, default="climbbot3.xml", help="Path to MuJoCo XML")
    parser.add_argument("--n", type=int, default=500, help="Number of samples per arm")
    parser.add_argument("--out", type=str, default="sampled_goals.png", help="Output image file")
    parser.add_argument("--csv", type=str, default=None, help="Optional CSV to save sampled points")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    args = parser.parse_args()
    xml_path = "/Users/aaronthomas/Desktop/Engineering_Projects/Climbing Robot/assets/robot_mjcf.xml"

    # instantiate environment (make sure xml path exists)
    env = ClimbBotEnv(xml_path=xml_path, render_mode=None, debug=False)
    env.reset(seed=args.seed)

    # Sample
    print(f"Sampling {args.n} points per arm...")
    pts_r = sample_goals(env, args.n, arm="r", seed=args.seed)
    pts_l = sample_goals(env, args.n, arm="l", seed=args.seed+1)

    # optionally save CSV
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["arm", "x", "y", "z"])
            for p in pts_r:
                writer.writerow(["r", p[0], p[1], p[2]])
            for p in pts_l:
                writer.writerow(["l", p[0], p[1], p[2]])
        print(f"Wrote samples to {args.csv}")

    # Plotting
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts_r[:,0], pts_r[:,1], pts_r[:,2], s=10, alpha=0.7, label="right targets", marker="o")
    ax.scatter(pts_l[:,0], pts_l[:,1], pts_l[:,2], s=10, alpha=0.7, label="left targets", marker="^")

    # draw workspace boxes if available
    try:
        if hasattr(env, "workspace_r_center") and hasattr(env, "workspace_r_half"):
            draw_box(ax, np.asarray(env.workspace_r_center), np.asarray(env.workspace_r_half), edge_color="red", alpha=0.15, label="R workspace")
        if hasattr(env, "workspace_l_center") and hasattr(env, "workspace_l_half"):
            draw_box(ax, np.asarray(env.workspace_l_center), np.asarray(env.workspace_l_half), edge_color="blue", alpha=0.15, label="L workspace")
    except Exception:
        pass

    # draw robot base position if available
    try:
        if env.base_id is not None:
            base_pos = np.array(env.data.xpos[env.base_id], dtype=np.float32)
            ax.scatter([base_pos[0]], [base_pos[1]], [base_pos[2]], color="k", s=50, label="base")
    except Exception:
        pass

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"Sampled goal clouds (N={args.n} per arm)")
    ax.legend()
    # tighten axes and equalize scale
    plt.tight_layout()
    ensure_axes_equal(ax)

    # Save and show
    fig.savefig(args.out, dpi=200)
    print(f"Saved figure to {args.out}")
    try:
        plt.show()
    except Exception:
        # non-interactive environment
        pass

if __name__ == "__main__":
    main()
