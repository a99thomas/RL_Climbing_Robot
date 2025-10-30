# scripts/inspect_samples.py
import numpy as np
from envs.varied_poses_env import ClimbBotEnv

env = ClimbBotEnv(xml_path="assets/robot_mjcf.xml", debug=False)
# try load cache; if missing, generate a modest set (fast)
if not env.load_workspace_samples_cache():
    print("No cache found — generating small sample set (2000) for inspection...")
    env.generate_workspace_samples(max_samples=200000, angle_samples=20, linear_samples=20, deterministic=True, save_cache=False)

r = env._workspace_samples["r"]
l = env._workspace_samples["l"]

def describe(pts, name):
    if pts is None or pts.shape[0] == 0:
        print(f"{name}: no points")
        return
    print(f"{name}: n={pts.shape[0]}")
    for i, ax in enumerate(["x","y","z"]):
        arr = pts[:, i]
        print(f"  {ax}: min={arr.min():.4f}, median={np.median(arr):.4f}, mean={arr.mean():.4f}, max={arr.max():.4f}, std={arr.std():.4f}")
    # basic histogram for x (10 bins)
    hist, edges = np.histogram(pts[:,0], bins=10)
    print("  x-hist counts:", hist.tolist())
    print("  x-edges:", [f"{e:.3f}" for e in edges])

print("Right samples:")
describe(r, "right")
print("Left samples:")
describe(l, "left")