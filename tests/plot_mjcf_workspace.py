# plot_mjcf_workspace.py
# Requires: mujoco (official python bindings), numpy, matplotlib
# Example run (from project root): python plot_mjcf_workspace_fixed.py

import numpy as np
import matplotlib.pyplot as plt
import mujoco
import os

# ---------- USER CONFIG ----------
MJCF_PATH = "/Users/aaronthomas/Desktop/Engineering_Projects/Climbing Robot/assets/robot_mjcf.xml"   # path to your MJCF file (replace with actual path)
MAIN_BODY_NAME = "assembly_7"
# site names for end-effectors (as in your MJCF)
RIGHT_SITE = "r_grip_site"
LEFT_SITE  = "l_grip_site"
# coarse sampling resolution for each joint (keep small to avoid too many samples)
ANGLE_SAMPLES = 20
LINEAR_SAMPLES = 20
# ----------------------------------

def name2id(model, obj_type, name):
    """Wrapper to find id; raises ValueError if not found."""
    idx = mujoco.mj_name2id(model, obj_type, name.encode() if isinstance(name, str) else name)
    if idx < 0:
        raise ValueError(f"Name '{name}' not found (obj_type={obj_type})")
    return idx

def joint_qpos_index(model, joint_name):
    # get joint id and the qpos address
    jid = name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    # model.jnt_qposadr gives qpos starting index for each joint
    return int(model.jnt_qposadr[jid]), int(jid)

def main():
    if not os.path.exists(MJCF_PATH):
        raise SystemExit(f"MJCF file not found: {MJCF_PATH}")

    # load model and allocate data
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data  = mujoco.MjData(model)

    # convert names to ids using correct mj_name2id
    main_body_id = name2id(model, mujoco.mjtObj.mjOBJ_BODY, MAIN_BODY_NAME)
    right_site_id = None
    left_site_id = None
    try:
        right_site_id = name2id(model, mujoco.mjtObj.mjOBJ_SITE, RIGHT_SITE)
    except Exception:
        print(f"Warning: site '{RIGHT_SITE}' not found.")
    try:
        left_site_id = name2id(model, mujoco.mjtObj.mjOBJ_SITE, LEFT_SITE)
    except Exception:
        print(f"Warning: site '{LEFT_SITE}' not found.")

    # joints we care about (from your mjcf): r1, r2, r3_1, l1, l2, l3_1
    joint_names = ["r1", "r2", "r3_1", "l1", "l2", "l3_1"]
    joint_qpos_addrs = {}
    for jn in joint_names:
        try:
            adr, jid = joint_qpos_index(model, jn)
            joint_qpos_addrs[jn] = adr
        except Exception:
            # If a joint is missing, skip it
            print(f"Joint '{jn}' not found in model; skipping it.")
    if len(joint_qpos_addrs) == 0:
        raise SystemExit("No joints found. Check names in MJCF and MJCF_PATH.")

    # Define sampling ranges for each joint; override if you want specific ranges
    # Revolute (r1, r2, l1, l2): use +-90 deg default (you can change)
    # Linear (r3_1, l3_1): use 0..0.134 from your MJCF defaults
    q_ranges = {}
    for jn in joint_qpos_addrs:
        if jn in ("r3_1", "l3_1"):
            q_ranges[jn] = np.linspace(0.0, 0.134, LINEAR_SAMPLES)    # linear slide
        else:
            q_ranges[jn] = np.linspace(-np.pi/2, np.pi/2, ANGLE_SAMPLES)  # revolute

    # Build a sampling grid but keep dimensionality manageable:
    # To avoid combinatorial explosion we'll sample joints independently
    # and also sample a small random subset of the full grid.
    # Strategy:
    #  - Create a coarse full-grid if total points <= 20000
    #  - Otherwise sample uniformly random points up to MAX_SAMPLES
    MAX_SAMPLES = 15000

    # compute full grid size
    grid_sizes = [len(q_ranges[jn]) for jn in joint_qpos_addrs]
    total_points = np.prod(grid_sizes)
    print("Joint sampling sizes:", {jn: len(q_ranges[jn]) for jn in joint_qpos_addrs})
    print("Total full-grid points would be:", total_points)

    # produce sample list of q vectors (qpos for full model)
    samples = []
    joint_list = list(joint_qpos_addrs.keys())

    if total_points <= MAX_SAMPLES:
        print("Using full grid sampling.")
        # use np.meshgrid to create grid
        grids = np.meshgrid(*[q_ranges[jn] for jn in joint_list], indexing="ij")
        flat = [g.ravel() for g in grids]
        for idx in range(flat[0].size):
            qvec = {}
            for k, jn in enumerate(joint_list):
                qvec[jn] = float(flat[k][idx])
            samples.append(qvec)
    else:
        print(f"Full grid too large; sampling {MAX_SAMPLES} random configurations.")
        rng = np.random.default_rng(12345)
        for _ in range(MAX_SAMPLES):
            qvec = {}
            for jn in joint_list:
                arr = q_ranges[jn]
                qvec[jn] = float(rng.choice(arr))
            samples.append(qvec)

    # prepare arrays to collect positions in main_body frame
    pos_list = []

    # set a baseline qpos array
    qpos = np.array(data.qpos, copy=True)

    # For each sample, set qpos entries for each joint and forward simulate kinematics
    for sidx, s in enumerate(samples):
        # reset qpos
        qpos[:] = 0.0
        # apply sample into the right qpos indices
        for jn, val in s.items():
            adr = joint_qpos_addrs[jn]
            # some joints may have multi-dof qpos (e.g., free joints) — handle scalars here
            qpos[adr] = val
        # copy into data.qpos and call mj_forward
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)   # compute positions

        # read site positions (in global/world frame)
        # convert to main-body relative coordinates: x_rel = world - main_body_world_pos, then rotate into main_body frame
        mb_pos = data.xpos[main_body_id]  # world position of the body frame origin
        # get main body orientation (rotation matrix)
        mb_xmat = data.xmat[main_body_id].reshape(3,3)  # row-major 3x3 rotation matrix

        # check each site
        for site_id, site_name in ((right_site_id, RIGHT_SITE), (left_site_id, LEFT_SITE)):
            if site_id is None:
                continue
            world_pos = data.site_xpos[site_id].copy()
            # position relative to main body origin, expressed in main-body coordinates:
            rel = world_pos - mb_pos
            # rotate into main body frame coordinates -> main body frame = R^T * rel (since xmat is from body to world)
            rel_in_body = mb_xmat.T.dot(rel)
            pos_list.append(rel_in_body)

    pos_arr = np.array(pos_list)
    if pos_arr.size == 0:
        raise SystemExit("No site positions were collected — verify site names and model.")

    # Plot workspace as 3D scatter
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pos_arr[:,0], pos_arr[:,1], pos_arr[:,2], s=2, alpha=0.6)
    ax.set_xlabel("x (m) — main body frame")
    ax.set_ylabel("y (m) — main body frame")
    ax.set_zlabel("z (m) — main body frame")
    ax.set_title("Sampled Workspace of Gripper Sites (relative to MAIN_BODY)")
    # set equal aspect ratio (approx)
    max_range = np.max(np.ptp(pos_arr, axis=0))
    mid = np.mean(pos_arr, axis=0)
    ax.set_xlim(mid[0]-max_range/2, mid[0]+max_range/2)
    ax.set_ylim(mid[1]-max_range/2, mid[1]+max_range/2)
    ax.set_zlim(mid[2]-max_range/2, mid[2]+max_range/2)
    plt.show()

if __name__ == "__main__":
    main()