# generate_workspace_goals.py
# Requires: mujoco (official python bindings), numpy, scipy, matplotlib (for optional plotting)
# Example:
#   python generate_workspace_goals.py
# Then import functions from this file in your env to sample goals.

import os
import numpy as np
import mujoco
from scipy.spatial import cKDTree, ConvexHull, Delaunay
import random

# ---------- USER CONFIG ----------
MJCF_PATH = "/Users/aaronthomas/Desktop/Engineering_Projects/Climbing Robot/assets/robot_mjcf.xml"
MAIN_BODY_NAME = "assembly_7"
RIGHT_SITE = "r_grip_site"
LEFT_SITE  = "l_grip_site"
ANGLE_SAMPLES = 20
LINEAR_SAMPLES = 20
MAX_SAMPLES = 15000
# voxel params
VOXEL_SIZE = 0.01   # meters; increase for coarser/less memory
VOXEL_PADDING = 0.02
# KDTree tolerance default
DEFAULT_EPS = 0.01
# ----------------------------------

def name2id(model, obj_type, name):
    idx = mujoco.mj_name2id(model, obj_type, name.encode() if isinstance(name, str) else name)
    if idx < 0:
        raise ValueError(f"Name '{name}' not found (obj_type={obj_type})")
    return idx

def joint_qpos_index(model, joint_name):
    jid = name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    return int(model.jnt_qposadr[jid]), int(jid)

def build_samples_from_model(mjcf_path,
                             main_body_name=MAIN_BODY_NAME,
                             right_site=RIGHT_SITE,
                             left_site=LEFT_SITE,
                             angle_samples=ANGLE_SAMPLES,
                             linear_samples=LINEAR_SAMPLES,
                             max_samples=MAX_SAMPLES):
    if not os.path.exists(mjcf_path):
        raise SystemExit(f"MJCF file not found: {mjcf_path}")

    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data  = mujoco.MjData(model)

    main_body_id = name2id(model, mujoco.mjtObj.mjOBJ_BODY, main_body_name)

    def maybe_site(name):
        try:
            return name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        except Exception:
            return None

    right_site_id = maybe_site(right_site)
    left_site_id = maybe_site(left_site)

    # joints of interest from your MJCF:
    joint_names = ["r1", "r2", "r3_1", "l1", "l2", "l3_1"]
    joint_qpos_addrs = {}
    for jn in joint_names:
        try:
            adr, _jid = joint_qpos_index(model, jn)
            joint_qpos_addrs[jn] = adr
        except Exception:
            print(f"Joint '{jn}' not found; skipping.")

    if len(joint_qpos_addrs) == 0:
        raise SystemExit("No joints found. Check MJCF.")

    # ranges
    q_ranges = {}
    for jn in joint_qpos_addrs:
        if jn in ("r3_1", "l3_1"):
            q_ranges[jn] = np.linspace(0.0, 0.134, linear_samples)
        else:
            q_ranges[jn] = np.linspace(-np.pi/2, np.pi/2, angle_samples)

    # sampling strategy (same as your original script)
    grid_sizes = [len(q_ranges[jn]) for jn in joint_qpos_addrs]
    total_points = int(np.prod(grid_sizes))
    joint_list = list(joint_qpos_addrs.keys())
    samples = []

    if total_points <= max_samples:
        grids = np.meshgrid(*[q_ranges[jn] for jn in joint_list], indexing="ij")
        flat = [g.ravel() for g in grids]
        for idx in range(flat[0].size):
            qvec = {jn: float(flat[k][idx]) for k, jn in enumerate(joint_list)}
            samples.append(qvec)
    else:
        rng = np.random.default_rng(12345)
        for _ in range(max_samples):
            qvec = {jn: float(rng.choice(q_ranges[jn])) for jn in joint_list}
            samples.append(qvec)

    # collect site positions relative to main body frame
    pos_list = []    # positions in main-body frame
    pos_world_list = []  # in world frame (for convenience)
    qpos = np.array(data.qpos, copy=True)
    for s in samples:
        qpos[:] = 0.0
        for jn, val in s.items():
            adr = joint_qpos_addrs[jn]
            qpos[adr] = val
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)

        mb_pos = data.xpos[main_body_id].copy()
        mb_xmat = data.xmat[main_body_id].reshape(3,3)

        for site_id in (right_site_id, left_site_id):
            if site_id is None:
                continue
            world_pos = data.site_xpos[site_id].copy()
            rel = world_pos - mb_pos
            rel_in_body = mb_xmat.T.dot(rel)
            pos_list.append(rel_in_body)
            pos_world_list.append(world_pos)

    pos_arr = np.array(pos_list)
    pos_world_arr = np.array(pos_world_list)
    if pos_arr.size == 0:
        raise SystemExit("No site positions collected. Verify site names in MJCF.")
    # Also return model/data and main-body transforms for mapping back to world
    return {
        "model": model,
        "data": data,
        "main_body_id": main_body_id,
        "pos_arr": pos_arr,
        "pos_world_arr": pos_world_arr,
        "mb_pos_world": data.xpos[main_body_id].copy(),
        "mb_xmat": data.xmat[main_body_id].reshape(3,3).copy(),
        "right_site_id": right_site_id,
        "left_site_id": left_site_id,
        "joint_qpos_addrs": joint_qpos_addrs
    }

# ---- Workspace representations & sampling utilities ----

def build_kdtree(pos_arr):
    return cKDTree(pos_arr)

def build_convex_hull_delaunay(pos_arr):
    hull = None
    delaunay = None
    try:
        if pos_arr.shape[0] >= 4:
            hull = ConvexHull(pos_arr)
            pts = pos_arr[hull.vertices]
            # Delaunay on hull vertices to get simplices (tetrahedra)
            delaunay = Delaunay(pts)
    except Exception as e:
        print("Convex hull / Delaunay build failed:", e)
    return hull, delaunay

def build_voxel_grid(points, voxel_size=VOXEL_SIZE, padding=VOXEL_PADDING):
    pts = np.asarray(points)
    mins = pts.min(axis=0) - padding
    maxs = pts.max(axis=0) + padding
    dims = np.ceil((maxs - mins) / voxel_size).astype(int) + 1
    idx = np.floor((pts - mins) / voxel_size).astype(int)
    occ = np.zeros(tuple(dims), dtype=bool)
    # clamp indices inside dims (safety)
    idx[:,0] = np.clip(idx[:,0], 0, dims[0]-1)
    idx[:,1] = np.clip(idx[:,1], 0, dims[1]-1)
    idx[:,2] = np.clip(idx[:,2], 0, dims[2]-1)
    occ[idx[:,0], idx[:,1], idx[:,2]] = True
    return mins, occ, voxel_size

# membership tests
def is_in_workspace_knn(kdtree, query_point, eps=DEFAULT_EPS):
    d, _ = kdtree.query(np.asarray(query_point), k=1)
    return float(d) <= float(eps)

def is_in_workspace_convex(delaunay, query_point):
    if delaunay is None:
        return False
    return bool(delaunay.find_simplex(np.atleast_2d(query_point)) >= 0)

def is_in_workspace_voxel(voxel_origin, voxel_occ, voxel_size, query_point):
    q = np.asarray(query_point)
    idx = np.floor((q - voxel_origin) / voxel_size).astype(int)
    if np.any(idx < 0) or np.any(idx >= np.array(voxel_occ.shape)):
        return False
    return bool(voxel_occ[idx[0], idx[1], idx[2]])

def is_in_workspace(query_point, method, helpers, **kwargs):
    """
    method: 'knn', 'convex', 'voxel', 'any'
    helpers: dict with entries kdtree, delaunay, voxel_origin, voxel_occ, voxel_size
    """
    if method == "knn":
        return is_in_workspace_knn(helpers["kdtree"], query_point, eps=kwargs.get("eps", DEFAULT_EPS))
    if method == "convex":
        return is_in_workspace_convex(helpers.get("delaunay"), query_point)
    if method == "voxel":
        return is_in_workspace_voxel(helpers["voxel_origin"], helpers["voxel_occ"], helpers["voxel_size"], query_point)
    if method == "any":
        return (is_in_workspace_knn(helpers["kdtree"], query_point, eps=kwargs.get("eps", DEFAULT_EPS))
                or is_in_workspace_convex(helpers.get("delaunay"), query_point)
                or is_in_workspace_voxel(helpers["voxel_origin"], helpers["voxel_occ"], helpers["voxel_size"], query_point))
    raise ValueError("Unknown method")

# ---- Goal samplers ----
def sample_goal_from_samples(pos_arr, mb_pos_world, mb_xmat, n=1, jitter=0.0, return_world=True):
    """
    Sample goals by picking random samples from pos_arr (main-body frame).
    jitter: meters applied in main-body frame
    returns list of dicts with keys: pos_body (3,), pos_world (3,), quat_world (4,)
    """
    out = []
    for _ in range(n):
        idx = random.randrange(len(pos_arr))
        p_body = pos_arr[idx].copy()
        if jitter > 0:
            p_body = p_body + np.random.normal(scale=jitter, size=3)
        # convert to world: world = R * p_body + mb_pos_world (since p_body expressed in main-body)
        p_world = mb_xmat.dot(p_body) + mb_pos_world
        # default quat = main-body orientation (identity transform to world already encoded in mb_xmat)
        # We'll return identity quaternion (1,0,0,0) for simplicity OR you can compute from mb_xmat if needed
        # Compute quaternion from mb_xmat (body to world) if you prefer; here return identity.
        quat_world = rotation_matrix_to_quat(mb_xmat)  # so gripper oriented same as main-body
        out.append({"pos_body": p_body, "pos_world": p_world, "quat_world": quat_world})
    return out

def sample_goal_convex(delaunay, hull_pts, mb_pos_world, mb_xmat, n=1):
    """
    Uniform-ish sample inside convex hull by sampling barycentric coords inside random simplex.
    hull_pts: pos_arr[hull.vertices]
    """
    if delaunay is None:
        return []
    out = []
    for _ in range(n):
        # choose a random simplex (tetrahedron)
        simplex_idx = random.randrange(len(delaunay.simplices))
        vertices = delaunay.points[delaunay.simplices[simplex_idx]]
        # sample barycentric uniformly in tetrahedron:
        # generate 4 random numbers, take their -ln to convert to exponential, normalize
        r = np.random.rand(4)
        # To get uniform barycentric: take random numbers u_i~Uniform(0,1) then use (u1**(1/3), u2**(1/2), u3) trick OR use exponentials:
        y = -np.log(np.random.rand(4))
        bary = y / np.sum(y)
        p_body = (bary[0]*vertices[0] + bary[1]*vertices[1] + bary[2]*vertices[2] + bary[3]*vertices[3])
        p_world = mb_xmat.dot(p_body) + mb_pos_world
        quat_world = rotation_matrix_to_quat(mb_xmat)
        out.append({"pos_body": p_body, "pos_world": p_world, "quat_world": quat_world})
    return out

def sample_goal_voxel(voxel_origin, voxel_occ, voxel_size, mb_pos_world, mb_xmat, n=1, jitter_frac=0.6):
    """
    Sample occupied voxels and jitter inside them.
    jitter_frac: fraction of voxel size used for jitter
    """
    occ_idx = np.array(np.nonzero(voxel_occ)).T
    if occ_idx.shape[0] == 0:
        return []
    out = []
    for _ in range(n):
        i = random.randrange(occ_idx.shape[0])
        ix, iy, iz = occ_idx[i]
        # center of voxel in main-body frame
        center = voxel_origin + voxel_size * (np.array([ix, iy, iz]) + 0.5)
        jitter = (np.random.rand(3) - 0.5) * voxel_size * jitter_frac
        p_body = center + jitter
        p_world = mb_xmat.dot(p_body) + mb_pos_world
        quat_world = rotation_matrix_to_quat(mb_xmat)
        out.append({"pos_body": p_body, "pos_world": p_world, "quat_world": quat_world})
    return out

# small helper: convert 3x3 rotation matrix to quaternion (w,x,y,z)
def rotation_matrix_to_quat(R):
    # From matrix to quaternion (robust-ish)
    m = np.asarray(R, dtype=float)
    tr = m[0,0] + m[1,1] + m[2,2]
    if tr > 0:
        S = np.sqrt(tr+1.0)*2
        w = 0.25 * S
        x = (m[2,1] - m[1,2]) / S
        y = (m[0,2] - m[2,0]) / S
        z = (m[1,0] - m[0,1]) / S
    elif (m[0,0] > m[1,1]) and (m[0,0] > m[2,2]):
        S = np.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2]) * 2
        w = (m[2,1] - m[1,2]) / S
        x = 0.25 * S
        y = (m[0,1] + m[1,0]) / S
        z = (m[0,2] + m[2,0]) / S
    elif m[1,1] > m[2,2]:
        S = np.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2]) * 2
        w = (m[0,2] - m[2,0]) / S
        x = (m[0,1] + m[1,0]) / S
        y = 0.25 * S
        z = (m[1,2] + m[2,1]) / S
    else:
        S = np.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1]) * 2
        w = (m[1,0] - m[0,1]) / S
        x = (m[0,2] + m[2,0]) / S
        y = (m[1,2] + m[2,1]) / S
        z = 0.25 * S
    return np.array([w, x, y, z], dtype=float)

# ---- Example main (build everything and produce example goals) ----
def main_example():
    print("Building samples from MJCF (this may take a while)...")
    info = build_samples_from_model(MJCF_PATH)
    pos_arr = info["pos_arr"]
    mb_pos = info["mb_pos_world"]
    mb_xmat = info["mb_xmat"]

    print("Collected", pos_arr.shape[0], "sampled positions (in main-body frame).")

    # build kd-tree
    kdtree = build_kdtree(pos_arr)
    hull, delaunay = build_convex_hull_delaunay(pos_arr)
    voxel_origin, voxel_occ, voxel_size = build_voxel_grid(pos_arr)

    helpers = {
        "kdtree": kdtree,
        "hull": hull,
        "delaunay": delaunay,
        "voxel_origin": voxel_origin,
        "voxel_occ": voxel_occ,
        "voxel_size": voxel_size
    }

    # generate some example goals
    goals_from_samples = sample_goal_from_samples(pos_arr, mb_pos, mb_xmat, n=5, jitter=0.005)
    goals_convex = sample_goal_convex(delaunay, pos_arr[hull.vertices] if hull is not None else pos_arr, mb_pos, mb_xmat, n=5)
    goals_voxel = sample_goal_voxel(voxel_origin, voxel_occ, voxel_size, mb_pos, mb_xmat, n=5)

    print("\nExample goals (from samples):")
    for g in goals_from_samples:
        b = g["pos_body"]; w = g["pos_world"]
        print(" body:", np.round(b,4), " world:", np.round(w,4))

    # membership checks for first convex goal
    if len(goals_convex) > 0:
        q = goals_convex[0]["pos_body"]
        print("\nMembership checks for first convex-sampled goal (in main-body frame):")
        print(" knn:", is_in_workspace(q, "knn", helpers, eps=DEFAULT_EPS))
        print(" convex:", is_in_workspace(q, "convex", helpers))
        print(" voxel:", is_in_workspace(q, "voxel", helpers))
        print(" any:", is_in_workspace(q, "any", helpers, eps=DEFAULT_EPS))

    # return helpers and example goals for reuse
    return helpers, {"samples": goals_from_samples, "convex": goals_convex, "voxel": goals_voxel}

if __name__ == "__main__":
    helpers, examples = main_example()
    # You can now use helpers to call is_in_workspace(...) or the samplers in your env.