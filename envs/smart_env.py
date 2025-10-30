# climbbot_env_fixed.py
import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

# ---- small util: convert 3x3 rotation matrix to quaternion (w, x, y, z) ----
def mat3_to_quat(mat3):
    # mat3: shape (3,3)
    m = mat3
    trace = m[0,0] + m[1,1] + m[2,2]
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2,1] - m[1,2]) * s
        y = (m[0,2] - m[2,0]) * s
        z = (m[1,0] - m[0,1]) * s
    else:
        if m[0,0] > m[1,1] and m[0,0] > m[2,2]:
            s = 2.0 * np.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2])
            w = (m[2,1] - m[1,2]) / s
            x = 0.25 * s
            y = (m[0,1] + m[1,0]) / s
            z = (m[0,2] + m[2,0]) / s
        elif m[1,1] > m[2,2]:
            s = 2.0 * np.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2])
            w = (m[0,2] - m[2,0]) / s
            x = (m[0,1] + m[1,0]) / s
            y = 0.25 * s
            z = (m[1,2] + m[2,1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1])
            w = (m[1,0] - m[0,1]) / s
            x = (m[0,2] + m[2,0]) / s
            y = (m[1,2] + m[2,1]) / s
            z = 0.25 * s
    return np.array([w, x, y, z], dtype=np.float32)

# ----------------------------
# Utilities for name -> id
# ----------------------------
def name2id(model, objtype, name):
    if objtype == 'site':
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if objtype == 'joint':
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if objtype == 'body':
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if objtype == 'actuator':
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    raise ValueError("unsupported objtype")

# ----------------------------
# Environment
# ----------------------------
class ClimbBotEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self,
             xml_path: str,
             ee_max_step: float = 0.03,
             dt: float = 0.001,
             sim_substeps: int = 10,
             max_steps: int = 1000,
             success_radius: float = 0.03,
             render_mode: str | None = None,
             debug: bool = False,
             **kwargs):
        """
        ClimbBotEnv constructor.

        Accepts `render_mode` and `debug` (and arbitrary kwargs) so it's compatible with
        Stable-Baselines3 / gym wrappers that pass these keywords.
        """
        # record render/debug preferences (used by gym API or for verbose logging)
        self.render_mode = render_mode
        self.debug = bool(debug)

        assert os.path.exists(xml_path), f"xml file not found: {xml_path}"
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # physics params
        self.dt = dt
        self.sim_substeps = sim_substeps
        self.model.opt.timestep = dt
        self.ee_max_step = float(ee_max_step)
        self.max_steps = int(max_steps)
        self.success_radius = float(success_radius)

        # cache ids (raise informative errors if missing)
        try:
            self.site_ids = {
                'r': name2id(self.model, 'site', 'r_grip_site'),
                'l': name2id(self.model, 'site', 'l_grip_site'),
                'base': name2id(self.model, 'site', 'base_site'),
            }
        except Exception as e:
            raise RuntimeError("Missing expected site in MJCF (r_grip_site, l_grip_site, base_site required).") from e

        # IK joints per arm
        self.joint_names_r = ["r1", "r2", "r3_1"]
        self.joint_names_l = ["l1", "l2", "l3_1"]
        try:
            self.joint_ids_r = [name2id(self.model, 'joint', n) for n in self.joint_names_r]
            self.joint_ids_l = [name2id(self.model, 'joint', n) for n in self.joint_names_l]
        except Exception as e:
            raise RuntimeError("Missing expected joint names (r1,r2,r3_1,l1,l2,l3_1).") from e

        # actuators (assume 1:1 mapping)
        self.actuator_names_r = ["r1_ctrl", "r2_ctrl", "r3_1_ctrl"]
        self.actuator_names_l = ["l1_ctrl", "l2_ctrl", "l3_1_ctrl"]
        try:
            self.actuator_ids_r = [name2id(self.model, 'actuator', n) for n in self.actuator_names_r]
            self.actuator_ids_l = [name2id(self.model, 'actuator', n) for n in self.actuator_names_l]
        except Exception:
            # allow missing actuator names (we'll still attempt to set data.ctrl by index later)
            self.actuator_ids_r = []
            self.actuator_ids_l = []

        # handhold bodies: hold_1 ... hold_N
        self.hold_bodies = []
        i = 1
        while True:
            name = f"hold_{i}"
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid == -1:
                break
            self.hold_bodies.append((name, bid))
            i += 1
        if len(self.hold_bodies) == 0:
            raise RuntimeError("No hold bodies found (expected hold_1...hold_N in MJCF)")

        # observation & action spaces
        obs_dim = 3 + 4 + 3 + 3 + 3 + 3 + 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        # bookkeeping
        self._step_count = 0
        self._right_target_idx = 0
        self._left_target_idx = 0
        self._rng = np.random.RandomState()

        # scratch for jacobian
        self._jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        self._jacr = np.zeros((3, self.model.nv), dtype=np.float64)

        # reset
        self.reset()

    # ----------------------------
    # helpers: robust attribute access for site orientation/velocity
    # ----------------------------
    def _site_quat(self, site_id):
        # prefer quaternion if available
        if hasattr(self.data, "site_xquat"):
            q = self.data.site_xquat[site_id].copy()
            return q.astype(np.float32)
        # fallback: site_xmat -> convert to quaternion
        if hasattr(self.data, "site_xmat"):
            mat_flat = self.data.site_xmat[site_id].copy()  # length 9
            mat3 = np.asarray(mat_flat).reshape(3,3)
            return mat3_to_quat(mat3)
        # last fallback: identity quaternion
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def _site_linvel(self, site_id):
        # try site_xvel (some bindings), else zeros
        if hasattr(self.data, "site_xvel"):
            return self.data.site_xvel[site_id].copy().astype(np.float32)
        # some bindings expose sensordata; ignore here
        return np.zeros(3, dtype=np.float32)

    # ----------------------------
    # other helpers
    # ----------------------------
    def _get_all_hold_positions(self):
        mujoco.mj_forward(self.model, self.data)
        poses = []
        for name, bid in self.hold_bodies:
            # body com/world pos via data.xpos (body com)
            poses.append(self.data.xpos[bid].copy())
        return np.stack(poses, axis=0)

    # ----------------------------
    # IK: Jacobian damped least squares (per-arm)
    # ----------------------------
    def _ik_solve_arm(self, site_id, joint_ids, desired_world_pos, max_iters=8, tol=1e-4, damping=1e-3):
        """
        Damped least squares using mj_jacSite with jacp (3 x nv) and jacr (3 x nv).
        Updates self.data.qpos during the solve and returns joint position targets.
        """
        # active DOF column indices (single-dof joints assumed)
        active_dofs = [self.model.jnt_dofadr[jid] for jid in joint_ids]

        for _ in range(max_iters):
            mujoco.mj_forward(self.model, self.data)
            p_cur = self.data.site_xpos[site_id].copy()
            err = desired_world_pos - p_cur
            if np.linalg.norm(err) < tol:
                break

            # compute jacobians into jacp and jacr (each shape 3 x nv)
            mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, site_id)
            Jp = self._jacp  # shape (3, nv)

            # reduce to active columns (3 x m)
            J_small = Jp[:, active_dofs]   # (3 x m)

            # normal equations with damping
            A = J_small.T @ J_small
            A += (damping**2) * np.eye(A.shape[0])
            b = J_small.T @ err
            try:
                dq = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                dq = np.linalg.lstsq(A, b, rcond=None)[0]

            # clamp step magnitude for stability
            max_step = 0.05
            nd = np.linalg.norm(dq)
            if nd > max_step:
                dq = dq * (max_step / (nd + 1e-12))

            # apply into qpos (assumes 1-DOF joints)
            for i, dof in enumerate(active_dofs):
                self.data.qpos[dof] += float(dq[i])

        # return qpos targets in same order as joint_ids
        q_targets = [self.data.qpos[self.model.jnt_qposadr[jid]] for jid in joint_ids]
        return np.array(q_targets, dtype=np.float32)

    # ----------------------------
    # Gym API
    # ----------------------------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng.seed(seed)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self._right_target_idx = 0
        self._left_target_idx = 0
        self._step_count = 0
        return self._get_obs(), {}

    def _get_obs(self):
        mujoco.mj_forward(self.model, self.data)
        base_pos = self.data.site_xpos[self.site_ids['base']].copy()
        base_quat = self._site_quat(self.site_ids['base'])
        base_linvel = self._site_linvel(self.site_ids['base'])

        rpos = self.data.site_xpos[self.site_ids['r']].copy()
        lpos = self.data.site_xpos[self.site_ids['l']].copy()
        all_holds = self._get_all_hold_positions()
        r_idx = min(self._right_target_idx, len(all_holds)-1)
        l_idx = min(self._left_target_idx, len(all_holds)-1)
        r_target = all_holds[r_idx].copy()
        l_target = all_holds[l_idx].copy()

        obs = np.concatenate([base_pos, base_quat, base_linvel, rpos, lpos, r_target, l_target]).astype(np.float32)
        return obs

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        assert action.shape == (6,)
        delta_r = action[0:3] * self.ee_max_step
        delta_l = action[3:6] * self.ee_max_step

        mujoco.mj_forward(self.model, self.data)
        p_r = self.data.site_xpos[self.site_ids['r']].copy()
        p_l = self.data.site_xpos[self.site_ids['l']].copy()
        p_r_target = p_r + delta_r
        p_l_target = p_l + delta_l

        all_holds = self._get_all_hold_positions()
        r_goal = all_holds[min(self._right_target_idx, len(all_holds)-1)]
        l_goal = all_holds[min(self._left_target_idx, len(all_holds)-1)]

        qpos_backup = self.data.qpos.copy()
        q_targets_r = self._ik_solve_arm(self.site_ids['r'], self.joint_ids_r, p_r_target,
                                         max_iters=6, tol=1e-4, damping=1e-3)
        # set actuators for right arm if we have their ids, else try best-effort mapping
        if len(self.actuator_ids_r) == len(q_targets_r) and len(self.actuator_ids_r) > 0:
            for act_id, qt in zip(self.actuator_ids_r, q_targets_r):
                self.data.ctrl[act_id] = float(qt)
        else:
            # fallback: attempt to write into ctrl first few indices
            for i, qt in enumerate(q_targets_r):
                if i < self.data.ctrl.shape[0]:
                    self.data.ctrl[i] = float(qt)

        # restore qpos before solving left arm
        self.data.qpos[:] = qpos_backup[:]
        mujoco.mj_forward(self.model, self.data)

        q_targets_l = self._ik_solve_arm(self.site_ids['l'], self.joint_ids_l, p_l_target,
                                         max_iters=6, tol=1e-4, damping=1e-3)
        if len(self.actuator_ids_l) == len(q_targets_l) and len(self.actuator_ids_l) > 0:
            for act_id, qt in zip(self.actuator_ids_l, q_targets_l):
                self.data.ctrl[act_id] = float(qt)
        else:
            # fallback: write after the indices used for right arm
            start = len(self.actuator_ids_r) if len(self.actuator_ids_r) > 0 else 3
            for i, qt in enumerate(q_targets_l):
                idx = start + i
                if idx < self.data.ctrl.shape[0]:
                    self.data.ctrl[idx] = float(qt)

        for _ in range(self.sim_substeps):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        obs = self._get_obs()

        mujoco.mj_forward(self.model, self.data)
        p_r = self.data.site_xpos[self.site_ids['r']].copy()
        p_l = self.data.site_xpos[self.site_ids['l']].copy()
        r_dist = np.linalg.norm(p_r - r_goal)
        l_dist = np.linalg.norm(p_l - l_goal)
        reward = -(r_dist + l_dist) - 0.01 * (np.linalg.norm(action) ** 2)

        done = False
        info = {}
        if r_dist < self.success_radius and self._right_target_idx < len(all_holds)-1:
            reward += 5.0
            self._right_target_idx += 1
        if l_dist < self.success_radius and self._left_target_idx < len(all_holds)-1:
            reward += 5.0
            self._left_target_idx += 1

        if self._step_count >= self.max_steps:
            done = True
            info["TimeLimit.truncated"] = True

        if (self._right_target_idx >= len(all_holds)-1) and (self._left_target_idx >= len(all_holds)-1):
            done = True
            reward += 50.0
            info["task_complete"] = True

        # Gymnasium expects (obs, reward, terminated, truncated, info)
        return obs, float(reward), done, False, info

    def render(self):
        try:
            if not hasattr(self, "_viewer") or self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.render()
        except Exception as e:
            print("Render error (viewer may be unavailable):", e)

    def close(self):
        try:
            if hasattr(self, "_viewer") and self._viewer is not None:
                self._viewer.close()
        except Exception:
            pass

    def seed(self, s=None):
        self._rng.seed(s)


# ----------------------------
# Quick usage example
# ----------------------------
if __name__ == "__main__":
    xml = "/Users/aaronthomas/Desktop/Engineering_Projects/Climbing Robot/assets/robot_mjcf.xml"   # replace with your actual MJCF path
    env = ClimbBotEnv(xml_path=xml)
    obs, _ = env.reset()
    print("obs shape", obs.shape)
    for i in range(100):
        a = env.action_space.sample() * 0.2  # small random actions
        obs, rew, done, truncated, info = env.step(a)
        if i % 10 == 0:
            print(f"step {i} reward {rew:.3f}")
        if done:
            print("done:", info)
            break
    env.close()
