import os
import gymnasium as gym
import numpy as np
import mujoco
import mujoco_viewer


class ClimbBotEnv(gym.Env):
    """
    Gymnasium environment for the climbing robot (MuJoCo).

    Observation vector (order):
       [ ee_r (3),
         ee_l (3),
         qpos_norm_included (M),
         qvel_included (K),
         target_r (3),
         target_l (3) ]

    Notes:
    - Floating-base (freejoint) qpos/qvel entries are EXCLUDED from the observation by default
      to keep the observation shape stable when you enable/disable a <freejoint />.
    - `max_actuator_velocity` is specified in units/second (rad/s or m/s) and is converted to a
      per-control-step delta using `control_timestep`.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, xml_path="climbbot3.xml", render_mode=None, debug=False):
        super().__init__()

        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML model not found: {xml_path}")

        # --- Load MuJoCo model & data ---
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Render state
        self.viewer = None
        self.render_mode = render_mode
        self.debug = debug

        # --- safe name lookup helper ---
        def _safe_name2id(model, obj_type, name):
            try:
                return mujoco.mj_name2id(model, obj_type, name)
            except Exception:
                return None

        # Site and body IDs used in this env (safe)
        self.r_grip_id = _safe_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "r_grip_site")
        self.l_grip_id = _safe_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "l_grip_site")
        self.base_id = _safe_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "assembly_7")

        # build joint-name list (best-effort)
        try:
            self._joint_names = [n.decode() if isinstance(n, bytes) else str(n) for n in list(self.model.jnt_names)]
        except Exception:
            self._joint_names = [f"j{jid}" for jid in range(int(self.model.njnt))]

        # Fixed default targets (world frame)
        self.target_r = np.array([0.7219, -0.1873, 0.0548], dtype=np.float32)
        self.target_l = np.array([0.5221,  0.0956, 0.0549], dtype=np.float32)

        # minimum allowed Z height for any target (in meters)
        self.min_target_z = 0.1

        # ---------- target management & workspace limits ----------
        # If True, when an end-effector enters the success radius the env immediately samples a new target.
        self.auto_advance_target = True
        # meters for considering a target reached
        self.success_radius = 0.03

        # workspace centers and half-extents (axis-aligned box)
        self.workspace_center = np.array([0.7, -0.2, 0.05], dtype=np.float32)
        self.workspace_half_extents = np.array([0.12, 0.12, 0.06], dtype=np.float32)

        self.workspace_r_center = self.workspace_center.copy()
        self.workspace_r_half = self.workspace_half_extents.copy()
        self.workspace_l_center = np.array([0.52, 0.095, 0.055], dtype=np.float32)
        self.workspace_l_half = self.workspace_half_extents.copy()

        # Pending target queue (for next episode; useful with VecEnv)
        self._pending_target_r = None
        self._pending_target_l = None

        # ---------- actuator setpoint velocity limits (units/sec) ----------
        # Default: 1.0 units/s for all actuators; you can override per-actuator later:
        self.max_actuator_velocity = np.ones(int(self.model.nu), dtype=np.float32)

        # current setpoint applied to actuators (initialize from current ctrl if possible)
        try:
            cp = np.array(self.data.ctrl, dtype=np.float32)
            if cp.shape[0] == self.model.nu:
                self._current_setpoint = cp.copy()
            else:
                self._current_setpoint = np.zeros(self.model.nu, dtype=np.float32)
        except Exception:
            self._current_setpoint = np.zeros(self.model.nu, dtype=np.float32)

        # Action space from actuator control ranges (n_act, 2)
        act_low = self.model.actuator_ctrlrange[:, 0].astype(np.float32)
        act_high = self.model.actuator_ctrlrange[:, 1].astype(np.float32)
        self.action_space = gym.spaces.Box(low=act_low, high=act_high, dtype=np.float32)

        # Internal RNG
        self.np_random = None

        # Basic bookkeeping
        self.reward = 0.0

        # Timesteps: sim vs control
        self.sim_timestep = float(self.model.opt.timestep)   # e.g. 0.001
        self.control_timestep = 0.02                         # default 20 ms per agent step = 50 Hz
        self.frame_skip = max(1, int(round(self.control_timestep / self.sim_timestep)))

        if debug:
            print(f"[env] sim_timestep={self.sim_timestep:.4f}, "
                  f"control_timestep={self.control_timestep:.3f}, "
                  f"frame_skip={self.frame_skip}")

        # Which joint names to exclude from obs (floating base typical name).
        # If your freejoint is named differently, add it here.
        self._excluded_joint_names = ["floating_base"]

        # Precompute per-joint qpos/qvel address indices for normalization and exclusion
        self._prepare_joint_qpos_slices()

        # Ensure MuJoCo derived data available
        mujoco.mj_forward(self.model, self.data)

        # Create a sample observation and set observation_space to match it exactly.
        sample_obs = self._get_obs()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=sample_obs.shape, dtype=np.float32
        )

    # ---------------------------
    # Joint qpos slice helper
    # ---------------------------
    def _prepare_joint_qpos_slices(self):
        """
        Build list of joint slices (start, length, low, high), mark excluded joints,
        and build include index masks for qpos and qvel.
        """
        njnt = int(self.model.njnt)
        nq = int(self.model.nq)
        nv = int(self.model.nv)

        fallback_low = -np.pi
        fallback_high = np.pi
        slices = []

        qpos_include_mask = np.ones(nq, dtype=bool)
        # default qvel include: all (we'll try to mask per-joint if dof info available)
        qvel_include_mask = np.ones(nv, dtype=bool)

        have_dof_info = hasattr(self.model, "jnt_dofadr") and hasattr(self.model, "jnt_dofnum")

        for jid in range(njnt):
            start = int(self.model.jnt_qposadr[jid])
            if jid < njnt - 1:
                next_start = int(self.model.jnt_qposadr[jid + 1])
                length = max(1, next_start - start)
            else:
                length = max(1, nq - start)

            try:
                jlow = float(self.model.jnt_range[jid, 0])
                jhigh = float(self.model.jnt_range[jid, 1])
                if np.isclose(jlow, jhigh):
                    jlow, jhigh = fallback_low, fallback_high
            except Exception:
                jlow, jhigh = fallback_low, fallback_high

            try:
                jname = self.model.jnt_names[jid].decode() if isinstance(self.model.jnt_names[jid], bytes) else str(self.model.jnt_names[jid])
            except Exception:
                jname = f"j{jid}"

            excluded = (jname in self._excluded_joint_names)

            slices.append({
                "jid": jid,
                "name": jname,
                "start": start,
                "length": length,
                "low": jlow,
                "high": jhigh,
                "excluded": excluded
            })

            if excluded:
                qpos_include_mask[start : start + length] = False

            if have_dof_info:
                dof_start = int(self.model.jnt_dofadr[jid])
                dof_num = int(self.model.jnt_dofnum[jid])
                if excluded:
                    qvel_include_mask[dof_start : dof_start + dof_num] = False

        self._jnt_qpos_slices = slices
        self._qpos_include_mask = qpos_include_mask
        self._qpos_include_idx = np.nonzero(qpos_include_mask)[0]

        # build qvel include idx (fallback heuristics if dof info not present)
        try:
            if have_dof_info:
                self._qvel_include_idx = np.nonzero(qvel_include_mask)[0]
            else:
                # best-effort: if nq == nv, use same included indices; otherwise include all qvel
                if int(self.model.nq) == int(self.model.nv):
                    self._qvel_include_idx = self._qpos_include_idx.copy()
                else:
                    self._qvel_include_idx = np.arange(int(self.model.nv))
        except Exception:
            self._qvel_include_idx = np.arange(int(self.model.nv))

    # ---------------------------
    # Reset / Step / Rendering
    # ---------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        mujoco.mj_resetData(self.model, self.data)

        # Apply queued next-episode targets (if any)
        if getattr(self, "_pending_target_r", None) is not None:
            self.target_r = self._pending_target_r.astype(np.float32)
            self._pending_target_r = None
        if getattr(self, "_pending_target_l", None) is not None:
            self.target_l = self._pending_target_l.astype(np.float32)
            self._pending_target_l = None

        # small initial noise
        self.data.qpos[:] += 0.01 * self.np_random.standard_normal(self.model.nq)
        self.data.qvel[:] = np.zeros(self.model.nv, dtype=np.float64)

        # Re-init current setpoint to current ctrl or qpos to avoid jumps
        try:
            cp = np.array(self.data.ctrl, dtype=np.float32)
            if cp.shape[0] == self.model.nu:
                self._current_setpoint = cp.copy()
            else:
                qp = np.array(self.data.qpos[:self.model.nu], dtype=np.float32)
                if qp.shape[0] == self.model.nu:
                    self._current_setpoint = qp.copy()
                else:
                    self._current_setpoint = np.zeros(self.model.nu, dtype=np.float32)
        except Exception:
            self._current_setpoint = np.zeros(self.model.nu, dtype=np.float32)

        mujoco.mj_forward(self.model, self.data)

        # If no explicit targets were set, ensure targets obey Z-floor & workspace clipping
        self.target_r = self._clip_to_workspace(self.target_r, self.workspace_r_center, self.workspace_r_half)
        self.target_l = self._clip_to_workspace(self.target_l, self.workspace_l_center, self.workspace_l_half)

        self.reward = 0.0
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        # Ensure action shape and bounds and apply to simulation
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        if action.shape[0] != self.model.nu:
            raise ValueError(f"Action length {action.shape[0]} does not match model.nu {self.model.nu}")

        # Convert stored max_actuator_velocity (units/sec) -> per-control-step max delta
        max_delta = self.max_actuator_velocity * float(self.control_timestep)

        # compute desired delta from current setpoint
        desired_delta = action - self._current_setpoint

        # clamp per-actuator delta
        clamped_delta = np.clip(desired_delta, -max_delta, max_delta)

        # update the internal setpoint (this is what we actually send to the position servos)
        self._current_setpoint = self._current_setpoint + clamped_delta

        # apply the (velocity-limited) setpoint to the simulator
        self.data.ctrl[:] = self._current_setpoint

        # optional: store last raw action for diagnostics / reward shaping
        self._last_action = action.copy()

        # Step simulation for frame_skip internal steps
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        # Compute observation and reward
        obs = self._get_obs()
        self.reward = self._compute_reward()

        # Termination checks
        terminated, truncated = self._check_termination()

        if self.render_mode == "human":
            self.render()

        info = {}
        return obs, float(self.reward), bool(terminated), bool(truncated), info

    # ---------------------------
    # Observations
    # ---------------------------
    def _get_obs(self):
        """
        Build observation using only the included qpos/qvel indices (floating base excluded).
        Order is:
          [ee_r (3), ee_l (3), qpos_norm_included (M), qvel_included (K), target_r (3), target_l (3)]
        """

        # Raw qpos and qvel from MuJoCo
        qpos_full = np.array(self.data.qpos, dtype=np.float32)
        qvel_full = np.array(self.data.qvel, dtype=np.float32)

        # select included qpos/qvel entries
        try:
            qpos = qpos_full[self._qpos_include_idx]
        except Exception:
            qpos = qpos_full.copy()

        try:
            qvel = qvel_full[self._qvel_include_idx]
        except Exception:
            qvel = qvel_full.copy()

        # Normalize qpos per included-joint slices
        qpos_norm = np.zeros_like(qpos, dtype=np.float32)
        off = 0
        for s in self._jnt_qpos_slices:
            if s.get("excluded", False):
                continue
            l = s["length"]
            low = s["low"]
            high = s["high"]
            mid = 0.5 * (low + high)
            half_span = 0.5 * (high - low)
            if np.isclose(half_span, 0.0):
                mid = 0.0
                half_span = np.pi
            slice_vals = qpos[off : off + l].astype(np.float32)
            qpos_norm[off : off + l] = (slice_vals - mid) / half_span
            off += l
            if off >= qpos.shape[0]:
                break

        qpos_norm = np.clip(qpos_norm, -5.0, 5.0)

        # End-effector positions (safe access)
        ee_r = np.zeros(3, dtype=np.float32)
        ee_l = np.zeros(3, dtype=np.float32)
        try:
            if self.r_grip_id is not None:
                ee_r = np.array(self.data.site_xpos[self.r_grip_id], dtype=np.float32)
        except Exception:
            pass
        try:
            if self.l_grip_id is not None:
                ee_l = np.array(self.data.site_xpos[self.l_grip_id], dtype=np.float32)
        except Exception:
            pass

        # Build obs in fixed order
        obs = np.concatenate([
            ee_r, ee_l,
            qpos_norm, qvel,
            self.target_r, self.target_l
        ]).astype(np.float32)

        # if self.debug:
        #     print(f"[obs] shape={obs.shape}, ee_r={ee_r}, ee_l={ee_l}")

        return obs

    # ---------------------------
    # Reward & Termination
    # ---------------------------
    def _compute_reward(self):
        """Distance-based reward to targets; supports auto-advance of targets when reached."""
        ee_r = np.array(self.data.site_xpos[self.r_grip_id], dtype=np.float32) if self.r_grip_id is not None else np.zeros(3, dtype=np.float32)
        ee_l = np.array(self.data.site_xpos[self.l_grip_id], dtype=np.float32) if self.l_grip_id is not None else np.zeros(3, dtype=np.float32)

        dist_r = float(np.linalg.norm(ee_r - self.target_r))
        dist_l = float(np.linalg.norm(ee_l - self.target_l))

        # dense reward (negative distances)
        reward = - (dist_r + dist_l)

        # small penalty on commanded action magnitude (stabilizes)
        act_pen = 0.0
        if hasattr(self, "_last_action"):
            act_pen = 1e-3 * float(np.sum(np.square(self._last_action)))
            reward -= act_pen

        # detect reach
        reached_r = dist_r < self.success_radius
        reached_l = dist_l < self.success_radius

        # If auto_advance_target is enabled, sample new targets immediately for reached arms
        if self.auto_advance_target:
            if reached_r:
                # sample around right workspace center, ensure clipped & Z-floor enforced
                new_r = self.sample_target(rng=self.np_random, workspace_center=self.workspace_r_center, radius=np.min(self.workspace_r_half))
                self.target_r = new_r.astype(np.float32)
            if reached_l:
                new_l = self.sample_target(rng=self.np_random, workspace_center=self.workspace_l_center, radius=np.min(self.workspace_l_half))
                self.target_l = new_l.astype(np.float32)

        # success bonuses
        if reached_r:
            reward += 1.5
        if reached_l:
            reward += 1.5
        if (reached_r and reached_l):
            reward += 5.0

        # if self.debug:
        #     print(f"[reward] d_r={dist_r:.3f}, d_l={dist_l:.3f}, act_pen={act_pen:.5f}, reward={reward:.3f}")

        return float(reward)

    def _check_termination(self):
        ee_r = np.array(self.data.site_xpos[self.r_grip_id], dtype=np.float32) if self.r_grip_id is not None else np.zeros(3, dtype=np.float32)
        ee_l = np.array(self.data.site_xpos[self.l_grip_id], dtype=np.float32) if self.l_grip_id is not None else np.zeros(3, dtype=np.float32)
        base_z = float(self.data.xpos[self.base_id][2]) if self.base_id is not None else 1.0

        terminated = False
        truncated = False

        # Safety: if base center of mass falls below a threshold => episode ends (negative signal)
        if base_z < 0.05:
            terminated = True

        # Time truncation
        if float(self.data.time) > 20.0:
            truncated = True

        if self.debug and (terminated or truncated):
            if base_z < 0.05:
                print("Terminated: base fell!")
            if truncated:
                print("Truncated: time limit hit.")

        return terminated, truncated

    # ---------------------------
    # Targets / sampling helpers
    # ---------------------------
    def _clip_to_workspace(self, point, center, half_extents):
        p = np.asarray(point, dtype=np.float32).reshape(3,)
        minv = center - half_extents
        maxv = center + half_extents
        p = np.minimum(np.maximum(p, minv), maxv)
        # enforce Z floor
        if p[2] < self.min_target_z:
            p[2] = self.min_target_z
        return p

    def set_target(self, right=None, left=None, clip_to_workspace=True):
        if right is not None:
            v = np.asarray(right, dtype=np.float32).reshape(3,)
            if clip_to_workspace:
                v = self._clip_to_workspace(v, self.workspace_r_center, self.workspace_r_half)
            self.target_r = v
            print(f"Set right target to {self.target_r}")
        if left is not None:
            v = np.asarray(left, dtype=np.float32).reshape(3,)
            if clip_to_workspace:
                v = self._clip_to_workspace(v, self.workspace_l_center, self.workspace_l_half)
            self.target_l = v
            print(f"Set left target to {self.target_l}")

    def set_target_next_episode(self, right=None, left=None, clip_to_workspace=True):
        if right is not None:
            v = np.asarray(right, dtype=np.float32).reshape(3,)
            if clip_to_workspace:
                v = self._clip_to_workspace(v, self.workspace_r_center, self.workspace_r_half)
            self._pending_target_r = v
        if left is not None:
            v = np.asarray(left, dtype=np.float32).reshape(3,)
            if clip_to_workspace:
                v = self._clip_to_workspace(v, self.workspace_l_center, self.workspace_l_half)
            self._pending_target_l = v

    def sample_target(self, rng=None, workspace_center=None, radius=0.5):
        """
        Sample a candidate target around workspace_center within +/- radius in each axis,
        then clip to the environment's workspace box to ensure it's reachable.
        """
        if rng is None:
            rng = self.np_random or np.random
        if workspace_center is None:
            workspace_center = self.workspace_center
            half = self.workspace_half_extents
        else:
            # pick half-extents per the requested center (right/left) if you like; default uses workspace_half_extents
            half = self.workspace_half_extents
        # sample uniform offset in cube [-radius, radius]^3
        offset = rng.uniform(low=-radius, high=radius, size=3).astype(np.float32)
        candidate = np.asarray(workspace_center, dtype=np.float32) + offset
        # clip to the appropriate workspace box (and enforce Z floor)
        return self._clip_to_workspace(candidate, workspace_center, half)

    # ---------------------------
    # Rendering & closing
    # ---------------------------
    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
            self.viewer.render()
        elif self.render_mode == "rgb_array":
            if self.viewer is None:
                self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data, offscreen=True)
            return self.viewer.read_pixels(depth=False)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None