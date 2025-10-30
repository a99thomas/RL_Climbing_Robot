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


    def __init__(self, xml_path="climbbot3.xml", render_mode=None, debug=False,
                max_episode_steps: int = 1000, ik_apply_mode="actuator_ctrl"):
        """
        Gymnasium environment for the climbing robot (MuJoCo), modified for EE-based control.
        
        Args:
            xml_path: Path to the MuJoCo XML.
            render_mode: "human" or "rgb_array"
            debug: print verbose info
            max_episode_steps: episode cutoff
            ik_apply_mode: "actuator_ctrl" (preferred) or "teleport_qpos" (debug only)
        """

        super().__init__()

        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML model not found: {xml_path}")

        # --- Load MuJoCo model & data ---
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Rendering
        self.viewer = None
        self.render_mode = render_mode
        self.debug = debug

        # --- Safe name lookup helper ---
        def _safe_name2id(model, obj_type, name):
            try:
                return mujoco.mj_name2id(model, obj_type, name)
            except Exception:
                return None

        # Site and body IDs
        self.r_grip_id = _safe_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "r_grip_site")
        self.l_grip_id = _safe_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "l_grip_site")
        self.base_id = _safe_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_site")

        # Build joint-name list
        try:
            self._joint_names = [n.decode() if isinstance(n, bytes) else str(n)
                                for n in list(self.model.jnt_names)]
        except Exception:
            self._joint_names = [f"j{jid}" for jid in range(int(self.model.njnt))]

        # Fixed default targets (world frame)
        self.target_r = np.array([0.7219, -0.1873, 0.0548], dtype=np.float32)
        self.target_l = np.array([0.5221,  0.0956, 0.0549], dtype=np.float32)
        self.min_target_z = 0.1

        # Target management
        self.auto_advance_target = True
        self.success_radius = 0.005
        self._pending_target_r = None
        self._pending_target_l = None

        # Actuator limits (units/sec)
        self.max_actuator_velocity = 0.001 * np.ones(int(self.model.nu), dtype=np.float32)

        # Current actuator setpoint
        try:
            cp = np.array(self.data.ctrl, dtype=np.float32)
            if cp.shape[0] == self.model.nu:
                self._current_setpoint = cp.copy()
            else:
                self._current_setpoint = np.zeros(self.model.nu, dtype=np.float32)
        except Exception:
            self._current_setpoint = np.zeros(self.model.nu, dtype=np.float32)

        # Internal RNG
        self.np_random = None

        # Bookkeeping
        self.reward = 0.0

        # --- Simulation / Control timestep setup ---
        self.sim_timestep = float(self.model.opt.timestep)
        self.control_timestep = 0.002
        self.frame_skip = max(1, int(round(self.control_timestep / self.sim_timestep)))
        self.control_timestep = float(self.frame_skip * self.sim_timestep)

        if debug:
            print(f"[env] sim_timestep={self.sim_timestep:.6f}, "
                f"control_timestep={self.control_timestep:.6f}, "
                f"frame_skip={self.frame_skip}")

        # Floating-base exclusion
        self._excluded_joint_names = ["floating_base"]

        # Precompute joint indices
        self._prepare_joint_qpos_slices()

        # Ensure forward kinematics is valid
        mujoco.mj_forward(self.model, self.data)

        # Episode limits
        self._max_episode_steps = int(max_episode_steps)
        self._elapsed_steps = 0
        self.episode_reward = 0.0

        # --- IK-related additions ---
        self.ik_apply_mode = ik_apply_mode  # "actuator_ctrl" or "teleport_qpos"
        self._build_actuator_joint_map()    # create actuator->joint mapping

        # --- Action/Observation spaces ---
        # New: action = desired end-effector positions (relative to robot base)
        ee_bound = 1.0  # meters (rough workspace limit; tune as needed)
        act_low = -ee_bound * np.ones((6,), dtype=np.float32)
        act_high = ee_bound * np.ones((6,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=act_low, high=act_high, dtype=np.float32)

        # Observation space built from sample obs
        sample_obs = self._get_obs()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=sample_obs.shape, dtype=np.float32
        )

        if debug:
            print("[env] Action space (EE mode):", self.action_space)
            print("[env] Obs shape:", self.observation_space.shape)


    def _build_actuator_joint_map(self):
        """
        Attempt to map each actuator to a qpos index (qpos adr) so we can set ctrl
        via actuator target = desired joint qpos. This is best-effort and uses
        model.actuator_trnid (object ref for each actuator).
        Result stored in self._actuator_to_qpos_index (np.array of length model.nu).
        """
        nu = int(self.model.nu)
        nq = int(self.model.nq)
        self._actuator_to_qpos_index = -1 * np.ones(nu, dtype=int)

        try:
            # actuator_trnid is shape (nu, 2) : (objtype, objid)
            # we'll handle the typical case where the actuator targets a joint.
            for a in range(nu):
                trnid = self.model.actuator_trnid[a]  # two ints: (objtype, objid)
                objtype, objid = int(trnid[0]), int(trnid[1])
                # compare with mjtObj enum for joint
                if objtype == int(mujoco.mjtObj.mjOBJ_JOINT):
                    j = objid
                    if 0 <= j < int(self.model.njnt):
                        qposadr = int(self.model.jnt_qposadr[j])
                        # sanity check bounds
                        if 0 <= qposadr < nq:
                            self._actuator_to_qpos_index[a] = qposadr
        except Exception:
            # if anything fails, leave map as -1
            pass

        # build reverse mapping for convenience: qpos index -> actuator index (first match)
        self._qpos_to_actuator = dict()
        for a, qidx in enumerate(self._actuator_to_qpos_index):
            if qidx >= 0 and qidx not in self._qpos_to_actuator:
                self._qpos_to_actuator[qidx] = a

        if self.debug:
            print("[_build_actuator_joint_map] actuator->qpos:", self._actuator_to_qpos_index)
            print("[_build_actuator_joint_map] qpos->actuator:", self._qpos_to_actuator)


    # ---- Observation changes ----
    def _get_obs(self):
        """
        New observation layout (suggested; you can change order):
          [ base_world_pos (3),
            ee_r_rel_to_base (3),
            ee_l_rel_to_base (3),
            handhold_positions_rel_to_base (N*3)  # N = number of holds we found
            qpos_norm_included (M),
            qvel_included (K),
            target_r (3),
            target_l (3) ]
        The returned vector is flattened, with handholds appended in name order (hold_1..hold_6)
        If the number of handholds is variable, you may wish to fix N to a maximum and pad with zeros.
        """
        # base & end-effectors (world)
        base_world = np.zeros(3, dtype=np.float32)
        if self.base_id is not None:
            try:
                base_world = np.array(self.data.xpos[self.base_id], dtype=np.float32)
            except Exception:
                pass

        ee_r_world = np.zeros(3, dtype=np.float32)
        ee_l_world = np.zeros(3, dtype=np.float32)
        if self.r_grip_id is not None:
            try:
                ee_r_world = np.array(self.data.site_xpos[self.r_grip_id], dtype=np.float32)
            except Exception:
                pass
        if self.l_grip_id is not None:
            try:
                ee_l_world = np.array(self.data.site_xpos[self.l_grip_id], dtype=np.float32)
            except Exception:
                pass

        # convert to base-relative
        ee_r_rel = ee_r_world - base_world
        ee_l_rel = ee_l_world - base_world

        # handhold bodies: collect hold_1..hold_6 if present, in numeric order
        handhold_rel_list = []
        for hid in range(1, 7):
            try:
                bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"hold_{hid}")
            except Exception:
                bid = None
            if bid is None or bid < 0:
                # no such body => append zeros
                handhold_rel_list.append(np.zeros(3, dtype=np.float32))
            else:
                try:
                    pos = np.array(self.data.xpos[bid], dtype=np.float32)
                    handhold_rel_list.append(pos - base_world)
                except Exception:
                    handhold_rel_list.append(np.zeros(3, dtype=np.float32))

        handhold_rel = np.concatenate(handhold_rel_list).astype(np.float32)  # length 18 (6 * 3)

        # qpos/qvel (unchanged logic)
        qpos_full = np.array(self.data.qpos, dtype=np.float32)
        qvel_full = np.array(self.data.qvel, dtype=np.float32)
        try:
            qpos = qpos_full[self._qpos_include_idx]
        except Exception:
            qpos = qpos_full.copy()
        try:
            qvel = qvel_full[self._qvel_include_idx]
        except Exception:
            qvel = qvel_full.copy()

        # Normalize qpos per included-joint slices (same as before)
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

        # pack final observation (order described above)
        obs = np.concatenate([
            base_world.astype(np.float32),
            ee_r_rel.astype(np.float32),
            ee_l_rel.astype(np.float32),
            handhold_rel.astype(np.float32),
            qpos_norm.astype(np.float32),
            qvel.astype(np.float32),
            self.target_r.astype(np.float32),
            self.target_l.astype(np.float32)
        ]).astype(np.float32)

        return obs


    # ---- IK helper (numerical Jacobian + DLS) ----
    def _ik_solve_arm(self, desired_ee_world, arm="r", max_iters=8, tol=1e-3, damping=1e-3):
        """
        Faster analytic-Jacobian IK using MuJoCo's Jacobian for the grip site.
        Returns same dict as the numerical version: {'success', 'qpos_try', 'qpos_indices'}
        """
        # pick patterns and preferred_site like before
        if arm == "l":
            patterns = ["l_", "_l", "left", "left_arm", "arm_l", "shoulder_l", "elbow_l", "wrist_l"]
            site_id = self.l_grip_id
        else:
            patterns = ["r_", "_r", "right", "right_arm", "arm_r", "shoulder_r", "elbow_r", "wrist_r"]
            site_id = self.r_grip_id

        # collect arm joint qpos indices (same as numeric version)
        arm_slices = []
        for s in self._jnt_qpos_slices:
            if s.get("excluded", False):
                continue
            name = str(s.get("name", "")).lower()
            if any(pat in name for pat in patterns):
                arm_slices.append(s)
        if len(arm_slices) == 0:
            arm_slices = [s for s in self._jnt_qpos_slices if not s.get("excluded", False)]

        qpos_indices = []
        for s in arm_slices:
            start = int(s["start"]); length = int(s["length"])
            for i in range(length):
                qpos_indices.append(start + i)
        qpos_indices = np.array(qpos_indices, dtype=int)

        full_qpos = np.array(self.data.qpos, dtype=np.float64).copy()
        q_try = full_qpos.copy()

        # allocate Jacobian storage: MuJoCo typically returns jacp (3 x nv) and jacr (3 x nv)
        nv = int(self.model.nv)
        jacp = np.zeros((3, nv), dtype=np.float64)
        jacr = np.zeros((3, nv), dtype=np.float64)

        def read_ee():
            try:
                return np.array(self.data.site_xpos[site_id], dtype=np.float64)
            except Exception:
                return None

        success = False
        for it in range(max_iters):
            # set q_try
            self.data.qpos[:] = q_try
            if self.data.qvel.shape[0] == self.model.nv:
                self.data.qvel[:] = np.zeros(self.model.nv, dtype=np.float64)
            mujoco.mj_forward(self.model, self.data)

            ee_now = read_ee()
            if ee_now is None:
                break
            err = desired_ee_world.astype(np.float64) - ee_now
            err_norm = float(np.linalg.norm(err))
            if err_norm < tol:
                success = True
                break

            # compute analytic jacobian for site (jacp columns = dpos/dq for all dofs)
            # NOTE: mujoco binding name may be mj_jac or mj_jacSite; adjust if needed.
            try:
                mujoco.mj_jac(self.model, self.data, jacp, jacr, site_id)   # jacp: (3, nv)
            except Exception:
                # fallback to other API name, if present (try mj_jacSite)
                try:
                    mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site_id)
                except Exception:
                    # if no analytic jac available, abort to fallback numeric
                    break

            # extract columns only for the DOFs we will change
            J_full = jacp  # 3 x nv
            J = J_full[:, qpos_indices]  # 3 x n_arm_dofs

            # DLS: delta = J^T (J J^T + lambda^2 I)^{-1} err
            JJt = J.dot(J.T)
            lam2I = (damping**2) * np.eye(3)
            try:
                inv = np.linalg.inv(JJt + lam2I)
                delta_small = J.T.dot(inv.dot(err))
            except np.linalg.LinAlgError:
                delta_small = np.linalg.pinv(J).dot(err)

            # clamp step
            max_step = 0.05
            delta_small = np.clip(delta_small, -max_step, max_step)
            q_try[qpos_indices] = q_try[qpos_indices] + delta_small

        # restore
        try:
            self.data.qpos[:] = full_qpos
            mujoco.mj_forward(self.model, self.data)
        except Exception:
            pass

        return {"success": success, "qpos_try": q_try, "qpos_indices": qpos_indices}



    # ---- Step: accept EE position actions and apply IK ----
    def step(self, action):
        """
        Accepts action as 6D vector: [ee_r_rel (3), ee_l_rel (3)], in robot-base frame.
        Converts to world frame, solves IK -> joint targets, then applies targets.
        If action length equals model.nu (original actuator space) we preserve previous behavior.
        """
        action = np.asarray(action, dtype=np.float32)

        # Detect whether agent is sending ee positions (6) or original actuator controls
        if action.shape[0] == 6:
            # determine base world pos
            base_world = np.zeros(3, dtype=np.float32)
            if self.base_id is not None:
                try:
                    base_world = np.array(self.data.xpos[self.base_id], dtype=np.float32)
                except Exception:
                    pass

            # interpret action as ee targets relative to base
            ee_r_rel = action[0:3].astype(np.float64)
            ee_l_rel = action[3:6].astype(np.float64)
            desired_r_world = (ee_r_rel + base_world).astype(np.float64)
            desired_l_world = (ee_l_rel + base_world).astype(np.float64)

            # Solve IK for each arm
            res_r = self._ik_solve_arm(desired_r_world, arm="r", max_iters=6, tol=1e-3, damping=1e-3)
            res_l = self._ik_solve_arm(desired_l_world, arm="l", max_iters=6, tol=1e-3, damping=1e-3)

            # if either fail, we still try to apply whatever we have
            qpos_target_full = None
            if res_r.get("qpos_try") is not None and res_l.get("qpos_try") is not None:
                # Average results (they are both full-length qpos arrays)
                # We choose to prefer each result's modified indices to avoid overwriting other arm's changes
                qpos_target_full = np.array(self.data.qpos, dtype=np.float64)
                # write right arm qpos
                qr = res_r["qpos_try"]
                qri = res_r["qpos_indices"]
                qpos_target_full[qri] = qr[qri]
                # write left arm qpos
                ql = res_l["qpos_try"]
                qli = res_l["qpos_indices"]
                qpos_target_full[qli] = ql[qli]
            else:
                # If one succeeded but the other didn't, prefer the successful solution
                if res_r.get("qpos_try") is not None:
                    qpos_target_full = res_r["qpos_try"]
                elif res_l.get("qpos_try") is not None:
                    qpos_target_full = res_l["qpos_try"]
                else:
                    qpos_target_full = None

            # apply IK result
            if qpos_target_full is not None:
                if self.ik_apply_mode == "actuator_ctrl":
                    # Map qpos_target_full into actuator ctrl setpoints (self._current_setpoint)
                    # self._current_setpoint is length model.nu
                    try:
                        cp = self._current_setpoint.copy()
                    except Exception:
                        cp = np.zeros(self.model.nu, dtype=np.float32)

                    applied_any = False
                    for qidx, qval in enumerate(qpos_target_full):
                        # try to find actuator that controls this qpos index
                        if qidx in self._qpos_to_actuator:
                            aidx = self._qpos_to_actuator[qidx]
                            try:
                                cp[aidx] = float(qval)  # set actuator ctrl to joint position
                                applied_any = True
                            except Exception:
                                pass
                    if applied_any:
                        # clamp per-actuator velocity limit as before: compute delta and apply gradually across sim steps
                        action_to_pass = cp.astype(np.float32)
                        # reuse original code logic to ramp from self._current_setpoint -> action_to_pass
                        # compute desired_delta and clamp per control step
                        max_delta_per_control = self.max_actuator_velocity * float(self.control_timestep)
                        desired_delta = action_to_pass - self._current_setpoint
                        clamped_delta_control = np.clip(desired_delta, -max_delta_per_control, max_delta_per_control)
                        target_setpoint = self._current_setpoint + clamped_delta_control
                        per_sim_max_delta = self.max_actuator_velocity * float(self.sim_timestep)
                        for _ in range(self.frame_skip):
                            remaining = target_setpoint - self._current_setpoint
                            step_delta = np.clip(remaining, -per_sim_max_delta, per_sim_max_delta)
                            self._current_setpoint = self._current_setpoint + step_delta
                            self.data.ctrl[:] = self._current_setpoint
                            mujoco.mj_step(self.model, self.data)
                        # record last_action for reward penalty
                        self._last_action = action_to_pass.copy()
                    else:
                        # No actuator mapping available: fall back to teleport (warn)
                        if self.debug:
                            print("[step] actuator mapping failed; no qpos->actuator mapping; falling back to teleport")
                        self.data.qpos[:] = qpos_target_full
                        if self.data.qvel.shape[0] == self.model.nv:
                            self.data.qvel[:] = np.zeros(self.model.nv, dtype=np.float64)
                        mujoco.mj_forward(self.model, self.data)
                        self._last_action = np.zeros_like(self._last_action) if hasattr(self, "_last_action") else np.zeros(self.model.nu)
                else:
                    # teleport qpos mode (debugging)
                    self.data.qpos[:] = qpos_target_full
                    if self.data.qvel.shape[0] == self.model.nv:
                        self.data.qvel[:] = np.zeros(self.model.nv, dtype=np.float64)
                    mujoco.mj_forward(self.model, self.data)
                    self._last_action = np.zeros_like(self._last_action) if hasattr(self, "_last_action") else np.zeros(self.model.nu)
            else:
                # IK failed completely: do nothing (but step simulation with zero change)
                if self.debug:
                    print("[step] IK failed for both arms; skipping actuator update")

            # After IK-based stepping we continue to compute observation/reward etc
        else:
            action = np.asarray(action, dtype=np.float32)
            action = np.clip(action, self.action_space.low, self.action_space.high)
            if action.shape[0] != self.model.nu:
                raise ValueError(f"Action length {action.shape[0]} does not match model.nu {self.model.nu}")

            max_delta_per_control = self.max_actuator_velocity * float(self.control_timestep)
            desired_delta = action - self._current_setpoint
            clamped_delta_control = np.clip(desired_delta, -max_delta_per_control, max_delta_per_control)
            target_setpoint = self._current_setpoint + clamped_delta_control
            per_sim_max_delta = self.max_actuator_velocity * float(self.sim_timestep)
            for _ in range(self.frame_skip):
                remaining = target_setpoint - self._current_setpoint
                step_delta = np.clip(remaining, -per_sim_max_delta, per_sim_max_delta)
                self._current_setpoint = self._current_setpoint + step_delta
                self.data.ctrl[:] = self._current_setpoint
                mujoco.mj_step(self.model, self.data)
            self._last_action = action.copy()

        # --- rest of your original step code to compute obs, reward, termination, return ---
        obs = self._get_obs()
        self.reward = self._compute_reward()
        self._elapsed_steps += 1
        self.episode_reward += float(self.reward)
        terminated, truncated = self._check_termination()
        if self._elapsed_steps >= self._max_episode_steps:
            truncated = True
        if self.render_mode == "human":
            self.render()
        info = {"elapsed_steps": int(self._elapsed_steps), "episode_reward": float(self.episode_reward)}
        return obs, float(self.reward), bool(terminated), bool(truncated), info


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
        self.target_r = self.sample_target(rng=self.np_random, arm="r", max_attempts=100)
        self.target_l = self.sample_target(rng=self.np_random, arm="l", max_attempts=100)

        # reset episode bookkeeping
        self.reward = 0.0
        self._elapsed_steps = 0
        self.episode_reward = 0.0
        self._last_action = np.zeros(self.model.nu, dtype=np.float32)

        obs = self._get_obs()
        info = {}
        return obs, info

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
                new_r = self.sample_target(rng=self.np_random, arm="r", max_attempts=100)
                self.target_r = new_r.astype(np.float32)
            if reached_l:
                new_l = self.sample_target(rng=self.np_random, arm="l", max_attempts=100)
                self.target_l = new_l.astype(np.float32)

        # success bonuses
        if reached_r:
            reward += 1.5
            print(ee_r)
            print("Reached right target! New target is", self.target_r)
        if reached_l:
            reward += 1.5
            print(ee_l)
            print("Reached left target! New target is", self.target_l)

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
    
    def sample_target(self, rng=None, arm="r", max_attempts=50, perturb_amount=0.0):
        """
        Sample a reachable goal without permanently modifying the simulator state.

        - Saves/restores qpos/qvel while probing candidate joint configurations.
        - Optionally perturbs the sampled EE position by up to `perturb_amount` (meters)
          to avoid targets that are exactly the sampled FK pose.
        """
        # RNG
        if rng is None:
            rng = getattr(self, "np_random", None)
        if rng is None:
            rng = np.random

        preferred_site = self.r_grip_id if arm != "l" else self.l_grip_id

        if arm == "l":
            patterns = ["l_", "_l", "left", "left_arm", "arm_l", "shoulder_l", "elbow_l", "wrist_l"]
        else:
            patterns = ["r_", "_r", "right", "right_arm", "arm_r", "shoulder_r", "elbow_r", "wrist_r"]

        # pick joint slices that look like they belong to the arm
        arm_slices = []
        for s in self._jnt_qpos_slices:
            if s.get("excluded", False):
                continue
            name = str(s.get("name", "")).lower()
            if any(pat in name for pat in patterns):
                arm_slices.append(s)

        if len(arm_slices) == 0:
            arm_slices = [s for s in self._jnt_qpos_slices if not s.get("excluded", False)]

        # store current qpos/qvel so we can restore them after probing
        try:
            backup_qpos = np.array(self.data.qpos, dtype=np.float64).copy()
            backup_qvel = np.array(self.data.qvel, dtype=np.float64).copy()
        except Exception:
            backup_qpos = None
            backup_qvel = None

        qpos_template = np.array(self.data.qpos, dtype=np.float32)

        sampled_world = None
        for attempt in range(max_attempts):
            qpos_try = qpos_template.copy()
            for s in arm_slices:
                start = int(s["start"])
                length = int(s["length"])
                low = float(s.get("low", -np.pi))
                high = float(s.get("high", np.pi))
                if np.isclose(low, high):
                    low, high = -np.pi, np.pi
                try:
                    sample_vals = rng.uniform(low, high, size=(length,))
                except Exception:
                    sample_vals = np.random.uniform(low, high, size=(length,))
                qpos_try[start : start + length] = sample_vals

            # write probe qpos/qvel, forward kinematics
            try:
                self.data.qpos[:] = qpos_try
                if self.data.qvel.shape[0] == self.model.nv:
                    self.data.qvel[:] = np.zeros(self.model.nv, dtype=np.float64)
                mujoco.mj_forward(self.model, self.data)
            except Exception:
                # restore and continue
                if backup_qpos is not None:
                    self.data.qpos[:] = backup_qpos
                if backup_qvel is not None and self.data.qvel.shape[0] == backup_qvel.shape[0]:
                    self.data.qvel[:] = backup_qvel
                try:
                    mujoco.mj_forward(self.model, self.data)
                except Exception:
                    pass
                continue

            # read EE world pos
            ee_world = None
            if preferred_site is not None:
                try:
                    ee_world = np.array(self.data.site_xpos[preferred_site], dtype=np.float32)
                except Exception:
                    ee_world = None

            if ee_world is None:
                other_site = self.l_grip_id if preferred_site == self.r_grip_id else self.r_grip_id
                if other_site is not None:
                    try:
                        ee_world = np.array(self.data.site_xpos[other_site], dtype=np.float32)
                    except Exception:
                        ee_world = None

            if ee_world is None:
                # fallback to base pos
                if self.base_id is not None:
                    try:
                        ee_world = np.array(self.data.xpos[self.base_id], dtype=np.float32)
                    except Exception:
                        ee_world = np.zeros(3, dtype=np.float32)
                else:
                    ee_world = np.zeros(3, dtype=np.float32)

            # optionally perturb the sampled EE so it's not exactly the FK pose
            if perturb_amount and perturb_amount > 0.0:
                try:
                    delta = rng.uniform(-perturb_amount, perturb_amount, size=(3,))
                except Exception:
                    delta = np.random.uniform(-perturb_amount, perturb_amount, size=(3,))
                ee_world = ee_world + delta

            # enforce min Z
            if ee_world[2] < self.min_target_z:
                if self.debug:
                    print(f"[sample_target] attempt {attempt+1}/{max_attempts} rejected: z={ee_world[2]:.4f} < min_target_z={self.min_target_z}")
                # restore and continue
                if backup_qpos is not None:
                    self.data.qpos[:] = backup_qpos
                if backup_qvel is not None and self.data.qvel.shape[0] == backup_qvel.shape[0]:
                    self.data.qvel[:] = backup_qvel
                try:
                    mujoco.mj_forward(self.model, self.data)
                except Exception:
                    pass
                continue

            sampled_world = ee_world
            # restore the original sim state before breaking out
            if backup_qpos is not None:
                self.data.qpos[:] = backup_qpos
            if backup_qvel is not None and self.data.qvel.shape[0] == backup_qvel.shape[0]:
                self.data.qvel[:] = backup_qvel
            try:
                mujoco.mj_forward(self.model, self.data)
            except Exception:
                pass
            break

        # final fallback: if none found, use base pos and ensure min z
        if sampled_world is None:
            if self.base_id is not None:
                try:
                    sampled_world = np.array(self.data.xpos[self.base_id], dtype=np.float32)
                except Exception:
                    sampled_world = np.zeros(3, dtype=np.float32)
            else:
                sampled_world = np.zeros(3, dtype=np.float32)
            if sampled_world[2] < self.min_target_z:
                sampled_world[2] = self.min_target_z

        # set environment target
        if arm == "l":
            self.target_l = np.asarray(sampled_world, dtype=np.float32)
            if self.debug:
                print(f"Sampled new l target: {self.target_l}")
        else:
            self.target_r = np.asarray(sampled_world, dtype=np.float32)
            if self.debug:
                print(f"Sampled new r target: {self.target_r}")

        return np.asarray(sampled_world, dtype=np.float32)


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