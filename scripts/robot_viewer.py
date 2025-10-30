import mujoco
import mujoco.viewer
import time

# --- Configuration ---
XML_FILE_PATH = "/Users/aaronthomas/Desktop/Engineering_Projects/Climbing Robot/assets/robot_mjcf.xml"
simulation_speed = 1.0  # 1.0 = real time
# ---------------------

try:
    model = mujoco.MjModel.from_xml_path(XML_FILE_PATH)
    data = mujoco.MjData(model)
except Exception as e:
    print(f"Error loading XML file: {e}")
    exit(1)

with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = time.time()
    last_sim_time = 0.0
    r_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "r_grip_site")
    l_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "l_grip_site")

    while viewer.is_running():
        # --- Detect reset and resync clocks ---
        if data.time < last_sim_time:
            # Simulation was reset — shift the start_time
            start_time = time.time() - (data.time / simulation_speed)

        last_sim_time = data.time
        # ---------------------------------------

        # Real time vs simulation time sync
        elapsed_real_time = time.time() - start_time
        target_real_time = data.time / simulation_speed

        time_to_wait = target_real_time - elapsed_real_time
        if time_to_wait > 0:
            time.sleep(time_to_wait)

        # Advance physics
        mujoco.mj_step(model, data)

        r_grip_pos = data.site_xpos[r_site_id]
        l_grip_pos = data.site_xpos[l_site_id]
        print(f"r_grip_site position: {r_grip_pos}", f"l_grip_site position: {l_grip_pos}")
        # print("Actuator forces:", data.actuator_force[2])
        # body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "assembly_12_collision_1_2")
        # print("External force on assembly_12 gripper:", data.cfrc_ext[body_id])


        # print(data.qpos)

        # Render
        viewer.sync()
