import numpy as np
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envs.varied_poses_env import ClimbBotEnv

# --- Path to your model ---
MODEL_PATH = "/Users/aaronthomas/Desktop/Engineering_Projects/Climbing Robot/assets/robot_mjcf.xml"  # <-- Adjust if needed

# --- Create environment ---
env = ClimbBotEnv(xml_path=MODEL_PATH, render_mode="human", debug=True)

# --- Reset environment ---
obs, info = env.reset()
print("Initial Observation shape:", obs.shape)
print("Initial Info:", info)

# --- Run a few steps with random actions ---
for step in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {step}: reward={reward:.3f}, terminated={terminated}, info={info}")

    if terminated:
        print("Terminated due to base fall.")
        break

env.close()
