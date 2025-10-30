#!/usr/bin/env python3
"""
Visualize a trained PPO ClimbBot model in the MuJoCo simulator.
"""
import os
import time
import numpy as np
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ------------------ CONFIG ------------------
XML_PATH = "/Users/aaronthomas/Desktop/Engineering_Projects/Climbing Robot/assets/robot_mjcf.xml"
SAVE_DIR = "./models"
MODEL_NAME = "best_model.zip"
VECNORM_NAME = "vecnormalize.pkl"
N_EPISODES = 5
RENDER_DELAY = 0.001  # seconds between steps (for visualization)
# --------------------------------------------

from envs.smart_env import ClimbBotEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def make_env(xml_path, seed=None, render_mode="human"):
    """Helper to make an environment for evaluation."""
    def _init():
        env = ClimbBotEnv(xml_path=xml_path, render_mode=render_mode, debug=False)
        env.reset(seed=seed)
        return env
    return _init


def main():
    model_path = os.path.join(SAVE_DIR, MODEL_NAME)
    vecnorm_path = os.path.join(SAVE_DIR, VECNORM_NAME)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    # Create a fresh env with rendering enabled (vectorized -> shape (1, obs_dim))
    eval_env = DummyVecEnv([make_env(XML_PATH, seed=0, render_mode="human")])

    # Try loading VecNormalize stats (if they exist)
    if os.path.exists(vecnorm_path):
        print(f"Loading VecNormalize stats from {vecnorm_path}")
        eval_env = VecNormalize.load(vecnorm_path, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False
    else:
        print("⚠️ VecNormalize stats not found — running without normalization.")

    # Load trained model. Pass env to allow some wrappers to be attached (optional)
    print(f"Loading model from {model_path}")
    model = PPO.load(model_path, env=eval_env)

    # --- Evaluation loop that correctly handles vectorized outputs ---
    for ep in range(N_EPISODES):
        # reset returns an array of observations for vectorized envs (shape (n_envs, ...))
        obs = eval_env.reset()  # typically shaped (1, obs_dim)
        total_reward = 0.0
        step = 0

        done = False
        while True:
            # model.predict accepts batched observations for vec envs
            action, _ = model.predict(obs, deterministic=True)

            # step returns batched arrays (obs, rewards, terminateds, truncateds, infos)
            step_out = eval_env.step(action)
            # For Gymnasium-compatible vec envs this is a 5-tuple
            if len(step_out) == 5:
                obs, rewards, terminateds, truncateds, infos = step_out
            else:
                # fallback (older 4-tuple): obs, rewards, dones, infos
                obs, rewards, dones, infos = step_out
                # emulate terminateds/truncateds
                terminateds = dones
                truncateds = np.zeros_like(np.asarray(dones), dtype=bool)

            # rewards/terminateds/truncateds are arrays with length == n_envs (here n_envs==1)
            # Extract scalar values for the single-env case:
            r = float(np.asarray(rewards).ravel()[0])
            term = bool(np.asarray(terminateds).ravel()[0])
            trunc = bool(np.asarray(truncateds).ravel()[0])

            total_reward += r
            step += 1

            # Render: your environment already renders inside step() when render_mode=="human".
            # If it doesn't, you could call eval_env.render() here (but DummyVecEnv.render proxies to the inner env).
            # small sleep to let the viewer update (increase if rendering stutters)
            time.sleep(RENDER_DELAY)

            if term or trunc:
                break

        print(f"Episode {ep+1}: total reward = {total_reward:.3f}, steps = {step}")

    eval_env.close()
    print("✅ Evaluation complete.")



if __name__ == "__main__":
    main()
