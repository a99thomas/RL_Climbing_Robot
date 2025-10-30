#!/usr/bin/env python3
"""
Train PPO on ClimbBotEnv (no command-line arguments needed).
Just edit the config section below.
"""
import os
import numpy as np

import torch as th

import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ------------------ CONFIG ------------------
XML_PATH = "/Users/aaronthomas/Desktop/Engineering_Projects/Climbing Robot/assets/robot_mjcf.xml"
SAVE_DIR = "./models"
TOTAL_TIMESTEPS = 1_000_000
N_ENVS = 1
SEED = 0
EVAL_FREQ = 50_000
# --------------------------------------------

from envs.smart_env import ClimbBotEnv

# Stable-Baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback


def make_env(xml_path, seed, render_mode=None):
    """Factory for environment instances (for vectorized training)."""
    def _init():
        env = ClimbBotEnv(xml_path=xml_path, render_mode=render_mode, debug=False)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Choose environment type (parallel or single)
    if N_ENVS > 1:
        env_fns = [make_env(XML_PATH, SEED + i) for i in range(N_ENVS)]
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv([make_env(XML_PATH, SEED)])

    # Normalize observations and rewards
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Evaluation environment
    eval_env = DummyVecEnv([make_env(XML_PATH, SEED + 1000)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=SAVE_DIR,
        log_path=SAVE_DIR,
        eval_freq=EVAL_FREQ // max(1, N_ENVS),
        deterministic=True,
        render=False,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=100_000 // max(1, N_ENVS),
        save_path=SAVE_DIR,
        name_prefix="checkpoint"
    )

    # PPO policy network configuration
    policy_kwargs = dict(
    net_arch=dict(pi=[256, 256], vf=[256, 256]),  # actor / critic separate networks
    activation_fn=th.nn.ReLU
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        seed=SEED,
        n_steps=2048 // max(1, N_ENVS),
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        policy_kwargs=policy_kwargs,
        tensorboard_log=os.path.join(SAVE_DIR, "tb"),
    )

    # ---- Train ----
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[eval_callback, checkpoint_cb])

    # ---- Save ----
    model_path = os.path.join(SAVE_DIR, "ppo_climbbot_final.zip")
    model.save(model_path)
    print(f"\n✅ Saved model to {model_path}")

    vecnorm_path = os.path.join(SAVE_DIR, "vecnormalize.pkl")
    vec_env.save(vecnorm_path)
    print(f"✅ Saved VecNormalize stats to {vecnorm_path}")

    # Cleanup
    vec_env.close()
    eval_env.close()
    print("🎯 Training complete.")


if __name__ == "__main__":
    main()
