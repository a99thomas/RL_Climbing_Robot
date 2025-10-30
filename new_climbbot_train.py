# train_climbbot_robust.py
import os
import argparse
import multiprocessing as mp
import traceback

import numpy as np
import gymnasium as gym
import torch as th

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.logger import configure

from envs.climbbot_env import ClimbBotEnv

# Config
XML_PATH = "/Users/aaronthomas/Desktop/Engineering_Projects/Climbing Robot/assets/robot_mjcf.xml"
LOG_DIR = "logs/climbbot"
MODEL_DIR = os.path.join(LOG_DIR, "models")
TENSORBOARD_DIR = os.path.join(LOG_DIR, "tb")
SEED = 42
N_ENVS = 1            # set to 1 by default for macOS stability; raise if you know subprocesses work
TOTAL_TIMESTEPS = 500_000
EVAL_FREQ = 10_000
N_EVAL_EPISODES = 5
USE_VEC_NORMALIZE = True

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_DIR, exist_ok=True)

def make_env(rank=0, seed=SEED, xml_path=XML_PATH, render_mode=None, debug=False):
    """Return an env factory. Do not wrap with Monitor here (VecMonitor will be used)."""
    def _init():
        env = ClimbBotEnv(xml_path=xml_path, render_mode=render_mode, debug=debug)
        try:
            env.reset(seed=seed + rank)
        except TypeError:
            try:
                env.np_random, _ = gym.utils.seeding.np_random(seed + rank)
            except Exception:
                pass
        return env
    return _init

def build_vec_env(n_envs, xml_path, seed, debug=False):
    env_fns = [make_env(rank=i, seed=seed, xml_path=xml_path, debug=debug) for i in range(n_envs)]
    if n_envs > 1:
        try:
            vec = SubprocVecEnv(env_fns)
            return vec, True
        except Exception:
            print("SubprocVecEnv failed to initialize; falling back to DummyVecEnv.")
            traceback.print_exc()
            vec = DummyVecEnv(env_fns)
            return vec, False
    else:
        vec = DummyVecEnv(env_fns)
        return vec, False

def create_eval_vec(xml_path, seed):
    # Single eval env wrapped as DummyVecEnv to match Vec interface
    eval_fn = lambda: ClimbBotEnv(xml_path=xml_path, render_mode=None, debug=False)
    eval_vec = DummyVecEnv([eval_fn])
    try:
        eval_vec.reset(seed=seed + 999)
    except Exception:
        pass
    # Wrap with VecMonitor for consistent interface
    eval_vec = VecMonitor(eval_vec)
    return eval_vec

def safe_close_vec(vec, name="vec_env"):
    if vec is None:
        return
    try:
        vec.close()
    except BrokenPipeError:
        print(f"Warning: BrokenPipeError closing {name} (workers likely dead). Ignoring.")
    except Exception as e:
        print(f"Exception closing {name}: {e}")
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=str, default=XML_PATH)
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--n-envs", type=int, default=N_ENVS)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # TensorBoard logger
    new_logger = configure(TENSORBOARD_DIR, ["stdout", "tensorboard"])

    # Build training vec env (with fallback)
    vec_env = None
    eval_vec = None
    try:
        vec_env, used_subproc = build_vec_env(args.n_envs, args.xml, SEED, debug=args.debug)
        # Monitor + VecNormalize
        vec_env = VecMonitor(vec_env)
        if USE_VEC_NORMALIZE:
            vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

        # Build evaluation vec
        eval_vec = create_eval_vec(args.xml, SEED)
        # If training uses VecNormalize, mirror normalization to eval (copy obs_rms)
        if USE_VEC_NORMALIZE:
            try:
                # Create eval VecNormalize wrapper (do not track rewards)
                eval_vec = VecNormalize(eval_vec, norm_obs=True, norm_reward=False, clip_obs=10.0)
                # Copy running stats from training vec_env (if present)
                if hasattr(vec_env, "obs_rms") and hasattr(eval_vec, "obs_rms"):
                    eval_vec.obs_rms = vec_env.obs_rms
                if hasattr(vec_env, "ret_rms") and hasattr(eval_vec, "ret_rms"):
                    eval_vec.ret_rms = vec_env.ret_rms
            except Exception:
                print("Warning: Failed to copy VecNormalize stats to eval_vec. Proceeding without copy.")
                traceback.print_exc()

        # Ensure best_model_path exists before callbacks
        best_model_path = os.path.join(MODEL_DIR, "best_model")
        os.makedirs(best_model_path, exist_ok=True)

        # Policy kwargs
        policy_kwargs = dict(
            activation_fn=th.nn.Tanh,
            net_arch=dict(pi=[256, 256], vf=[256, 256])
        )

        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            verbose=1,
            seed=SEED,
            n_steps=2048 // max(1, args.n_envs),
            batch_size=64,
            learning_rate=3e-4,
            policy_kwargs=policy_kwargs,
            tensorboard_log=TENSORBOARD_DIR,
        )
        model.set_logger(new_logger)

        # Callbacks
        checkpoint_callback = CheckpointCallback(save_freq=200_000, save_path=MODEL_DIR, name_prefix="checkpoint")
        eval_callback = EvalCallback(
            eval_env=eval_vec,
            best_model_save_path=best_model_path,
            log_path=os.path.join(LOG_DIR, "eval"),
            eval_freq=EVAL_FREQ,
            n_eval_episodes=N_EVAL_EPISODES,
            deterministic=True,
            render=False
        )
        callbacks = [checkpoint_callback, eval_callback]

        # Train
        model.learn(total_timesteps=args.timesteps, callback=callbacks)

        # Save final
        model.save(os.path.join(MODEL_DIR, "final_model"))
        if USE_VEC_NORMALIZE and hasattr(vec_env, "save"):
            try:
                vecnorm_path = os.path.join(MODEL_DIR, "vecnormalize.pkl")
                vec_env.save(vecnorm_path)
                print("Saved VecNormalize stats to", vecnorm_path)
            except Exception:
                pass

    except Exception:
        print("Unhandled exception during training:")
        traceback.print_exc()
    finally:
        # Robust close
        safe_close_vec(vec_env, "training_vec_env")
        safe_close_vec(eval_vec, "eval_vec_env")

if __name__ == "__main__":
    # On macOS it's safer to use spawn; forcing it globally can cause issues if module not importable in child,
    # so only set it if it's not already 'spawn'.
    try:
        current = mp.get_start_method(allow_none=True)
        if current != "spawn":
            mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # start method already set; ignore
        pass

    main()