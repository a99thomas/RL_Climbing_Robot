#!/usr/bin/env python3
"""
varied_goal_train.py

PPO training for ClimbBotEnv with varied goals and resume support.

Usage:
  # fresh training
  python varied_goal_train.py --timesteps 2000000

  # generate workspace samples (once) and then train:
  python varied_goal_train.py --gen-samples --max-samples 20000

  # resume training from checkpoint (and provide matching vecnormalize file)
  python varied_goal_train.py --resume models/ppo_best.zip --vecnorm models/vecnormalize.pkl
"""

import os
import argparse
import multiprocessing as mp
import traceback
import time

import numpy as np
import gymnasium as gym
import torch as th

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.logger import configure

from envs.varied_poses_env import ClimbBotEnv

# ---------- Config ----------
XML_PATH = "/Users/aaronthomas/Desktop/Engineering_Projects/Climbing Robot/assets/robot_mjcf.xml"
LOG_DIR = "logs/climbbot_ppo"
MODEL_DIR = os.path.join(LOG_DIR, "models")
TB_DIR = os.path.join(LOG_DIR, "tb")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TB_DIR, exist_ok=True)

SEED = 42
N_ENVS = 1                     # set 1 on macOS for stability
TOTAL_TIMESTEPS = 20_000_000
EVAL_FREQ = 50_000
N_EVAL_EPISODES = 6
USE_VEC_NORMALIZE = True

# PPO policy architecture
policy_kwargs = dict(
    activation_fn=th.nn.Tanh,
    net_arch=dict(pi=[256, 256], vf=[256, 256])
)
# ----------------------------

def make_env(rank=0, seed=SEED, xml_path=XML_PATH, debug=False):
    def _init():
        env = ClimbBotEnv(xml_path=xml_path, render_mode=None, debug=debug)
        # ensure per-episode varied goals (env helper)
        env.set_sample_goal_on_reset(True)
        # optionally enable curriculum: env.enable_curriculum(True, curriculum_steps=10000, target_radius=0.12)
        env.enable_curriculum(False)
        # seed the env deterministically
        try:
            env.reset(seed=seed + rank)
        except TypeError:
            env.np_random, _ = gym.utils.seeding.np_random(seed + rank)
        return env
    return _init

def build_vec_env(n_envs, xml_path, seed, debug=False):
    env_fns = [make_env(rank=i, seed=seed, xml_path=xml_path, debug=debug) for i in range(n_envs)]
    if n_envs > 1:
        try:
            vec = SubprocVecEnv(env_fns)
            used_subproc = True
        except Exception:
            traceback.print_exc()
            vec = DummyVecEnv(env_fns)
            used_subproc = False
    else:
        vec = DummyVecEnv(env_fns)
        used_subproc = False
    return vec, used_subproc

def create_eval_vec(xml_path, seed):
    eval_fn = lambda: ClimbBotEnv(xml_path=xml_path, render_mode=None, debug=False)
    eval_vec = DummyVecEnv([eval_fn])
    try:
        eval_vec.reset(seed=seed + 999)
    except Exception:
        pass
    return eval_vec

def safe_close(vec, name="vec"):
    if vec is None:
        return
    try:
        vec.close()
    except BrokenPipeError:
        print(f"Warning: BrokenPipeError closing {name} (worker may have died). Ignoring.")
    except Exception:
        traceback.print_exc()

def _count_samples_dict(samples_dict):
    """
    Accepts the dict returned by generate_workspace_samples and returns (r_n, l_n)
    using either legacy keys ('r','l') or new keys ('r_world','l_world').
    """
    if samples_dict is None:
        return 0, 0
    r = samples_dict.get("r_world", None) or samples_dict.get("r", None)
    l = samples_dict.get("l_world", None) or samples_dict.get("l", None)
    r_n = int(r.shape[0]) if (r is not None and getattr(r, "shape", (0,))[0] > 0) else 0
    l_n = int(l.shape[0]) if (l is not None and getattr(l, "shape", (0,))[0] > 0) else 0
    return r_n, l_n

def generate_samples_once(xml_path, max_samples=15000, angle_samples=20, linear_samples=20,
                          deterministic=True, save_cache=True, save_qvecs=False, debug=False):
    """
    Create a temporary env to generate and save workspace samples using env.generate_workspace_samples.
    This ensures the cache file exists for all workers to load.
    """
    print(f"[samples] Generating workspace samples: max_samples={max_samples}, angle={angle_samples}, linear={linear_samples}, deterministic={deterministic}, save_cache={save_cache}, save_qvecs={save_qvecs}")
    env = None
    t0 = time.time()
    try:
        env = ClimbBotEnv(xml_path=xml_path, render_mode=None, debug=debug)
        samples = env.generate_workspace_samples(
            max_samples=max_samples,
            angle_samples=angle_samples,
            linear_samples=linear_samples,
            deterministic=deterministic,
            save_cache=save_cache,
            save_qvecs=save_qvecs
        )
        r_n, l_n = _count_samples_dict(samples)
        print(f"[samples] Generation complete: right={r_n} samples, left={l_n} samples. time={time.time()-t0:.1f}s")
        # check cache file exists and is readable
        try:
            if save_cache and os.path.exists(env.workspace_samples_cache):
                print(f"[samples] Cache saved to: {env.workspace_samples_cache}")
            elif save_cache:
                print("[samples] Warning: save_cache=True but cache file not found after generation.")
        except Exception:
            pass
        return True
    except Exception:
        print("[samples] Generation failed with exception:")
        traceback.print_exc()
        return False
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

def main(args):
    # logger
    new_logger = configure(TB_DIR, ["stdout", "tensorboard"])

    # Optionally pre-generate workspace samples before creating vectorized envs
    if args.gen_samples:
        ok = generate_samples_once(
            xml_path=args.xml,
            max_samples=args.max_samples,
            angle_samples=args.angle_samples,
            linear_samples=args.linear_samples,
            deterministic=args.deterministic_samples,
            save_cache=args.save_samples,
            save_qvecs=args.save_qvecs,
            debug=args.debug
        )
        if not ok:
            print("[main] Warning: sample generation failed; continuing anyway.")
        else:
            print("[main] Workspace samples generated (or loaded) successfully.")

    # build training & eval vecs
    vec_env, used_subproc = build_vec_env(args.n_envs, args.xml, SEED, debug=args.debug)
    vec_env = VecMonitor(vec_env)
    # vec normalization: either load from provided file (when resuming) or create new
    if USE_VEC_NORMALIZE:
        if args.vecnorm and os.path.exists(args.vecnorm):
            print("Loading VecNormalize from:", args.vecnorm)
            vec_env = VecNormalize.load(args.vecnorm, vec_env)
        else:
            vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_vec = create_eval_vec(args.xml, SEED)
    eval_vec = VecMonitor(eval_vec)
    if USE_VEC_NORMALIZE:
        # wrap eval to use same obs normalization but not update it
        if args.vecnorm and os.path.exists(args.vecnorm):
            eval_vec = VecNormalize.load(args.vecnorm, eval_vec)
            eval_vec.training = False
            eval_vec.norm_reward = False
        else:
            # create fresh eval VecNormalize and copy stats after model created
            eval_vec = VecNormalize(eval_vec, norm_obs=True, norm_reward=False, clip_obs=10.0)
            eval_vec.training = False

    # Create or load model
    model = None
    if args.resume and os.path.exists(args.resume):
        print("Resuming PPO from:", args.resume)
        # load and attach new env for continued training
        model = PPO.load(args.resume, env=vec_env, device="auto")
        model.set_env(vec_env)
    else:
        # Create new PPO model
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            verbose=1,
            seed=SEED,
            n_steps=2048 // max(1, args.n_envs),
            batch_size=64,
            learning_rate=3e-4,
            clip_range=0.2,
            ent_coef=0.0,
            policy_kwargs=policy_kwargs,
            tensorboard_log=TB_DIR,
        )
    model.set_logger(new_logger)

    # If eval_vec exists and vecnormalize was created after model creation, copy obs_rms to eval
    if USE_VEC_NORMALIZE and hasattr(vec_env, "obs_rms") and hasattr(eval_vec, "obs_rms"):
        try:
            eval_vec.obs_rms = vec_env.obs_rms
            if hasattr(vec_env, "ret_rms") and hasattr(eval_vec, "ret_rms"):
                eval_vec.ret_rms = vec_env.ret_rms
        except Exception:
            pass

    # callbacks
    best_model_path = os.path.join(MODEL_DIR, "ppo_best")
    os.makedirs(best_model_path, exist_ok=True)
    checkpoint_cb = CheckpointCallback(save_freq=100_000, save_path=MODEL_DIR, name_prefix="ppo_ckpt")
    eval_cb = EvalCallback(
        eval_env=eval_vec,
        best_model_save_path=best_model_path,
        log_path=os.path.join(LOG_DIR, "eval"),
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False
    )

    # train
    try:
        model.learn(total_timesteps=args.timesteps, callback=[checkpoint_cb, eval_cb])
    except KeyboardInterrupt:
        print("Interrupted by user; saving model.")
    finally:
        # save model + vecnorm
        model.save(os.path.join(MODEL_DIR, "ppo_final"))
        print("Saved final model to", os.path.join(MODEL_DIR, "ppo_final"))
        if USE_VEC_NORMALIZE and hasattr(vec_env, "save"):
            try:
                vecnorm_path = args.vecnorm if args.vecnorm else os.path.join(MODEL_DIR, "vecnormalize.pkl")
                vec_env.save(vecnorm_path)
                print("Saved VecNormalize to", vecnorm_path)
            except Exception:
                traceback.print_exc()

        safe_close(vec_env, "train_vec")
        safe_close(eval_vec, "eval_vec")

if __name__ == "__main__":
    # spawn is safer on macOS
    try:
        current = mp.get_start_method(allow_none=True)
        if current != "spawn":
            mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=str, default=XML_PATH)
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--n-envs", type=int, default=N_ENVS)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--vecnorm", type=str, default="")
    parser.add_argument("--debug", action="store_true")

    # New sampling options
    parser.add_argument("--gen-samples", action="store_true", help="Generate workspace samples before training (use env.generate_workspace_samples).")
    parser.add_argument("--max-samples", type=int, default=15000, help="Max number of q configurations to sample when generating workspace samples.")
    parser.add_argument("--angle-samples", type=int, default=20, help="Discretization count for angular joints when generating samples.")
    parser.add_argument("--linear-samples", type=int, default=20, help="Discretization count for linear/large-range joints when generating samples.")
    parser.add_argument("--deterministic-samples", action="store_true", help="Use deterministic RNG for sampling (reproducible).")
    parser.add_argument("--save-samples", dest="save_samples", action="store_true", help="Save generated samples to cache (workspace_samples.npz).")
    parser.add_argument("--no-save-samples", dest="save_samples", action="store_false", help="Do not save generated samples to cache.")
    parser.set_defaults(save_samples=True)
    parser.add_argument("--save-qvecs", action="store_true", help="Also save qpos vectors that produced each sample (useful for debugging).")

    args = parser.parse_args()

    main(args)