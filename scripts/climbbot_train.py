# train_climbbot.py  (corrected)
import os
import multiprocessing as mp
import time
from typing import Callable
import numpy as np

# Stable Baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy

# Your environment
from envs.climbbot_env import ClimbBotEnv

# ---------- Config ----------
LOGDIR = "logs/climbbot_ppo"
os.makedirs(LOGDIR, exist_ok=True)

NUM_CPU = 5             # <-- start with 1 on macOS; set >1 only after confirming things work
TOTAL_TIMESTEPS = 500_000
EVAL_FREQ = 50_000
N_EVAL_EPISODES = 1
SEED = 42

POLICY_KWARGS = dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])])

# ---------- Helper to create envs ----------
def make_env_fn(seed: int = 0, render_mode: str = None) -> Callable:
    def _init():
        env = ClimbBotEnv(xml_path="/Users/aaronthomas/Desktop/Engineering_Projects/Climbing Robot/climbbot/robot_mjcf.xml", render_mode=render_mode, debug=False)
        # gymnasium envs can be seeded via reset, but also provide seed() to keep compatibility
        try:
            env.reset(seed=seed)
        except Exception:
            pass
        return env
    return _init

# ---------- Render callback (unchanged) ----------
class RenderEvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq: int, n_eval_episodes: int = 1, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.last_eval_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_eval_step >= self.eval_freq:
            self.last_eval_step = self.num_timesteps
            if self.verbose:
                print(f"\n[RenderEvalCallback] Running evaluation at step {self.num_timesteps} ...")
            mean_reward, std_reward = evaluate_policy(self.model, self.eval_env, n_eval_episodes=1, render=False, deterministic=True)
            if self.verbose:
                print(f"[RenderEvalCallback] Eval (no render) mean_reward={mean_reward:.3f} +- {std_reward:.3f}")

            for ep in range(self.n_eval_episodes):
                obs, info = self.eval_env.reset()
                done = False
                total_r = 0.0
                steps = 0
                while True:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    total_r += float(reward)
                    steps += 1
                    try:
                        self.eval_env.render()
                    except Exception:
                        print("[RenderEvalCallback] Warning: render() failed during eval render.")
                    if terminated or truncated:
                        if self.verbose:
                            print(f"[RenderEvalCallback] Eval episode {ep+1} finished: reward={total_r:.3f}, steps={steps}")
                        break
        return True

    def _on_training_end(self) -> None:
        try:
            self.eval_env.close()
        except Exception:
            pass

# ---------- Main training ----------
def main():
    # macOS note: ensure "spawn" start method to avoid fork-related issues
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # start method already set; ignore
        pass

    # Create vectorized envs
    seeds = [SEED + i for i in range(NUM_CPU)]
    if NUM_CPU > 1:
        env_fns = [make_env_fn(seed=s, render_mode=None) for s in seeds]
        train_env = SubprocVecEnv(env_fns)
    else:
        env_fns = [make_env_fn(seed=SEED, render_mode=None)]
        train_env = DummyVecEnv(env_fns)

    train_env = VecMonitor(train_env)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # Evaluation env (must be non-vectorized and renderable)
    eval_env = ClimbBotEnv(xml_path="/Users/aaronthomas/Desktop/Engineering_Projects/Climbing Robot/climbbot/robot_mjcf.xml", render_mode=None, debug=False)

    # Logger
    tmp_path = os.path.join(LOGDIR, "sb3_logs")
    new_logger = configure(tmp_path, ["stdout", "tensorboard"])

    model = PPO(
        "MlpPolicy",
        train_env,
        policy_kwargs=POLICY_KWARGS,
        verbose=1,
        tensorboard_log=LOGDIR,
        seed=SEED,
        ent_coef=0.0,
        learning_rate=3e-4,
        n_steps=2048 // max(1, NUM_CPU),
        batch_size=64,
        n_epochs=10,
        gae_lambda=0.95,
        device="cpu",
    )
    model.set_logger(new_logger)

    render_cb = RenderEvalCallback(eval_env=eval_env, eval_freq=EVAL_FREQ, n_eval_episodes=N_EVAL_EPISODES, verbose=1)

    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=render_cb)
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        model.save(os.path.join(LOGDIR, "ppo_climbbot_final"))
        train_env.close()
        eval_env.close()

if __name__ == "__main__":
    main()
