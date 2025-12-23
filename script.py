import os, random, math
import gym
import d4rl  # noqa: F401
import d4rl.gym_mujoco  # noqa: F401
import numpy as np
import torch
from omegaconf import OmegaConf

from env_utils import ObservationWrapperGym, denormalize_obs
from utils import load_base_policy

# Register the same resolvers used in train_dsrl.py
OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil)
OmegaConf.register_new_resolver("round_down", math.floor)
# Register a dummy 'now' resolver to avoid the error when loading the config
from datetime import datetime
OmegaConf.register_new_resolver("now", lambda fmt: datetime.now().strftime(fmt), replace=True)

cfg = OmegaConf.load("cfg/gym/dsrl_walker.yaml")
OmegaConf.resolve(cfg)

random.seed(cfg.seed)
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)

env = gym.make(cfg.env_name)
env = ObservationWrapperGym(env, cfg.normalization_path)
obs_min = env.obs_min
obs_max = env.obs_max

base_policy = load_base_policy(cfg)

speed_index = 8
episodes = 10
max_steps = int(cfg.env.max_episode_steps)
act_steps = int(cfg.act_steps)
action_dim = int(cfg.action_dim)

def _reset_env(env):
    out = env.reset()
    return out[0] if isinstance(out, tuple) else out

def _step_env(env, action):
    out = env.step(action)
    if len(out) == 4:
        return out
    obs, reward, done, truncated, info = out
    return obs, reward, done or truncated, info

speeds = []
for _ in range(episodes):
    obs = _reset_env(env)
    done = False
    steps = 0
    while not done and steps < max_steps:
        noise = torch.randn(1, act_steps, action_dim, device=cfg.device)
        obs_tensor = torch.tensor(obs, device=cfg.device, dtype=torch.float32).unsqueeze(0)
        action_seq = base_policy(obs_tensor, noise, return_numpy=True)[0]
        for a in action_seq:
            obs, reward, done, info = _step_env(env, a)
            raw_obs = denormalize_obs(obs, obs_min, obs_max)
            speeds.append(float(raw_obs[speed_index]))
            steps += 1
            if done or steps >= max_steps:
                break

speeds = np.array(speeds, dtype=np.float32)
print(f"[speed_ref] samples={len(speeds)} mean={np.mean(speeds):.6f} std={np.std(speeds):.6f}")
print(f"[speed_ref] p25={np.percentile(speeds, 25):.6f} p50={np.percentile(speeds, 50):.6f} p75={np.percentile(speeds, 75):.6f}")