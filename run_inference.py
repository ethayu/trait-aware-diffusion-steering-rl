import os
import random
import math

import gym
import d4rl  # noqa: F401
import d4rl.gym_mujoco  # noqa: F401
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from stable_baselines3 import SAC, DSRL
from stable_baselines3.common.vec_env import DummyVecEnv

from env_utils import (
    ObservationWrapperGym,
    ObservationWrapperRobomimic,
    TraitWrapperGym,
    ActionChunkWrapper,
    make_robomimic_env,
)
from utils import load_base_policy


OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil)
OmegaConf.register_new_resolver("round_down", math.floor)

base_path = os.path.dirname(os.path.abspath(__file__))


def make_single_env(cfg):
    if cfg.env_name in ["halfcheetah-medium-v2", "hopper-medium-v2", "walker2d-medium-v2"]:
        env = gym.make(cfg.env_name)
        env = ObservationWrapperGym(env, cfg.normalization_path)
        traits_cfg = getattr(cfg, "traits", None)
        if traits_cfg is not None and getattr(traits_cfg, "enabled", False):
            env = TraitWrapperGym(env, traits_cfg)
    elif cfg.env_name in ["lift", "can", "square", "transport"]:
        env = make_robomimic_env(
            env=cfg.env_name,
            normalization_path=cfg.normalization_path,
            low_dim_keys=cfg.env.wrappers.robomimic_lowdim.low_dim_keys,
            dppo_path=cfg.dppo_path,
        )
        env = ObservationWrapperRobomimic(env, reward_offset=cfg.env.reward_offset)
    else:
        raise ValueError(f"Unsupported env_name: {cfg.env_name}")

    env = ActionChunkWrapper(env, cfg, max_episode_steps=cfg.env.max_episode_steps)
    return env


def _get_trait_wrapper(vec_env):
    base = vec_env
    if hasattr(base, "venv"):
        base = base.venv
    env = base.envs[0]
    while hasattr(env, "env"):
        if isinstance(env, TraitWrapperGym):
            return env
        env = env.env
    if isinstance(env, TraitWrapperGym):
        return env
    return None


def run_episodes(model, vec_env, cfg, episodes, deterministic):
    returns = []
    metric_agg = {}  # To track trait_speed_target_speed, trait_thigh_gap_gap, etc.
    max_steps = cfg.env.max_episode_steps
    record_video = bool(cfg.get("record_video", False))
    video_dir = cfg.get("video_dir", "videos_inference")
    video_episodes = int(cfg.get("video_episodes", 2))
    video_fps = int(cfg.get("video_fps", 30))
    video_indices = set(range(min(video_episodes, episodes)))
    writer = None
    if record_video:
        os.makedirs(video_dir, exist_ok=True)
        import imageio
        writer = imageio
    for ep_idx in range(episodes):
        obs = vec_env.reset()
        done = np.array([False])
        ep_return = 0.0
        steps = 0
        frames = []
        if record_video and ep_idx in video_indices:
            render_env = vec_env
            if hasattr(render_env, "venv"):
                render_env = render_env.venv
            render_env = render_env.envs[0]
        while not done[0] and steps < max_steps:
            if cfg.algorithm == "dsrl_sac":
                action, _ = model.predict(obs, deterministic=deterministic)
            else:
                action, _ = model.predict_diffused(obs, deterministic=deterministic)
            obs, rewards, dones, infos = vec_env.step(action)
            
            # Aggregate trait metrics from the first env in the vector
            info = infos[0]
            for k, v in info.items():
                if k.startswith("trait_") and not isinstance(v, (np.ndarray, list)):
                    if k not in metric_agg:
                        metric_agg[k] = []
                    metric_agg[k].append(float(v))

            if record_video and writer is not None and ep_idx in video_indices:
                frame = render_env.render(mode="rgb_array")
                frames.append(frame)
            ep_return += float(rewards[0])
            done = dones
            steps += 1
        returns.append(ep_return)
        if record_video and writer is not None and len(frames) > 0 and ep_idx in video_indices:
            video_path = os.path.join(video_dir, f"episode_{ep_idx}.mp4")
            with writer.get_writer(video_path, fps=video_fps) as video:
                for frame in frames:
                    video.append_data(frame)
    # Calculate means for all collected metrics
    avg_metrics = {k: np.mean(v) for k, v in metric_agg.items()}
    return returns, avg_metrics


@hydra.main(
    config_path=os.path.join(base_path, "cfg/gym"),
    config_name="dsrl_walker.yaml",
    version_base=None,
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    cfg._set_flag("struct", False)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    vec_env = DummyVecEnv([lambda: make_single_env(cfg)])
    trait_wrapper = _get_trait_wrapper(vec_env)
    if trait_wrapper is not None:
        trait_values = cfg.get("trait_values", None)
        trait_mask = cfg.get("trait_mask", None)
        if trait_values is not None or trait_mask is not None:
            trait_wrapper.set_traits(values=trait_values, mask=trait_mask)

    base_policy = load_base_policy(cfg)
    if cfg.algorithm == "dsrl_sac":
        ModelCls = SAC
    elif cfg.algorithm == "dsrl_na":
        ModelCls = DSRL
    else:
        raise ValueError(f"Unsupported algorithm: {cfg.algorithm}")

    model_path = cfg.get("model_path", "")
    if not model_path:
        raise ValueError("You must set model_path for inference.")
    if not os.path.isabs(model_path):
        model_path = os.path.abspath(model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model_path does not exist: {model_path}")

    load_kwargs = {}
    if cfg.algorithm == "dsrl_na":
        load_kwargs.update(
            diffusion_act_dim=(cfg.act_steps, cfg.action_dim),
            noise_critic_grad_steps=cfg.train.noise_critic_grad_steps,
            critic_backup_combine_type=cfg.train.critic_backup_combine_type,
        )
    load_device = getattr(cfg, "device", "auto")
    
    # Disable observation space check to allow loading models with different trait bounds
    # and to allow testing OOD values.
    custom_objects = {
        "observation_space": vec_env.observation_space,
        "action_space": vec_env.action_space,
    }
    
    model = ModelCls.load(
        model_path, 
        env=vec_env, 
        device=load_device, 
        custom_objects=custom_objects, 
        **load_kwargs
    )
    if cfg.algorithm == "dsrl_na" and getattr(model, "diffusion_policy", None) is None:
        model.diffusion_policy = base_policy

    episodes = int(cfg.get("eval_episodes", 10))
    deterministic = bool(cfg.get("deterministic_eval", False))
    returns, metrics = run_episodes(model, vec_env, cfg, episodes, deterministic)
    print(f"[run_inference] episodes={episodes} mean_return={np.mean(returns):.3f} std_return={np.std(returns):.3f}")
    for k, v in metrics.items():
        print(f"[run_inference] metric_{k}={v:.4f}")
    print("[run_inference] returns", returns)


if __name__ == "__main__":
    main()