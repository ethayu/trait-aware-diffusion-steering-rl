import os
import random
import math
from typing import Optional

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
    ActionChunkWrapper,
    make_robomimic_env,
)


# here register same OmegaConf resolvers as in train_dsrl.py so that ${eval:...}
# and similar interpolations work when loading the Hopper config.
OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil)
OmegaConf.register_new_resolver("round_down", math.floor)

base_path = os.path.dirname(os.path.abspath(__file__))


def make_single_env(cfg):
    """
    Create a single environment with the same wrappers as used during training.
    """
    if cfg.env_name in ["halfcheetah-medium-v2", "hopper-medium-v2", "walker2d-medium-v2"]:
        env = gym.make(cfg.env_name)
        env = ObservationWrapperGym(env, cfg.normalization_path)
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


def _get_render_env(vec_env):
    """
    Get the underlying (non-vector) env used for rendering.
    should work with plain DummyVecEnv or a VecEnvWrapper on top of it.
    """
    base = vec_env
    # unwrap one level if this is a VecEnvWrapper
    if hasattr(base, "venv"):
        base = base.venv
    # DummyVecEnv exposes envs list
    return base.envs[0]


def record_episodes(
    model,
    vec_env,
    cfg,
    total_episodes: int = 10,
    save_first_last: int = 2,
    output_dir: Optional[str] = None,
):
    """
    Run episodes and save the first and last `save_first_last` as MP4 videos.
    Need to make it just save the first ones since theres no training here.

    videos saved here:
        <output_dir>/hopper_ep{idx}.mp4
    """
    import imageio

    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "videos")
    os.makedirs(output_dir, exist_ok=True)

    total_episodes = max(total_episodes, save_first_last * 2)
    indices_to_save = set(
        list(range(save_first_last))
        + list(range(total_episodes - save_first_last, total_episodes))
    )

    render_env = _get_render_env(vec_env)

    for ep in range(total_episodes):
        obs = vec_env.reset()
        done = np.array([False])
        frames = []
        steps = 0

        while not done[0] and steps < cfg.env.max_episode_steps:
            frame = render_env.render(mode="rgb_array")
            if ep in indices_to_save:
                frames.append(frame)

            if cfg.algorithm == "dsrl_sac":
                action, _ = model.predict(obs, deterministic=True)
            else:
                action, _ = model.predict_diffused(obs, deterministic=True)

            obs, rewards, dones, infos = vec_env.step(action)
            done = dones
            steps += 1

        if ep in indices_to_save and len(frames) > 0:
            video_path = os.path.join(output_dir, f"hopper_ep{ep}.mp4")
            with imageio.get_writer(video_path, fps=30) as writer:
                for f in frames:
                    writer.append_data(f)
            print(f"[record_hopper_rollouts] Saved episode {ep} to {video_path}")


@hydra.main(
    config_path=os.path.join(base_path, "cfg/gym"),
    config_name="dsrl_hopper.yaml",
    version_base=None,
)
def main(cfg: OmegaConf):
    """
    run like this
        python record_hopper_rollouts.py \\
            model_path=/absolute/path/to/checkpoint_or_final.zip \\
            record_episodes=10

    this will save MP4s for episodes 0,1 and 8,9 into a `videos/` folder inside
    the current Hydra run directory.
    """
    OmegaConf.resolve(cfg)

    # make config mutable for extra CLI keys like model_path, record_episodes
    cfg._set_flag("struct", False)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    vec_env = DummyVecEnv([lambda: make_single_env(cfg)])

    algo = cfg.algorithm
    if algo == "dsrl_sac":
        ModelCls = SAC
    elif algo == "dsrl_na":
        ModelCls = DSRL
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")

    model_path = cfg.get("model_path", "")
    if not model_path:
        raise ValueError(
            "You must pass a trained checkpoint path via CLI, e.g. "
            "'model_path=/abs/path/to/checkpoint.zip'"
        )

    if not os.path.isabs(model_path):
        model_path = os.path.abspath(model_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model_path does not exist: {model_path}")

    print(f"[record_hopper_rollouts] Loading model from: {model_path}")

    # for DSRL, some constructor-only kwargs (like diffusion_act_dim) are not
    # always stored correctly in the checkpoint. We override them from cfg here
    load_kwargs = {}
    if algo == "dsrl_na":
        load_kwargs.update(
            diffusion_act_dim=(cfg.act_steps, cfg.action_dim),
            noise_critic_grad_steps=cfg.train.noise_critic_grad_steps,
            critic_backup_combine_type=cfg.train.critic_backup_combine_type,
        )

    model = ModelCls.load(model_path, env=vec_env, **load_kwargs)

    # where to place videos: by default inside the current Hydra run dir
    output_dir = os.path.join(os.getcwd(), "videos")
    total_episodes = int(cfg.get("record_episodes", 10))

    record_episodes(
        model=model,
        vec_env=vec_env,
        cfg=cfg,
        total_episodes=total_episodes,
        save_first_last=2,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()


