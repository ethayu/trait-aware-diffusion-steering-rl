import os
import numpy as np
from omegaconf import OmegaConf
import torch
import hydra
import sys
import gym
import gymnasium
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnvWrapper
import json

from dppo.env.gym_utils.wrapper import wrapper_dict
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
from traits import get_base_reward_fn, get_trait_reward_fn


def denormalize_obs(
	norm_obs: np.ndarray,
	obs_min: np.ndarray,
	obs_max: np.ndarray,
) -> np.ndarray:
	"""
	Map normalized obs values back to raw values.
	"""
	if obs_min is None or obs_max is None:
		return np.asarray(norm_obs, dtype=np.float32)
	raw_obs = ((norm_obs / 2.0) + 0.5) * (obs_max - obs_min + 1e-6) + obs_min
	return np.asarray(raw_obs, dtype=np.float32)


def make_robomimic_env(render=False, env='square', normalization_path=None, low_dim_keys=None, dppo_path=None):
	wrappers = OmegaConf.create({
		'robomimic_lowdim': {
			'normalization_path': normalization_path,
			'low_dim_keys': low_dim_keys,
		},
	})
	obs_modality_dict = {
		"low_dim": (
			wrappers.robomimic_image.low_dim_keys
			if "robomimic_image" in wrappers
			else wrappers.robomimic_lowdim.low_dim_keys
		),
		"rgb": (
			wrappers.robomimic_image.image_keys
			if "robomimic_image" in wrappers
			else None
		),
	}
	if obs_modality_dict["rgb"] is None:
		obs_modality_dict.pop("rgb")
	ObsUtils.initialize_obs_modality_mapping_from_dict(obs_modality_dict)
	robomimic_env_cfg_path = f'{dppo_path}/cfg/robomimic/env_meta/{env}.json'
	with open(robomimic_env_cfg_path, "r") as f:
		env_meta = json.load(f)
	env_meta["reward_shaping"] = False
	env = EnvUtils.create_env_from_metadata(
		env_meta=env_meta,
		render=False,
		render_offscreen=render,
		use_image_obs=False,
	)
	env.env.hard_reset = False
	for wrapper, args in wrappers.items():
		env = wrapper_dict[wrapper](env, **args)
	return env


class ObservationWrapperRobomimic(gym.Env):
	def __init__(
		self,
		env,
		reward_offset=1,
	):
		self.env = env
		self.action_space = env.action_space
		self.observation_space = env.observation_space
		self.reward_offset = reward_offset

	def seed(self, seed=None):
		if seed is not None:
			np.random.seed(seed=seed)
		else:
			np.random.seed()

	def reset(self, **kwargs):
		options = kwargs.get("options", {})
		new_seed = options.get("seed", None)
		if new_seed is not None:
			self.seed(seed=new_seed)
		raw_obs = self.env.reset()
		obs = raw_obs['state'].flatten()
		return obs

	def step(self, action):
		raw_obs, reward, done, info = self.env.step(action)
		reward = (reward - self.reward_offset)
		obs = raw_obs['state'].flatten()
		return obs, reward, done, info

	def render(self, mode="human", **kwargs):
		return self.env.render(mode=mode, **kwargs)
	

class ObservationWrapperGym(gym.Env):
	def __init__(
		self,
		env,
		normalization_path,
	):
		self.env = env
		self.action_space = env.action_space
		self.observation_space = env.observation_space
		normalization = np.load(normalization_path)
		self.obs_min = normalization["obs_min"]
		self.obs_max = normalization["obs_max"]
		self.action_min = normalization["action_min"]
		self.action_max = normalization["action_max"]

	def seed(self, seed=None):
		if seed is not None:
			np.random.seed(seed=seed)
		else:
			np.random.seed()

	def reset(self, **kwargs):
		options = kwargs.get("options", {})
		new_seed = options.get("seed", None)
		if new_seed is not None:
			self.seed(seed=new_seed)
		raw_obs = self.env.reset()
		obs = self.normalize_obs(raw_obs)
		return obs

	def step(self, action):
		raw_action = self.unnormalize_action(action)
		raw_obs, reward, done, info = self.env.step(raw_action)
		obs = self.normalize_obs(raw_obs)
		return obs, reward, done, info

	def render(self, mode="human", **kwargs):
		return self.env.render(mode=mode, **kwargs)
	
	def normalize_obs(self, obs):
		return 2 * ((obs - self.obs_min) / (self.obs_max - self.obs_min + 1e-6) - 0.5)

	def unnormalize_action(self, action):
		action = (action + 1) / 2
		return action * (self.action_max - self.action_min) + self.action_min
	

class TraitWrapperGym(gym.Env):
	"""
	Append trait values and a binary trait mask to the observation, and apply
	trait-aware reward shaping: r = r_old + sum_i mask_i * lambda_i * r_i(...).
	"""

	def __init__(self, env, traits_cfg):
		self.env = env
		self.action_space = env.action_space
		self.obs_min = getattr(env, "obs_min", None)
		self.obs_max = getattr(env, "obs_max", None)

		if OmegaConf.is_config(traits_cfg):
			traits_cfg = OmegaConf.to_container(traits_cfg, resolve=True)
		self.traits_cfg = traits_cfg or {}
		self.trait_defs = list(self.traits_cfg.get("defs", []))
		self.n_traits = len(self.trait_defs)
		self.trait_dims = []
		self.trait_value_mins = []
		self.trait_value_maxs = []
		for trait_def in self.trait_defs:
			dim, v_min, v_max = self._trait_bounds_for_def(trait_def)
			self.trait_dims.append(dim)
			self.trait_value_mins.append(v_min)
			self.trait_value_maxs.append(v_max)
		self.total_trait_dim = int(sum(self.trait_dims))
		self.mask_mode = self.traits_cfg.get("mask_mode", "all_on")
		self.resample_on_reset = bool(self.traits_cfg.get("resample_on_reset", True))

		self.fixed_values = self.traits_cfg.get("fixed_values", None)
		self.fixed_mask = self.traits_cfg.get("fixed_mask", None)
		self.override_values = None
		self.override_mask = None
		self.current_traits = None
		self.current_mask = None
		self.prev_raw_obs = None
		self.base_reward_fn = None

		self.trait_slices = []
		start = 0
		for dim in self.trait_dims:
			self.trait_slices.append((start, start + dim))
			start += dim
		self.reward_fns = []
		for trait_def in self.trait_defs:
			trait_name = trait_def.get("name", None)
			if trait_name is None:
				raise ValueError("Each trait must define name.")
			self.reward_fns.append(get_trait_reward_fn(trait_name))
		base_reward_name = self.traits_cfg.get("base_reward_fn", None)
		if base_reward_name is not None:
			self.base_reward_fn = get_base_reward_fn(base_reward_name)

		env_low = env.observation_space.low
		env_high = env.observation_space.high
		trait_low, trait_high = self._trait_value_bounds()
		mask_low = np.zeros(self.n_traits, dtype=np.float32)
		mask_high = np.ones(self.n_traits, dtype=np.float32)
		low = np.concatenate([env_low, trait_low, mask_low], axis=0)
		high = np.concatenate([env_high, trait_high, mask_high], axis=0)
		self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

	def _trait_bounds_for_def(self, trait_def):
		v_min = trait_def.get("value_min", -1.0)
		v_max = trait_def.get("value_max", 1.0)
		v_min_arr = np.asarray(v_min, dtype=np.float32)
		v_max_arr = np.asarray(v_max, dtype=np.float32)
		if v_min_arr.ndim == 0 and v_max_arr.ndim == 0:
			v_min_arr = np.array([float(v_min_arr)], dtype=np.float32)
			v_max_arr = np.array([float(v_max_arr)], dtype=np.float32)
		else:
			v_min_arr = np.atleast_1d(v_min_arr).astype(np.float32)
			v_max_arr = np.atleast_1d(v_max_arr).astype(np.float32)
			if v_min_arr.shape != v_max_arr.shape:
				name = trait_def.get("name", "trait")
				raise ValueError(f"value_min/value_max shape mismatch for {name}.")
		dim = int(v_min_arr.shape[0])
		return dim, v_min_arr, v_max_arr

	def _trait_value_bounds(self):
		lows = []
		highs = []
		for v_min, v_max in zip(self.trait_value_mins, self.trait_value_maxs):
			lows.extend(v_min.tolist())
			highs.extend(v_max.tolist())
		return np.asarray(lows, dtype=np.float32), np.asarray(highs, dtype=np.float32)

	def seed(self, seed=None):
		if hasattr(self.env, "seed"):
			self.env.seed(seed)
		if seed is not None:
			np.random.seed(seed=seed)
		else:
			np.random.seed()

	def _sample_trait_values(self):
		if self.fixed_values is not None:
			return np.asarray(self.fixed_values, dtype=np.float32).reshape(-1)
		values = []
		for dim, v_min, v_max in zip(self.trait_dims, self.trait_value_mins, self.trait_value_maxs):
			sample = np.asarray(np.random.uniform(v_min, v_max), dtype=np.float32)
			if dim == 1:
				values.append(float(sample[0]))
			else:
				values.extend(sample.tolist())
		return np.asarray(values, dtype=np.float32)

	def _sample_trait_mask(self):
		if self.fixed_mask is not None:
			return np.asarray(self.fixed_mask, dtype=np.float32).reshape(-1)
		if self.mask_mode == "random":
			mask_prob = self.traits_cfg.get("mask_prob", 1.0)
			if isinstance(mask_prob, (list, tuple)):
				mask_prob = np.asarray(mask_prob, dtype=np.float32)
				if mask_prob.shape[0] != self.n_traits:
					mask_prob = np.full(self.n_traits, float(mask_prob[0]))
			else:
				mask_prob = np.full(self.n_traits, float(mask_prob))
			return (np.random.uniform(0.0, 1.0, size=(self.n_traits,)) < mask_prob).astype(np.float32)
		return np.ones(self.n_traits, dtype=np.float32)

	def set_traits(self, values=None, mask=None):
		if values is not None:
			self.override_values = np.asarray(values, dtype=np.float32).reshape(-1)
			self.current_traits = self.override_values.copy()
		if mask is not None:
			self.override_mask = np.asarray(mask, dtype=np.float32).reshape(-1)
			self.current_mask = self.override_mask.copy()

	def clear_traits(self):
		self.override_values = None
		self.override_mask = None

	def _ensure_traits(self, resample=False):
		if self.override_values is not None:
			self.current_traits = self.override_values.copy()
		elif self.current_traits is None or resample:
			self.current_traits = self._sample_trait_values()

		if self.override_mask is not None:
			self.current_mask = self.override_mask.copy()
		elif self.current_mask is None or resample:
			self.current_mask = self._sample_trait_mask()

	def _augment_obs(self, obs):
		traits = self.current_traits if self.current_traits is not None else np.zeros(self.total_trait_dim, dtype=np.float32)
		mask = self.current_mask if self.current_mask is not None else np.ones(self.n_traits, dtype=np.float32)
		return np.concatenate([obs, traits, mask], axis=-1)

	def _trait_value(self, trait_idx):
		start, end = self.trait_slices[trait_idx]
		vals = self.current_traits[start:end]
		if end - start == 1:
			return float(vals[0])
		return vals

	def _compute_trait_reward(self, raw_obs, trait_idx, env_info, action):
		trait_def = self.trait_defs[trait_idx]
		trait_val = self._trait_value(trait_idx)
		reward_fn = self.reward_fns[trait_idx]
		trait_info = {
			"trait_name": trait_def.get("name", f"trait_{trait_idx}"),
			"trait_index": trait_idx,
			"trait_value": trait_val,
			"prev_raw_obs": self.prev_raw_obs,
			"env_info": env_info,
			"action": action,
		}
		return reward_fn(raw_obs, trait_info)

	def reset(self, **kwargs):
		options = kwargs.get("options", {})
		new_seed = options.get("seed", None)
		if new_seed is not None:
			self.seed(seed=new_seed)
		norm_obs = self.env.reset()
		self._ensure_traits(resample=self.resample_on_reset or self.current_traits is None)
		raw_obs = denormalize_obs(norm_obs, self.obs_min, self.obs_max)
		self.prev_raw_obs = raw_obs
		obs = self._augment_obs(norm_obs)
		return obs

	def step(self, action):
		norm_obs, reward, done, info = self.env.step(action)
		self._ensure_traits(resample=False)
		raw_obs = denormalize_obs(norm_obs, self.obs_min, self.obs_max)
		env_reward = float(reward)
		if self.base_reward_fn is not None:
			base_info = {
				"prev_raw_obs": self.prev_raw_obs,
				"env_info": info,
				"action": action,
				"original_reward": env_reward,
			}
			base_reward = float(self.base_reward_fn(raw_obs, base_info))
		else:
			base_reward = env_reward
		shaped_reward = float(base_reward)
		trait_rewards = []
		trait_metrics = {}
		for i in range(self.n_traits):
			mask_val = float(self.current_mask[i]) if self.current_mask is not None else 1.0
			if mask_val <= 0.0:
				trait_rewards.append(0.0)
				continue
			trait_reward, metrics = self._compute_trait_reward(raw_obs, i, info, action)
			lambda_i = float(self.trait_defs[i].get("lambda", self.trait_defs[i].get("weight", 1.0)))
			weighted = mask_val * lambda_i * float(trait_reward)
			trait_rewards.append(weighted)
			for key, val in metrics.items():
				trait_metrics[f"trait_{self.trait_defs[i].get('name', i)}_{key}"] = float(val)
			shaped_reward += weighted

		obs = self._augment_obs(norm_obs)
		info = dict(info)
		info["env_reward"] = env_reward
		info["base_reward"] = float(base_reward)
		info["shaped_reward"] = float(shaped_reward)
		info["trait_values"] = self.current_traits.copy() if self.current_traits is not None else None
		info["trait_mask"] = self.current_mask.copy() if self.current_mask is not None else None
		info["trait_rewards"] = trait_rewards
		info["raw_obs"] = raw_obs.copy()
		if self.prev_raw_obs is not None:
			info["prev_raw_obs"] = self.prev_raw_obs.copy()
		info.update(trait_metrics)
		self.prev_raw_obs = raw_obs
		return obs, shaped_reward, done, info

	def render(self, mode="human", **kwargs):
		if hasattr(self.env, "render"):
			return self.env.render(mode=mode, **kwargs)
		raise NotImplementedError


class ActionChunkWrapper(gymnasium.Env):
	def __init__(self, env, cfg, max_episode_steps=300):
		self.max_episode_steps = max_episode_steps
		self.env = env
		self.act_steps = cfg.act_steps
		self.action_space = spaces.Box(
			low=np.tile(env.action_space.low, cfg.act_steps),
			high=np.tile(env.action_space.high, cfg.act_steps),
			dtype=np.float32
		)
		self.observation_space = env.observation_space
		self.count = 0

	def reset(self, seed=None):
		obs = self.env.reset(seed=seed)
		self.count = 0
		return obs, {}
	
	def step(self, action):
		if len(action.shape) == 1:
			action = action.reshape(self.act_steps, -1)
		obs_ = []
		reward_ = []
		done_ = []
		info_ = []
		done_i = False
		for i in range(action.shape[0]):
			self.count += 1
			obs_i, reward_i, done_i, info_i = self.env.step(action[i])
			obs_.append(obs_i)
			reward_.append(reward_i)
			done_.append(done_i)
			info_.append(info_i)
		obs = obs_[-1]
		reward = sum(reward_)
		done = np.max(done_)
		info = info_[-1]
		if self.count >= self.max_episode_steps:
			done = True
		if done:
			info['terminal_observation'] = obs
		return obs, reward, done, False, info

	def render(self, mode="human", **kwargs):
		return self.env.render(mode=mode, **kwargs)
	
	def close(self):
		return

	def set_traits(self, values=None, mask=None):
		if hasattr(self.env, "set_traits"):
			return self.env.set_traits(values=values, mask=mask)
		return None

	def clear_traits(self):
		if hasattr(self.env, "clear_traits"):
			return self.env.clear_traits()
		return None
	

class DiffusionPolicyEnvWrapper(VecEnvWrapper):
	def __init__(self, env, cfg, base_policy):
		super().__init__(env)
		self.action_horizon = cfg.act_steps
		self.action_dim = cfg.action_dim
		# Base observation dimension expected by the diffusion policy.
		# This allows us to later extend the observation space (e.g., by appending traits)
		# while still feeding only the original state to the diffusion model.
		self.base_obs_dim = cfg.obs_dim
		self.action_space = spaces.Box(
			low=-cfg.train.action_magnitude*np.ones(self.action_dim*self.action_horizon),
			high=cfg.train.action_magnitude*np.ones(self.action_dim*self.action_horizon),
			dtype=np.float32
		)
		# use the wrapped VecEnv observation space (which may include traits)
		self.observation_space = self.venv.observation_space
		self.obs_dim = int(np.prod(self.observation_space.shape))
		self.env = env
		self.device = cfg.model.device
		self.base_policy = base_policy
		self.obs = None

	def step_async(self, actions):
		actions = torch.tensor(actions, device=self.device, dtype=torch.float32)
		actions = actions.view(-1, self.action_horizon, self.action_dim)
		# Only pass the base part of the observation to the diffusion policy!!!
		diffused_actions = self.base_policy(self.obs[..., :self.base_obs_dim], actions)
		self.venv.step_async(diffused_actions)

	def step_wait(self):
		obs, rewards, dones, infos = self.venv.step_wait()
		self.obs = torch.tensor(obs, device=self.device, dtype=torch.float32)
		obs_out = self.obs
		return obs_out.detach().cpu().numpy(), rewards, dones, infos

	def reset(self):
		obs = self.venv.reset()
		self.obs = torch.tensor(obs, device=self.device, dtype=torch.float32)
		obs_out = self.obs
		return obs_out.detach().cpu().numpy()
	
