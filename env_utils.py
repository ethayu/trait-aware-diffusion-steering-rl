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


def compute_joint_pref_reward(
	base_reward: float,
	joint_angle: float,
	p: np.ndarray,
	neutral_angle: float = 0.0,
	lambda_joint: float = 1.0,
) -> float:
	"""
	compute a preference-shaped reward based on a single joint angle.

	args:
		base_reward: Original environment reward (scalar).
		joint_angle: Joint angle in physical units (e.g. radians).
		p: Preference vector; we currently use the first two dims:
		   p[0]: weight for "default" behavior (base reward),
		   p[1]: weight for "joint constraint" preference.
		   We internally normalize p[:2] so w0 + w1 ~= 1.
		neutral_angle: Desired / neutral joint angle in the same units as joint_angle
			(e.g. radians for Hopper's foot joint).
		lambda_joint: Scale factor for the joint-angle penalty term.
	"""
	# p should be a 1D numpy array
	if isinstance(p, (list, tuple)):
		p = np.asarray(p, dtype=np.float32)
	if p.ndim > 1:
		p = p.reshape(-1)

	if p.shape[0] < 2:
		return float(base_reward)

	# use only first two dims for now, but keep p as a vector for future extensions. 
	# because theoretically you could set p1 = 1 - p0 and get the same effect.
	w_base = float(p[0])
	w_joint = float(p[1])
	total = w_base + w_joint
	if total > 0.0:
		w_base /= total
		w_joint /= total
	else:
		w_base, w_joint = 1.0, 0.0

	# Joint penalty: prefer angles close to neutral_angle.
	# squared penalty so large deviations are punished more strongly.
	joint_angle = float(joint_angle)
	delta = joint_angle - float(neutral_angle)
	joint_penalty = -(delta ** 2)

	base_reward = float(base_reward)
	# Two "reward components":
	# - r_base: original env reward
	# - r_joint: env reward plus a joint-angle penalty term
	r_base = base_reward
	r_joint = base_reward + lambda_joint * joint_penalty

	# final reward is a normalized mixture
	shaped_reward = w_base * r_base + w_joint * r_joint
	return shaped_reward


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


class PreferenceWrapperGym(gym.Env):
	"""
	Append a preference vector p to the normalized observation and optionally
	apply preference-shaped reward using a joint angle (e.g., Hopper foot joint).

	This keeps the underlying env (e.g., Hopper-v2 with ObservationWrapperGym) unchanged,
	but exposes an extended observation [state; p] to downstream wrappers / agents.
	"""

	def __init__(
		self,
		env,
		pref_dim: int = 2,
		p_min: float = -1.0,
		p_max: float = 1.0,
		joint_index: int = 4,
		neutral_angle: float = 0.0,
		lambda_joint: float = 1.0,
		fixed_pref=None,
	):
		self.env = env
		self.pref_dim = pref_dim
		self.p_min = p_min
		self.p_max = p_max
		self.joint_index = joint_index
		self.neutral_angle = neutral_angle
		self.lambda_joint = lambda_joint
		self.fixed_pref = None
		if fixed_pref is not None:
			fixed_pref = np.asarray(fixed_pref, dtype=np.float32).reshape(-1)
			assert fixed_pref.shape[0] == self.pref_dim, f"Expected fixed_pref of dim {self.pref_dim}, got {fixed_pref.shape[0]}"
			self.fixed_pref = fixed_pref
		self.action_space = env.action_space
		# if the wrapped env is an ObservationWrapperGym, we can access the
		# normalization stats to recover the raw joint angle from the normalized one.
		self.obs_min = getattr(env, "obs_min", None)
		self.obs_max = getattr(env, "obs_max", None)
		self.reset_joint_angle = None

		# extend the observation space by pref_dim dimensions for p.
		env_low = env.observation_space.low
		env_high = env.observation_space.high
		p_low = np.full(pref_dim, p_min, dtype=np.float32)
		p_high = np.full(pref_dim, p_max, dtype=np.float32)
		low = np.concatenate([env_low, p_low], axis=0)
		high = np.concatenate([env_high, p_high], axis=0)
		self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

		self.current_pref = None

	def seed(self, seed=None):
		if hasattr(self.env, "seed"):
			self.env.seed(seed)
		if seed is not None:
			np.random.seed(seed=seed)
		else:
			np.random.seed()

	def _sample_pref(self):
		# sometimes I want to fix preference
		if self.fixed_pref is not None:
			return self.fixed_pref.copy()
		# Otherwise, sample p uniformly in [p_min, p_max]^pref_dim.
		# but maybe we could eventualy start sampling according to perimeters?
		# and then the holdout would be on an interpolation of the perimeters.
		return np.random.uniform(self.p_min, self.p_max, size=(self.pref_dim,)).astype(np.float32)

	def set_pref(self, p):
		"""
		Manually set the current preference vector (used e.g. at eval time).
		"""
		p = np.asarray(p, dtype=np.float32).reshape(-1)
		assert p.shape[0] == self.pref_dim, f"Expected p of dim {self.pref_dim}, got {p.shape[0]}"
		self.current_pref = p

	def _augment_obs(self, obs):
		if self.current_pref is None:
			self.current_pref = self._sample_pref()
		return np.concatenate([obs, self.current_pref], axis=-1)

	def reset(self, **kwargs):
		options = kwargs.get("options", {})
		new_seed = options.get("seed", None)
		if new_seed is not None:
			self.seed(seed=new_seed)
		raw_obs = self.env.reset()
		# compute and store the raw joint angle at reset (position 0) if possible.
		norm_angle = float(raw_obs[self.joint_index])
		if self.obs_min is not None and self.obs_max is not None:
			j_min = float(self.obs_min[self.joint_index])
			j_max = float(self.obs_max[self.joint_index])
			self.reset_joint_angle = ((norm_angle / 2.0) + 0.5) * (j_max - j_min + 1e-6) + j_min
		else:
			self.reset_joint_angle = norm_angle
		if self.current_pref is None:
			self.current_pref = self._sample_pref()
		obs = self._augment_obs(raw_obs)
		return obs

	def step(self, action):
		raw_obs, reward, done, info = self.env.step(action)

		# pply joint-angle based preference shaping to the reward using the base observation
		# (before concatenating p). raw_obs here is normalized by ObservationWrapperGym,
		# so we optionally unnormalize it back to the physical joint angle using obs_min/obs_max.
		norm_angle = float(raw_obs[self.joint_index])
		if self.obs_min is not None and self.obs_max is not None:
			j_min = float(self.obs_min[self.joint_index])
			j_max = float(self.obs_max[self.joint_index])
			# norm = 2 * ((raw - min) / (max - min + 1e-6) - 0.5)
			# => raw = ((norm / 2 + 0.5) * (max - min + 1e-6)) + min
			joint_angle_raw = ((norm_angle / 2.0) + 0.5) * (j_max - j_min + 1e-6) + j_min
		else:
			joint_angle_raw = norm_angle
		foot_dev = None
		if self.lambda_joint != 0.0 and self.current_pref is not None:
			foot_dev = float(abs(joint_angle_raw - self.neutral_angle))
			shaped_reward = compute_joint_pref_reward(
				base_reward=reward,
				joint_angle=joint_angle_raw,
				p=self.current_pref,
				neutral_angle=self.neutral_angle,
				lambda_joint=self.lambda_joint,
			)
		else:
			shaped_reward = reward

		obs = self._augment_obs(raw_obs)
		info = dict(info)
		info["pref"] = self.current_pref.copy() if self.current_pref is not None else None
		info["base_reward"] = float(reward)
		info["shaped_reward"] = float(shaped_reward)
		info["foot_angle_raw"] = float(joint_angle_raw)
		if self.reset_joint_angle is not None:
			info["foot_angle_reset"] = float(self.reset_joint_angle)
		if foot_dev is not None:
			info["foot_angle_dev"] = foot_dev
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
	

class DiffusionPolicyEnvWrapper(VecEnvWrapper):
	def __init__(self, env, cfg, base_policy):
		super().__init__(env)
		self.action_horizon = cfg.act_steps
		self.action_dim = cfg.action_dim
		# Base observation dimension expected by the diffusion policy.
		# This allows us to later extend the observation space (e.g., by appending preferences)
		# while still feeding only the original state to the diffusion model.
		self.base_obs_dim = cfg.obs_dim
		self.action_space = spaces.Box(
			low=-cfg.train.action_magnitude*np.ones(self.action_dim*self.action_horizon),
			high=cfg.train.action_magnitude*np.ones(self.action_dim*self.action_horizon),
			dtype=np.float32
		)
		# use the wrapped VecEnv observation space (which may include preferences)
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
	
