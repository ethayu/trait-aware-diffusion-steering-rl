import os
import csv
from typing import Optional

import torch
import wandb
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import hydra
from omegaconf import OmegaConf


class DPPOBasePolicyWrapper:
	def __init__(self, base_policy):
		self.base_policy = base_policy
		# xpose the underlying diffusion model's observation dimension so that
		# downstream algorithms (e.g., DSRL) can slice extended observations
		# back to the base state dim before calling the diffusion policy.
		if hasattr(base_policy, "obs_dim"):
			self.obs_dim = base_policy.obs_dim
		
	def __call__(self, obs, initial_noise, return_numpy=True):
		# if the env observation has been extended (e.g., with traits),
		# slice back to the base diffusion model's obs_dim before calling it.
		if hasattr(self, "obs_dim"):
			obs = obs[..., : self.obs_dim]
		cond = {
			"state": obs,
			"noise_action": initial_noise,
		}
		with torch.no_grad():
			samples = self.base_policy(cond=cond, deterministic=True)
		diffused_actions = (samples.trajectories.detach())
		if return_numpy:
			diffused_actions = diffused_actions.cpu().numpy()
		return diffused_actions	


def load_base_policy(cfg):
	base_policy = hydra.utils.instantiate(cfg.model)
	base_policy = base_policy.eval()
	return DPPOBasePolicyWrapper(base_policy)


class LoggingCallback(BaseCallback):
	def __init__(self, 
		action_chunk=4, 
		log_freq=1000,
		use_wandb=True, 
		eval_env=None, 
		eval_freq=70, 
		eval_episodes=2, 
		verbose=0, 
		rew_offset=0, 
		num_train_env=1,
		num_eval_env=1,
		algorithm='dsrl_sac',
		max_steps=-1,
		deterministic_eval=False,
		log_dir: Optional[str] = None,
		trait_cfg: Optional[dict] = None,
	):
		super().__init__(verbose)
		self.action_chunk = action_chunk
		self.log_freq = log_freq
		self.episode_rewards = []
		self.episode_lengths = []
		self.use_wandb = use_wandb
		self.eval_env = eval_env
		self.eval_episodes = eval_episodes
		self.eval_freq = eval_freq
		self.log_count = 0
		self.total_reward = 0
		self.rew_offset = rew_offset
		self.total_timesteps = 0
		self.num_train_env = num_train_env
		self.num_eval_env = num_eval_env
		self.episode_success = np.zeros(self.num_train_env)
		self.episode_completed = np.zeros(self.num_train_env)
		self.algorithm = algorithm
		self.max_steps = max_steps
		self.deterministic_eval = deterministic_eval
		self.log_dir = log_dir
		if OmegaConf.is_config(trait_cfg):
			trait_cfg = OmegaConf.to_container(trait_cfg, resolve=True)
		self.trait_cfg = trait_cfg or {}
		self.trait_schedule = self.trait_cfg.get("schedule", None)
		self.trait_logging = self.trait_cfg.get("logging", {})
		self.trait_logging_enabled = bool(self.trait_logging.get("enabled", False))
		self.trait_logging_auto_plots = bool(self.trait_logging.get("auto_plots", True))
		self.trait_defs = list(self.trait_cfg.get("defs", []))
		self.trait_names = [t.get("name", f"trait_{i}") for i, t in enumerate(self.trait_defs)]
		self.trait_dims = [self._trait_dim_from_def(t) for t in self.trait_defs]
		self.trait_slices = []
		start = 0
		for dim in self.trait_dims:
			self.trait_slices.append((start, start + dim))
			start += dim
		self._trait_phase_index = 0
		self._trait_phase_start = 0
		self._trait_best_eval = None
		self._trait_plateau_count = 0
		self._current_trait_mask = None
		self._eval_count = 0
		self._trait_value_accum = [[] for _ in self.trait_defs]
		self._trait_mask_accum = [[] for _ in self.trait_defs]
		self._trait_reward_accum = [[] for _ in self.trait_defs]
		self._trait_metric_accum = {}
		self._action_norm_accum = []
		self._action_trait_pairs = [[] for _ in self.trait_defs]
		self.local_log_path = None
		if self.log_dir is not None:
			os.makedirs(self.log_dir, exist_ok=True)
			self.local_log_path = os.path.join(self.log_dir, "trait_metrics.csv")
			if not os.path.exists(self.local_log_path):
				with open(self.local_log_path, mode="w", newline="") as f:
					writer = csv.writer(f)
					writer.writerow([
						"step",
						"timesteps",
						"ep_len_mean",
						"ep_rew_mean",
						"base_rew_mean_step",
						"shaped_rew_mean_step",
					])

	def _trait_dim_from_def(self, trait_def):
		v_min = trait_def.get("value_min", -1.0)
		v_min_arr = np.asarray(v_min, dtype=np.float32)
		if v_min_arr.ndim == 0:
			return 1
		return int(np.atleast_1d(v_min_arr).shape[0])

	def _on_training_start(self):
		self._apply_current_trait_mask()
		return True

	def _apply_current_trait_mask(self):
		if not self.trait_schedule:
			return
		phases = self.trait_schedule.get("phases", [])
		if not phases:
			return
		phase = phases[min(self._trait_phase_index, len(phases) - 1)]
		mask = phase.get("mask", None)
		if mask is None:
			return
		self._current_trait_mask = list(mask)
		if self.training_env is not None:
			self.training_env.env_method("set_traits", mask=mask)
		if self.eval_env is not None:
			self.eval_env.env_method("set_traits", mask=mask)

	def _maybe_advance_trait_phase(self, eval_reward):
		if not self.trait_schedule:
			return
		phases = self.trait_schedule.get("phases", [])
		if not phases or self._trait_phase_index >= len(phases) - 1:
			return
		min_steps = self.trait_schedule.get("min_steps", 0) # at least many steps to wait before advancing to the next phase
		patience = self.trait_schedule.get("patience", 0) # how many evals (episodes) without improvement to consider as convergence
		min_delta = self.trait_schedule.get("min_delta", 0.0) # what counts as improvement (we want `patience` evals without improvement to consider as convergence)
		phase_min_steps = phases[self._trait_phase_index].get("min_steps", min_steps)
		if (self.total_timesteps - self._trait_phase_start) < phase_min_steps:
			return
		if self._trait_best_eval is None or eval_reward > (self._trait_best_eval + min_delta):
			self._trait_best_eval = eval_reward
			self._trait_plateau_count = 0
			return
		self._trait_plateau_count += 1
		if self._trait_plateau_count >= patience:
			self._trait_phase_index += 1
			self._trait_phase_start = self.total_timesteps
			self._trait_best_eval = None
			self._trait_plateau_count = 0
			self._apply_current_trait_mask()

	def _on_step(self):
		# per-step / per-env logging
		for info in self.locals['infos']:
			if 'episode' in info:
				self.episode_rewards.append(info['episode']['r'])
				self.episode_lengths.append(info['episode']['l'])
		base_rewards = []
		shaped_rewards = []
		trait_keys = []
		for info in self.locals['infos']:
			if isinstance(info, dict):
				if 'base_reward' in info:
					base_rewards.append(info['base_reward'])
				if 'shaped_reward' in info:
					shaped_rewards.append(info['shaped_reward'])
				if 'trait_rewards' in info and info['trait_rewards'] is not None:
					trait_rewards = info['trait_rewards']
					for i, reward in enumerate(trait_rewards):
						if i < len(self._trait_reward_accum):
							self._trait_reward_accum[i].append(reward)
				if 'trait_values' in info and info['trait_values'] is not None:
					values = np.asarray(info['trait_values'], dtype=np.float32).reshape(-1)
					for i, (start, end) in enumerate(self.trait_slices):
						if end <= len(values):
							val = values[start:end]
							val_scalar = float(np.mean(val))
							self._trait_value_accum[i].append(val_scalar)
				if 'trait_mask' in info and info['trait_mask'] is not None:
					mask = np.asarray(info['trait_mask'], dtype=np.float32).reshape(-1)
					for i, mask_val in enumerate(mask):
						if i < len(self._trait_mask_accum):
							self._trait_mask_accum[i].append(float(mask_val))
				for key, val in info.items():
					if key.startswith("trait_") and isinstance(val, (int, float)):
						self._trait_metric_accum.setdefault(key, []).append(float(val))
		actions = self.locals.get("actions", None)
		if actions is not None:
			actions = np.asarray(actions)
			if actions.size > 0:
				action_flat = actions.reshape(actions.shape[0], -1)
				action_norms = np.linalg.norm(action_flat, axis=1)
				self._action_norm_accum.extend(action_norms.tolist())
				if len(self.trait_slices) > 0:
					infos = self.locals['infos']
					for i, info in enumerate(infos):
						if not isinstance(info, dict):
							continue
						if 'trait_values' not in info:
							continue
						values = np.asarray(info['trait_values'], dtype=np.float32).reshape(-1)
						if i >= len(action_norms):
							break
						for t_idx, (start, end) in enumerate(self.trait_slices):
							if end <= len(values):
								val = values[start:end]
								val_scalar = float(np.mean(val))
								self._action_trait_pairs[t_idx].append((val_scalar, float(action_norms[i])))

		rew = self.locals['rewards']
		self.total_reward += np.mean(rew)
		self.episode_success[rew > -self.rew_offset] = 1
		self.episode_completed[self.locals['dones']] = 1
		self.total_timesteps += self.action_chunk * self.model.n_envs
		if self.n_calls % self.log_freq == 0:
			if len(self.episode_rewards) > 0:
				if self.use_wandb:
					self.log_count += 1
					log_payload = {
						"train/ep_len_mean": np.mean(self.episode_lengths),
						"train/success_rate": np.sum(self.episode_success) / np.sum(self.episode_completed),
						"train/ep_rew_mean": np.mean(self.episode_rewards),
						"train/rew_mean": np.mean(self.total_reward),
						"train/timesteps": self.total_timesteps,
						"train/ent_coef": self.locals['self'].logger.name_to_value['train/ent_coef'],
						"train/actor_loss": self.locals['self'].logger.name_to_value['train/actor_loss'],
						"train/critic_loss": self.locals['self'].logger.name_to_value['train/critic_loss'],
						"train/ent_coef_loss": self.locals['self'].logger.name_to_value['train/ent_coef_loss'],
					}
					if len(base_rewards) > 0:
						log_payload["train/base_rew_mean_step"] = float(np.mean(base_rewards))
					if len(shaped_rewards) > 0:
						log_payload["train/shaped_rew_mean_step"] = float(np.mean(shaped_rewards))
					if len(base_rewards) > 0 and len(shaped_rewards) > 0:
						log_payload["train/shaping_delta_mean_step"] = float(np.mean(shaped_rewards) - np.mean(base_rewards))
					if self._current_trait_mask is not None:
						log_payload["train/trait_phase_index"] = self._trait_phase_index
						for i, m in enumerate(self._current_trait_mask):
							log_payload[f"train/trait_mask_{self.trait_names[i]}"] = float(m)
					for i, name in enumerate(self.trait_names):
						if len(self._trait_reward_accum[i]) > 0:
							log_payload[f"train/trait_reward_{name}"] = float(np.mean(self._trait_reward_accum[i]))
						if len(self._trait_value_accum[i]) > 0:
							log_payload[f"train/trait_value_{name}"] = float(np.mean(self._trait_value_accum[i]))
						if len(self._trait_mask_accum[i]) > 0:
							log_payload[f"train/trait_mask_mean_{name}"] = float(np.mean(self._trait_mask_accum[i]))
					for key, vals in self._trait_metric_accum.items():
						if len(vals) > 0:
							log_payload[f"train/{key}_mean"] = float(np.mean(vals))
					if len(self._action_norm_accum) > 0:
						log_payload["train/action_norm_mean"] = float(np.mean(self._action_norm_accum))
						log_payload["train/action_norm_std"] = float(np.std(self._action_norm_accum))
						for i, name in enumerate(self.trait_names):
							pairs = self._action_trait_pairs[i]
							if len(pairs) > 1:
								trait_vals = np.array([p[0] for p in pairs], dtype=np.float32)
								action_vals = np.array([p[1] for p in pairs], dtype=np.float32)
								if np.std(trait_vals) > 0 and np.std(action_vals) > 0:
									corr = float(np.corrcoef(trait_vals, action_vals)[0, 1])
									log_payload[f"train/action_norm_corr_{name}"] = corr

					wandb.log(log_payload, step=self.log_count)
					if np.sum(self.episode_completed) > 0:
						wandb.log({
							"train/success_rate": np.sum(self.episode_success) / np.sum(self.episode_completed),
						}, step=self.log_count)
					if self.algorithm == 'dsrl_na':
						wandb.log({
							"train/noise_critic_loss": self.locals['self'].logger.name_to_value['train/noise_critic_loss'],
						}, step=self.log_count)
				if self.local_log_path is not None:
					with open(self.local_log_path, mode="a", newline="") as f:
						writer = csv.writer(f)
						writer.writerow([
							self.log_count,
							self.total_timesteps,
							float(np.mean(self.episode_lengths)),
							float(np.mean(self.episode_rewards)),
							float(np.mean(base_rewards)) if len(base_rewards) > 0 else "",
							float(np.mean(shaped_rewards)) if len(shaped_rewards) > 0 else "",
						])
				self.episode_rewards = []
				self.episode_lengths = []
				self.total_reward = 0
				self.episode_success = np.zeros(self.num_train_env)
				self.episode_completed = np.zeros(self.num_train_env)
				self._trait_value_accum = [[] for _ in self.trait_defs]
				self._trait_mask_accum = [[] for _ in self.trait_defs]
				self._trait_reward_accum = [[] for _ in self.trait_defs]
				self._trait_metric_accum = {}
				self._action_norm_accum = []
				self._action_trait_pairs = [[] for _ in self.trait_defs]

		if self.n_calls % self.eval_freq == 0:
			eval_reward = self.evaluate(self.locals['self'], deterministic=False)
			self._eval_count += 1
			if eval_reward is not None:
				self._maybe_advance_trait_phase(eval_reward)
				self._maybe_run_trait_sweeps(self.locals['self'])
			if self.deterministic_eval:
				self.evaluate(self.locals['self'], deterministic=True)
		return True
	
	def evaluate(self, agent, deterministic=False):
		if self.eval_episodes > 0:
			env = self.eval_env
			with torch.no_grad():
				success, rews = [], []
				rew_total, total_ep = 0, 0
				rew_ep = np.zeros(self.num_eval_env)
				for i in range(self.eval_episodes):
					obs = env.reset()
					success_i = np.zeros(obs.shape[0])
					r = []
					for _ in range(self.max_steps):
						if self.algorithm == 'dsrl_sac':
							action, _ = agent.predict(obs, deterministic=deterministic)
						elif self.algorithm == 'dsrl_na':
							action, _ = agent.predict_diffused(obs, deterministic=deterministic)
						next_obs, reward, done, info = env.step(action)
						obs = next_obs
						rew_ep += reward
						rew_total += sum(rew_ep[done])
						rew_ep[done] = 0 
						total_ep += np.sum(done)
						success_i[reward > -self.rew_offset] = 1
						r.append(reward)
					success.append(success_i.mean())
					rews.append(np.mean(np.array(r)))
					print(f'eval episode {i} at timestep {self.total_timesteps}')
				success_rate = np.mean(success)
				if total_ep > 0:
					avg_rew = rew_total / total_ep
				else:
					avg_rew = 0
				if self.use_wandb:
					name = 'eval'
					if deterministic:
						wandb.log({
							f"{name}/success_rate_deterministic": success_rate,
							f"{name}/reward_deterministic": avg_rew,
						}, step=self.log_count)
					else:
						wandb.log({
							f"{name}/success_rate": success_rate,
							f"{name}/reward": avg_rew,
							f"{name}/timesteps": self.total_timesteps,
						}, step=self.log_count)
				return avg_rew
		return None

	def _maybe_run_trait_sweeps(self, agent):
		if not self.trait_logging_enabled or self.eval_env is None:
			return
		sweep_interval = int(self.trait_logging.get("sweep_interval", 1))
		if sweep_interval <= 0 or (self._eval_count % sweep_interval) != 0:
			return
		self._run_trait_sweep(agent)
		if self.trait_logging.get("cross_mask_eval", True):
			self._run_cross_mask_eval(agent)
		self._restore_eval_traits()

	def _restore_eval_traits(self):
		if self.eval_env is None:
			return
		self.eval_env.env_method("clear_traits")
		self._apply_current_trait_mask()

	def _set_eval_traits(self, values, mask):
		if self.eval_env is None:
			return
		self.eval_env.env_method("set_traits", values=values, mask=mask)

	def _evaluate_with_traits(self, agent, values, mask, episodes, deterministic):
		self._set_eval_traits(values, mask)
		env = self.eval_env
		with torch.no_grad():
			rew_total, total_ep = 0.0, 0
			rew_ep = np.zeros(self.num_eval_env)
			for _ in range(episodes):
				obs = env.reset()
				for _ in range(self.max_steps):
					if self.algorithm == 'dsrl_sac':
						action, _ = agent.predict(obs, deterministic=deterministic)
					elif self.algorithm == 'dsrl_na':
						action, _ = agent.predict_diffused(obs, deterministic=deterministic)
					next_obs, reward, done, info = env.step(action)
					obs = next_obs
					rew_ep += reward
					rew_total += sum(rew_ep[done])
					rew_ep[done] = 0
					total_ep += np.sum(done)
			if total_ep > 0:
				avg_rew = rew_total / total_ep
			else:
				avg_rew = 0.0
		return float(avg_rew)

	def _trait_value_ranges(self):
		ranges = []
		for trait_def in self.trait_defs:
			v_min = trait_def.get("value_min", None)
			v_max = trait_def.get("value_max", None)
			if v_min is None or v_max is None:
				return None
			v_min_arr = np.asarray(v_min, dtype=np.float32)
			v_max_arr = np.asarray(v_max, dtype=np.float32)
			if v_min_arr.ndim != 0 or v_max_arr.ndim != 0:
				return None
			ranges.append((float(v_min_arr), float(v_max_arr)))
		return ranges

	def _run_trait_sweep(self, agent):
		if not self.use_wandb:
			return
		ranges = self._trait_value_ranges()
		if ranges is None:
			return
		points = int(self.trait_logging.get("sweep_points", 5))
		max_points = int(self.trait_logging.get("sweep_max_points", 25))
		episodes = int(self.trait_logging.get("sweep_episodes", 2))
		deterministic = bool(self.trait_logging.get("sweep_deterministic", True))
		grid_axes = [np.linspace(v_min, v_max, points) for v_min, v_max in ranges]
		grid = np.array(np.meshgrid(*grid_axes)).T.reshape(-1, len(grid_axes))
		if grid.shape[0] > max_points:
			idx = np.random.choice(grid.shape[0], size=max_points, replace=False)
			grid = grid[idx]
		table = wandb.Table(columns=[*self.trait_names, "return"])
		sweep_values = []
		sweep_returns = []
		mask = [1.0] * len(self.trait_names)
		for vals in grid:
			avg_rew = self._evaluate_with_traits(agent, vals.tolist(), mask, episodes, deterministic)
			table.add_data(*[float(v) for v in vals], float(avg_rew))
			sweep_values.append([float(v) for v in vals])
			sweep_returns.append(float(avg_rew))
		wandb.log({"eval/trait_sweep": table}, step=self.log_count)
		if self.trait_logging_auto_plots:
			if len(self.trait_names) == 2:
				heatmap = wandb.plot.heatmap(
					table,
					self.trait_names[0],
					self.trait_names[1],
					"return",
					title="eval/trait_sweep_heatmap",
				)
				wandb.log({"eval/trait_sweep_heatmap": heatmap}, step=self.log_count)
			for i, name in enumerate(self.trait_names):
				by_value = {}
				for vals, ret in zip(sweep_values, sweep_returns):
					by_value.setdefault(vals[i], []).append(ret)
				x_vals = sorted(by_value.keys())
				y_vals = [float(np.mean(by_value[x])) for x in x_vals]
				line_table = wandb.Table(
					data=[[x, y] for x, y in zip(x_vals, y_vals)],
					columns=[name, "return"],
				)
				line = wandb.plot.line(
					line_table,
					x=name,
					y="return",
					title=f"eval/trait_elasticity_{name}",
				)
				wandb.log({f"eval/trait_elasticity_{name}": line}, step=self.log_count)

	def _run_cross_mask_eval(self, agent):
		if not self.use_wandb:
			return
		ranges = self._trait_value_ranges()
		if ranges is None:
			return
		episodes = int(self.trait_logging.get("cross_mask_episodes", 2))
		deterministic = bool(self.trait_logging.get("cross_mask_deterministic", True))
		values = [(v_min + v_max) / 2.0 for v_min, v_max in ranges]
		masks = []
		if self.trait_schedule and self.trait_schedule.get("phases", None):
			for phase in self.trait_schedule["phases"]:
				mask = phase.get("mask", None)
				if mask is not None and mask not in masks:
					masks.append(mask)
		if not masks:
			masks = [[1.0] * len(self.trait_names)]
		table = wandb.Table(columns=["mask", "return"])
		for mask in masks:
			avg_rew = self._evaluate_with_traits(agent, values, mask, episodes, deterministic)
			mask_key = "_".join(str(int(m)) for m in mask)
			wandb.log({f"eval/mask_{mask_key}_reward": avg_rew}, step=self.log_count)
			table.add_data(mask_key, float(avg_rew))
		wandb.log({"eval/mask_sweep": table}, step=self.log_count)
		if self.trait_logging_auto_plots:
			bar = wandb.plot.bar(
				table,
				"mask",
				"return",
				title="eval/mask_sweep_bar",
			)
			wandb.log({"eval/mask_sweep_bar": bar}, step=self.log_count)

	def set_timesteps(self, timesteps):
		self.total_timesteps = timesteps



def collect_rollouts(model, env, num_steps, base_policy, cfg):
	obs = env.reset()
	for i in range(num_steps):
		noise = torch.randn(cfg.env.n_envs, cfg.act_steps, cfg.action_dim).to(device=cfg.device)
		if cfg.algorithm == 'dsrl_sac':
			noise[noise < -cfg.train.action_magnitude] = -cfg.train.action_magnitude
			noise[noise > cfg.train.action_magnitude] = cfg.train.action_magnitude
		action = base_policy(torch.tensor(obs, device=cfg.device, dtype=torch.float32), noise)
		next_obs, reward, done, info = env.step(action)
		if cfg.algorithm == 'dsrl_na':
			action_store = action
		elif cfg.algorithm == 'dsrl_sac':
			action_store = noise.detach().cpu().numpy()
		action_store = action_store.reshape(-1, action_store.shape[1] * action_store.shape[2])
		if cfg.algorithm == 'dsrl_sac':
			action_store = model.policy.scale_action(action_store)
		model.replay_buffer.add(
				obs=obs,
				next_obs=next_obs,
				action=action_store,
				reward=reward,
				done=done,
				infos=info,
			)
		obs = next_obs
	model.replay_buffer.final_offline_step()
	


def load_offline_data(model, offline_data_path, n_env):
	# this function should only be applied with dsrl_na
	offline_data = np.load(offline_data_path)
	obs = offline_data['states']
	next_obs = offline_data['states_next']
	actions = offline_data['actions']
	rewards = offline_data['rewards']
	terminals = offline_data['terminals']
	for i in range(int(obs.shape[0]/n_env)):
		model.replay_buffer.add(
					obs=obs[n_env*i:n_env*i+n_env],
					next_obs=next_obs[n_env*i:n_env*i+n_env],
					action=actions[n_env*i:n_env*i+n_env],
					reward=rewards[n_env*i:n_env*i+n_env],
					done=terminals[n_env*i:n_env*i+n_env],
					infos=[{}] * n_env,
				)
	model.replay_buffer.final_offline_step()
