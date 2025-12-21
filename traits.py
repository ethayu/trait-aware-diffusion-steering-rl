from typing import Callable, Dict, Tuple

import numpy as np


def thigh_gap(raw_obs: np.ndarray, info: dict) -> Tuple[float, dict]:
	"""
	Penalize the difference between two joint angles beyond a threshold.
	Trait value is the max allowed gap.
	API: https://gymnasium.farama.org/environments/mujoco/walker2d/
	"""
	left_idx, right_idx = 5, 2
	diff = float(abs(raw_obs[left_idx] - raw_obs[right_idx]))
	threshold = float(info.get("trait_value", 0.0))
	penalty = max(0.0, diff - threshold)
	reward = -(penalty ** 2)
	metrics = {"gap": diff, "target": threshold}
	return reward, metrics


def speed_target(raw_obs: np.ndarray, info: dict) -> Tuple[float, dict]:
	"""
	Penalize deviation from a target forward speed.
	Trait value is a multiplier of speed_ref.
	API: https://gymnasium.farama.org/environments/mujoco/walker2d/
	"""
	speed = float(raw_obs[8])
	speed_ref = 1.0 # TODO: @aneesh figure out speed_ref
	trait_value = float(info.get("trait_value", 0.0))
	target = trait_value * speed_ref
	error = speed - target
	reward = -(error ** 2)
	metrics = {"speed": speed, "target": target}
	return reward, metrics


TRAIT_REWARD_FNS: Dict[str, Callable[[np.ndarray, dict], Tuple[float, dict]]] = {
	"thigh_gap": thigh_gap,
	"speed_target": speed_target,
}

def healthy_reward(raw_obs: np.ndarray, info: dict) -> float:
	"""
	Return 1.0 for every step.
	"""
	return 1.0


BASE_REWARD_FNS: Dict[str, Callable[[np.ndarray, dict], float]] = {
	"healthy_reward": healthy_reward,
}


def get_trait_reward_fn(name: str) -> Callable[[np.ndarray, dict], Tuple[float, dict]]:
	if name not in TRAIT_REWARD_FNS:
		raise KeyError(f"Unknown trait name '{name}'. Define it in traits.py.")
	return TRAIT_REWARD_FNS[name]


def get_base_reward_fn(name: str) -> Callable[[np.ndarray, dict], float]:
	if name not in BASE_REWARD_FNS:
		raise KeyError(f"Unknown base_reward_fn '{name}'. Define it in traits.py.")
	return BASE_REWARD_FNS[name]
