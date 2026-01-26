from typing import Callable, Dict, Tuple

import numpy as np


def thigh_gap(raw_obs: np.ndarray, info: dict) -> Tuple[float, dict]:
	"""
	Penalize the squared distance to a target joint angle difference.
	Trait value is the desired gap.
	API: https://gymnasium.farama.org/environments/mujoco/walker2d/
	"""
	left_idx, right_idx = 5, 2
	diff = float(abs(raw_obs[left_idx] - raw_obs[right_idx]))
	target = float(info.get("trait_value", 0.0))
	error = diff - target
	reward = -(error ** 2)
	metrics = {"gap": diff, "target": target}
	return reward, metrics


def thigh_gap_max(raw_obs: np.ndarray, info: dict) -> Tuple[float, dict]:
	"""
	Reward the absolute difference between thigh joint angles directly.
	Use squared difference to strongly incentivize larger gaps.
	"""
	left_idx, right_idx = 5, 2
	diff = float(abs(raw_obs[left_idx] - raw_obs[right_idx]))
	weight = float(info.get("trait_value", 1.0))
	# Squared term makes large gaps significantly more rewarding
	reward = (diff ** 2) * weight
	metrics = {"gap": diff, "weight": weight}
	return reward, metrics


def stride_max(raw_obs: np.ndarray, info: dict) -> Tuple[float, dict]:
	"""
	Reward a combination of speed and actual horizontal foot separation.
	Uses MuJoCo internal geom_xpos if available, otherwise falls back to thigh angle proxy.
	"""
	speed = float(raw_obs[8]) # Forward velocity
	
	# Try to get actual foot separation from info (injected by TraitWrapperGym)
	foot_sep = info.get("foot_sep", None)
	if foot_sep is not None:
		spread = float(foot_sep)
	else:
		# Fallback to proxy if MuJoCo internals aren't accessible
		spread = float(abs(np.sin(raw_obs[2]) - np.sin(raw_obs[5])))
	
	weight = float(info.get("trait_value", 1.0))
	# Reward is actual separation scaled by speed
	reward = (spread * speed) * weight
	metrics = {"stride_spread": spread, "speed": speed}
	return reward, metrics


def stride_td(raw_obs: np.ndarray, info: dict) -> Tuple[float, dict]:
	"""
	Reward real stride length only at the moment of touchdown.
	Uses a quadratic boost for larger strides and top-K filtering to discourage shuffling.
	Enforces alternating steps by only rewarding spatial overtaking.
	"""
	speed = max(float(raw_obs[8]), 0.0)
	
	# Configuration for the "push"
	s_min = 0.0 # Ignore steps smaller than 25cm (filters shuffles)
	s_cap = 1.50 # Cap the progress at 70cm (requested by user)
	alpha = 3.0  # Quadratic boost factor (user suggested 1-5)

	# Only pay on touchdown
	r_stride = 0.0
	# Collect all valid touchdown strides in the chunk
	# These are now 'overtake' strides (current_foot_x - other_foot_last_td_x)
	all_strides = info.get("strides_r", []) + info.get("strides_l", [])
	
	# Top-K filtering: only reward the best 2 strides in the chunk
	best_strides = sorted(all_strides, reverse=True)[:2]
	
	for s in best_strides:
		# x is excess: ranges from 0 to (s_cap - s_min)
		x = np.clip(s, s_min, s_cap) - s_min
		if x > 0:
			# Quadratic reward: x + alpha * x^2
			r_stride += (x + alpha * (x ** 2))

	# Gate by speed rather than multiply to avoid incentive coupling
	v_min = 0.1  # Minimum forward speed threshold (m/s)
	if speed < v_min:
		return 0.0, {"speed": speed, "stride_td_quad": r_stride, "speed_gated": 1.0}
	
	weight = float(info.get("trait_value", 1.0))
	reward = weight * r_stride
	return reward, {"speed": speed, "stride_td_quad": r_stride, "speed_gated": 0.0}
   



def reach_td(raw_obs: np.ndarray, info: dict) -> Tuple[float, dict]:
	"""
	Reward landing the swing foot far ahead of the pelvis.
	Directly pushes the "stretch" behavior.
	Uses top-K filtering to reward only the best reaches in a chunk.
	"""
	reach_min = 0.0
	reach_cap = 0.50

	r = 0.0
	# Collect all valid reach values from the chunk
	all_reaches = info.get("reaches_r", []) + info.get("reaches_l", [])
	
	# Top-K filtering: only reward the best 2 reaches in the chunk
	best_reaches = sorted(all_reaches, reverse=True)[:2]
	
	for reach in best_reaches:
		r += np.clip(reach, reach_min, reach_cap) - reach_min

	weight = float(info.get("trait_value", 1.0))
	return r * weight, {"reach_td": float(r)}


def landing_sep_max(raw_obs: np.ndarray, info: dict) -> Tuple[float, dict]:
	"""
	Reward landing the swing foot far from the stationary foot.
	Uses top-K filtering and thresholding to target shuffling.
	"""
	# Configuration for the "don't land close" push
	d_min = 0.0 # Ignore separations smaller than 25cm
	d_cap = 100.0 # High cap as requested
	
	r = 0.0
	all_seps = info.get("seps_r", []) + info.get("seps_l", [])
	
	# Top-K filtering: only reward the best 2 separations in the chunk
	best_seps = sorted(all_seps, reverse=True)[:2]
	
	for sep in best_seps:
		# x is excess separation: ranges from 0 to (d_cap - d_min)
		x = np.clip(sep, d_min, d_cap) - d_min
		r += x
		
	weight = float(info.get("trait_value", 1.0))
	return r * weight, {"landing_sep_capped": r}


def flight_penalty(raw_obs: np.ndarray, info: dict) -> Tuple[float, dict]:
	"""
	Penalize the agent for having both feet off the ground (flight phase).
	Uses actual contact data if available, otherwise falls back to torso height proxy.
	"""
	flight = info.get("flight", None)
	if flight is not None:
		# Use boolean contact signal
		penalty = -10.0 if flight else 0.0
	else:
		# Fallback to torso height (z) proxy
		z = float(raw_obs[0])
		penalty = -((max(z - 1.3, 0.0)) ** 2)
	
	weight = float(info.get("trait_value", 1.0))
	return penalty * weight, {}


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


def speed_max(raw_obs: np.ndarray, info: dict) -> Tuple[float, dict]:
	"""
	Reward forward speed directly.
	Trait value is a weight/multiplier for the speed reward.
	"""
	speed = float(raw_obs[8])
	# trait_value here acts as a scaling factor if needed, 
	# but the primary goal is maximizing speed.
	weight = float(info.get("trait_value", 1.0))
	reward = speed * weight
	metrics = {"speed": speed, "weight": weight}
	return reward, metrics


TRAIT_REWARD_FNS: Dict[str, Callable[[np.ndarray, dict], Tuple[float, dict]]] = {
	"thigh_gap": thigh_gap,
	"thigh_gap_max": thigh_gap_max,
	"stride_max": stride_max,
	"stride_td": stride_td,
	"reach_td": reach_td,
	"landing_sep_max": landing_sep_max,
	"flight_penalty": flight_penalty,
	"speed_target": speed_target,
	"speed_max": speed_max,
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
