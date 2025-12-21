import argparse

import gym
import d4rl  # noqa: F401
import d4rl.gym_mujoco  # noqa: F401


def _joint_type_dofs(jnt_type: int):
	if jnt_type == 0:  # free
		return 7, 6
	if jnt_type == 1:  # ball
		return 4, 3
	if jnt_type in (2, 3):  # slide, hinge
		return 1, 1
	return 1, 1


def _build_qpos_qvel_names(model):
	qpos_names = [""] * model.nq
	qvel_names = [""] * model.nv
	for j in range(model.njnt):
		if hasattr(model, "joint_id2name"):
			name = model.joint_id2name(j)
		else:
			name = model.joint_names[j] if hasattr(model, "joint_names") else None
		if name is None:
			name = f"joint{j}"
		qpos_start = int(model.jnt_qposadr[j])
		qvel_start = int(model.jnt_dofadr[j])
		qpos_count, qvel_count = _joint_type_dofs(int(model.jnt_type[j]))
		for i in range(qpos_count):
			if qpos_start + i < len(qpos_names):
				qpos_names[qpos_start + i] = f"{name}:qpos{i}"
		for i in range(qvel_count):
			if qvel_start + i < len(qvel_names):
				qvel_names[qvel_start + i] = f"{name}:qvel{i}"
	return qpos_names, qvel_names


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--env", default="walker2d-medium-v2")
	args = parser.parse_args()

	env = gym.make(args.env)
	model = env.unwrapped.model
	qpos_names, qvel_names = _build_qpos_qvel_names(model)

	print(f"env={args.env} nq={model.nq} nv={model.nv}")
	print("Observation mapping (obs -> qpos[1:], qvel):")
	obs_idx = 0
	for qpos_idx in range(1, model.nq):
		name = qpos_names[qpos_idx] or f"qpos{qpos_idx}"
		print(f"obs[{obs_idx}] = qpos[{qpos_idx}] ({name})")
		obs_idx += 1
	for qvel_idx in range(model.nv):
		name = qvel_names[qvel_idx] or f"qvel{qvel_idx}"
		print(f"obs[{obs_idx}] = qvel[{qvel_idx}] ({name})")
		obs_idx += 1
	env.close()


if __name__ == "__main__":
	main()
