# Trait-Aware Diffusion Trait Steering (TADSRL)
This repository is a research fork of DSRL that focuses on making a frozen diffusion policy steerable at test time by conditioning the noise policy on explicit traits. It keeps the diffusion policy fixed and learns a trait-aware noise policy with reward shaping.

Core idea:
```
r = base_reward + sum_i m_i * lambda_i * r_i(...)
```
where `t` is a vector of trait values and `m` is a binary mask that turns traits on/off.

## Setup
1. Clone this repository
```
git clone --recurse-submodules <this-repo>
cd diffusion-trait-steering
```
2. Create a conda environment
```
conda create -n tadsrl python=3.9 -y
conda activate tadsrl
```
3. Install DPPO (diffusion policies)
```
cd dppo
pip install -e .
pip install -e .[gym]
cd ..
```
4. Install Stable Baselines3 (DSRL implementation)
```
cd stable-baselines3
pip install -e .
cd ..
```

Download diffusion policy checkpoints for DSRL from the original project and place them in `./dppo/log`:
https://drive.google.com/drive/folders/1kzC49RRFOE7aTnJh_7OvJ1K5XaDmtuh1

## Trait-Aware Training (TADSRL)
Traits and schedules live in `cfg/gym/dsrl_walker.yaml` under `traits`. Trait reward functions live in `traits.py`.

Run Walker2d training:
```
python train_dsrl.py --config-path=cfg/gym --config-name=dsrl_walker.yaml
```

### Define traits (Python)
Each trait reward is a Python function that receives raw (unnormalized) observations:
```
def thigh_gap(raw_obs, info):
    return reward, {"gap": gap}
```
Traits are registered by name in `traits.py`.

### Base reward override (optional)
You can replace the environment reward per step:
```
traits:
  base_reward_fn: healthy_reward
```
Base reward functions also live in `traits.py`.

### Phased mask training
Train traits incrementally with a mask schedule:
```
traits:
  schedule:
    min_steps: 250000
    patience: 3
    min_delta: 0.0
    phases:
      - mask: [1, 0]
      - mask: [0, 1]
      - mask: [1, 1]
```

## Inference with Traits
Use `run_inference.py` to set arbitrary trait values and masks at test time:
```
python run_inference.py --config-path=cfg/gym --config-name=dsrl_walker.yaml \
  model_path=/abs/path/to/ft_policy_XXXX_steps.zip \
  trait_values=[0.6,1.2] trait_mask=[1,1] eval_episodes=5
```

To record videos:
```
python run_inference.py --config-path=cfg/gym --config-name=dsrl_walker.yaml \
  model_path=/abs/path/to/ft_policy_XXXX_steps.zip \
  record_video=true video_dir=videos_inference video_episodes=2
```

## Trait-Aware Logging (W&B)
Logging is configured under `traits.logging` in `cfg/gym/dsrl_walker.yaml`. It includes:
- Per-trait reward/value/mask statistics and shaping delta.
- Action norm stats and correlation with traits.
- Eval sweeps over trait values and cross-mask evals.
- Auto-generated W&B plots (heatmap, elasticity, mask bar).

## Notes
- Trait values are sampled per episode. Mask phases control which traits are active.
- `speed_ref` (for Walker2d speed trait) should be estimated from the frozen policy and set in `traits.py`.

## Acknowledgements
This fork builds on [DSRL](https://diffusion-steering.github.io), [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3), and [DPPO](https://github.com/irom-princeton/dppo).

## Citation (DSRL)
```
@article{wagenmaker2025steering,
  author    = {Wagenmaker, Andrew and Nakamoto, Mitsuhiko and Zhang, Yunchu and Park, Seohong and Yagoub, Waleed and Nagabandi, Anusha and Gupta, Abhishek and Levine, Sergey},
  title     = {Steering Your Diffusion Policy with Latent Space Reinforcement Learning},
  journal   = {Conference on Robot Learning (CoRL)},
  year      = {2025},
}
```
