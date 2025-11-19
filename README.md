<div align="center">

# Steering Your Diffusion Policy with Latent Space Reinforcement Learning (DSRL)

## [[website](https://diffusion-steering.github.io)]      [[pdf](https://arxiv.org/pdf/2506.15799)]

</div>


<p align="center">
  <a href="https://colinqiyangli.github.io/qc/">
    <img alt="teaser figure" src="./assets/teaser.png" width="90%">
  </a>
</p>

## Overview
Diffusion steering via reinforcement learning (DSRL) is a lightweight and efficient method for RL finetuning of diffusion and flow policies. Rather than modifying the weights of the diffusion/flow policy, DSRL instead modifies the noise distribution sampled from to begin the denoising process.


## Installation
1. Clone repository
```
git clone --recurse-submodules git@github.com:ajwagen/dsrl.git
cd dsrl
```
2. Create conda environment
```
conda create -n dsrl python=3.9 -y
conda activate dsrl
```
3. Install our fork of DPPO 
```
cd dppo
pip install -e .
pip install -e .[robomimic]
pip install -e .[gym]
cd ..
```
4. Install our fork of Stable Baselines3
```
cd stable-baselines3
pip install -e .
cd ..
```
The diffusion policy checkpoints for the Robomimic and Gym experiments can be found [here](https://drive.google.com/drive/folders/1kzC49RRFOE7aTnJh_7OvJ1K5XaDmtuh1?usp=share_link). Download the contents of this folder and place in `./dppo/log`.

## Running DSRL
To run DSRL on Robomimic, call
```
python train_dsrl.py --config-path=cfg/robomimic --config-name=dsrl_can.yaml
```
where `dsrl_can.yaml` is set to the config file for the desired task. Similarly, for Gym, call
```
python train_dsrl.py --config-path=cfg/gym --config-name=dsrl_hopper.yaml
```
where `dsrl_hopper.yaml` is set to the config file for the desired task.

## Applying DSRL to new settings
It is straightforward to apply DSRL to new settings. Doing this typically requires:
- Access to a diffusion or flow policy with the ability to control the noise initializing the denoising process. Note that if using a diffusion policy it must be sampled from with DDIM sampling.
-  In the case of `DSRL-NA`, the diffusion/flow policy is passed to the `SACDiffusionNoise` algorithm, and then this algorithm is simply run on a standard gym environment. 
- In the case of `DSRL-SAC`, it is recommended that you write a wrapper around your environment which transforms the action space from the original action space to the noise space of the diffusion/noise policy. Here, the noise action given to the environment wrapper is then denoised through the diffusion policy, and this denoised action is played on the original environment, all of which is performed within the environment wrapper. See the `DiffusionPolicyEnvWrapper` in `env_utils.py` for an example of this. 



### Tips for hyperparameter tuning
The following may be helpful in tuning DSRL on new settings:
- Typically the key hyperparameters to tune are `action_magnitude` and `utd`. `action_magnitude` controls how large a noise value can be played in the noise action space, and `utd` is the number of gradient steps taken per update. Typically setting `action_magnitude` around 1.5 and `utd` around 20 performs effectively, but for best performance these should be tuned on new environments.
- As described in the paper, there are two primary variants of the algorithm: `DSRL-NA` and `DSRL-SAC`. `DSRL-SAC` simply runs `SAC` with the action space the noise space of the diffusion policy, while `DSRL-NA` distills a Q-function learned on the original action space (see the paper for further details). In general `DSRL-NA` is more sample efficient and should be preferred to `DSRL-SAC`, however `DSRL-SAC` is somewhat more computationally efficient in settings where speed is critical.
- DSRL typically performs best when using relatively large actor and critic networks. A reasonable value here is typically using a 3-layer MLP of width 2048. Tuning the size can sometimes lead to further gains. 

## Acknowledgements
Our implementation of DSRL is built on top of [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3). For our diffusion policy implementation, we utilize the implementation given in the [DPPO](https://github.com/irom-princeton/dppo) codebase.

## Citation
```
@article{wagenmaker2025steering,
  author    = {Wagenmaker, Andrew and Nakamoto, Mitsuhiko and Zhang, Yunchu and Park, Seohong and Yagoub, Waleed and Nagabandi, Anusha and Gupta, Abhishek and Levine, Sergey},
  title     = {Steering Your Diffusion Policy with Latent Space Reinforcement Learning},
  journal   = {Conference on Robot Learning (CoRL)},
  year      = {2025},
}
```
