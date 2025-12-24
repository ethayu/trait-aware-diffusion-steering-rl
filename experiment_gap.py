import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt

# configuration
MODEL_PATH = "/home/asethi04/proj/diffusion-trait-steering/logs/gym-dsrl/gym_walker_dsrl_2025-12-24_02-55-54_1/2025-12-24_02-55-54_1/checkpoint/ft_policy_40000_steps.zip"
GAP_TARGETS = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21]
FIXED_SPEED = 1.0
EPISODES = 5
GPU = "5"

results = []

for gap in GAP_TARGETS:
    print(f"\n>>> Testing Thigh Gap Target: {gap}")
    cmd = [
        "python", "run_inference.py",
        "--config-name=dsrl_walker.yaml",
        f"model_path={MODEL_PATH}",
        f"trait_values=[{gap}, {FIXED_SPEED}]",
        "trait_mask=[1, 1]",
        f"eval_episodes={EPISODES}",
        "deterministic_eval=True",
        "record_video=False"
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = GPU
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    
    mean_gap = None
    mean_return = None
    
    for line in process.stdout:
        print(line, end="")
        if "mean_return=" in line:
            try:
                mean_return = float(line.split("mean_return=")[1].split()[0])
            except (IndexError, ValueError):
                pass
        if "metric_trait_thigh_gap_gap=" in line:
            try:
                mean_gap = float(line.split("metric_trait_thigh_gap_gap=")[1])
            except (IndexError, ValueError):
                pass
            
    process.wait()
    results.append({"target": gap, "actual": mean_gap, "return": mean_return})

# plotting
targets = [r["target"] for r in results]
actuals = [r["actual"] for r in results if r["actual"] is not None]
valid_targets_actual = [r["target"] for r in results if r["actual"] is not None]
returns = [r["return"] for r in results if r["return"] is not None]
valid_targets_return = [r["target"] for r in results if r["return"] is not None]

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
if actuals:
    plt.plot(valid_targets_actual, actuals, 'o-')
plt.xlabel("Thigh Gap Target")
plt.ylabel("Actual Thigh Gap")
plt.title("Gap Interpolation")
plt.grid(True)

plt.subplot(1, 2, 2)
if returns:
    plt.plot(valid_targets_return, returns, 'o-r')
plt.xlabel("Thigh Gap Target")
plt.ylabel("Mean Return")
plt.title("Return vs Gap Target")
plt.grid(True)

plt.tight_layout()
plt.savefig("experiment_gap_results.png")
print("\nDone! Results saved to experiment_gap_results.png")

