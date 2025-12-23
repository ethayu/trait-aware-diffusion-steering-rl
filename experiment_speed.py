import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt

# configuration
MODEL_PATH = "/home/aneeshe/projects/diffusion-trait-steering/logs/gym-dsrl/gym_walker_dsrl_2025-12-22_18-36-41_1/2025-12-22_18-36-41_1/checkpoint/ft_policy_16000_steps.zip"
SPEED_TARGETS = [0.5, 0.8, 1.0, 1.2, 1.5]
FIXED_GAP = 0.6
EPISODES = 10
GPU = "5"

results = []

for speed in SPEED_TARGETS:
    print(f"\n>>> Testing Speed Multiplier: {speed}")
    cmd = [
        "python", "run_inference.py",
        "--config-name=dsrl_walker.yaml",
        f"model_path={MODEL_PATH}",
        f"trait_values=[{FIXED_GAP}, {speed}]",
        "trait_mask=[1, 1]",
        f"eval_episodes={EPISODES}",
        "deterministic_eval=True",
        "record_video=False"
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = GPU
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    
    mean_speed = None
    mean_return = None
    
    for line in process.stdout:
        print(line, end="")
        if "mean_return=" in line:
            try:
                mean_return = float(line.split("mean_return=")[1].split()[0])
            except (IndexError, ValueError):
                pass
        if "metric_trait_speed_target_speed=" in line:
            try:
                mean_speed = float(line.split("metric_trait_speed_target_speed=")[1])
            except (IndexError, ValueError):
                pass
            
    process.wait()
    results.append({"target": speed, "actual": mean_speed, "return": mean_return})

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
plt.xlabel("Speed Target Multiplier")
plt.ylabel("Actual Forward Speed")
plt.title("Speed Interpolation")
plt.grid(True)

plt.subplot(1, 2, 2)
if returns:
    plt.plot(valid_targets_return, returns, 'o-r')
plt.xlabel("Speed Target Multiplier")
plt.ylabel("Mean Return")
plt.title("Return vs Speed Target")
plt.grid(True)

plt.tight_layout()
plt.savefig("experiment_speed_results.png")
print("\nDone! Results saved to experiment_speed_results.png")

