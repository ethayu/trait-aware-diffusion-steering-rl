import gym
import d4rl
try:
    env = gym.make('walker2d-medium-v2')
    model = env.unwrapped.model
    print("Geom names:")
    for i in range(model.ngeom):
        print(f"{i}: {model.geom_id2name(i)}")
except Exception as e:
    print(f"Error: {e}")

