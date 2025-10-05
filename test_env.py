import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print("Changed working directory to:", os.getcwd())

from src.environment.dogfight_env import DogfightEnv

env = DogfightEnv()
obs, _ = env.reset()

for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print('Episode ended')
        break

print('Test passed. Final obs:', obs[:4], 'Reward:', reward)