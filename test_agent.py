from haulage_env import HaulTruckEnv
from stable_baselines3 import PPO

# Load trained model
model = PPO.load("haul_truck_ppo")

# Create the environment
env = HaulTruckEnv()

obs = env.reset()
done = False
total_reward = 0

while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    env.render()
    print("Reward:", reward)
    total_reward += reward

print("Total Reward Earned:", total_reward)
