from haulage_env import HaulTruckEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# Step 1: Initialize your custom environment
env = HaulTruckEnv()

# Step 2: Check if the environment follows Gym standards
check_env(env)

# Step 3: Initialize the PPO (Proximal Policy Optimization) agent
model = PPO("MlpPolicy", env, verbose=1)

# Step 4: Train the agent
model.learn(total_timesteps=10000)

# Step 5: Save the trained model
model.save("haul_truck_ppo")
print(" Model trained and saved as haul_truck_ppo.zip")
