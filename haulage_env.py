import gym
from gym import spaces
import numpy as np

class HaulTruckEnv(gym.Env):
    """
    Custom Environment for training a mining haul truck using reinforcement learning.
    The truck moves on a grid to pick up a load and deliver it to a drop-off point.
    """

    def __init__(self, grid_size=5, max_steps=50):
        super(HaulTruckEnv, self).__init__()

        self.grid_size = grid_size     # Size of the mine layout (e.g., 5x5)
        self.max_steps = max_steps     # Maximum number of steps per episode

        # Define 4 possible actions:
        # 0 = up, 1 = right, 2 = down, 3 = left
        self.action_space = spaces.Discrete(4)

        # Observation = truck's x, y position + has_load (0 or 1)
        # For example, [2, 3, 1] means truck is at (2,3) and carrying a load
        self.observation_space = spaces.MultiDiscrete([grid_size, grid_size, 2])

        # Define where the truck picks up and drops off the load
        self.pickup_point = (0, 0)                        # Top-left corner
        self.dropoff_point = (grid_size - 1, grid_size - 1)  # Bottom-right corner

        self.reset()  # Initialize environment state

    def reset(self):
        """
        Reset the environment to the starting position:
        - Truck starts at (0, grid_size - 1) [top-right]
        - No load initially
        - Step count reset to 0
        """
        self.truck_pos = [0, self.grid_size - 1]
        self.has_load = 0
        self.steps = 0

        return np.array(self.truck_pos + [self.has_load])

    def step(self, action):
        """
        Perform an action (move) and return the next state, reward, done, and info.
        """
        # --- Move truck based on action ---
        if action == 0 and self.truck_pos[1] > 0:              # UP
            self.truck_pos[1] -= 1
        elif action == 1 and self.truck_pos[0] < self.grid_size - 1:  # RIGHT
            self.truck_pos[0] += 1
        elif action == 2 and self.truck_pos[1] < self.grid_size - 1:  # DOWN
            self.truck_pos[1] += 1
        elif action == 3 and self.truck_pos[0] > 0:             # LEFT
            self.truck_pos[0] -= 1

        # --- Reward system ---
        reward = -1  # Default time penalty for each move (to encourage faster delivery)

        # If the truck reaches the pickup point and it's empty
        if tuple(self.truck_pos) == self.pickup_point and self.has_load == 0:
            self.has_load = 1
            reward += 10  # Reward for picking up the load

        # If the truck reaches the drop-off point and it's carrying a load
        if tuple(self.truck_pos) == self.dropoff_point and self.has_load == 1:
            self.has_load = 0
            reward += 20  # Reward for successfully delivering the load

        # --- Bookkeeping ---
        self.steps += 1
        done = self.steps >= self.max_steps  # End if max steps reached

        # Return new state, reward earned, whether episode is done, and info (empty for now)
        return np.array(self.truck_pos + [self.has_load]), reward, done, {}

    def render(self, mode='human'):
        """
        Print the current state of the environment (visual debug mode).
        Shows where the truck is, with or without a load.
        """
        # Build a blank grid
        grid = [[" ." for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        # Truck's current location
        x, y = self.truck_pos
        grid[y][x] = " T" if self.has_load else " t"

        # Pickup and drop-off locations
        grid[self.pickup_point[1]][self.pickup_point[0]] = " P"
        grid[self.dropoff_point[1]][self.dropoff_point[0]] = " D"

        # Print the grid to console
        print("\n".join(["".join(row) for row in grid]))
        print(f"Truck at: {self.truck_pos}, Load: {self.has_load}, Steps: {self.steps}")
