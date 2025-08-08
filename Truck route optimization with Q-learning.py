import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional

@dataclass
class TruckState:
    """Represents the current state of a mining truck"""
    position: Tuple[int, int]  # (x, y) coordinates
    fuel_level: float  # 0.0 to 1.0
    cargo_load: float  # 0.0 to 1.0
    destination: Tuple[int, int]  # target position

class MiningEnvironment:
    """
    Simplified mining environment for route optimization
    
    This represents our RL Environment - where the truck operates
    """
    
    def __init__(self, width=10, height=10):
        self.width = width
        self.height = height
        
        # Create mine layout
        self.obstacles = set()  # Rocks, equipment, etc.
        self.fuel_stations = [(2, 2), (7, 8)]  # Refueling points
        self.loading_zones = [(1, 1), (8, 1)]  # Where trucks get loaded
        self.dump_zones = [(1, 8), (8, 8)]     # Where trucks dump cargo
        
        # Add some obstacles
        self.obstacles.update([(3, 3), (4, 4), (5, 5), (6, 6)])
        
        # Possible actions: up, down, left, right, refuel, load, dump
        self.actions = ['up', 'down', 'left', 'right', 'refuel', 'load', 'dump']
        
    def get_valid_actions(self, state: TruckState) -> List[str]:
        """Get valid actions from current state"""
        valid = []
        x, y = state.position
        
        # Movement actions
        if y > 0 and (x, y-1) not in self.obstacles:
            valid.append('up')
        if y < self.height-1 and (x, y+1) not in self.obstacles:
            valid.append('down')
        if x > 0 and (x-1, y) not in self.obstacles:
            valid.append('left')
        if x < self.width-1 and (x+1, y) not in self.obstacles:
            valid.append('right')
            
        # Special actions
        if state.position in self.fuel_stations and state.fuel_level < 1.0:
            valid.append('refuel')
        if state.position in self.loading_zones and state.cargo_load < 1.0:
            valid.append('load')
        if state.position in self.dump_zones and state.cargo_load > 0.0:
            valid.append('dump')
            
        return valid
    
    def step(self, state: TruckState, action: str) -> Tuple[TruckState, float, bool]:
        """
        Execute action and return new state, reward, and done flag
        
        This is the core RL interaction - Agent takes action, Environment responds
        """
        x, y = state.position
        new_state = TruckState(
            position=state.position,
            fuel_level=state.fuel_level,
            cargo_load=state.cargo_load,
            destination=state.destination
        )
        
        reward = 0.0
        done = False
        
        # Fuel consumption for movement
        fuel_cost = 0.02
        if state.cargo_load > 0.5:  # Heavier loads use more fuel
            fuel_cost *= 1.5
            
        # Execute action
        if action == 'up':
            new_state.position = (x, y-1)
            new_state.fuel_level = max(0, state.fuel_level - fuel_cost)
        elif action == 'down':
            new_state.position = (x, y+1)
            new_state.fuel_level = max(0, state.fuel_level - fuel_cost)
        elif action == 'left':
            new_state.position = (x-1, y)
            new_state.fuel_level = max(0, state.fuel_level - fuel_cost)
        elif action == 'right':
            new_state.position = (x+1, y)
            new_state.fuel_level = max(0, state.fuel_level - fuel_cost)
        elif action == 'refuel':
            new_state.fuel_level = 1.0
            reward += 5  # Small reward for maintaining fuel
        elif action == 'load':
            new_state.cargo_load = 1.0
            reward += 10  # Reward for loading cargo
        elif action == 'dump':
            new_state.cargo_load = 0.0
            reward += 50  # Big reward for completing delivery!
            done = True
            
        # Calculate rewards
        if new_state.fuel_level <= 0:
            reward -= 100  # Big penalty for running out of fuel
            done = True
            
        # Distance-based reward (encourage moving toward destination)
        old_dist = abs(x - state.destination[0]) + abs(y - state.destination[1])
        new_dist = abs(new_state.position[0] - state.destination[0]) + abs(new_state.position[1] - state.destination[1])
        
        if new_dist < old_dist:
            reward += 1  # Small reward for getting closer
        else:
            reward -= 0.5  # Small penalty for moving away
            
        # Time penalty (encourage efficiency)
        reward -= 1
        
        return new_state, reward, done

class QLearningAgent:
    """
    Q-Learning Agent for mining truck route optimization
    
    Q-Learning learns the value of taking each action in each state
    """
    
    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.learning_rate = learning_rate    # How fast we learn (alpha)
        self.discount_factor = discount_factor  # How much we value future rewards (gamma)
        self.epsilon = epsilon                # Exploration rate
        
        # Q-table: maps (state, action) -> value
        self.q_table = defaultdict(lambda: defaultdict(float))
        
    def state_to_key(self, state: TruckState) -> str:
        """Convert state to string key for Q-table"""
        return f"{state.position}_{state.fuel_level:.1f}_{state.cargo_load:.1f}_{state.destination}"
    
    def choose_action(self, state: TruckState, valid_actions: List[str]) -> str:
        """
        Choose action using epsilon-greedy policy
        
        Epsilon-greedy balances exploration vs exploitation:
        - With probability epsilon: explore (random action)
        - With probability 1-epsilon: exploit (best known action)
        """
        if random.random() < self.epsilon:
            return random.choice(valid_actions)  # Explore
        else:
            # Exploit - choose best action
            state_key = self.state_to_key(state)
            best_action = max(valid_actions, 
                            key=lambda a: self.q_table[state_key][a])
            return best_action
    
    def update_q_value(self, state: TruckState, action: str, reward: float, 
                      next_state: TruckState, valid_next_actions: List[str], done: bool):
        """
        Update Q-value using Q-learning formula:
        Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        """
        state_key = self.state_to_key(state)
        next_state_key = self.state_to_key(next_state)
        
        current_q = self.q_table[state_key][action]
        
        if done:
            # No future rewards if episode is done
            target = reward
        else:
            # Future reward is discounted max Q-value of next state
            max_next_q = max([self.q_table[next_state_key][a] for a in valid_next_actions], 
                           default=0)
            target = reward + self.discount_factor * max_next_q
        
        # Update Q-value
        self.q_table[state_key][action] += self.learning_rate * (target - current_q)

def train_mining_truck(episodes=1000):
    """Train the mining truck using Q-learning"""
    
    env = MiningEnvironment()
    agent = QLearningAgent()
    
    # Track training progress
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(episodes):
        # Start each episode with truck at loading zone
        start_pos = random.choice(env.loading_zones)
        destination = random.choice(env.dump_zones)
        
        state = TruckState(
            position=start_pos,
            fuel_level=1.0,
            cargo_load=0.0,
            destination=destination
        )
        
        total_reward = 0
        steps = 0
        max_steps = 100  # Prevent infinite episodes
        
        while steps < max_steps:
            # Get valid actions
            valid_actions = env.get_valid_actions(state)
            if not valid_actions:
                break
                
            # Agent chooses action
            action = agent.choose_action(state, valid_actions)
            
            # Environment responds
            next_state, reward, done = env.step(state, action)
            
            # Agent learns from experience
            next_valid_actions = env.get_valid_actions(next_state)
            agent.update_q_value(state, action, reward, next_state, next_valid_actions, done)
            
            total_reward += reward
            state = next_state
            steps += 1
            
            if done:
                break
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        # Reduce exploration over time
        if episode % 100 == 0:
            agent.epsilon = max(0.01, agent.epsilon * 0.95)
            avg_reward = np.mean(episode_rewards[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            print(f"Episode {episode}: Avg Reward = {avg_reward:.2f}, Avg Length = {avg_length:.2f}")
    
    return agent, episode_rewards, episode_lengths

def visualize_training(rewards, lengths):
    """Visualize training progress"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot rewards
    ax1.plot(rewards)
    ax1.set_title('Training Progress: Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.grid(True)
    
    # Plot episode lengths
    ax2.plot(lengths)
    ax2.set_title('Training Progress: Episode Length')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps to Complete')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def test_trained_agent(agent, episodes=10):
    """Test the trained agent"""
    env = MiningEnvironment()
    
    print("\nTesting trained agent...")
    for episode in range(episodes):
        start_pos = random.choice(env.loading_zones)
        destination = random.choice(env.dump_zones)
        
        state = TruckState(
            position=start_pos,
            fuel_level=1.0,
            cargo_load=0.0,
            destination=destination
        )
        
        path = [state.position]
        total_reward = 0
        steps = 0
        
        # Use trained policy (no exploration)
        original_epsilon = agent.epsilon
        agent.epsilon = 0  # No exploration during testing
        
        while steps < 50:
            valid_actions = env.get_valid_actions(state)
            if not valid_actions:
                break
                
            action = agent.choose_action(state, valid_actions)
            next_state, reward, done = env.step(state, action)
            
            path.append(next_state.position)
            total_reward += reward
            state = next_state
            steps += 1
            
            if done:
                break
        
        agent.epsilon = original_epsilon  # Restore exploration rate
        
        print(f"Test Episode {episode + 1}: "
              f"Path length = {len(path)}, "
              f"Total reward = {total_reward:.2f}, "
              f"Success = {state.cargo_load == 0.0}")

if __name__ == "__main__":
    print("Training Mining Truck Route Optimization with Q-Learning...")
    print("=" * 60)
    
    # Train the agent
    trained_agent, rewards, lengths = train_mining_truck(episodes=1000)
    
    # Visualize training progress
    visualize_training(rewards, lengths)
    
    # Test the trained agent
    test_trained_agent(trained_agent)
    
    print("\nTraining complete! The truck has learned to:")
    print("1. Navigate around obstacles")
    print("2. Manage fuel efficiently")
    print("3. Complete delivery tasks")
    print("4. Optimize routes for minimal time and fuel consumption")