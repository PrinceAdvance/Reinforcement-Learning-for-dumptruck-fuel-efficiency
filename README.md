# Mining Truck Route Optimization with Reinforcement Learning

A reinforcement learning system that teaches autonomous mining trucks to optimize their routes for fuel efficiency, delivery time, and operational safety.

## 🎯 Project Overview

This project demonstrates how Q-learning can be applied to real-world mining operations. The system trains virtual mining trucks to navigate complex mine environments while managing fuel consumption and completing cargo delivery tasks efficiently.

### Key Features
- **Q-Learning Implementation**: Classic reinforcement learning algorithm
- **Realistic Mining Environment**: Obstacles, fuel stations, loading/dump zones
- **Multi-Objective Optimization**: Balances fuel efficiency, time, and safety
- **Progressive Learning**: Trucks improve performance through trial and error
- **Visualization Tools**: Track training progress and performance metrics

## 🏗️ System Architecture

### Environment Components
- **Grid-based Mine Layout**: 10x10 navigable area with obstacles
- **Loading Zones**: Where empty trucks pick up cargo
- **Dump Zones**: Delivery destinations for loaded trucks  
- **Fuel Stations**: Refueling points to prevent breakdowns
- **Dynamic Obstacles**: Rocks, equipment, and blocked paths

### Agent (Mining Truck)
- **State Space**: Position, fuel level, cargo status, destination
- **Action Space**: Movement (up/down/left/right) + operations (load/dump/refuel)
- **Learning Algorithm**: Q-learning with epsilon-greedy exploration
- **Reward System**: Incentivizes efficiency and penalizes failures

## 🚀 Quick Start

### Prerequisites
```bash
pip install numpy matplotlib
```

### Basic Usage

#### Option 1: Jupyter Notebook (Recommended for Learning)
```python
# Cell 1: Setup
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
# ... paste all class definitions ...

# Cell 2: Quick Training
agent, rewards, lengths = train_mining_truck(episodes=500)

# Cell 3: Visualize Progress
visualize_training(rewards, lengths)

# Cell 4: Test Performance  
test_trained_agent(agent, episodes=10)
```

#### Option 2: Python Script
```bash
python mining_truck_rl.py
```

### Expected Output
```
Training Mining Truck Route Optimization with Q-Learning...
Episode 0: Avg Reward = -45.23, Avg Length = 28.45
Episode 100: Avg Reward = -32.18, Avg Length = 22.33
...
Episode 900: Avg Reward = 12.78, Avg Length = 12.89

Testing trained agent...
Test Episode 1: Path length = 8, Total reward = 45.20, Success = True
```

## 🧠 Reinforcement Learning Concepts

### Q-Learning Fundamentals
- **Q-Table**: Stores value of each action in each state
- **Exploration vs Exploitation**: Epsilon-greedy policy balances learning new strategies vs using known good ones
- **Temporal Difference Learning**: Updates estimates based on immediate rewards and future value predictions

### Key Parameters
```python
learning_rate = 0.1      # How fast the agent learns (α)
discount_factor = 0.95   # Importance of future rewards (γ)  
epsilon = 0.1           # Exploration rate (ε-greedy)
```

### Reward Structure
| Action | Reward | Purpose |
|--------|--------|---------|
| Successful delivery | +50 | Primary objective |
| Loading cargo | +10 | Intermediate progress |
| Refueling | +5 | Maintenance behavior |
| Moving toward goal | +1 | Navigation guidance |
| Running out of fuel | -100 | Critical failure |
| Each step | -1 | Efficiency pressure |

## 📊 Performance Metrics

### Training Progress
- **Episode Rewards**: Should improve from -100 to +50 over 1000 episodes
- **Episode Length**: Should decrease from 50+ steps to 8-15 steps
- **Success Rate**: Should approach 100% after training

### Visualization
The system generates two key plots:
1. **Reward Progression**: Shows learning curve over episodes
2. **Efficiency Improvement**: Tracks reduction in steps needed

## 🔧 Customization Options

### Environment Modifications
```python
# Adjust mine size
env = MiningEnvironment(width=15, height=15)

# Add more obstacles
env.obstacles.update([(5,5), (6,6), (7,7)])

# Relocate stations
env.fuel_stations = [(3,3), (12,12)]
```

### Hyperparameter Tuning
```python
# More aggressive learning
agent = QLearningAgent(learning_rate=0.2, epsilon=0.3)

# Longer-term planning
agent = QLearningAgent(discount_factor=0.99)
```

### Extended Training
```python
# Production-level training
train_mining_truck(episodes=5000)
```

## 🚛 Real-World Applications

### Current Capabilities
- **Route Optimization**: Finds shortest safe paths
- **Fuel Management**: Learns when and where to refuel
- **Task Completion**: Successfully delivers cargo loads
- **Obstacle Avoidance**: Navigates around static barriers

### Potential Extensions
- **Multi-Agent Coordination**: Multiple trucks avoiding collisions
- **Dynamic Environments**: Weather, traffic, equipment changes  
- **Advanced RL Algorithms**: Deep Q-Networks, Actor-Critic methods
- **Real-Time Integration**: Connect to actual mining fleet management

## 📈 Scaling Considerations

### For Larger Operations
- **State Space**: May need function approximation for continuous spaces
- **Action Space**: Could include speed control, route planning
- **Multiple Objectives**: Pareto optimization for competing goals
- **Computational Resources**: Deep RL for complex environments

### Production Deployment
```python
# Save trained model
import pickle
with open('trained_truck_agent.pkl', 'wb') as f:
    pickle.dump(trained_agent, f)

# Load for production use
with open('trained_truck_agent.pkl', 'rb') as f:
    production_agent = pickle.load(f)
```

## 🔍 Troubleshooting

### Common Issues

**Poor Learning Performance**
- Check reward structure - are incentives aligned?
- Adjust learning rate (try 0.05-0.3 range)
- Increase exploration (epsilon = 0.2-0.4)
- Extend training episodes (2000+)

**Agent Gets Stuck**
- Verify valid actions are available from all states
- Check obstacle placement doesn't block critical paths
- Ensure fuel stations are reachable

**No Matplotlib Output**
```python
# Add to notebook
%matplotlib inline

# Or use different backend
import matplotlib
matplotlib.use('TkAgg')
```

## 📚 Learning Resources

### Reinforcement Learning Concepts
1. **Q-Learning**: Value-based method that learns action values
2. **Epsilon-Greedy**: Simple exploration strategy
3. **Temporal Difference**: Learning from prediction errors
4. **Markov Decision Process**: Mathematical framework for sequential decisions

### Next Steps
- Implement Deep Q-Networks (DQN) for larger state spaces
- Add Priority Experience Replay for faster learning  
- Explore Policy Gradient methods (Actor-Critic)
- Study multi-agent reinforcement learning

## 🤝 Contributing

Ideas for contributions:
- Additional mining scenarios (underground tunnels, weather effects)
- Advanced RL algorithms (PPO, A3C, Rainbow DQN)
- Real-world data integration
- Performance optimization
- Web-based visualization interface

## 📄 License

This project is for educational purposes. Feel free to modify and extend for learning reinforcement learning concepts.

## 🎓 Educational Value

This project teaches:
- **RL Fundamentals**: States, actions, rewards, policies
- **Q-Learning Algorithm**: Value iteration and temporal difference learning
- **Environment Design**: How to model real-world problems for RL
- **Training Process**: Exploration, exploitation, and convergence
- **Performance Analysis**: Interpreting learning curves and metrics

Perfect for students, researchers, or engineers wanting hands-on RL experience with practical applications!
