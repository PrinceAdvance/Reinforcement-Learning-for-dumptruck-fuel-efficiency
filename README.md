# 🚛 Autonomous Haul Truck Route Optimization

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
[![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-Latest-green)](https://stable-baselines3.readthedocs.io/)
[![OpenAI Gym](https://img.shields.io/badge/OpenAI-Gym-red)](https://gym.openai.com/)

> Leveraging Reinforcement Learning to optimize autonomous mining truck routes for efficient material transportation.

## 📖 Overview

This project demonstrates the application of Reinforcement Learning (RL) to optimize autonomous haulage truck routes in mining operations. Using Proximal Policy Optimization (PPO), the system learns to navigate efficiently between pickup and dump points while considering time and resource constraints.

![Mining Environment Visualization](path/to/environment_image.png)

## 🎯 Features

- **Custom OpenAI Gym Environment**: Simulated mine layout with configurable grid size
- **Intelligent Route Planning**: Autonomous pathfinding between pickup and dump points
- **Realistic Reward System**: Carefully crafted rewards to encourage optimal behavior
- **PPO Implementation**: Utilizing Stable-Baselines3 for robust learning
- **Performance Monitoring**: Track and visualize training progress

## 🏗️ Architecture

### Environment Details
```
 P . . . t    Legend:
 . . . . .    P = Pickup Point
 . . . . .    D = Dump Point
 . . . . .    t = Empty Truck
 . . . . D    T = Loaded Truck
```

### Reward Structure
| Action | Reward |
|--------|---------|
| Load Pickup | +10 |
| Successful Dump | +20 |
| Movement | -1 |

## 🚀 Getting Started

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/PrinceAdvance/mining-rl-project.git
cd mining-rl-project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Project Structure
```
mining_rl_project/
├── haulage_env.py     # OpenAI Gym environment implementation
├── train_agent.py     # Training script using PPO
├── test_agent.py      # Model evaluation script
├── requirements.txt   # Project dependencies
└── README.md         # Documentation
```

## 🎮 Usage

### Training the Agent
```bash
python train_agent.py
```
This will initiate the training process for 10,000 timesteps and save the trained model as `haul_truck_ppo.zip`.

### Testing the Model
```bash
python test_agent.py
```
Watch your trained agent navigate the environment and observe its decision-making process.

## 📈 Future Roadmap

- [ ] Implement dynamic obstacles and terrain features
- [ ] Add variable elevation and fuel consumption modeling
- [ ] Support for multiple trucks and collision avoidance
- [ ] Integration with real-time mining telemetry data
- [ ] Enhanced visualization and monitoring tools

## 📚 Learning Resources

- [Introduction to Reinforcement Learning](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [OpenAI Gym Tutorial](https://gym.openai.com/docs/)
- [PPO Algorithm Explained](https://openai.com/blog/openai-baselines-ppo/)


## 🙋‍♂️ Author

**Prince Advance** - *Initial work* - [GitHub](https://github.com/PrinceAdvance)

> A beginner in Reinforcement Learning, working on mining automation tools with AI.

---
⭐ If you find this project helpful, please consider giving it a star!
