# 🚁 Multi-Agent Deep Reinforcement Learning for Autonomous Drone Fleet Coordination

## 📌 Overview

This project implements a **Multi-Agent Deep Reinforcement Learning (MADRL)** framework to enable autonomous coordination among multiple drones. Each drone acts as an intelligent agent that learns to navigate, avoid collisions, and complete tasks efficiently in a shared environment.

The system compares three reinforcement learning algorithms:

* Deep Q-Network (DQN)
* Proximal Policy Optimization (PPO)
* Multi-Agent Deep Deterministic Policy Gradient (MADDPG)

---

## 🎯 Objectives

* Develop a decentralized multi-drone coordination system
* Enable collision-free navigation and efficient path planning
* Optimize energy consumption using learning-based strategies
* Compare RL algorithms based on performance metrics

---

## 🧠 Technologies Used

* **Language:** Python
* **Libraries:** PyTorch, NumPy, Matplotlib
* **Environment:** Custom Simulation / OpenAI Gym / PettingZoo
* **Tools:** Jupyter Notebook / VS Code

---

## ⚙️ System Architecture

The system consists of:

* Simulation Environment (multi-drone setup with obstacles & targets)
* RL Agents (DQN, PPO, MADDPG)
* Training Module
* Visualization Module (graphs & trajectories)

---

## 🚀 Features

* Decentralized multi-agent learning
* Collision avoidance and safe navigation
* Energy-efficient path planning
* Scalable for multiple drones
* Performance visualization

---

## 📊 Results Summary

* **DQN:** Fast convergence, highest reward
* **MADDPG:** Best coordination & lowest collision rate
* **PPO:** Stable but slower learning

---

## 📁 Project Structure

```
├── drone_env.py        # Simulation environment
├── dqn_agent.py       # DQN implementation
├── ppo_agent.py       # PPO implementation
├── maddpg_agent.py    # MADDPG implementation
├── train_research.py  # Training script
├── run_project.py     # Main execution file
```

---

## ▶️ How to Run

1. Clone the repository

```
git clone <repo-link>
cd <repo-folder>
```

2. Install dependencies

```
pip install -r requirements.txt
```

3. Run the project

```
python run_project.py
```

---

## 📈 Evaluation Metrics

* Cumulative Reward
* Collision Rate
* Energy Consumption
* Task Completion Efficiency

---

## 🌍 Applications

* Drone delivery systems
* Disaster management
* Surveillance & security
* Smart city monitoring
* Environmental data collection

---

## 🔮 Future Scope

* Real-world drone integration
* 3D navigation environments
* Advanced communication between agents
* Computer vision integration

---
