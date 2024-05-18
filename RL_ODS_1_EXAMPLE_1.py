# -*- coding: utf-8 -*-
"""
Created on Sat May 18 13:59:49 2024

@author: Jaime
"""
"""
Proposal 3. Improvement of aid distribution systems: The proposal addresses
the need to improve the logistics and distribution of food aid and essential 
resources, especially in highly vulnerable areas. The implementation of a 
reinforcement learning algorithm is proposed that would optimize the entire 
aid supply chain, from inventory planning to effective delivery on the ground. 
This algorithm would adapt and learn from each distribution cycle to continually 
improve in aspects such as selecting distribution routes, assigning resources 
to different areas, and scheduling deliveries. The main objective is to ensure 
that aid is delivered in a timely and efficient manner to those who need it 
most, minimizing waste and improving the overall efficiency of the process. 
This proposal is vital to expanding the reach and effectiveness of humanitarian 
efforts, which in turn directly supports Sustainable Development Goal 1: No 
Poverty, by ensuring that limited resources are used in the most impactful 
way possible.
"""
#==============================================================================
# SDG 1: End Poverty (Proposal 1)
#==============================================================================

import numpy as np
import pandas as pd
import gym
from gym import spaces
from stable_baselines3 import DQN
import networkx as nx
import matplotlib.pyplot as plt
import random

class SupplyChainEnv(gym.Env):
    def __init__(self, num_areas=5, num_resources=100, num_types=3, max_capacity=10, budget=500, seed=None):
        super(SupplyChainEnv, self).__init__()
        self.num_areas = num_areas
        self.num_resources = num_resources
        self.num_types = num_types
        self.max_capacity = max_capacity
        self.budget = budget
        
        # Define the action space and observation space
        self.action_space = spaces.MultiDiscrete([num_areas, num_types])
        self.observation_space = spaces.Box(low=0, high=num_resources, shape=(num_areas, num_types), dtype=np.int)
        
        # Set the seed for reproducibility
        if seed:
            np.random.seed(seed)
            random.seed(seed)
        
        self.reset()

    def reset(self):
        # Initialize the state, demand, cost matrix, time windows, priorities, terrain types, and incidents
        self.state = np.zeros((self.num_areas, self.num_types))
        self.demand = np.random.randint(1, 10, (self.num_areas, self.num_types))
        self.cost_matrix = np.random.randint(1, 10, (self.num_areas, self.num_areas))
        self.time_windows = np.random.randint(0, 24, self.num_areas)
        self.priorities = np.random.randint(1, 5, self.num_areas)
        self.terrain_types = np.random.choice(['flat', 'mountain', 'urban'], self.num_areas)
        self.done = False
        self.budget_used = 0
        self.time_elapsed = 0
        self.incidents = [random.choice([True, False]) for _ in range(self.num_areas)]
        return self.state

    def step(self, action):
        area, resource_type = action
        reward = 0
        
        # Update the state based on the action taken
        if self.state[area, resource_type] < self.demand[area, resource_type]:
            cost = self.cost_matrix[0, area] * (1.5 if self.terrain_types[area] == 'mountain' else 1.0)
            if self.budget_used + cost <= self.budget:
                self.state[area, resource_type] += 1
                reward = 1 * self.priorities[area]
                self.budget_used += cost
                self.time_elapsed += random.randint(1, 3) + (5 if self.incidents[area] else 0)
            
        # Check if the simulation is done
        if np.sum(self.state) >= self.num_resources or self.budget_used >= self.budget:
            self.done = True

        return self.state, reward, self.done, {}

    def render(self, mode='human'):
        # Print the current state of the environment
        for i in range(self.num_areas):
            print(f"Area {i} distribution: {self.state[i]}, Demand: {self.demand[i]}, Time Window: {self.time_windows[i]}, Incident: {self.incidents[i]}, Priority: {self.priorities[i]}, Terrain: {self.terrain_types[i]}")

# Create environment
env = SupplyChainEnv()

# Train the model
model = DQN('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

# Evaluate and visualize the model
obs = env.reset()
for _ in range(20):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

# Visualization of results
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(12, 25))

# Resource distribution graph
for i in range(env.num_areas):
    ax1.bar(f'Area {i}', env.state[i].sum(), label=f'Demand {i}: {env.demand[i].sum()}')
ax1.set_ylabel('Resources Distributed')
ax1.set_title('Final Distribution of Resources')
ax1.legend()

# Cost distribution graph
areas = np.arange(env.num_areas)
costs = [env.cost_matrix[0, i] for i in areas]
ax2.bar(areas, costs, color='orange')
ax2.set_ylabel('Cost per Area')
ax2.set_xlabel('Areas')
ax2.set_title('Cost of Distribution per Area')

# Delivery time graph
times = [random.randint(1, 3) + (5 if env.incidents[i] else 0) for i in areas]
ax3.bar(areas, times, color='green')
ax3.set_ylabel('Time per Area')
ax3.set_xlabel('Areas')
ax3.set_title('Time of Delivery per Area')

# Delivery priority graph
priorities = [env.priorities[i] for i in areas]
ax4.bar(areas, priorities, color='purple')
ax4.set_ylabel('Priority')
ax4.set_xlabel('Areas')
ax4.set_title('Priority of Delivery per Area')

# Terrain types graph
terrain_colors = {'flat': 'blue', 'mountain': 'brown', 'urban': 'gray'}
terrain_types = [terrain_colors[env.terrain_types[i]] for i in areas]
ax5.bar(areas, [1]*len(areas), color=terrain_types)
ax5.set_ylabel('Terrain Type')
ax5.set_xlabel('Areas')
ax5.set_title('Terrain Types of Areas')
ax5.set_yticks([])

plt.tight_layout()
plt.show()

# Evaluation metrics
total_resources_distributed = np.sum(env.state)
total_demand = np.sum(env.demand)
efficiency = total_resources_distributed / total_demand * 100

print(f"Total Resources Distributed: {total_resources_distributed}")
print(f"Total Demand: {total_demand}")
print(f"Distribution Efficiency: {efficiency:.2f}%")
print(f"Budget Used: {env.budget_used}")
print(f"Total Time Elapsed: {env.time_elapsed}")
print(f"Incidents: {env.incidents}")
