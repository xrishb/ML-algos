#!/usr/bin/env python
# Q-Learning Implementation - A basic reinforcement learning algorithm

import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.colors import ListedColormap

class GridWorld:
    """
    A simple 5x5 grid world environment with:
    - Agent starting at the bottom-left (0,0)
    - Goal at the top-right (4,4)
    - Some obstacles/walls in the environment
    - Actions: up, right, down, left
    - Rewards: -1 for each step, -10 for hitting a wall, +100 for reaching the goal
    """
    
    def __init__(self, size=5):
        self.size = size
        # Define the grid world
        self.grid = np.zeros((size, size))
        
        # Set the goal position (top-right)
        self.goal = (size-1, size-1)
        self.grid[self.goal] = 2
        
        # Set obstacles/walls
        self.walls = [(1, 1), (2, 1), (3, 1), (1, 3), (2, 3), (3, 3)]
        for wall in self.walls:
            self.grid[wall] = 1
            
        # Define possible actions: up, right, down, left
        self.actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self.action_names = ['Up', 'Right', 'Down', 'Left']
        
        # Starting position (bottom-left)
        self.agent_position = (0, 0)
        
        # Track if episode is done
        self.done = False
    
    def reset(self):
        """Reset the environment to initial state."""
        self.agent_position = (0, 0)
        self.done = False
        return self.agent_position
    
    def step(self, action_idx):
        """Take an action and return new state, reward, and done flag."""
        if self.done:
            return self.agent_position, 0, True
        
        # Get the action direction
        action = self.actions[action_idx]
        
        # Calculate new position
        new_position = (
            self.agent_position[0] + action[0], 
            self.agent_position[1] + action[1]
        )
        
        # Check if the new position is valid (within grid and not a wall)
        if (0 <= new_position[0] < self.size and 
            0 <= new_position[1] < self.size and 
            new_position not in self.walls):
            
            self.agent_position = new_position
            
            # Check if reached the goal
            if self.agent_position == self.goal:
                self.done = True
                return self.agent_position, 100, True
            
            # Default step reward
            return self.agent_position, -1, False
        else:
            # Hitting a wall or going out of bounds
            return self.agent_position, -10, False
    
    def render(self, q_table=None):
        """Visualize the grid world."""
        # Create a colormap: 0=empty, 1=wall, 2=goal, 3=agent
        cmap = ListedColormap(['white', 'black', 'green', 'blue'])
        
        # Create a copy of the grid for visualization
        vis_grid = self.grid.copy()
        
        # Add agent to the visualization grid
        if vis_grid[self.agent_position] == 0:  # Only if it's an empty cell
            vis_grid[self.agent_position] = 3
            
        plt.figure(figsize=(10, 8))
        
        # Main grid visualization
        plt.subplot(1, 2, 1)
        plt.imshow(vis_grid, cmap=cmap, interpolation='none')
        
        # Add grid lines
        plt.grid(True, color='black', linestyle='-', linewidth=1)
        plt.xticks(np.arange(-.5, self.size, 1), [])
        plt.yticks(np.arange(-.5, self.size, 1), [])
        
        # Adjust grid lines to match cell boundaries
        ax = plt.gca()
        ax.set_xticks(np.arange(-.5, self.size, 1), minor=True)
        ax.set_yticks(np.arange(-.5, self.size, 1), minor=True)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        plt.title("Grid World")
        
        # If Q-table is provided, visualize the policy
        if q_table is not None:
            plt.subplot(1, 2, 2)
            
            # Create a visual representation of the best action for each state
            policy_grid = np.zeros((self.size, self.size))
            values_grid = np.zeros((self.size, self.size))
            
            for i in range(self.size):
                for j in range(self.size):
                    if (i, j) in self.walls:
                        policy_grid[i, j] = -1  # Wall
                        values_grid[i, j] = -np.inf
                    elif (i, j) == self.goal:
                        policy_grid[i, j] = 4  # Goal
                        values_grid[i, j] = 100
                    else:
                        # Get best action (highest Q-value)
                        state_idx = i * self.size + j
                        best_action = np.argmax(q_table[state_idx])
                        policy_grid[i, j] = best_action
                        values_grid[i, j] = np.max(q_table[state_idx])
            
            # Show policy grid
            cmap_policy = ListedColormap(['black', 'red', 'yellow', 'purple', 'blue', 'green'])
            plt.imshow(policy_grid, cmap=cmap_policy, interpolation='none')
            
            # Add arrows to indicate policy direction
            for i in range(self.size):
                for j in range(self.size):
                    if (i, j) not in self.walls and (i, j) != self.goal:
                        action_idx = int(policy_grid[i, j])
                        if action_idx == 0:  # Up
                            plt.arrow(j, i, 0, -0.3, head_width=0.2, head_length=0.2, fc='white', ec='white')
                        elif action_idx == 1:  # Right
                            plt.arrow(j, i, 0.3, 0, head_width=0.2, head_length=0.2, fc='white', ec='white')
                        elif action_idx == 2:  # Down
                            plt.arrow(j, i, 0, 0.3, head_width=0.2, head_length=0.2, fc='white', ec='white')
                        elif action_idx == 3:  # Left
                            plt.arrow(j, i, -0.3, 0, head_width=0.2, head_length=0.2, fc='white', ec='white')
            
            # Add grid lines
            plt.grid(True, color='black', linestyle='-', linewidth=1)
            plt.xticks(np.arange(-.5, self.size, 1), [])
            plt.yticks(np.arange(-.5, self.size, 1), [])
            
            # Adjust grid lines to match cell boundaries
            ax = plt.gca()
            ax.set_xticks(np.arange(-.5, self.size, 1), minor=True)
            ax.set_yticks(np.arange(-.5, self.size, 1), minor=True)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            
            plt.title("Policy (Arrows show best action)")
        
        plt.tight_layout()
        plt.savefig('reinforcement_learning/q_learning_visualization.png')
        plt.close()


def q_learning(env, episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1):
    """
    Q-learning algorithm implementation.
    
    Args:
        env: The environment (GridWorld instance)
        episodes: Number of episodes to train
        alpha: Learning rate
        gamma: Discount factor
        epsilon: Exploration rate (epsilon-greedy policy)
    
    Returns:
        Q-table: The learned action-value function
    """
    # Initialize Q-table with zeros
    num_states = env.size * env.size
    num_actions = len(env.actions)
    q_table = np.zeros((num_states, num_actions))
    
    # To track rewards per episode for visualization
    rewards_per_episode = []
    
    for episode in range(episodes):
        # Reset environment for new episode
        state = env.reset()
        state_idx = state[0] * env.size + state[1]  # Convert 2D state to index
        
        total_reward = 0
        done = False
        
        while not done:
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                # Explore: select random action
                action = np.random.randint(num_actions)
            else:
                # Exploit: select best action from Q-table
                action = np.argmax(q_table[state_idx])
            
            # Take action and observe new state and reward
            next_state, reward, done = env.step(action)
            next_state_idx = next_state[0] * env.size + next_state[1]
            
            # Update Q-value using the Bellman equation
            q_table[state_idx, action] = q_table[state_idx, action] + alpha * (
                reward + gamma * np.max(q_table[next_state_idx]) - q_table[state_idx, action]
            )
            
            # Move to the next state
            state = next_state
            state_idx = next_state_idx
            
            total_reward += reward
            
            if done:
                break
        
        # Decay epsilon over time for less exploration
        epsilon = max(0.01, epsilon * 0.995)
        
        # Record rewards
        rewards_per_episode.append(total_reward)
        
        # Print progress
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{episodes}, Reward: {total_reward}, Epsilon: {epsilon:.4f}")
    
    # Plot learning progress
    plt.figure(figsize=(10, 6))
    plt.plot(rewards_per_episode)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Q-Learning Performance')
    plt.grid(True)
    plt.savefig('reinforcement_learning/q_learning_rewards.png')
    plt.close()
    
    return q_table


def test_policy(env, q_table, render=True):
    """Test the learned policy."""
    state = env.reset()
    state_idx = state[0] * env.size + state[1]
    
    done = False
    total_reward = 0
    steps = 0
    
    print("\nTesting policy:")
    
    while not done and steps < 100:  # Limit steps to avoid infinite loops
        # Select action with highest Q-value
        action = np.argmax(q_table[state_idx])
        
        # Take the action
        next_state, reward, done = env.step(action)
        next_state_idx = next_state[0] * env.size + next_state[1]
        
        # Print action and result
        print(f"Step {steps+1}: Action={env.action_names[action]}, " 
              f"State={next_state}, Reward={reward}")
        
        # Update state
        state = next_state
        state_idx = next_state_idx
        
        total_reward += reward
        steps += 1
        
        # Render environment (optional)
        if render:
            env.render(q_table)
            time.sleep(0.5)  # Pause to visualize each step
    
    print(f"\nTotal steps: {steps}")
    print(f"Total reward: {total_reward}")
    
    return total_reward


if __name__ == "__main__":
    # Create environment
    env = GridWorld(size=5)
    
    print("Starting Q-learning in a 5x5 Grid World...")
    print("Goal: Reach the top-right corner from the bottom-left corner")
    print("Obstacles: Several walls to navigate around")
    print("Actions: Up, Right, Down, Left")
    print("Rewards: -1 per step, -10 for hitting walls, +100 for reaching goal")
    
    # Initial visualization
    env.render()
    
    # Train agent
    print("\nTraining agent...")
    q_table = q_learning(env, episodes=500, alpha=0.1, gamma=0.99, epsilon=0.1)
    
    # Test the learned policy
    print("\nTesting learned policy...")
    test_policy(env, q_table, render=True)
    
    # Show final policy visualization
    env.render(q_table)
    
    print("\nReinforcement learning complete!")
    print("Visualizations saved to: reinforcement_learning/q_learning_visualization.png")
    print("Learning curve saved to: reinforcement_learning/q_learning_rewards.png") 