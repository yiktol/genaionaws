# Reinforcement Learning Demonstration with Streamlit

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import random
from matplotlib.colors import ListedColormap
import pandas as pd

# Set page config
st.set_page_config(page_title="Reinforcement Learning Demo", layout="wide", page_icon="🤖")

# Title and introduction
st.title("🤖 Reinforcement Learning Demonstration")
st.markdown("""
This application demonstrates how reinforcement learning works using a Q-learning algorithm. 
The agent learns to navigate through a grid-world environment from a starting point to a goal while avoiding obstacles.
""")

# Sidebar
st.sidebar.header("Configuration")

# Environment Parameters
st.sidebar.subheader("Environment Parameters")
grid_size = st.sidebar.slider("Grid Size", min_value=5, max_value=12, value=8, step=1)
obstacle_density = st.sidebar.slider("Obstacle Density", min_value=0.0, max_value=0.4, value=0.2, step=0.05)

# Q-learning Parameters
st.sidebar.subheader("Q-Learning Parameters")
alpha = st.sidebar.slider("Learning Rate (Alpha)", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
gamma = st.sidebar.slider("Discount Factor (Gamma)", min_value=0.1, max_value=1.0, value=0.9, step=0.1)
epsilon = st.sidebar.slider("Exploration Rate (Epsilon)", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
num_episodes = st.sidebar.slider("Number of Episodes", min_value=10, max_value=1000, value=100, step=10)

# Define GridWorld environment
class GridWorld:
    def __init__(self, size=8, obstacle_density=0.2):
        self.size = size
        self.grid = np.zeros((size, size))
        self.start = (0, 0)
        self.goal = (size-1, size-1)
        
        # Set obstacles
        num_obstacles = int(obstacle_density * size * size)
        obstacles = 0
        while obstacles < num_obstacles:
            x, y = random.randint(0, size-1), random.randint(0, size-1)
            if (x, y) != self.start and (x, y) != self.goal and self.grid[x, y] != -1:
                self.grid[x, y] = -1  # -1 represents an obstacle
                obstacles += 1
        
        # Set goal
        self.grid[self.goal] = 1  # 1 represents the goal
        
        # Actions: up, right, down, left
        self.actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self.action_names = ["Up", "Right", "Down", "Left"]
    
    def reset(self):
        self.current_state = self.start
        return self.current_state
    
    def step(self, action):
        # Compute next state
        next_state = (self.current_state[0] + self.actions[action][0], 
                       self.current_state[1] + self.actions[action][1])
        
        # Check if out of bounds
        if next_state[0] < 0 or next_state[0] >= self.size or next_state[1] < 0 or next_state[1] >= self.size:
            next_state = self.current_state
            reward = -1
            done = False
        # Check if hit obstacle
        elif self.grid[next_state] == -1:
            next_state = self.current_state
            reward = -1
            done = False
        # Check if reached goal
        elif next_state == self.goal:
            reward = 10
            done = True
        # Regular move
        else:
            reward = -0.1  # Small penalty to encourage shorter paths
            done = False
        
        self.current_state = next_state
        return next_state, reward, done
    
    def is_valid_state(self, state):
        x, y = state
        if 0 <= x < self.size and 0 <= y < self.size and self.grid[x, y] != -1:
            return True
        return False

# Q-Learning Agent
class QLearningAgent:
    def __init__(self, env, alpha=0.5, gamma=0.9, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}
        
        # Initialize Q-table
        for i in range(env.size):
            for j in range(env.size):
                if env.is_valid_state((i, j)):
                    self.q_table[(i, j)] = [0.0, 0.0, 0.0, 0.0]
    
    def choose_action(self, state):
        if random.random() < self.epsilon:
            # Exploration: choose a random action
            return random.randint(0, 3)
        else:
            # Exploitation: choose the best action according to Q-table
            return np.argmax(self.q_table[state])
    
    def update_q_table(self, state, action, reward, next_state):
        if next_state in self.q_table:
            # Q(s,a) = Q(s,a) + alpha * [R + gamma * max(Q(s',a')) - Q(s,a)]
            self.q_table[state][action] += self.alpha * (
                reward + self.gamma * max(self.q_table[next_state]) - self.q_table[state][action]
            )
    
    def train(self, episodes):
        rewards_history = []
        steps_history = []
        
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            done = False
            steps = 0
            
            while not done and steps < self.env.size * self.env.size * 2:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                
                if next_state in self.q_table:
                    self.update_q_table(state, action, reward, next_state)
                
                state = next_state
                total_reward += reward
                steps += 1
            
            rewards_history.append(total_reward)
            steps_history.append(steps)
        
        return rewards_history, steps_history
    
    def get_policy(self):
        policy = np.zeros((self.env.size, self.env.size))
        
        for i in range(self.env.size):
            for j in range(self.env.size):
                if (i, j) in self.q_table:
                    policy[i, j] = np.argmax(self.q_table[i, j])
                else:
                    policy[i, j] = -1  # Obstacle or invalid state
        
        return policy
    
    def get_value_function(self):
        value_function = np.zeros((self.env.size, self.env.size))
        
        for i in range(self.env.size):
            for j in range(self.env.size):
                if (i, j) in self.q_table:
                    value_function[i, j] = max(self.q_table[i, j])
                else:
                    value_function[i, j] = -10  # Obstacle or invalid state
        
        return value_function

# Function to visualize the environment
def visualize_env(env, policy=None, value_function=None, agent_path=None, q_values=None, selected_state=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Create colormap for the grid
    cmap = ListedColormap(['white', 'red', 'green'])
    ax.imshow(env.grid, cmap=cmap, interpolation='nearest')
    
    # Draw grid lines
    for i in range(env.size + 1):
        ax.axhline(i - 0.5, color='black', linewidth=1)
        ax.axvline(i - 0.5, color='black', linewidth=1)
    
    # Add labels for cells
    for i in range(env.size):
        for j in range(env.size):
            if env.grid[i, j] == -1:
                ax.text(j, i, "🧱", fontsize=15, ha='center', va='center')
            elif (i, j) == env.goal:
                ax.text(j, i, "🏁", fontsize=15, ha='center', va='center')
            elif (i, j) == env.start:
                ax.text(j, i, "🚀", fontsize=15, ha='center', va='center')
    
    # Draw policy arrows
    if policy is not None:
        for i in range(env.size):
            for j in range(env.size):
                if policy[i, j] != -1 and (i, j) != env.goal:
                    dx, dy = env.actions[int(policy[i, j])]
                    ax.arrow(j, i, dy * 0.3, dx * 0.3, head_width=0.2, head_length=0.2, fc='blue', ec='blue')
    
    # Draw value function as text
    if value_function is not None:
        for i in range(env.size):
            for j in range(env.size):
                if env.grid[i, j] != -1 and (i, j) != env.goal:
                    val = value_function[i, j]
                    ax.text(j, i - 0.25, f"{val:.2f}", fontsize=8, ha='center', va='center', color='purple')
    
    # Draw agent path
    if agent_path:
        path_x = [p[1] for p in agent_path]
        path_y = [p[0] for p in agent_path]
        ax.plot(path_x, path_y, 'o-', color='blue', markersize=5, alpha=0.7)
    
    # Draw Q-values for selected state
    if q_values and selected_state:
        i, j = selected_state
        if (i, j) in q_values:
            q_vals = q_values[(i, j)]
            # Up
            ax.text(j, i - 0.4, f"{q_vals[0]:.2f}", fontsize=8, ha='center', color='red')
            # Right
            ax.text(j + 0.4, i, f"{q_vals[1]:.2f}", fontsize=8, va='center', color='red')
            # Down
            ax.text(j, i + 0.4, f"{q_vals[2]:.2f}", fontsize=8, ha='center', color='red')
            # Left
            ax.text(j - 0.4, i, f"{q_vals[3]:.2f}", fontsize=8, va='center', color='red')
    
    ax.set_xticks(range(env.size))
    ax.set_yticks(range(env.size))
    ax.set_title("Grid World Environment")
    
    return fig

# Function to create a heatmap of the state values
def create_value_heatmap(env, value_function):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Create a masked array where obstacles are masked
    masked_values = np.ma.masked_array(value_function, env.grid == -1)
    
    # Create the heatmap
    cmap = plt.cm.viridis
    cmap.set_bad('gray', 1.)
    im = ax.imshow(masked_values, cmap=cmap, interpolation='nearest')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('State Value')
    
    # Draw grid lines
    for i in range(env.size + 1):
        ax.axhline(i - 0.5, color='black', linewidth=1)
        ax.axvline(i - 0.5, color='black', linewidth=1)
    
    # Add labels for special cells
    for i in range(env.size):
        for j in range(env.size):
            if (i, j) == env.goal:
                ax.text(j, i, "🏁", fontsize=15, ha='center', va='center')
            elif (i, j) == env.start:
                ax.text(j, i, "🚀", fontsize=15, ha='center', va='center')
    
    ax.set_xticks(range(env.size))
    ax.set_yticks(range(env.size))
    ax.set_title("State Value Heatmap")
    
    return fig

# Function to run an episode with the trained agent
def run_episode(env, agent, epsilon=0):
    state = env.reset()
    done = False
    path = [state]
    total_reward = 0
    
    while not done and len(path) < env.size * env.size * 2:
        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            action = np.argmax(agent.q_table[state])
        
        next_state, reward, done = env.step(action)
        path.append(next_state)
        total_reward += reward
        state = next_state
    
    return path, total_reward

# Main app content
def main():
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Train Agent", "Visualize Policy", "Test Agent", "About RL"])
    
    with tab1:
        st.header("Train the Q-Learning Agent")
        
        # Create the environment and agent
        env = GridWorld(size=grid_size, obstacle_density=obstacle_density)
        agent = QLearningAgent(env, alpha=alpha, gamma=gamma, epsilon=epsilon)
        
        # Training section
        col1, col2 = st.columns([2, 3])
        
        with col1:
            st.subheader("Initial Environment")
            fig_initial = visualize_env(env)
            st.pyplot(fig_initial)
            
        with col2:
            train_button = st.button("Train Agent", type="primary")
            
            if train_button:
                progress_bar = st.progress(0)
                rewards_placeholder = st.empty()
                
                # Run training with progress updates
                rewards = []
                steps = []
                
                for episode in range(1, num_episodes + 1):
                    state = env.reset()
                    total_reward = 0
                    done = False
                    num_steps = 0
                    
                    while not done and num_steps < env.size * env.size * 2:
                        action = agent.choose_action(state)
                        next_state, reward, done = env.step(action)
                        
                        if next_state in agent.q_table:
                            agent.update_q_table(state, action, reward, next_state)
                        
                        state = next_state
                        total_reward += reward
                        num_steps += 1
                    
                    rewards.append(total_reward)
                    steps.append(num_steps)
                    
                    # Update progress
                    progress_bar.progress(episode / num_episodes)
                    
                    # Show rewards trend every 10 episodes
                    if episode % 10 == 0 or episode == num_episodes:
                        fig_rewards, ax = plt.subplots(figsize=(8, 4))
                        ax.plot(range(1, episode + 1), rewards)
                        ax.set_xlabel('Episode')
                        ax.set_ylabel('Total Reward')
                        ax.set_title('Training Rewards Over Time')
                        rewards_placeholder.pyplot(fig_rewards)
                
                st.success(f"Training completed! Agent trained for {num_episodes} episodes.")
                
                # Display final training metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Average Reward (Last 10 Episodes)", f"{np.mean(rewards[-10:]):.2f}")
                
                with col2:
                    st.metric("Average Steps (Last 10 Episodes)", f"{np.mean(steps[-10:]):.1f}")
                
                # Display training data in a DataFrame
                training_data = pd.DataFrame({
                    'Episode': range(1, num_episodes + 1),
                    'Total Reward': rewards,
                    'Steps': steps
                })
                st.subheader("Training Data")
                st.dataframe(training_data)
        
        # Display the learned Q-table
        st.subheader("Learned Q-Table")
        q_table_data = []
        for state, values in agent.q_table.items():
            q_table_data.append({
                'State': str(state),
                'Up': f"{values[0]:.2f}",
                'Right': f"{values[1]:.2f}",
                'Down': f"{values[2]:.2f}",
                'Left': f"{values[3]:.2f}",
                'Best Action': env.action_names[np.argmax(values)]
            })
        
        st.dataframe(pd.DataFrame(q_table_data))
    
    with tab2:
        st.header("Visualize The Learned Policy")
        
        try:  # Only run this if the agent has been trained
            policy = agent.get_policy()
            value_function = agent.get_value_function()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Policy Visualization")
                fig_policy = visualize_env(env, policy=policy)
                st.pyplot(fig_policy)
            
            with col2:
                st.subheader("Value Function Heatmap")
                fig_heatmap = create_value_heatmap(env, value_function)
                st.pyplot(fig_heatmap)
            
            # State inspection
            st.subheader("Inspect State Q-Values")
            st.write("Select a cell to see the Q-values for each action at that state")
            
            col1, col2 = st.columns(2)
            with col1:
                selected_x = st.slider("Row (X)", 0, env.size - 1, 0)
            with col2:
                selected_y = st.slider("Column (Y)", 0, env.size - 1, 0)
            
            selected_state = (selected_x, selected_y)
            
            if selected_state in agent.q_table:
                q_values = agent.q_table[selected_state]
                st.write(f"Q-values at state {selected_state}:")
                
                q_data = pd.DataFrame({
                    'Action': ['Up', 'Right', 'Down', 'Left'],
                    'Q-Value': q_values,
                    'Is Best': [i == np.argmax(q_values) for i in range(4)]
                })
                
                st.dataframe(q_data, hide_index=True)
                
                fig_state = visualize_env(env, q_values=agent.q_table, selected_state=selected_state)
                st.pyplot(fig_state)
            else:
                st.error("This state is an obstacle or out of bounds. No Q-values available.")
        
        except NameError:
            st.warning("Please train the agent first in the Training tab.")
    
    with tab3:
        st.header("Test the Trained Agent")
        
        try:  # Only run this if the agent has been trained
            col1, col2 = st.columns([1, 2])
            
            with col1:
                test_epsilon = st.slider("Test Exploration Rate", 0.0, 1.0, 0.0, 0.01, 
                                         help="Set to 0 for fully exploiting the learned policy, or higher for more exploration")
                run_test = st.button("Run Test Episode")
            
            if run_test:
                with col2:
                    path, total_reward = run_episode(env, agent, epsilon=test_epsilon)
                    
                    st.write(f"Episode completed! Total reward: {total_reward:.2f}, Path length: {len(path)}")
                    
                    fig_test = visualize_env(env, policy=agent.get_policy(), agent_path=path)
                    st.pyplot(fig_test)
                    
                    # Display path details
                    path_data = []
                    for i, state in enumerate(path):
                        if i < len(path) - 1:
                            next_state = path[i+1]
                            # Determine the action taken
                            dx = next_state[0] - state[0]
                            dy = next_state[1] - state[1]
                            action_idx = env.actions.index((dx, dy)) if (dx, dy) in env.actions else -1
                            action_name = env.action_names[action_idx] if action_idx != -1 else "None"
                        else:
                            action_name = "Goal Reached" if state == env.goal else "Terminated"
                        
                        path_data.append({
                            'Step': i,
                            'State': str(state),
                            'Action': action_name
                        })
                    
                    st.subheader("Path Details")
                    st.dataframe(pd.DataFrame(path_data))
        
        except NameError:
            st.warning("Please train the agent first in the Training tab.")
    
    with tab4:
        st.header("About Reinforcement Learning")
        
        st.markdown("""
        ### What is Reinforcement Learning?
        
        Reinforcement Learning (RL) is a type of machine learning where an agent learns to make sequences of decisions by interacting with an environment to maximize a cumulative reward.
        
        ### Key Concepts of RL
        
        1. **Agent**: The decision-maker that interacts with the environment (in our demo, the agent navigates the grid)
        2. **Environment**: The world the agent interacts with (the grid world with obstacles)
        3. **State**: The current situation of the agent (the position in the grid)
        4. **Action**: What the agent can do in a given state (move up, right, down, left)
        5. **Reward**: Feedback from the environment after taking an action (penalties for obstacles, reward for reaching the goal)
        6. **Policy**: The strategy the agent follows to decide actions (derived from the Q-table)
        7. **Value Function**: The expected cumulative reward from a state following a policy
        
        ### Q-Learning Algorithm
        
        Q-Learning is a value-based RL algorithm that learns the value of an action in a particular state. The "Q" refers to the "quality" of an action in a given state.
        
        The core update equation is:
        
        Q(s,a) = Q(s,a) + α[r + γ * max(Q(s',a')) - Q(s,a)]
        
        Where:
        - Q(s,a) is the Q-value for state s and action a
        - α is the learning rate
        - r is the reward received
        - γ is the discount factor for future rewards
        - s' is the next state
        - a' represents all possible actions in the next state
        
        ### Exploration vs. Exploitation
        
        - **Exploration**: Trying new actions to discover better strategies (controlled by epsilon)
        - **Exploitation**: Using known good strategies to maximize reward
        
        The demo uses an ε-greedy approach, where the agent explores with probability ε and exploits with probability 1-ε.
        """)
        
        st.image("https://upload.wikimedia.org/wikipedia/commons/1/1b/Reinforcement_learning_diagram.svg", 
                 caption="Reinforcement Learning Framework", width=400)

# Run the app
if __name__ == "__main__":
    main()
