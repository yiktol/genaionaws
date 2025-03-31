# Enhanced Interactive Reinforcement Learning Demonstration with Streamlit


import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import random
from matplotlib.colors import ListedColormap
import pandas as pd
import altair as alt
from PIL import Image
import io
import base64
from streamlit_drawable_canvas import st_canvas
import copy

# Set page config
st.set_page_config(
    page_title="Interactive Reinforcement Learning", 
    layout="wide", 
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4A56E2;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.8rem;
        color: #5D6970;
        margin-top: 1.5rem;
    }
    .highlight {
        background-color: #F0F2F6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4A56E2;
    }
    .stButton button {
        background-color: #4A56E2;
        color: white;
    }
    .stButton button:hover {
        background-color: #3A46C2;
    }
    .reward-positive {
        color: green;
        font-weight: bold;
    }
    .reward-negative {
        color: red;
        font-weight: bold;
    }
    .stProgress > div > div > div > div {
        background-color: #4A56E2;
    }
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    .q-value-best {
        font-weight: bold;
        color: #4A56E2;
    }
    .grid-cell {
        width: 50px;
        height: 50px;
        border: 1px solid black;
        display: flex;
        justify-content: center;
        align-items: center;
        font-size: 24px;
        cursor: pointer;
    }
    .canvas-container {
        border: 2px solid #4A56E2;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Title and introduction
st.markdown('<h1 class="main-header">🤖 Interactive Reinforcement Learning Playground</h1>', unsafe_allow_html=True)

st.markdown("""
<div class="highlight">
Learn how reinforcement learning works by training an AI agent to navigate through a customizable environment! 
This interactive demo uses Q-learning, a foundational RL algorithm, to show how agents learn by trial and error.
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'env' not in st.session_state:
    st.session_state.env = None
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'training_complete' not in st.session_state:
    st.session_state.training_complete = False
if 'rewards_history' not in st.session_state:
    st.session_state.rewards_history = []
if 'steps_history' not in st.session_state:
    st.session_state.steps_history = []
if 'custom_grid' not in st.session_state:
    st.session_state.custom_grid = None
if 'current_episode' not in st.session_state:
    st.session_state.current_episode = 0
if 'simulation_speed' not in st.session_state:
    st.session_state.simulation_speed = 0.5
if 'agent_path' not in st.session_state:
    st.session_state.agent_path = []

# Define GridWorld environment
class GridWorld:
    def __init__(self, size=8, obstacle_density=0.2, custom_grid=None, start=None, goal=None):
        self.size = size
        
        if custom_grid is not None:
            self.grid = copy.deepcopy(custom_grid)
        else:
            self.grid = np.zeros((size, size))
            # Set obstacles
            num_obstacles = int(obstacle_density * size * size)
            obstacles = 0
            while obstacles < num_obstacles:
                x, y = random.randint(0, size-1), random.randint(0, size-1)
                if (x == 0 and y == 0) or (x == size-1 and y == size-1):
                    continue
                if self.grid[x, y] != -1:
                    self.grid[x, y] = -1  # -1 represents an obstacle
                    obstacles += 1
        
        # Set start and goal positions
        if start:
            self.start = start
        else:
            self.start = (0, 0)
            
        if goal:
            self.goal = goal
        else:
            self.goal = (size-1, size-1)
            
        # Ensure start and goal positions are clear
        if custom_grid is not None:
            self.grid[self.start] = 0
            self.grid[self.goal] = 1
            
        # Set goal reward in grid
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
        self.visit_counts = {}
        
        # Initialize Q-table and visit counts
        for i in range(env.size):
            for j in range(env.size):
                if env.is_valid_state((i, j)):
                    self.q_table[(i, j)] = [0.0, 0.0, 0.0, 0.0]
                    self.visit_counts[(i, j)] = 0
    
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
            # Update visit count
            self.visit_counts[state] += 1
    
    def train(self, episodes, callback=None):
        rewards_history = []
        steps_history = []
        
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            done = False
            steps = 0
            path = [state]
            
            while not done and steps < self.env.size * self.env.size * 2:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                path.append(next_state)
                
                if next_state in self.q_table:
                    self.update_q_table(state, action, reward, next_state)
                
                state = next_state
                total_reward += reward
                steps += 1
            
            rewards_history.append(total_reward)
            steps_history.append(steps)
            
            # Call the callback function if provided
            if callback:
                callback(episode, total_reward, steps, path)
        
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
    
    def get_visit_heatmap(self):
        visit_map = np.zeros((self.env.size, self.env.size))
        
        for i in range(self.env.size):
            for j in range(self.env.size):
                if (i, j) in self.visit_counts:
                    visit_map[i, j] = self.visit_counts[(i, j)]
                else:
                    visit_map[i, j] = 0  # Obstacle or invalid state
        
        return visit_map

# Function to visualize the environment
def visualize_env(env, policy=None, value_function=None, agent_path=None, q_values=None, selected_state=None, highlight_state=None, visit_counts=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Create a base grid for drawing
    grid_vis = np.zeros_like(env.grid)
    grid_vis[env.grid == -1] = 1  # Obstacles
    grid_vis[env.goal] = 2  # Goal
    
    # Create colormap for the grid
    cmap = ListedColormap(['white', 'darkgray', 'lightgreen'])
    ax.imshow(grid_vis, cmap=cmap, interpolation='nearest')
    
    # Draw grid lines
    for i in range(env.size + 1):
        ax.axhline(i - 0.5, color='black', linewidth=1)
        ax.axvline(i - 0.5, color='black', linewidth=1)
    
    # Add emoji labels for cells
    for i in range(env.size):
        for j in range(env.size):
            if env.grid[i, j] == -1:
                ax.text(j, i, "🧱", fontsize=15, ha='center', va='center')
            elif (i, j) == env.goal:
                ax.text(j, i, "🏁", fontsize=15, ha='center', va='center')
            elif (i, j) == env.start:
                ax.text(j, i, "🚀", fontsize=15, ha='center', va='center')
            
            # Add visit counts if provided
            if visit_counts is not None and (i, j) in visit_counts and visit_counts[(i, j)] > 0:
                # Display visit count in smaller text in the corner
                ax.text(j + 0.35, i - 0.35, f"{visit_counts[(i, j)]}", 
                        fontsize=7, ha='right', va='top', color='purple', alpha=0.7)
    
    # Highlight the current state if provided
    if highlight_state:
        circle = patches.Circle((highlight_state[1], highlight_state[0]), 0.4, 
                                 color='blue', fill=True, alpha=0.3)
        ax.add_patch(circle)
    
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
                    ax.text(j, i - 0.25, f"{val:.1f}", fontsize=8, ha='center', va='center', color='purple')
    
    # Draw agent path
    if agent_path:
        path_x = [p[1] for p in agent_path]
        path_y = [p[0] for p in agent_path]
        ax.plot(path_x, path_y, 'o-', color='blue', markersize=6, alpha=0.7, linewidth=2)
        
        # Add agent at the end of the path
        if len(agent_path) > 0:
            last_pos = agent_path[-1]
            agent_circle = patches.Circle((last_pos[1], last_pos[0]), 0.3, color='blue', fill=True)
            ax.add_patch(agent_circle)
    
    # Draw Q-values for selected state
    if q_values and selected_state:
        i, j = selected_state
        if (i, j) in q_values:
            q_vals = q_values[(i, j)]
            best_q = max(q_vals)
            
            # Draw background highlight for the selected state
            highlight = patches.Rectangle((j-0.5, i-0.5), 1, 1, fill=True, color='yellow', alpha=0.2)
            ax.add_patch(highlight)
            
            # Up
            arrow_color = 'red' if q_vals[0] == best_q else 'gray'
            ax.arrow(j, i-0.2, 0, -0.2, head_width=0.15, head_length=0.1, fc=arrow_color, ec=arrow_color)
            ax.text(j, i - 0.4, f"{q_vals[0]:.1f}", fontsize=7, ha='center', color=arrow_color)
            
            # Right
            arrow_color = 'red' if q_vals[1] == best_q else 'gray'
            ax.arrow(j+0.2, i, 0.2, 0, head_width=0.15, head_length=0.1, fc=arrow_color, ec=arrow_color)
            ax.text(j + 0.4, i, f"{q_vals[1]:.1f}", fontsize=7, va='center', color=arrow_color)
            
            # Down
            arrow_color = 'red' if q_vals[2] == best_q else 'gray'
            ax.arrow(j, i+0.2, 0, 0.2, head_width=0.15, head_length=0.1, fc=arrow_color, ec=arrow_color)
            ax.text(j, i + 0.4, f"{q_vals[2]:.1f}", fontsize=7, ha='center', color=arrow_color)
            
            # Left
            arrow_color = 'red' if q_vals[3] == best_q else 'gray'
            ax.arrow(j-0.2, i, -0.2, 0, head_width=0.15, head_length=0.1, fc=arrow_color, ec=arrow_color)
            ax.text(j - 0.4, i, f"{q_vals[3]:.1f}", fontsize=7, va='center', color=arrow_color)
    
    ax.set_xticks(range(env.size))
    ax.set_yticks(range(env.size))
    ax.set_title("Grid World Environment")
    plt.tight_layout()
    
    return fig

# Function to create a heatmap of the state values
def create_value_heatmap(env, value_data, title="State Value Heatmap", vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Create a masked array where obstacles are masked
    masked_values = np.ma.masked_array(value_data, env.grid == -1)
    
    # Create the heatmap
    if vmin is None:
        vmin = np.min(masked_values)
    if vmax is None:
        vmax = np.max(masked_values)
    
    cmap = plt.cm.viridis
    cmap.set_bad('gray', 1.)
    im = ax.imshow(masked_values, cmap=cmap, interpolation='nearest', vmin=vmin, vmax=vmax)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Value')
    
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
            elif env.grid[i, j] == -1:
                ax.text(j, i, "🧱", fontsize=15, ha='center', va='center')
    
    ax.set_xticks(range(env.size))
    ax.set_yticks(range(env.size))
    ax.set_title(title)
    
    return fig

# Function to run an episode with the trained agent
def run_episode(env, agent, epsilon=0, callback=None):
    state = env.reset()
    done = False
    path = [state]
    total_reward = 0
    step = 0
    
    while not done and step < env.size * env.size * 2:
        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            action = np.argmax(agent.q_table[state])
        
        next_state, reward, done = env.step(action)
        path.append(next_state)
        total_reward += reward
        state = next_state
        step += 1
        
        if callback:
            callback(step, state, action, reward, done)
    
    return path, total_reward

# Function to convert the grid drawing to a numpy array
def process_grid_drawing(canvas_result, grid_size):
    # Get the image data from the canvas result
    img_data = canvas_result.image_data
    
    # Calculate cell size
    cell_height = img_data.shape[0] // grid_size
    cell_width = img_data.shape[1] // grid_size
    
    # Create a grid to store the processed data
    grid = np.zeros((grid_size, grid_size))
    
    # Process each cell
    for i in range(grid_size):
        for j in range(grid_size):
            # Extract the cell from the image
            cell = img_data[i*cell_height:(i+1)*cell_height, j*cell_width:(j+1)*cell_width]
            
            # Check if the cell is drawn (obstacle)
            # We consider a cell as an obstacle if it has some threshold of non-white pixels
            if np.mean(cell[:, :, 0]) < 200:  # If red channel is low, it's likely drawn
                grid[i, j] = -1  # Mark as obstacle
    
    return grid

# Custom Streamlit components

# Function to create a visual grid editor
def grid_editor(grid_size, default_grid=None):
    st.markdown("### Design Your Environment")
    st.markdown("Draw obstacles on the grid. Click and drag to create walls.")
    
    # Set up the canvas
    canvas_width = min(600, grid_size * 50)  # 50px per cell
    canvas_height = canvas_width  # Square canvas
    
    # Create the canvas
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 1)",
        stroke_width=25,
        stroke_color="rgba(0, 0, 0, 1)",
        background_color="#FFFFFF",
        height=canvas_height,
        width=canvas_width,
        drawing_mode="freedraw",
        key="canvas",
    )
    
    # Process the canvas result when available
    if canvas_result.image_data is not None:
        custom_grid = process_grid_drawing(canvas_result, grid_size)
        return custom_grid
    
    return None

# Function to display a metric with a tooltip
def metric_with_tooltip(label, value, tooltip, delta=None):
    html = f"""
    <div style="padding: 10px; background: #f0f2f6; border-radius: 5px; margin-bottom: 10px;">
        <small style="color: #555;">{label}</small>
        <div style="font-size: 1.5rem; font-weight: bold;">{value}</div>
        <div class="tooltip">ℹ️
            <span class="tooltiptext">{tooltip}</span>
        </div>
    """
    if delta:
        html += f'<small style="color: {"green" if float(delta) > 0 else "red"};">{"+" if float(delta) > 0 else ""}{delta}</small>'
    html += "</div>"
    
    return st.markdown(html, unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    
    # Environment Parameters
    st.subheader("🌍 Environment")
    grid_size = st.slider("Grid Size", min_value=4, max_value=15, value=8, step=1)
    obstacle_density = st.slider("Obstacle Density", min_value=0.0, max_value=0.4, value=0.2, step=0.05, 
                                help="Percentage of the grid that will be filled with obstacles (only for random environment)")
    
    env_setup = st.radio("Environment Setup", ["Random", "Custom"], 
                        help="Choose between a randomly generated environment or design your own")
    
    # Q-learning Parameters
    st.subheader("🧠 Learning Parameters")
    st.markdown('<div class="highlight">These parameters control how the agent learns:</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        alpha = st.slider("Learning Rate (α)", min_value=0.1, max_value=1.0, value=0.5, step=0.1, 
                         help="How quickly the agent incorporates new information (higher = faster learning but less stable)")
    
    with col2:
        gamma = st.slider("Discount Factor (γ)", min_value=0.1, max_value=1.0, value=0.9, step=0.1,
                        help="How much the agent values future rewards (higher = more long-term planning)")
    
    epsilon_initial = st.slider("Initial Exploration Rate (ε)", min_value=0.01, max_value=1.0, value=0.3, step=0.05,
                             help="Probability of taking a random action (higher = more exploration)")
    
    epsilon_decay = st.checkbox("Use Epsilon Decay", value=True,
                              help="Gradually reduce exploration rate over time")
    
    if epsilon_decay:
        min_epsilon = st.slider("Minimum Exploration Rate", min_value=0.0, max_value=0.2, value=0.01, step=0.01)
    
    # Training Parameters
    st.subheader("⚙️ Training")
    num_episodes = st.slider("Number of Episodes", min_value=10, max_value=2000, value=200, step=10,
                           help="How many training episodes to run")
    
    render_training = st.checkbox("Visualize Training Process", value=True,
                                help="Show the agent's learning in real-time (slower but more informative)")
    
    st.session_state.simulation_speed = st.slider("Simulation Speed", 0.0, 2.0, 0.5, 0.1,
                                               help="Speed of the visualization (higher = faster)")
    
    # Reset button
    if st.button("Reset Environment & Agent"):
        st.session_state.env = None
        st.session_state.agent = None
        st.session_state.training_complete = False
        st.session_state.rewards_history = []
        st.session_state.steps_history = []
        st.session_state.custom_grid = None
        st.session_state.current_episode = 0
        st.session_state.agent_path = []
        st.success("Environment and agent have been reset!")

# Main app content
def main():
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🛠️ Setup & Train", "📊 Analyze Results", "🎮 Interactive Testing", "📚 Learn About RL"])
    
    with tab1:
        st.markdown('<h2 class="sub-header">Setup Environment & Train Agent</h2>', unsafe_allow_html=True)
        
        # Environment setup section
        col1, col2 = st.columns([3, 2])
        
        with col1:
            if env_setup == "Custom":
                st.markdown('<div class="highlight">Design your own grid world by drawing obstacles:</div>', unsafe_allow_html=True)
                custom_grid = grid_editor(grid_size, st.session_state.custom_grid)
                if custom_grid is not None:
                    st.session_state.custom_grid = custom_grid
                
                if st.button("Create Environment"):
                    # Create environment from custom grid
                    st.session_state.env = GridWorld(size=grid_size, custom_grid=st.session_state.custom_grid)
                    # Create agent
                    st.session_state.agent = QLearningAgent(st.session_state.env, alpha=alpha, gamma=gamma, epsilon=epsilon_initial)
                    st.success("Custom environment created! You can now train the agent.")
            else:
                if st.button("Generate Random Environment"):
                    # Create random environment
                    st.session_state.env = GridWorld(size=grid_size, obstacle_density=obstacle_density)
                    # Create agent
                    st.session_state.agent = QLearningAgent(st.session_state.env, alpha=alpha, gamma=gamma, epsilon=epsilon_initial)
                    st.success("Random environment created! You can now train the agent.")
        
        with col2:
            if st.session_state.env is not None:
                st.markdown("### Environment Preview")
                fig = visualize_env(st.session_state.env)
                st.pyplot(fig)
            else:
                st.info("No environment created yet. Please create one first.")
        
        # Training section
        st.markdown('<h3 class="sub-header">Train the Agent</h3>', unsafe_allow_html=True)
        
        if st.session_state.env is not None and st.session_state.agent is not None:
            col1, col2 = st.columns([3, 2])
            
            with col1:
                if not st.session_state.training_complete:
                    # Training controls
                    start_training = st.button("Start Training")
                    
                    # Progress indicators
                    progress_bar = st.progress(0)
                    episode_counter = st.empty()
                    reward_metric = st.empty()
                    steps_metric = st.empty()
                    
                    # Visualization placeholders
                    path_vis = st.empty()
                    reward_plot = st.empty()
                    
                    if start_training:
                        # Callback to update visuals during training
                        def training_callback(episode, reward, steps, path):
                            st.session_state.current_episode = episode + 1
                            progress = (episode + 1) / num_episodes
                            progress_bar.progress(progress)
                            
                            episode_counter.markdown(f"**Episode: {episode + 1}/{num_episodes}** ({(progress*100):.1f}%)")
                            
                            # Update metrics every few episodes or at the end
                            if episode % 5 == 0 or episode == num_episodes - 1:
                                recent_rewards = st.session_state.rewards_history[-10:] if len(st.session_state.rewards_history) > 10 else st.session_state.rewards_history
                                avg_reward = np.mean(recent_rewards) if recent_rewards else 0
                                
                                reward_metric.markdown(f"**Reward: {reward:.2f}** (Avg last 10: {avg_reward:.2f})")
                                steps_metric.markdown(f"**Steps: {steps}**")
                            
                            # Save the training data
                            st.session_state.rewards_history.append(reward)
                            st.session_state.steps_history.append(steps)
                            
                            # Visualize the current episode if enabled
                            if render_training and (episode % 5 == 0 or episode == num_episodes - 1):
                                fig = visualize_env(st.session_state.env, agent_path=path)
                                path_vis.pyplot(fig)
                                
                                # Update the rewards chart
                                chart_data = pd.DataFrame({
                                    'Episode': range(1, len(st.session_state.rewards_history) + 1),
                                    'Reward': st.session_state.rewards_history
                                })
                                
                                # Create a line chart with a smoothed trend line
                                chart = alt.Chart(chart_data).mark_line(point=True).encode(
                                    x='Episode',
                                    y=alt.Y('Reward', scale=alt.Scale(zero=False))
                                )
                                
                                # Add a trend line
                                if len(chart_data) > 5:
                                    trend = alt.Chart(chart_data).transform_regression(
                                        'Episode', 'Reward'
                                    ).mark_line(color='red').encode(
                                        x='Episode',
                                        y='Reward'
                                    )
                                    reward_plot.altair_chart(chart + trend, use_container_width=True)
                                else:
                                    reward_plot.altair_chart(chart, use_container_width=True)
                                
                                # Control the visualization speed
                                time.sleep(1.0 - min(0.99, st.session_state.simulation_speed))
                                
                            # Update epsilon if using decay
                            if epsilon_decay:
                                decay_factor = 1.0 - (episode / num_episodes)
                                current_epsilon = max(min_epsilon, epsilon_initial * decay_factor)
                                st.session_state.agent.epsilon = current_epsilon
                        
                        # Train the agent with the callback
                        rewards, steps = st.session_state.agent.train(num_episodes, callback=training_callback)
                        
                        # Mark training as complete
                        st.session_state.training_complete = True
                        
                        # Final update
                        progress_bar.progress(1.0)
                        st.success(f"Training completed! The agent learned over {num_episodes} episodes.")
                else:
                    st.success("Training already completed. You can analyze results or test the agent.")
                    if st.button("Train Again"):
                        st.session_state.training_complete = False
                        st.session_state.rewards_history = []
                        st.session_state.steps_history = []
                        st.session_state.current_episode = 0
                        st.experimental_rerun()
            
            with col2:
                if st.session_state.training_complete:
                    # Display training summary
                    recent_rewards = st.session_state.rewards_history[-10:] if len(st.session_state.rewards_history) > 10 else st.session_state.rewards_history
                    avg_reward = np.mean(recent_rewards)
                    max_reward = max(st.session_state.rewards_history)
                    min_steps = min(st.session_state.steps_history[-10:]) if len(st.session_state.steps_history) > 10 else min(st.session_state.steps_history)
                    
                    st.markdown("### Training Results")
                    metric_with_tooltip("Average Reward (Last 10)", f"{avg_reward:.2f}", 
                                      "Higher rewards indicate better performance")
                    metric_with_tooltip("Best Episode Reward", f"{max_reward:.2f}", 
                                      "The highest reward achieved during training")
                    metric_with_tooltip("Shortest Path (Steps)", f"{min_steps}", 
                                      "Fewer steps indicate a more efficient path to the goal")
        else:
            st.info("Please create an environment first before training the agent.")
    
    with tab2:
        st.markdown('<h2 class="sub-header">Analyze Learning Results</h2>', unsafe_allow_html=True)
        
        if st.session_state.training_complete and st.session_state.agent is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Learned Policy")
                policy = st.session_state.agent.get_policy()
                fig_policy = visualize_env(st.session_state.env, policy=policy)
                st.pyplot(fig_policy)
                
                st.markdown("""
                <div class="highlight">
                <strong>Policy Visualization:</strong> Arrows show the optimal action in each state according to what the agent has learned.
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Radio button for what to display on heatmap
                heatmap_type = st.radio("Heatmap Type", ["State Values", "Visit Count"], 
                                      help="Choose what data to visualize on the heatmap")
                
                if heatmap_type == "State Values":
                    st.markdown("### Value Function Heatmap")
                    value_function = st.session_state.agent.get_value_function()
                    fig_heatmap = create_value_heatmap(st.session_state.env, value_function, 
                                                     title="State Value Heatmap")
                    st.pyplot(fig_heatmap)
                    
                    st.markdown("""
                    <div class="highlight">
                    <strong>Value Heatmap:</strong> Shows the expected total reward from each state.
                    Brighter colors indicate higher values (more desirable states).
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("### Visit Count Heatmap")
                    visit_counts = st.session_state.agent.get_visit_heatmap()
                    fig_visits = create_value_heatmap(st.session_state.env, visit_counts, 
                                                    title="State Visit Frequency", 
                                                    vmin=0, vmax=np.max(visit_counts)*0.8)
                    st.pyplot(fig_visits)
                    
                    st.markdown("""
                    <div class="highlight">
                    <strong>Visit Heatmap:</strong> Shows how often the agent visited each state during training.
                    Brighter colors indicate more frequently visited states.
                    </div>
                    """, unsafe_allow_html=True)
            
            # Show the learning curves
            st.markdown("### Learning Curves")
            
            learning_data = pd.DataFrame({
                'Episode': range(1, len(st.session_state.rewards_history) + 1),
                'Reward': st.session_state.rewards_history,
                'Steps': st.session_state.steps_history
            })
            
            # Create a smoothed version for the trend lines
            window_size = min(15, len(learning_data) // 5) if len(learning_data) > 30 else 1
            learning_data['Reward_Smoothed'] = learning_data['Reward'].rolling(window=window_size, center=True).mean()
            learning_data['Steps_Smoothed'] = learning_data['Steps'].rolling(window=window_size, center=True).mean()
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Rewards chart
                rewards_chart = alt.Chart(learning_data).mark_line(color='blue', opacity=0.3).encode(
                    x='Episode',
                    y=alt.Y('Reward', scale=alt.Scale(zero=False), title='Episode Reward')
                ).properties(
                    title='Rewards Over Time',
                    width=350,
                    height=250
                )
                
                # Add smoothed trend line
                rewards_trend = alt.Chart(learning_data).mark_line(color='blue', strokeWidth=3).encode(
                    x='Episode',
                    y=alt.Y('Reward_Smoothed', scale=alt.Scale(zero=False))
                )
                
                st.altair_chart(rewards_chart + rewards_trend, use_container_width=True)
            
            with col2:
                # Steps chart
                steps_chart = alt.Chart(learning_data).mark_line(color='green', opacity=0.3).encode(
                    x='Episode',
                    y=alt.Y('Steps', scale=alt.Scale(zero=False), title='Steps to Goal')
                ).properties(
                    title='Steps Per Episode',
                    width=350,
                    height=250
                )
                
                # Add smoothed trend line
                steps_trend = alt.Chart(learning_data).mark_line(color='green', strokeWidth=3).encode(
                    x='Episode',
                    y=alt.Y('Steps_Smoothed', scale=alt.Scale(zero=False))
                )
                
                st.altair_chart(steps_chart + steps_trend, use_container_width=True)
            
            st.markdown("""
            <div class="highlight">
            <strong>Learning Curves:</strong> These charts show how the agent's performance improved over time.
            <ul>
                <li><strong>Rewards</strong>: Higher rewards indicate better performance</li>
                <li><strong>Steps</strong>: Fewer steps indicate more efficient paths to the goal</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # State inspection section
            st.markdown("### 🔍 Inspect Individual States")
            st.write("Select a cell to see the Q-values for each action at that state")
            
            col1, col2 = st.columns(2)
            with col1:
                selected_x = st.slider("Row (X)", 0, st.session_state.env.size - 1, 0)
            with col2:
                selected_y = st.slider("Column (Y)", 0, st.session_state.env.size - 1, 0)
            
            selected_state = (selected_x, selected_y)
            
            if selected_state in st.session_state.agent.q_table:
                q_values = st.session_state.agent.q_table[selected_state]
                best_action_idx = np.argmax(q_values)
                
                # Display the Q-values in a more visual way
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    st.markdown("#### Q-Values Visualization")
                    fig_state = visualize_env(st.session_state.env, q_values=st.session_state.agent.q_table, 
                                            selected_state=selected_state)
                    st.pyplot(fig_state)
                
                with col2:
                    st.markdown("#### State Details")
                    st.markdown(f"**Position:** Row {selected_x}, Column {selected_y}")
                    st.markdown(f"**Best Action:** {st.session_state.env.action_names[best_action_idx]} (Q = {q_values[best_action_idx]:.2f})")
                    st.markdown(f"**Visit Count:** {st.session_state.agent.visit_counts.get(selected_state, 0)}")
                    
                    # Q-values table with visual indicators
                    q_data = []
                    for i, (action, q_val) in enumerate(zip(st.session_state.env.action_names, q_values)):
                        is_best = i == best_action_idx
                        q_data.append({
                            "Action": action,
                            "Q-Value": f"{q_val:.2f}",
                            "Is Best": "✅" if is_best else ""
                        })
                    
                    st.table(pd.DataFrame(q_data))
            else:
                st.error("This state is an obstacle or out of bounds. No Q-values available.")
        
        else:
            st.info("Please train the agent first before analyzing results.")
    
    with tab3:
        st.markdown('<h2 class="sub-header">Interactive Agent Testing</h2>', unsafe_allow_html=True)
        
        if st.session_state.training_complete and st.session_state.agent is not None:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("### Test Settings")
                test_epsilon = st.slider("Test Exploration Rate", 0.0, 1.0, 0.0, 0.01, 
                                       help="Set to 0 for fully exploiting the learned policy, or higher for more exploration")
                
                interactive_mode = st.radio("Testing Mode", ["Automatic", "Step-by-Step"], 
                                          help="Run a full episode automatically or step through it manually")
                
                run_test = st.button("Run Test")
                step_button = st.empty()
                reset_button = st.empty()
            
            with col2:
                # Placeholder for the visualization
                path_vis = st.empty()
                metrics_container = st.empty()
            
            if run_test or 'test_running' in st.session_state:
                if run_test:
                    # Reset the test state
                    st.session_state.agent_path = [st.session_state.env.reset()]
                    st.session_state.test_running = True
                    st.session_state.test_done = False
                    st.session_state.test_reward = 0
                    st.session_state.test_step = 0
                
                # Display current state
                fig = visualize_env(st.session_state.env, policy=st.session_state.agent.get_policy(), 
                                  agent_path=st.session_state.agent_path)
                path_vis.pyplot(fig)
                
                # Display metrics
                with metrics_container:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Steps", f"{st.session_state.test_step}")
                    col2.metric("Total Reward", f"{st.session_state.test_reward:.2f}")
                    status = "Goal Reached! 🎉" if st.session_state.test_done and st.session_state.agent_path[-1] == st.session_state.env.goal else \
                             "Failed" if st.session_state.test_done else "In Progress"
                    col3.metric("Status", status)
                
                if interactive_mode == "Automatic" and not st.session_state.test_done:
                    # Automatically run the episode
                    state = st.session_state.agent_path[-1]
                    
                    # Choose action
                    if random.random() < test_epsilon:
                        action = random.randint(0, 3)
                    else:
                        if state in st.session_state.agent.q_table:
                            action = np.argmax(st.session_state.agent.q_table[state])
                        else:
                            action = random.randint(0, 3)
                    
                    # Take a step
                    next_state, reward, done = st.session_state.env.step(action)
                    
                    # Update the path and metrics
                    st.session_state.agent_path.append(next_state)
                    st.session_state.test_reward += reward
                    st.session_state.test_step += 1
                    st.session_state.test_done = done
                    
                    # Re-render
                    time.sleep(0.5 - min(0.4, st.session_state.simulation_speed))
                    st.experimental_rerun()
                
                elif interactive_mode == "Step-by-Step":
                    # Manual stepping
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if not st.session_state.test_done:
                            if step_button.button("Take Step"):
                                state = st.session_state.agent_path[-1]
                                
                                # Choose action
                                if random.random() < test_epsilon:
                                    action = random.randint(0, 3)
                                else:
                                    if state in st.session_state.agent.q_table:
                                        action = np.argmax(st.session_state.agent.q_table[state])
                                    else:
                                        action = random.randint(0, 3)
                                
                                # Take a step
                                next_state, reward, done = st.session_state.env.step(action)
                                
                                # Update the path and metrics
                                st.session_state.agent_path.append(next_state)
                                st.session_state.test_reward += reward
                                st.session_state.test_step += 1
                                st.session_state.test_done = done
                                
                                # Re-render
                                st.experimental_rerun()
                    
                    with col2:
                        if reset_button.button("Reset Test"):
                            # Reset the test
                            if 'test_running' in st.session_state:
                                del st.session_state.test_running
                            st.experimental_rerun()
            
            # Display path details if a test has been run
            if 'test_running' in st.session_state and len(st.session_state.agent_path) > 1:
                st.markdown("### Path Details")
                
                # Create a table of the path
                path_data = []
                for i, state in enumerate(st.session_state.agent_path):
                    if i < len(st.session_state.agent_path) - 1:
                        next_state = st.session_state.agent_path[i+1]
                        # Determine the action taken
                        dx = next_state[0] - state[0]
                        dy = next_state[1] - state[1]
                        action_idx = -1
                        for j, (adx, ady) in enumerate(st.session_state.env.actions):
                            if adx == dx and ady == dy:
                                action_idx = j
                        
                        action_name = st.session_state.env.action_names[action_idx] if action_idx != -1 else "None"
                    else:
                        action_name = "Goal Reached" if state == st.session_state.env.goal else "Current Position"
                    
                    path_data.append({
                        'Step': i,
                        'Position': f"({state[0]}, {state[1]})",
                        'Action': action_name
                    })
                
                # Display the table
                st.dataframe(pd.DataFrame(path_data), hide_index=True)
        
        else:
            st.info("Please train the agent first before testing it.")
    
    with tab4:
        st.markdown('<h2 class="sub-header">Learn About Reinforcement Learning</h2>', unsafe_allow_html=True)
        
        # RL Fundamentals
        st.markdown("### The Basics of Reinforcement Learning")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            Reinforcement Learning (RL) is a type of machine learning where an agent learns to make decisions
            by taking actions in an environment to maximize a reward signal.
            
            #### Key Components:
            
            1. **Agent**: The learner or decision-maker (in our demo, the entity navigating the grid)
            2. **Environment**: The world the agent interacts with (our grid world)
            3. **State**: The current situation of the agent (position in the grid)
            4. **Action**: What the agent can do in a given state (move up, down, left, or right)
            5. **Reward**: Feedback from the environment (penalty for obstacles, reward for reaching goal)
            6. **Policy**: Strategy the agent follows to decide actions (derived from the Q-table)
            """)
        
        with col2:
            st.image("https://upload.wikimedia.org/wikipedia/commons/1/1b/Reinforcement_learning_diagram.svg", 
                   caption="RL Cycle: Agent-Environment Interaction", width=250)
        
        # Q-Learning Algorithm
        st.markdown("### Q-Learning Explained")
        
        st.markdown("""
        Q-Learning is a value-based reinforcement learning algorithm that learns the value of taking a specific action in a particular state.
        
        #### The Q-Learning Update Formula:
        
        Q(s, a) ← Q(s, a) + α [R + γ · max Q(s', a') - Q(s, a)]
        
        Where:
        - Q(s, a): Q-value for state s and action a
        - α (alpha): Learning rate - how quickly new information overrides old information
        - R: Reward received for taking action a in state s
        - γ (gamma): Discount factor - importance of future rewards
        - max Q(s', a'): Maximum Q-value for the next state across all possible actions
        
        #### How Q-Learning Works in Our Grid World:
        
        1. **Initialization**: Start with all Q-values at zero (no knowledge)
        2. **Exploration vs Exploitation**: The agent sometimes takes random actions (exploration) based on epsilon, and sometimes chooses the best-known action (exploitation)
        3. **Learning**: After each action, update the Q-value based on the reward and the best possible future Q-value
        4. **Convergence**: Over time, the Q-table converges to the optimal policy if the agent explores enough
        """)
        
        # Interactive learning element
        st.markdown("### Try a Simple Q-Learning Step:")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            simple_alpha = st.slider("Learning Rate (α)", 0.0, 1.0, 0.5, 0.1, key="simple_alpha")
            simple_gamma = st.slider("Discount Factor (γ)", 0.0, 1.0, 0.9, 0.1, key="simple_gamma")
            
        with col2:
            st.markdown("""
            Suppose our agent is in a state with a current Q-value of **0.0** for action "move right".
            The agent takes this action and receives a reward of **-0.1**.
            The next state has a maximum Q-value of **5.0**.
            """)
        
        with col3:
            # Calculate the new Q-value
            current_q = 0.0
            reward = -0.1
            next_max_q = 5.0
            
            new_q = current_q + simple_alpha * (reward + simple_gamma * next_max_q - current_q)
            
            st.metric("New Q-Value", f"{new_q:.2f}")
            
            # Show the calculation
            st.markdown(f"""
            **Calculation:**  
            Q(s,a) = 0.0 + {simple_alpha:.1f} × (-0.1 + {simple_gamma:.1f} × 5.0 - 0.0)  
            = {new_q:.2f}
            """)
        
        # Applications of RL
        st.markdown("### Real-World Applications of Reinforcement Learning")
        
        app_tabs = st.tabs(["🎮 Games", "🤖 Robotics", "📈 Business", "🏥 Healthcare"])
        
        with app_tabs[0]:
            st.markdown("""
            ### Games and Game AI
            - **AlphaGo/AlphaZero**: Defeated world champions in Go using RL
            - **Video Game NPCs**: Creating more intelligent, adaptive opponents
            - **Game Testing**: Automated game testers that find bugs and exploits
            """)
            
            st.image("https://www.kdnuggets.com/images/reinforcement-learning-fig1-700.jpg", 
                    caption="AlphaGo playing against Lee Sedol", width=400)
            
        with app_tabs[1]:
            st.markdown("""
            ### Robotics and Control
            - **Robot Navigation**: Teaching robots to move through complex environments
            - **Industrial Automation**: Optimizing manufacturing processes
            - **Self-Driving Cars**: Learning to navigate and make real-time decisions
            """)
            
            st.image("https://miro.medium.com/max/1400/1*hQhXxyvmdPLKKcxuSAILEQ.png", 
                    caption="Robot learning to walk through reinforcement learning", width=400)
            
        with app_tabs[2]:
            st.markdown("""
            ### Business and Economics
            - **Trading Algorithms**: Learning optimal trading strategies
            - **Resource Allocation**: Optimizing distribution of goods and services
            - **Dynamic Pricing**: Adjusting prices based on demand and other factors
            """)
            
        with app_tabs[3]:
            st.markdown("""
            ### Healthcare
            - **Treatment Optimization**: Finding optimal treatment regimens for diseases like cancer
            - **Drug Discovery**: Guiding the search for new pharmaceutical compounds
            - **Personalized Medicine**: Tailoring treatments to individual patient characteristics
            """)
        
        # Further Learning
        st.markdown("### 📚 Resources for Learning More")
        
        st.markdown("""
        - **Books**:
          - "Reinforcement Learning: An Introduction" by Richard S. Sutton and Andrew G. Barto
          - "Deep Reinforcement Learning Hands-On" by Maxim Lapan
          
        - **Online Courses**:
          - David Silver's Reinforcement Learning Course (UCL)
          - CS285: Deep Reinforcement Learning (UC Berkeley)
          - Practical RL (Higher School of Economics)
          
        - **Frameworks & Libraries**:
          - OpenAI Gym/Gymnasium: Standard environments for RL research
          - Stable Baselines3: Reliable implementations of RL algorithms
          - TensorFlow Agents and PyTorch RL: Deep RL implementations
        """)

# Run the app
if __name__ == "__main__":
    main()
