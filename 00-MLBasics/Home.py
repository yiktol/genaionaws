# AI Learning Path: Understanding Modern Intelligent Systems


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="AI Learning Path",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 28px;
        font-weight: bold;
        color: #0D47A1;
        margin-top: 30px;
        margin-bottom: 15px;
    }
    .concept-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .highlight-text {
        color: #1E88E5;
        font-weight: bold;
    }
    .info-box {
        background-color: #e3f2fd;
        border-left: 5px solid #1E88E5;
        padding: 15px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Sidebar
    with st.sidebar:
        st.title("Navigation")
        page = st.radio("Go to", [
            "Introduction",
            "AI vs ML vs DL vs GenAI",
            "Learning Paradigms",
            "ML Terminology",
            "Industry Use Cases"
        ])
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("This interactive guide explains the fundamental concepts behind modern intelligent systems.")
        st.markdown("© 2025 AI Learning Path")

    # Main content based on selection
    if page == "Introduction":
        introduction()
    elif page == "AI vs ML vs DL vs GenAI":
        ai_ml_dl_genai()
    elif page == "Learning Paradigms":
        learning_paradigms()
    elif page == "ML Terminology":
        ml_terminology()
    elif page == "Industry Use Cases":
        industry_use_cases()

def introduction():
    st.markdown('<div class="main-header">Understanding Modern Intelligent Systems</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        Welcome to this comprehensive guide on artificial intelligence and its related fields. 
        This interactive application will help you understand the various concepts, methodologies, 
        and applications that form the foundation of modern intelligent systems.
        
        ### What You'll Learn
        
        - The distinctions between AI, ML, DL, and Generative AI
        - Various learning paradigms in machine learning
        - Key terminology used in the field
        - Real-world applications across different industries
        """)
        
        st.markdown('<div class="info-box">Use the sidebar navigation to explore different topics.</div>', unsafe_allow_html=True)
    
    with col2:
        # Creating a simple AI evolution hierarchy chart
        fig, ax = plt.figure(figsize=(5, 6)), plt.subplot(111)
        
        data = {
            'Level': [4, 3, 2, 1],
            'Technology': ['Generative AI', 'Deep Learning', 'Machine Learning', 'Artificial Intelligence'],
            'Width': [0.6, 0.7, 0.8, 1.0]
        }
        
        ax.barh(data['Level'], data['Width'], color=['#FF9800', '#4CAF50', '#2196F3', '#3F51B5'])
        ax.set_yticks(data['Level'])
        ax.set_yticklabels(data['Technology'])
        ax.set_xlim(0, 1.1)
        ax.set_xticks([])
        ax.set_title('AI Evolution Hierarchy')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        st.pyplot(fig)

def ai_ml_dl_genai():
    st.markdown('<div class="main-header">AI vs ML vs DL vs Generative AI</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Understanding the relationships and differences between these fields is crucial for anyone working in data science or AI.
    Let's break down each concept:
    """)
    
    # Creating tabs for each concept
    tab1, tab2, tab3, tab4 = st.tabs(["Artificial Intelligence", "Machine Learning", "Deep Learning", "Generative AI"])
    
    with tab1:
        st.markdown('<div class="concept-box">', unsafe_allow_html=True)
        st.markdown("### Artificial Intelligence (AI)")
        st.markdown("""
        **Definition:** The broadest field that deals with creating systems capable of performing tasks that typically require human intelligence.
        
        **Key Characteristics:**
        - Problem-solving and reasoning
        - Knowledge representation
        - Planning and decision-making
        - Natural language processing
        - Perception (vision, speech)
        - Ability to manipulate the environment
        
        **Examples:** Expert systems, game-playing AI (like AlphaGo), virtual assistants
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="concept-box">', unsafe_allow_html=True)
        st.markdown("### Machine Learning (ML)")
        st.markdown("""
        **Definition:** A subset of AI focused on creating systems that learn from data without being explicitly programmed.
        
        **Key Characteristics:**
        - Data-driven approach
        - Statistical modeling
        - Pattern recognition
        - Algorithm-based predictions
        - Improves with experience/data
        
        **Common Algorithms:**
        - Linear/Logistic Regression
        - Decision Trees
        - Random Forests
        - Support Vector Machines
        - K-means Clustering
        
        **Examples:** Recommendation systems, spam filters, fraud detection
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="concept-box">', unsafe_allow_html=True)
        st.markdown("### Deep Learning (DL)")
        st.markdown("""
        **Definition:** A specialized subset of machine learning that uses multi-layered neural networks to analyze various data types.
        
        **Key Characteristics:**
        - Uses neural networks with many layers (hence "deep")
        - Automatic feature extraction
        - Requires large amounts of data
        - Computationally intensive
        - Handles unstructured data well
        
        **Common Architectures:**
        - Convolutional Neural Networks (CNNs)
        - Recurrent Neural Networks (RNNs)
        - Long Short-Term Memory (LSTM) networks
        - Transformers
        
        **Examples:** Image recognition, speech recognition, natural language processing
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<div class="concept-box">', unsafe_allow_html=True)
        st.markdown("### Generative AI")
        st.markdown("""
        **Definition:** A subset of deep learning focused on creating new content that resembles its training data.
        
        **Key Characteristics:**
        - Creates novel outputs (text, images, audio, etc.)
        - Based on patterns learned from training data
        - Often uses advanced neural network architectures
        - Balances creativity with coherence
        
        **Common Architectures:**
        - Generative Adversarial Networks (GANs)
        - Variational Autoencoders (VAEs)
        - Transformer-based models (GPT, BERT)
        - Diffusion models
        
        **Examples:** ChatGPT, DALL-E, Stable Diffusion, Midjourney, voice synthesis
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Visual representation of the relationships
    st.markdown("### Relationship Between Fields")
    
    fig, ax = plt.figure(figsize=(10, 6)), plt.subplot(111)
    
    # Create a series of nested circles
    ai_circle = plt.Circle((0.5, 0.5), 0.4, color='#3F51B5', alpha=0.2)
    ml_circle = plt.Circle((0.5, 0.5), 0.3, color='#2196F3', alpha=0.3)
    dl_circle = plt.Circle((0.5, 0.5), 0.2, color='#4CAF50', alpha=0.3)
    gen_circle = plt.Circle((0.5, 0.5), 0.1, color='#FF9800', alpha=0.4)
    
    ax.add_patch(ai_circle)
    ax.add_patch(ml_circle)
    ax.add_patch(dl_circle)
    ax.add_patch(gen_circle)
    
    # Add labels
    plt.text(0.5, 0.89, 'Artificial Intelligence', ha='center', va='center', fontsize=12, fontweight='bold')
    plt.text(0.5, 0.77, 'Machine Learning', ha='center', va='center', fontsize=11, fontweight='bold')
    plt.text(0.5, 0.65, 'Deep Learning', ha='center', va='center', fontsize=10, fontweight='bold')
    plt.text(0.5, 0.5, 'Generative AI', ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Set the limits and remove axes
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    
    st.pyplot(fig)
    
    st.markdown("""
    ### Key Takeaways
    
    - **Artificial Intelligence** is the broadest field encompassing all techniques that enable machines to mimic human intelligence.
    - **Machine Learning** is a subset of AI focused specifically on algorithms that can learn from data.
    - **Deep Learning** is a specialized subset of machine learning using neural networks with multiple layers.
    - **Generative AI** is the newest frontier, focusing on creating new content rather than just analyzing existing data.
    
    Each field builds upon the foundations of the broader categories while adding its own specialized techniques and approaches.
    """)

def learning_paradigms():
    st.markdown('<div class="main-header">Learning Paradigms in AI</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Machine learning algorithms can be categorized based on how they learn from data. 
    Understanding these learning paradigms is crucial for selecting the right approach for a specific problem.
    """)
    
    # Creating expandable sections for each paradigm
    with st.expander("Supervised Learning", expanded=True):
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            ### Supervised Learning
            
            In supervised learning, algorithms learn from labeled training data to predict outputs for unseen data.
            
            **Key Characteristics:**
            - Requires labeled data (inputs paired with correct outputs)
            - Goal is to learn a mapping function from input to output
            - Performance can be clearly measured
            
            **Common Algorithms:**
            - Linear and Logistic Regression
            - Support Vector Machines
            - Decision Trees and Random Forests
            - K-Nearest Neighbors
            - Neural Networks
            
            **Applications:**
            - Classification problems (spam detection, image recognition)
            - Regression problems (price prediction, sales forecasting)
            """)
        
        with col2:
            # Simple visualization for supervised learning
            fig, ax = plt.figure(figsize=(5, 4)), plt.subplot(111)
            
            # Generate some sample data
            np.random.seed(42)
            x = np.random.rand(30)
            y = 2 * x + np.random.normal(0, 0.1, 30)
            
            # Plot data points
            ax.scatter(x, y, color='blue', label='Training data')
            
            # Plot prediction line
            x_line = np.linspace(0, 1, 100)
            y_line = 2 * x_line
            ax.plot(x_line, y_line, color='red', label='Prediction model')
            
            ax.set_title('Supervised Learning Example')
            ax.set_xlabel('Feature')
            ax.set_ylabel('Target')
            ax.legend()
            
            st.pyplot(fig)
    
    with st.expander("Unsupervised Learning"):
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            ### Unsupervised Learning
            
            In unsupervised learning, algorithms identify patterns, relationships, or structures in unlabeled data.
            
            **Key Characteristics:**
            - Works with unlabeled data
            - Discovers hidden patterns or intrinsic structures
            - No explicit feedback on performance
            
            **Common Algorithms:**
            - K-means Clustering
            - Hierarchical Clustering
            - Principal Component Analysis (PCA)
            - t-SNE
            - Autoencoders
            
            **Applications:**
            - Customer segmentation
            - Anomaly detection
            - Dimensionality reduction
            - Feature learning
            - Recommendation systems
            """)
        
        with col2:
            # Simple visualization for unsupervised learning (clustering)
            fig, ax = plt.figure(figsize=(5, 4)), plt.subplot(111)
            
            # Generate some clustered data
            np.random.seed(42)
            cluster1_x = np.random.normal(0.3, 0.1, 20)
            cluster1_y = np.random.normal(0.3, 0.1, 20)
            cluster2_x = np.random.normal(0.7, 0.1, 20)
            cluster2_y = np.random.normal(0.7, 0.1, 20)
            
            # Plot clusters
            ax.scatter(cluster1_x, cluster1_y, color='blue', label='Cluster 1')
            ax.scatter(cluster2_x, cluster2_y, color='green', label='Cluster 2')
            
            # Draw cluster centers
            ax.scatter([0.3, 0.7], [0.3, 0.7], color='red', marker='X', s=100, label='Cluster centers')
            
            ax.set_title('Unsupervised Learning (Clustering)')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.legend()
            
            st.pyplot(fig)
    
    with st.expander("Reinforcement Learning"):
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            ### Reinforcement Learning
            
            In reinforcement learning, an agent learns to make decisions by interacting with an environment and receiving feedback in the form of rewards or penalties.
            
            **Key Characteristics:**
            - Based on actions, states, and rewards
            - Learning through trial and error
            - Delayed feedback (rewards may come after several steps)
            - Balance between exploration and exploitation
            
            **Common Algorithms:**
            - Q-learning
            - Deep Q-Networks (DQN)
            - Policy Gradient Methods
            - Actor-Critic Methods
            - Proximal Policy Optimization (PPO)
            
            **Applications:**
            - Game playing (AlphaGo, Dota 2)
            - Robotics and autonomous systems
            - Resource management
            - Personalized recommendations
            - Trading and finance
            """)
        
        with col2:
            # Simple visualization for reinforcement learning
            fig, ax = plt.figure(figsize=(5, 4)), plt.subplot(111)
            
            # Create a simple maze-like environment
            grid = np.zeros((5, 5))
            grid[0, 0] = 1  # Start
            grid[4, 4] = 2  # Goal
            grid[1:4, 2] = -1  # Obstacle
            
            # Display the grid
            cmap = plt.cm.colors.ListedColormap(['white', 'green', 'gold', 'red'])
            bounds = [-1.5, -0.5, 0.5, 1.5, 2.5]
            norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
            
            ax.imshow(grid, cmap=cmap, norm=norm)
            
            # Add a path
            # path_x = [0, 1, 2, 3, 4, 4, 4]
            # path_y = [0, 0, 0, 0, 0, 1, 2, 3, 4]
            # ax.plot(path_y[:5], path_x[:5], 'b-', linewidth=2)
            # ax.plot(path_y[4:], path_x[4:], 'b-', linewidth=2)
            
            # Fix the path coordinates to have matching dimensions
            path_x = [0, 1, 2, 3, 4, 4, 4]  # 7 points
            path_y = [0, 0, 0, 0, 0, 1, 2]  # 7 points matching path_x

            ax.plot(path_y[:5], path_x[:5], 'b-', linewidth=2)
            ax.plot(path_y[4:], path_x[4:], 'b-', linewidth=2)
            
            ax.set_title('Reinforcement Learning')
            ax.set_xticks([])
            ax.set_yticks([])
            
            st.pyplot(fig)
            


    
    with st.expander("Semi-Supervised Learning"):
        st.markdown("""
        ### Semi-Supervised Learning
        
        Semi-supervised learning combines elements of supervised and unsupervised learning, using a small amount of labeled data with a large amount of unlabeled data.
        
        **Key Characteristics:**
        - Uses both labeled and unlabeled data
        - Particularly useful when labeling data is expensive or time-consuming
        - Can achieve good performance with limited labeled examples
        
        **Common Approaches:**
        - Self-training
        - Multi-view training
        - Graph-based methods
        - Generative models
        
        **Applications:**
        - Text classification with limited annotations
        - Medical image analysis
        - Speech recognition
        - Web content classification
        """)
    
    with st.expander("Self-Supervised Learning"):
        st.markdown("""
        ### Self-Supervised Learning
        
        Self-supervised learning is a form of unsupervised learning where the system creates its own labels from the input data.
        
        **Key Characteristics:**
        - Generates supervisory signals from the data itself
        - No human annotations required
        - Learns useful representations that can transfer to downstream tasks
        
        **Common Approaches:**
        - Contrastive learning
        - Masked language modeling
        - Context prediction
        - Rotation prediction
        
        **Applications:**
        - Pre-training language models (BERT, GPT)
        - Computer vision feature learning
        - Speech representation learning
        - Multi-modal learning
        """)
    
    # Comparison table
    st.markdown("### Comparison of Learning Paradigms")
    
    comparison_data = {
        'Paradigm': ['Supervised', 'Unsupervised', 'Reinforcement', 'Semi-Supervised', 'Self-Supervised'],
        'Data Requirements': ['Labeled data', 'Unlabeled data', 'Environment feedback', 'Small labeled + large unlabeled', 'Unlabeled data'],
        'Goal': ['Predict outputs', 'Find patterns', 'Maximize rewards', 'Leverage unlabeled data', 'Create & learn from pseudo-labels'],
        'Example Use Case': ['Classification', 'Clustering', 'Game playing', 'Medical imaging', 'Language modeling']
    }
    
    df = pd.DataFrame(comparison_data)
    st.table(df)

def ml_terminology():
    st.markdown('<div class="main-header">Machine Learning Terminology</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Understanding the key terminology in machine learning is essential for effective communication in the field.
    Here are the most important terms and concepts you'll encounter:
    """)
    
    # Create two columns for better organization
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="sub-header">Data Terminology</div>', unsafe_allow_html=True)
        
        terms_data = [
            ("**Features (X)**", "Input variables or attributes used for making predictions."),
            ("**Labels/Targets (y)**", "Output variables that the model aims to predict."),
            ("**Training Set**", "Data used to train the model."),
            ("**Validation Set**", "Data used to tune hyperparameters and evaluate during training."),
            ("**Test Set**", "Unseen data used to evaluate the final model performance."),
            ("**Batch**", "A subset of training examples used in one iteration of model training."),
            ("**Data Preprocessing**", "Transforming raw data into a suitable format for modeling.")
        ]
        
        for term, definition in terms_data:
            st.markdown(f"{term}: {definition}")
        
        st.markdown('<div class="sub-header">Model Evaluation</div>', unsafe_allow_html=True)
        
        terms_eval = [
            ("**Accuracy**", "Proportion of correct predictions among the total predictions."),
            ("**Precision**", "Proportion of true positives among all positive predictions."),
            ("**Recall**", "Proportion of true positives among all actual positives."),
            ("**F1 Score**", "Harmonic mean of precision and recall."),
            ("**ROC-AUC**", "Area under the Receiver Operating Characteristic curve."),
            ("**Confusion Matrix**", "Table showing correct and incorrect classification counts."),
            ("**Mean Squared Error (MSE)**", "Average of squared differences between predictions and actual values.")
        ]
        
        for term, definition in terms_eval:
            st.markdown(f"{term}: {definition}")
    
    with col2:
        st.markdown('<div class="sub-header">Model Terminology</div>', unsafe_allow_html=True)
        
        terms_model = [
            ("**Parameters**", "Internal model variables learned during training."),
            ("**Hyperparameters**", "External configuration variables set before training."),
            ("**Bias**", "Error from wrong assumptions in the learning algorithm."),
            ("**Variance**", "Error from sensitivity to small fluctuations in the training set."),
            ("**Overfitting**", "Model performs well on training data but poorly on new data."),
            ("**Underfitting**", "Model is too simple to capture the underlying pattern."),
            ("**Regularization**", "Techniques to prevent overfitting (L1, L2 regularization).")
        ]
        
        for term, definition in terms_model:
            st.markdown(f"{term}: {definition}")
        
        st.markdown('<div class="sub-header">Learning Concepts</div>', unsafe_allow_html=True)
        
        terms_learning = [
            ("**Gradient Descent**", "Optimization algorithm to minimize the loss function."),
            ("**Backpropagation**", "Algorithm to calculate gradients in neural networks."),
            ("**Epoch**", "One complete pass through the entire training dataset."),
            ("**Learning Rate**", "Step size at each iteration of the optimization algorithm."),
            ("**Loss Function**", "Function that measures how well the model is performing."),
            ("**One-hot Encoding**", "Representing categorical variables as binary vectors."),
            ("**Feature Engineering**", "Process of creating new features from existing data.")
        ]
        
        for term, definition in terms_learning:
            st.markdown(f"{term}: {definition}")
    
    # Interactive glossary search
    st.markdown('<div class="sub-header">Glossary Search</div>', unsafe_allow_html=True)
    
    # Combine all terms and definitions
    all_terms = terms_data + terms_eval

    # Combine all terms and definitions
    all_terms = terms_data + terms_eval + terms_model + terms_learning
    
    # Create a dictionary for search
    term_dict = {term.replace("**", ""): definition for term, definition in all_terms}
    
    # Search box
    search_query = st.text_input("Search for a term")
    if search_query:
        results = {term: definition for term, definition in term_dict.items() 
                 if search_query.lower() in term.lower()}
        
        if results:
            st.markdown("### Search Results")
            for term, definition in results.items():
                st.markdown(f"**{term}**: {definition}")
        else:
            st.warning(f"No results found for '{search_query}'")
    
    # Visual representation of related terms
    st.markdown('<div class="sub-header">Common ML Workflow</div>', unsafe_allow_html=True)
    
    # Create a simple flow diagram
    stages = ['Data Collection', 'Preprocessing', 'Feature Engineering', 'Model Training', 'Validation', 'Testing', 'Deployment']
    
    fig, ax = plt.figure(figsize=(10, 2)), plt.subplot(111)
    
    # Plot the workflow stages
    for i, stage in enumerate(stages):
        ax.add_patch(plt.Rectangle((i, 0), 0.8, 0.8, fill=True, color=f'C{i}', alpha=0.7))
        ax.text(i + 0.4, 0.4, stage, ha='center', va='center', fontsize=9)
        if i < len(stages) - 1:
            ax.arrow(i + 0.85, 0.4, 0.1, 0, head_width=0.1, head_length=0.05, fc='black', ec='black')
    
    ax.set_xlim(-0.1, len(stages))
    ax.set_ylim(-0.1, 0.9)
    ax.axis('off')
    
    st.pyplot(fig)

def industry_use_cases():
    st.markdown('<div class="main-header">Industry Use Cases of Machine Learning</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Machine learning technologies are transforming industries across the board. 
    Let's explore some of the most impactful applications in different sectors.
    """)
    
    # Create tabs for different industries
    tabs = st.tabs(["Healthcare", "Finance", "Retail", "Manufacturing", "Transportation", "Entertainment"])
    
    with tabs[0]:  # Healthcare
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            ### Healthcare
            
            Machine learning is revolutionizing healthcare through improved diagnostics, personalized treatment, and operational efficiency.
            
            **Key Applications:**
            
            1. **Medical Imaging & Diagnostics**
               - Detecting abnormalities in X-rays, MRIs, and CT scans
               - Early detection of diseases like cancer and diabetes
               - Analyzing pathology slides
            
            2. **Drug Discovery**
               - Predicting molecular properties
               - Identifying potential drug candidates
               - Simulating drug interactions
            
            3. **Personalized Medicine**
               - Treatment recommendation systems
               - Predicting patient response to treatments
               - Genomics and precision medicine
            
            4. **Patient Monitoring**
               - Predicting patient deterioration
               - Remote monitoring via wearable devices
               - ICU management systems
            
            5. **Administrative Efficiency**
               - Optimizing hospital operations
               - Predicting patient readmissions
               - Resource allocation and staff scheduling
            """)
        
        with col2:
            st.image("https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=500&q=80", caption="Healthcare AI Applications")
            
            st.markdown("### Success Story")
            st.info("""
            **Google DeepMind's AI system** demonstrated the ability to detect over 50 eye diseases from optical coherence tomography (OCT) scans with accuracy matching world-leading expert doctors.
            """)
    
    with tabs[1]:  # Finance
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            ### Finance
            
            The financial sector leverages machine learning for risk assessment, fraud detection, and automated trading.
            
            **Key Applications:**
            
            1. **Fraud Detection**
               - Real-time transaction monitoring
               - Anomaly detection in spending patterns
               - Anti-money laundering systems
            
            2. **Algorithmic Trading**
               - High-frequency trading strategies
               - Market prediction models
               - Portfolio optimization
            
            3. **Credit Scoring**
               - Alternative credit assessment models
               - Default risk prediction
               - Loan approval automation
            
            4. **Customer Service**
               - Intelligent chatbots for banking
               - Personalized financial advice
               - Customer churn prediction
            
            5. **Risk Management**
               - Market risk assessment
               - Insurance underwriting
               - Compliance monitoring
            """)
        
        with col2:
            st.image("https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=500&q=80", caption="Financial AI Applications")
            
            st.markdown("### Success Story")
            st.info("""
            **JPMorgan's COiN platform** uses natural language processing to analyze legal documents and extract important data points, completing in seconds work that previously took lawyers 360,000 hours annually.
            """)
    
    with tabs[2]:  # Retail
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            ### Retail
            
            Machine learning helps retailers personalize customer experiences, optimize inventory, and streamline operations.
            
            **Key Applications:**
            
            1. **Recommendation Systems**
               - Product recommendations based on browsing history
               - Personalized marketing campaigns
               - Cross-selling and upselling opportunities
            
            2. **Inventory Management**
               - Demand forecasting
               - Optimal stock level prediction
               - Supply chain optimization
            
            3. **Price Optimization**
               - Dynamic pricing strategies
               - Competitive price monitoring
               - Promotion effectiveness analysis
            
            4. **Customer Experience**
               - Virtual shopping assistants
               - Visual search capabilities
               - Customer sentiment analysis
            
            5. **Store Operations**
               - Checkout-free shopping (e.g., Amazon Go)
               - In-store traffic analysis
               - Shelf monitoring systems
            """)
        
        with col2:
            st.image("https://images.unsplash.com/photo-1531973576160-7125cd663d86?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=500&q=80", caption="Retail AI Applications")
            
            st.markdown("### Success Story")
            st.info("""
            **Amazon's recommendation engine** drives 35% of total company revenue through its sophisticated ML algorithms that analyze user behavior and purchase history.
            """)
    
    with tabs[3]:  # Manufacturing
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            ### Manufacturing
            
            Machine learning is transforming production with predictive maintenance, quality control, and process optimization.
            

            **Key Applications:**
            
            1. **Predictive Maintenance**
               - Equipment failure prediction
               - Maintenance scheduling optimization
               - Anomaly detection in machinery
            
            2. **Quality Control**
               - Automated visual inspection
               - Defect detection and classification
               - Process variation analysis
            
            3. **Supply Chain Optimization**
               - Demand forecasting
               - Inventory management
               - Logistics optimization
            
            4. **Process Optimization**
               - Energy consumption reduction
               - Production parameter optimization
               - Yield improvement
            
            5. **Design Optimization**
               - Generative design
               - Material property prediction
               - Simulation and digital twins
            """)
        
        with col2:
            st.image("https://images.unsplash.com/photo-1581093450021-4a7360e9a6b5?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=500&q=80", caption="Manufacturing AI Applications")
            
            st.markdown("### Success Story")
            st.info("""
            **Siemens** implemented machine learning for predictive maintenance in their gas turbines, reducing unplanned downtime by 30% and maintenance costs by 20%.
            """)
    
    with tabs[4]:  # Transportation
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            ### Transportation
            
            From autonomous vehicles to traffic management, ML is reshaping how we move people and goods.
            
            **Key Applications:**
            
            1. **Autonomous Vehicles**
               - Self-driving cars and trucks
               - Computer vision for object detection
               - Path planning algorithms
            
            2. **Ride-Sharing Optimization**
               - Dynamic pricing models
               - Driver-rider matching algorithms
               - ETA prediction
            
            3. **Traffic Management**
               - Traffic flow prediction
               - Intelligent traffic signal control
               - Congestion management
            
            4. **Logistics and Delivery**
               - Route optimization
               - Delivery time estimation
               - Last-mile delivery automation
            
            5. **Maintenance and Safety**
               - Vehicle predictive maintenance
               - Driver behavior monitoring
               - Accident risk prediction
            """)
        
        with col2:
            st.image("https://images.unsplash.com/photo-1580273916550-e323be2ae537?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=500&q=80", caption="Transportation AI Applications")
            
            st.markdown("### Success Story")
            st.info("""
            **Waymo's autonomous vehicles** have driven over 20 million miles on public roads using machine learning algorithms for perception, prediction, and decision-making.
            """)
    
    with tabs[5]:  # Entertainment
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            ### Entertainment
            
            Machine learning powers content recommendations, creation, and personalized experiences in entertainment.
            
            **Key Applications:**
            
            1. **Content Recommendation**
               - Personalized streaming recommendations
               - Music discovery algorithms
               - Content curation
            
            2. **Content Creation**
               - AI-generated music
               - Script analysis and generation
               - Video game procedural content
            
            3. **Visual Effects**
               - Deep fake technology
               - Motion capture enhancement
               - Image and video enhancement
            
            4. **User Experience**
               - Personalized gaming experiences
               - Dynamic difficulty adjustment
               - Interactive storytelling
            
            5. **Marketing**
               - Audience segmentation
               - Content performance prediction
               - Optimal release timing
            """)
        
        with col2:
            st.image("https://images.unsplash.com/photo-1522869635100-9f4c5e86aa37?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=500&q=80", caption="Entertainment AI Applications")
            
            st.markdown("### Success Story")
            st.info("""
            **Netflix's recommendation system** saves the company an estimated $1 billion per year by reducing churn and helping users discover content they enjoy.
            """)
    
    # Cross-industry trends
    st.markdown('<div class="sub-header">Cross-Industry ML Trends</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### MLOps")
        st.markdown("""
        - Automated ML pipelines
        - Model version control
        - Continuous monitoring
        - Production deployment strategies
        """)
    
    with col2:
        st.markdown("### Explainable AI")
        st.markdown("""
        - Model interpretability tools
        - Feature importance analysis
        - Regulatory compliance
        - Building user trust
        """)
    
    with col3:
        st.markdown("### Edge AI")
        st.markdown("""
        - On-device processing
        - Reduced latency
        - Privacy preservation
        - Energy efficiency
        """)
    
    # Future outlook
    st.markdown('<div class="sub-header">Future Outlook</div>', unsafe_allow_html=True)
    
    st.markdown("""
    The industry applications of machine learning continue to expand as technology evolves. Several trends are shaping the future:
    
    1. **AI Democratization**: Low-code/no-code platforms making ML accessible to non-technical users
    
    2. **Multimodal Models**: Systems that can process multiple types of data (text, images, audio) simultaneously
    
    3. **Foundation Models**: Large pre-trained models that can be fine-tuned for specific applications
    
    4. **Human-AI Collaboration**: Moving from AI automation to AI augmentation of human capabilities
    
    5. **Responsible AI**: Growing focus on fairness, transparency, accountability, and sustainability
    """)

if __name__ == "__main__":
    main()
