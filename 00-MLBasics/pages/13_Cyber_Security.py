# Machine Learning for Cybersecurity Threat Detection Using Streamlit

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

# Set page configuration
st.set_page_config(
    page_title="Cybersecurity Threat Detection",
    page_icon="🔒",
    layout="wide"
)

# Application title
st.title("🔒 Cybersecurity Threat Detection System")
st.markdown("Detect anomalies in network traffic using machine learning")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Home", "Train Model", "Detect Threats", "About"])

# Generate synthetic network traffic data for demo purposes
@st.cache_data
def generate_sample_data(n_samples=10000):
    np.random.seed(42)
    
    # Normal traffic
    normal_traffic = pd.DataFrame({
        'packet_size': np.random.normal(500, 150, int(n_samples * 0.8)),
        'protocol': np.random.choice(['TCP', 'UDP', 'HTTP', 'HTTPS'], int(n_samples * 0.8)),
        'port': np.random.choice(range(1, 1025), int(n_samples * 0.8)),
        'duration': np.random.exponential(30, int(n_samples * 0.8)),
        'packet_count': np.random.poisson(100, int(n_samples * 0.8)),
        'source_ip_entropy': np.random.normal(3, 0.5, int(n_samples * 0.8)),
        'dest_ip_entropy': np.random.normal(3, 0.5, int(n_samples * 0.8)),
    })
    normal_traffic['label'] = 0  # Normal
    
    # Anomalous traffic
    anomalous_traffic = pd.DataFrame({
        'packet_size': np.random.choice([np.random.normal(100, 30), np.random.normal(2000, 300)], int(n_samples * 0.2)),
        'protocol': np.random.choice(['TCP', 'UDP', 'HTTP', 'HTTPS', 'SMB', 'DNS'], int(n_samples * 0.2)),
        'port': np.random.choice(list(range(1, 1025)) + list(range(4000, 10000)), int(n_samples * 0.2)),
        'duration': np.random.choice([np.random.exponential(1), np.random.exponential(300)], int(n_samples * 0.2)),
        'packet_count': np.random.choice([np.random.poisson(5), np.random.poisson(1000)], int(n_samples * 0.2)),
        'source_ip_entropy': np.random.choice([np.random.normal(1, 0.2), np.random.normal(5, 0.2)], int(n_samples * 0.2)),
        'dest_ip_entropy': np.random.choice([np.random.normal(1, 0.2), np.random.normal(5, 0.2)], int(n_samples * 0.2)),
    })
    anomalous_traffic['label'] = 1  # Anomalous
    
    # Combine and shuffle
    df = pd.concat([normal_traffic, anomalous_traffic], ignore_index=True)
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Convert protocol to numeric
    df['protocol'] = df['protocol'].map({'TCP': 0, 'UDP': 1, 'HTTP': 2, 'HTTPS': 3, 'SMB': 4, 'DNS': 5})
    
    return df

# Preprocess data
def preprocess_data(df):
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

# Train isolation forest model
def train_model(X_train, contamination=0.2):
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(X_train)
    return model

# Home page
if page == "Home":
    st.header("Welcome to the Cybersecurity Threat Detection System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### How it works
        This system uses machine learning to detect anomalies in network traffic data:
        1. **Data Collection**: Network traffic features are collected and processed
        2. **Anomaly Detection**: An Isolation Forest algorithm identifies unusual patterns
        3. **Alert Generation**: Potential threats are flagged for security analysts
        """)
        
    with col2:
        st.image("https://static.vecteezy.com/system/resources/previews/007/193/729/original/cyber-security-icon-for-protection-shield-thin-line-design-vector.jpg", width=300)
    
    st.markdown("### Sample Network Traffic Data")
    sample_data = generate_sample_data(1000)
    st.dataframe(sample_data.head(10))
    
    # Display some statistics
    st.subheader("Network Traffic Statistics")
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots()
        sns.histplot(data=sample_data, x='packet_size', hue='label', bins=30, ax=ax)
        ax.set_title('Packet Size Distribution')
        st.pyplot(fig)
        
    with col2:
        fig, ax = plt.subplots()
        sns.scatterplot(data=sample_data, x='packet_count', y='duration', hue='label', ax=ax)
        ax.set_title('Packet Count vs Duration')
        st.pyplot(fig)

# Train Model page
elif page == "Train Model":
    st.header("Train Anomaly Detection Model")
    
    st.write("Configure and train the Isolation Forest model for anomaly detection.")
    
    # Model parameters
    col1, col2 = st.columns(2)
    
    with col1:
        contamination = st.slider("Contamination (expected % of anomalies)", 0.01, 0.5, 0.2, 0.01)
        n_estimators = st.slider("Number of estimators", 50, 200, 100, 10)
        
    with col2:
        sample_size = st.slider("Training data size", 1000, 10000, 5000, 1000)
        test_size = st.slider("Test set percentage", 0.1, 0.4, 0.2, 0.05)
    
    # Train button
    if st.button("Train Model", type='primary'):
        with st.spinner("Generating and processing data..."):
            # Generate data
            data = generate_sample_data(sample_size)
            X, y, scaler = preprocess_data(data)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
        with st.spinner("Training model..."):
            # Train model
            model = IsolationForest(
                n_estimators=n_estimators,
                contamination=contamination,
                random_state=42
            )
            model.fit(X_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            # Convert predictions from {1: normal, -1: anomaly} to {0: normal, 1: anomaly}
            y_pred_binary = np.where(y_pred == 1, 0, 1)
            
            # Calculate some metrics
            from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
            
            cm = confusion_matrix(y_test, y_pred_binary)
            accuracy = accuracy_score(y_test, y_pred_binary)
            precision = precision_score(y_test, y_pred_binary)
            recall = recall_score(y_test, y_pred_binary)
            
        # Save the model and scaler
        joblib.dump(model, "isolation_forest_model.pkl")
        joblib.dump(scaler, "scaler.pkl")
        
        # Display results
        st.success("Model trained successfully!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Accuracy", f"{accuracy:.2f}")
            st.metric("Precision", f"{precision:.2f}")
            st.metric("Recall", f"{recall:.2f}")
            
        with col2:
            # Display confusion matrix
            fig, ax = plt.subplots(figsize=(6, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title('Confusion Matrix')
            ax.xaxis.set_ticklabels(['Normal', 'Anomaly'])
            ax.yaxis.set_ticklabels(['Normal', 'Anomaly'])
            st.pyplot(fig)

# Detect Threats page
elif page == "Detect Threats":
    st.header("Detect Network Threats")
    
    # Check if model exists
    model_path = "isolation_forest_model.pkl"
    scaler_path = "scaler.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.warning("Model not found! Please go to the 'Train Model' page and train a model first.")
    else:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        st.write("Enter network traffic parameters to analyze for potential threats:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            packet_size = st.number_input("Packet Size (bytes)", min_value=0, max_value=10000, value=500)
            protocol_options = {"TCP": 0, "UDP": 1, "HTTP": 2, "HTTPS": 3, "SMB": 4, "DNS": 5}
            protocol = st.selectbox("Protocol", list(protocol_options.keys()))
            protocol_numeric = protocol_options[protocol]
            
        with col2:
            port = st.number_input("Port", min_value=1, max_value=65535, value=443)
            duration = st.number_input("Connection Duration (seconds)", min_value=0.1, max_value=1000.0, value=30.0)
            
        with col3:
            packet_count = st.number_input("Packet Count", min_value=1, max_value=10000, value=100)
            source_ip_entropy = st.number_input("Source IP Entropy", min_value=0.0, max_value=8.0, value=3.0)
            dest_ip_entropy = st.number_input("Destination IP Entropy", min_value=0.0, max_value=8.0, value=3.0)
        
        # Create a single sample from input
        input_data = np.array([[packet_size, protocol_numeric, port, 
                              duration, packet_count, source_ip_entropy, 
                              dest_ip_entropy]])
        
        # Scale the input
        input_scaled = scaler.transform(input_data)
        
        if st.button("Analyze Traffic"):
            # Make prediction
            prediction = model.predict(input_scaled)
            score = model.decision_function(input_scaled)
            
            # Display result with nice formatting
            st.subheader("Analysis Result")
            
            if prediction[0] == -1:
                st.error("⚠️ ALERT: Potential Threat Detected!")
                st.markdown(f"""
                * **Anomaly Score**: {-score[0]:.4f} (higher is more anomalous)
                * **Confidence**: {min(100, (-score[0] * 20 + 50)):.1f}%
                """)
                
                # Explanation
                st.subheader("Potential Threat Details")
                explanations = []
                
                # Add some logic to explain why it might be flagged
                if packet_size < 100 or packet_size > 1500:
                    explanations.append(f"Unusual packet size ({packet_size} bytes)")
                
                if protocol in ["SMB"]:
                    explanations.append(f"{protocol} protocol may be used for lateral movement")
                
                if port > 1024 and port not in [3389, 8080, 8443]:
                    explanations.append(f"Uncommon port number ({port})")
                
                if packet_count > 500:
                    explanations.append(f"High number of packets ({packet_count})")
                
                if duration < 1:
                    explanations.append(f"Very short connection duration ({duration}s)")
                
                if source_ip_entropy < 1.5 or source_ip_entropy > 4.5:
                    explanations.append(f"Unusual source IP entropy ({source_ip_entropy:.2f})")
                
                if not explanations:
                    explanations.append("Complex pattern of multiple slight anomalies")
                
                for exp in explanations:
                    st.markdown(f"* {exp}")
                
                st.markdown("""
                **Recommended Action**: Review this traffic and consider blocking if malicious.
                """)
                
            else:
                st.success("✅ Normal Traffic Pattern")
                st.markdown(f"""
                * **Normality Score**: {score[0]:.4f} (higher is more normal)
                * **Confidence**: {min(100, (score[0] * 20 + 50)):.1f}%
                """)
            
            # Visualization of the decision
            st.subheader("Anomaly Score Visualization")
            fig, ax = plt.subplots(figsize=(10, 2))
            
            # Create color gradient
            cmap = plt.cm.RdYlGn
            norm = plt.Normalize(-0.5, 0.5)
            
            # Plot the score as a gauge
            score_val = score[0]
            plt.barh([0], [1], color=cmap(norm(score_val)))
            
            # Add a marker for the current score
            plt.scatter([0.5 + score_val/2], [0], color='black', s=150, zorder=5)
            
            # Remove axes
            plt.axis('off')
            
            # Add labels
            plt.text(0, 0, "Anomaly", ha='left', va='center', fontsize=12)
            plt.text(1, 0, "Normal", ha='right', va='center', fontsize=12)
            
            st.pyplot(fig)

# About page
elif page == "About":
    st.header("About This Application")
    
    st.markdown("""
    ### Cybersecurity Threat Detection with Machine Learning
    
    This application demonstrates how machine learning can be used to detect potential cyber threats
    by identifying anomalies in network traffic patterns.
    
    #### Features
    * **Isolation Forest Algorithm**: Detects data points that are different from normal patterns
    * **Real-time Analysis**: Input network traffic parameters to get immediate threat assessment
    * **Model Training**: Customize and train the model based on your security needs
    
    #### How Isolation Forest Works
    The Isolation Forest algorithm isolates observations by randomly selecting a feature and then randomly selecting 
    a split value between the maximum and minimum values of the selected feature. Since recursive partitioning 
    can be represented by a tree structure, the number of splittings required to isolate a sample is equivalent 
    to the path length from the root node to the terminating node.
    
    This path length, averaged over a forest of random trees, is a measure of normality and our decision function.
    Random partitioning produces noticeably shorter paths for anomalies, hence the name "Isolation Forest".
    
    #### In a Real-World Scenario
    In production environments, this system would:
    1. Ingest real network traffic data from firewalls, routers, and IDS/IPS systems
    2. Process and normalize the data in real-time
    3. Apply the trained model to detect anomalies
    4. Alert security teams of potential threats
    5. Update and refine the model based on feedback
    
    #### About the Developer
    This application was created as a demonstration of machine learning applications in cybersecurity.
    """)

st.sidebar.markdown("---")
st.sidebar.info(
    "This application demonstrates using machine learning for cybersecurity threat detection. "
    "The data used is synthetic and for demonstration purposes only."
)
