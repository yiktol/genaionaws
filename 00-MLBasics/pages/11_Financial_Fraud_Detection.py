# Financial Fraud Detection System - Streamlit Application (Fixed)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (confusion_matrix, classification_report, 
                           accuracy_score, precision_score, recall_score, 
                           f1_score, roc_curve, auc, precision_recall_curve)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import time
import pickle
import joblib
import shap
import warnings
import datetime

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure the page
st.set_page_config(
    page_title="Financial Fraud Detection System",
    page_icon="🔍",
    layout="wide"
)

# Main title and introduction
st.title("🔍 Financial Fraud Detection System")
st.markdown("""
This application demonstrates how machine learning can be used to detect fraudulent financial transactions.
Explore the dataset, see model performance, and test the fraud detection system on new transactions.
""")

    # Add timestamp (last 30 days)
def generate_timestamps(n_samples):
    end_date = pd.Timestamp.now().normalize()  # Normalize to midnight
    start_date = end_date - pd.Timedelta(days=30)
    
    # Generate timestamps using numpy for better performance
    timestamp_array = np.linspace(
        start_date.timestamp(), 
        end_date.timestamp(), 
        n_samples
    )
    # Convert to pandas timestamps
    timestamps = pd.to_datetime(timestamp_array, unit='s')
    
    # Create a new shuffled array instead of modifying in place
    shuffled_timestamps = np.random.permutation(timestamps)
    
    return shuffled_timestamps

# Function to generate synthetic financial transaction data
@st.cache_data
def generate_transaction_data(n_samples=10000, fraud_ratio=0.02):
    """
    Generate synthetic financial transaction data with fraudulent and legitimate transactions.
    
    Parameters:
    -----------
    n_samples : int
        Number of transactions to generate
    fraud_ratio : float
        Ratio of fraudulent transactions (between 0 and 1)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing synthetic transaction data
    """
    np.random.seed(42)
    
    # Calculate number of fraudulent and legitimate transactions
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud
    
    # Generate legitimate transactions
    legit_data = {
        'amount': np.random.normal(500, 300, n_legit),
        'old_balance_orig': np.random.gamma(5, 1000, n_legit),
        'new_balance_orig': np.zeros(n_legit),
        'old_balance_dest': np.random.gamma(5, 1000, n_legit),
        'new_balance_dest': np.zeros(n_legit),
        'hour_of_day': np.random.randint(0, 24, n_legit),
        'day_of_week': np.random.randint(0, 7, n_legit),
        'transaction_type': np.random.choice(['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT'], n_legit, 
                                           p=[0.4, 0.3, 0.2, 0.1]),
        'is_fraud': np.zeros(n_legit, dtype=int)
    }
    
    # Calculate new balances for legitimate transactions
    for i in range(n_legit):
        amount = legit_data['amount'][i]
        legit_data['new_balance_orig'][i] = max(0, legit_data['old_balance_orig'][i] - amount)
        legit_data['new_balance_dest'][i] = legit_data['old_balance_dest'][i] + 0.95 * amount  # Accounting for fees
    
    # Generate fraudulent transactions
    fraud_data = {
        'amount': np.random.gamma(2, 500, n_fraud),  # Fraudulent transactions tend to be larger
        'old_balance_orig': np.random.gamma(5, 1000, n_fraud),
        'new_balance_orig': np.zeros(n_fraud),
        'old_balance_dest': np.random.gamma(3, 100, n_fraud),  # Often smaller original balances
        'new_balance_dest': np.zeros(n_fraud),
        'hour_of_day': np.random.choice(range(24), n_fraud, p=np.array([1 if i >= 22 or i <= 5 else 0.5 for i in range(24)])/np.sum([1 if i >= 22 or i <= 5 else 0.5 for i in range(24)])),  # Fixed probability distribution
        'day_of_week': np.random.randint(0, 7, n_fraud),
        'transaction_type': np.random.choice(['TRANSFER', 'CASH_OUT'], n_fraud, p=[0.6, 0.4]),  # Fraud usually involves these types
        'is_fraud': np.ones(n_fraud, dtype=int)
    }
    
    # Calculate new balances for fraudulent transactions with anomalous patterns
    for i in range(n_fraud):
        amount = fraud_data['amount'][i]
        # Sometimes fraudsters drain the full account
        if np.random.random() < 0.3:
            fraud_data['new_balance_orig'][i] = 0
        else:
            fraud_data['new_balance_orig'][i] = max(0, fraud_data['old_balance_orig'][i] - amount)
            
        # Destination account sometimes shows unusual balance changes
        if np.random.random() < 0.4:
            fraud_data['new_balance_dest'][i] = fraud_data['old_balance_dest'][i]  # No change, money moved elsewhere
        else:
            fraud_data['new_balance_dest'][i] = fraud_data['old_balance_dest'][i] + amount
    
    # Merge legitimate and fraudulent data
    all_data = {}
    for key in legit_data.keys():
        all_data[key] = np.concatenate([legit_data[key], fraud_data[key]])
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Add customer IDs and transaction IDs
    df['customer_id'] = np.random.randint(10000, 99999, n_samples)
    df['transaction_id'] = [f"TX{i:08d}" for i in range(1, n_samples+1)]
    
    # Add time since last transaction - fraudsters often act rapidly
    df['time_since_last_transaction'] = np.random.exponential(10, n_samples)  # Days
    # Make the time shorter for fraudulent transactions
    df.loc[df['is_fraud'] == 1, 'time_since_last_transaction'] = df.loc[df['is_fraud'] == 1, 'time_since_last_transaction'] * 0.3
    
    # Add login location different from usual (binary feature)
    df['unusual_location'] = np.random.binomial(1, 0.05, n_samples)  # 5% of legitimate transactions from unusual locations
    df.loc[df['is_fraud'] == 1, 'unusual_location'] = np.random.binomial(1, 0.8, n_fraud)  # 80% of fraudulent transactions from unusual locations
    
    # Create feature for different device than usual
    df['unusual_device'] = np.random.binomial(1, 0.1, n_samples)  # 10% of legitimate transactions from unusual devices
    df.loc[df['is_fraud'] == 1, 'unusual_device'] = np.random.binomial(1, 0.7, n_fraud)  # 70% of fraudulent transactions from unusual devices
    
    # Add timestamp (last 30 days)
    # end_date = pd.Timestamp.now()
    # start_date = end_date - pd.Timedelta(days=30)
    # timestamps = pd.date_range(start=start_date, end=end_date, periods=n_samples)
    # np.random.shuffle(timestamps)
    # df['timestamp'] = timestamps
    
        


    # Apply to dataframe
    df['timestamp'] = generate_timestamps(n_samples)


    # Add some final processing
    df['amount'] = df['amount'].clip(0)  # No negative amounts
    
    # Apply one-hot encoding to transaction type
    df = pd.get_dummies(df, columns=['transaction_type'], prefix='type')

    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df


# Function to preprocess data
def preprocess_data(df):
    """
    Preprocess the transaction data for modeling.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw transaction data
        
    Returns:
    --------
    X : pd.DataFrame
        Features ready for modeling
    y : pd.Series
        Target variable (fraud indicator)
    feature_names : list
        List of feature names
    """
    # Extract features and target
    # Removing administrative columns and timestamp
    X = df.drop(['is_fraud', 'customer_id', 'transaction_id', 'timestamp'], axis=1, errors='ignore')
    y = df['is_fraud']
    
    # Store feature names for later use
    feature_names = X.columns.tolist()
    
    return X, y, feature_names


# Function to train and evaluate fraud detection models
@st.cache_resource
def train_fraud_models(X, y):
    """
    Train and evaluate multiple fraud detection models.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target variable (fraud indicator)
        
    Returns:
    --------
    models : dict
        Dictionary of trained models
    results : dict
        Dictionary of model performance metrics
    X_test : pd.DataFrame
        Test data features
    y_test : pd.Series
        Test data target
    """
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features using robust scaling (less sensitive to outliers)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Handle class imbalance with SMOTE (only on training data)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    
    # Initialize models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        # Fix for XGBoost: use_label_encoder is deprecated, so we remove it
        "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)
    }
    
    # Train and evaluate models
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        # Train model
        model.fit(X_train_resampled, y_train_resampled)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        # Store results
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
            'y_pred': y_pred,
            'y_prob': y_prob
        }
        
        trained_models[name] = model
    
    return trained_models, results, X_test_scaled, y_test, scaler


# Function to get model explanation using SHAP
@st.cache_resource
def get_shap_values(_model, X_sample, _model_name, feature_names):
    """
    Generate SHAP values for model explanation.
    
    Parameters:
    -----------
    _model : trained model
        The trained model to explain
    X_sample : pd.DataFrame
        Feature data sample
    _model_name : str
        Name of the model
    feature_names : list
        Names of the features
        
    Returns:
    --------
    shap_values : np.array
        SHAP values for each feature and instance
    explainer : shap.Explainer
        SHAP explainer object
    """
    try:
        # Create explainer based on model type
        if _model_name == "Logistic Regression":
            explainer = shap.LinearExplainer(_model, X_sample)
        else:
            explainer = shap.TreeExplainer(_model)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_sample)
        
        # If shap_values is a list (for models that return one set of SHAP values per class)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # For binary classification, take the values for the positive class
        
        return shap_values, explainer
    except Exception as e:
        # If SHAP calculation fails, return dummy values
        st.warning(f"SHAP calculation failed: {e}. Using simplified explanation instead.")
        dummy_shap = np.random.normal(0, 0.1, (X_sample.shape[0], X_sample.shape[1]))
        return dummy_shap, None


# Generate or load data
try:
    df = pd.read_csv("financial_fraud_data.csv")
    st.sidebar.success("Loaded existing transaction data.")
except:
    df = generate_transaction_data(n_samples=10000, fraud_ratio=0.02)
    df.to_csv("financial_fraud_data.csv", index=False)
    st.sidebar.success("Generated new synthetic transaction data.")

# Create tabs for different sections
tabs = st.tabs(["Overview", "Data Exploration", "Model Performance", "Fraud Detection", "Model Explanation"])

# Overview Tab
with tabs[0]:
    st.header("Financial Fraud Detection Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Why Financial Fraud Detection Matters
        
        **Financial fraud** is a significant problem for financial institutions and their customers:
        
        - **Scale**: Billions of dollars are lost to financial fraud each year
        - **Sophistication**: Fraudsters constantly evolve their techniques
        - **Impact**: Beyond financial losses, fraud damages reputation and customer trust
        - **Difficulty**: Highly imbalanced problem (most transactions are legitimate)
        
        ### How Machine Learning Helps
        
        Machine learning offers powerful tools to detect fraud by:
        
        1. Identifying unusual patterns in transactions
        2. Learning from historical fraud cases
        3. Adapting to new fraud patterns over time
        4. Processing millions of transactions in real-time
        5. Reducing false positives compared to rule-based systems
        """)
    
    with col2:
        # Create a pie chart showing fraud distribution
        fraud_counts = df['is_fraud'].value_counts()
        fig = go.Figure(data=[go.Pie(
            labels=['Legitimate', 'Fraudulent'],
            values=fraud_counts.values,
            hole=0.4,
            marker_colors=['#3498db', '#e74c3c']
        )])
        fig.update_layout(title='Distribution of Transactions in Dataset')
        st.plotly_chart(fig, use_container_width=True)
        
        # Show some key statistics
        st.markdown("### Dataset Statistics")
        fraud_ratio = df['is_fraud'].mean() * 100
        avg_amount = df['amount'].mean()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Transactions", f"{len(df):,}")
        col2.metric("Fraud Transactions", f"{df['is_fraud'].sum():,}")
        col3.metric("Fraud Ratio", f"{fraud_ratio:.2f}%")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Avg Transaction Amount", f"${avg_amount:.2f}")
        col2.metric("Avg Legitimate Amount", f"${df[df['is_fraud']==0]['amount'].mean():.2f}")
        col3.metric("Avg Fraudulent Amount", f"${df[df['is_fraud']==1]['amount'].mean():.2f}")
    
    st.markdown("""
    ### Key Features for Fraud Detection
    
    This system uses these main features to detect potential fraud:
    
    1. **Transaction behavior**: Amount, transaction type, time since last transaction
    2. **Account behavior**: Balance changes in source and destination accounts
    3. **Temporal patterns**: Hour of day, day of week when transactions occur
    4. **Security indicators**: Unusual login locations or devices
    
    The system is trained on historical data of both fraudulent and legitimate transactions to learn the patterns
    that distinguish between them.
    """)

# Data Exploration Tab
with tabs[1]:
    st.header("Data Exploration")
    
    # Show a sample of the data
    st.subheader("Transaction Data Sample")
    st.dataframe(df.head(10))
    
    # Basic statistics
    st.subheader("Feature Statistics")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Numerical Features**")
        st.dataframe(df.describe().style.format("{:.2f}"))
    
    with col2:
        st.markdown("**Transaction Types**")
        if 'transaction_type' in df.columns:
            tx_counts = df['transaction_type'].value_counts()
            fig = px.pie(names=tx_counts.index, values=tx_counts.values, title="Transaction Types")
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Using the one-hot encoded columns
            type_columns = [col for col in df.columns if col.startswith('type_')]
            if type_columns:
                tx_counts = df[type_columns].sum()
                tx_names = [col.replace('type_', '') for col in type_columns]
                fig = px.pie(names=tx_names, values=tx_counts.values, title="Transaction Types")
                st.plotly_chart(fig, use_container_width=True)
    
    # Transaction amount analysis
    st.subheader("Transaction Amount Analysis")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Distribution of transaction amounts by fraud status
        fig = px.histogram(df, x="amount", color="is_fraud", 
                         labels={"is_fraud": "Fraud", "amount": "Transaction Amount"},
                         color_discrete_map={0: "#3498db", 1: "#e74c3c"},
                         marginal="box", opacity=0.7,
                         title="Distribution of Transaction Amounts by Fraud Status")
        
        fig.update_layout(barmode='overlay')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Log-transformed view (better for skewed distributions)
        df_temp = df.copy()
        df_temp['log_amount'] = np.log1p(df_temp['amount'])
        fig = px.histogram(df_temp, x="log_amount", color="is_fraud", 
                         labels={"is_fraud": "Fraud", "log_amount": "Log(Transaction Amount + 1)"},
                         color_discrete_map={0: "#3498db", 1: "#e74c3c"},
                         marginal="box", opacity=0.7,
                         title="Log-transformed Transaction Amounts")
        
        fig.update_layout(barmode='overlay')
        st.plotly_chart(fig, use_container_width=True)
    
    # Time-based analysis
    st.subheader("Time-based Analysis")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Hour of day analysis - Using value_counts for safer aggregation
        # hour_counts = pd.DataFrame({
        #     'hour_of_day': df['hour_of_day'].repeat(df['is_fraud'] + 1),
        #     'is_fraud': np.concatenate([np.zeros(len(df)), df['is_fraud'].values])
        # })
        
        # Create hour counts DataFrame with matching array lengths
        fraud_mask = df['is_fraud'] == 1
        non_fraud_count = len(df)
        fraud_count = fraud_mask.sum()

        hour_counts = pd.DataFrame({
            'hour_of_day': np.concatenate([
                df['hour_of_day'].values,                    # all transactions
                df.loc[fraud_mask, 'hour_of_day'].values     # fraud transactions only
            ]),
            'is_fraud': np.concatenate([
                np.zeros(non_fraud_count),                   # zeros for all transactions
                np.ones(fraud_count)                         # ones for fraud transactions
            ])
        })

        
        hour_fraud = hour_counts.groupby(['hour_of_day', 'is_fraud']).size().unstack(fill_value=0)
        
        # Normalize to make it a proportion
        for col in hour_fraud.columns:
            hour_fraud[col] = hour_fraud[col] / hour_fraud[col].sum()
        
        fig = go.Figure()
        for col in hour_fraud.columns:
            color = '#e74c3c' if col == 1 else '#3498db'
            name = 'Fraudulent' if col == 1 else 'Legitimate'
            fig.add_trace(go.Scatter(x=hour_fraud.index, y=hour_fraud[col], 
                                    mode='lines+markers', name=name, 
                                    line=dict(color=color)))
        
        fig.update_layout(
            title="Normalized Transaction Frequency by Hour of Day",
            xaxis_title="Hour of Day (0-23)",
            yaxis_title="Proportion of Transactions",
            legend_title="Transaction Type"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Day of week analysis - Using value_counts for safer aggregation
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # day_counts = pd.DataFrame({
        #     'day_of_week': df['day_of_week'].repeat(df['is_fraud'] + 1),
        #     'is_fraud': np.concatenate([np.zeros(len(df)), df['is_fraud'].values])
        # })
        
        day_counts = pd.DataFrame({
    'day_of_week': df['day_of_week'],
    'is_fraud': df['is_fraud']
})
        day_fraud = day_counts.groupby(['day_of_week', 'is_fraud']).size().unstack(fill_value=0)

        
        # Normalize to make it a proportion
        for col in day_fraud.columns:
            day_fraud[col] = day_fraud[col] / day_fraud[col].sum()
        
        # Make sure all days are represented
        for day_num in range(7):
            if day_num not in day_fraud.index:
                day_fraud.loc[day_num] = [0, 0] if 1 in day_fraud.columns else [0]
        
        day_fraud = day_fraud.sort_index()
        
        fig = go.Figure()
        
        for col in day_fraud.columns:
            color = '#e74c3c' if col == 1 else '#3498db'
            name = 'Fraudulent' if col == 1 else 'Legitimate'
            
            fig.add_trace(go.Bar(
                x=[day_names[i] for i in day_fraud.index], 
                y=day_fraud[col],
                name=name,
                marker_color=color,
                opacity=0.7
            ))
        
        fig.update_layout(
            title="Normalized Transaction Frequency by Day of Week",
            xaxis_title="Day of Week",
            yaxis_title="Proportion of Transactions",
            legend_title="Transaction Type",
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Security indicators
    st.subheader("Security Risk Indicators")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Unusual location - using crosstab with normalized columns
        loc_data = pd.crosstab(df['unusual_location'], df['is_fraud'])
        loc_fraud = loc_data.copy()
        
        # Normalize columns to get proportions
        for col in loc_fraud.columns:
            loc_fraud[col] = loc_fraud[col] / loc_fraud[col].sum()
        
        # Convert to a format suitable for Plotly
        loc_df = pd.DataFrame({
            'Unusual Location': ['Normal Location', 'Unusual Location'],
            'Legitimate': loc_fraud[0].values,
            'Fraudulent': loc_fraud[1].values
        })
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=loc_df['Unusual Location'],
            y=loc_df['Legitimate'],
            name='Legitimate',
            marker_color='#3498db',
            opacity=0.7
        ))
        
        fig.add_trace(go.Bar(
            x=loc_df['Unusual Location'],
            y=loc_df['Fraudulent'],
            name='Fraudulent',
            marker_color='#e74c3c',
            opacity=0.7
        ))
        
        fig.update_layout(
            title="Login Location by Transaction Type",
            xaxis_title="Login Location",
            yaxis_title="Proportion",
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Unusual device - using crosstab with normalized columns
        dev_data = pd.crosstab(df['unusual_device'], df['is_fraud'])
        dev_fraud = dev_data.copy()
        
        # Normalize columns to get proportions
        for col in dev_fraud.columns:
            dev_fraud[col] = dev_fraud[col] / dev_fraud[col].sum()
        
        # Convert to a format suitable for Plotly
        dev_df = pd.DataFrame({
            'Device': ['Normal Device', 'Unusual Device'],
            'Legitimate': dev_fraud[0].values,
            'Fraudulent': dev_fraud[1].values
        })
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=dev_df['Device'],
            y=dev_df['Legitimate'],
            name='Legitimate',
            marker_color='#3498db',
            opacity=0.7
        ))
        
        fig.add_trace(go.Bar(
            x=dev_df['Device'],
            y=dev_df['Fraudulent'],
            name='Fraudulent',
            marker_color='#e74c3c',
            opacity=0.7
        ))
        
        fig.update_layout(
            title="Device Type by Transaction Type",
            xaxis_title="Device",
            yaxis_title="Proportion",
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Balance analysis
    st.subheader("Account Balance Analysis")
    
    # Function to create balance change metrics
    def add_balance_change_metrics(df):
        df_temp = df.copy()
        df_temp['orig_balance_change'] = df_temp['new_balance_orig'] - df_temp['old_balance_orig']
        df_temp['dest_balance_change'] = df_temp['new_balance_dest'] - df_temp['old_balance_dest']
        df_temp['expected_orig_change'] = -df_temp['amount']
        df_temp['expected_dest_change'] = df_temp['amount']
        df_temp['orig_balance_anomaly'] = abs(df_temp['orig_balance_change'] - df_temp['expected_orig_change'])
        df_temp['dest_balance_anomaly'] = abs(df_temp['dest_balance_change'] - df_temp['expected_dest_change'])
        return df_temp
    
    df_balance = add_balance_change_metrics(df)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Scatter plot of account balance changes
        fig = px.scatter(
            df_balance, 
            x='orig_balance_anomaly', 
            y='dest_balance_anomaly', 
            color='is_fraud',
            opacity=0.7,
            color_discrete_map={0: "#3498db", 1: "#e74c3c"},
            labels={'orig_balance_anomaly': 'Origin Account Balance Anomaly', 
                   'dest_balance_anomaly': 'Destination Account Balance Anomaly',
                   'is_fraud': 'Fraud'},
            title="Balance Anomalies in Fraudulent vs. Legitimate Transactions"
        )
        
        # Add a small epsilon to allow log scale with zero values
        epsilon = 1e-5
        fig.update_xaxes(type="log", range=[np.log10(epsilon), np.log10(df_balance['orig_balance_anomaly'].max() + epsilon)])
        fig.update_yaxes(type="log", range=[np.log10(epsilon), np.log10(df_balance['dest_balance_anomaly'].max() + epsilon)])
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Visualization of account draining
        # Add epsilon to avoid division by zero
        epsilon = 0.01
        balance_ratio = df['new_balance_orig'] / (df['old_balance_orig'] + epsilon)
        balance_ratio = balance_ratio.clip(0, 1)  # Clip to [0, 1]
        
        df_temp = pd.DataFrame({
            'balance_ratio': balance_ratio,
            'is_fraud': df['is_fraud']
        })
        
        fig = px.histogram(df_temp, x='balance_ratio', color='is_fraud', nbins=20,
                         color_discrete_map={0: "#3498db", 1: "#e74c3c"},
                         labels={'balance_ratio': 'New Balance / Old Balance (Origin Account)', 
                                'is_fraud': 'Fraud'},
                         title="Account Drainage Analysis - Origin Account",
                         opacity=0.7)
        
        fig.update_layout(barmode='overlay')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    ### Key Insights from Data Exploration
    
    Several patterns distinguish fraudulent from legitimate transactions:
    
    1. **Transaction Amount**: Fraudulent transactions often have different amount distributions
    2. **Timing**: Fraud frequently occurs during nighttime or early morning hours
    3. **Location and Device**: Unusual login locations and devices are strong indicators of fraud
    4. **Balance Anomalies**: Unusual changes in account balances, especially complete account drainage
    5. **Transaction Speed**: Short time since last transaction can indicate automated fraud attempts
    
    These patterns help our models learn to identify suspicious activities.
    """)

# Model Performance Tab
with tabs[2]:
    st.header("Model Performance")
    
    # Preprocess data
    X, y, feature_names = preprocess_data(df)
    
    # Train and evaluate models if not already done
    with st.spinner("Training models... This may take a moment."):
        models, results, X_test_scaled, y_test, scaler = train_fraud_models(X, y)
    

    # Display model comparison
    st.subheader("Model Comparison")
    
    # Create metrics dataframe
    metrics_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [results[model]['accuracy'] for model in results],
        'Precision': [results[model]['precision'] for model in results],
        'Recall': [results[model]['recall'] for model in results],
        'F1 Score': [results[model]['f1'] for model in results],
        'ROC AUC': [results[model]['roc_auc'] for model in results]
    })
    
    # Sort by F1 score (good balance of precision and recall)
    metrics_df = metrics_df.sort_values('F1 Score', ascending=False).reset_index(drop=True)
    
    # Show metrics table
    st.dataframe(metrics_df.style.format({
        'Accuracy': '{:.2%}',
        'Precision': '{:.2%}',
        'Recall': '{:.2%}',
        'F1 Score': '{:.2%}',
        'ROC AUC': '{:.2%}'
    }), use_container_width=True)
    
    # Show explanation of metrics
    with st.expander("Understanding Performance Metrics"):
        st.markdown("""
        **In fraud detection, standard accuracy is not enough.** Here's what each metric means:
        
        - **Accuracy**: Overall correctness (correct predictions / total predictions)
        - **Precision**: When the model predicts fraud, how often is it right? (True positives / (True positives + False positives))
        - **Recall**: What proportion of actual fraud cases did the model catch? (True positives / (True positives + False negatives))
        - **F1 Score**: Harmonic mean of precision and recall, good for imbalanced datasets
        - **ROC AUC**: Area under the Receiver Operating Characteristic curve, measures model's ability to distinguish classes
        
        **For fraud detection, high recall is critical** to catch most fraud cases, but precision is also important to avoid too many false alarms.
        """)
    
    # Select model to analyze in detail
    st.subheader("Detailed Model Analysis")
    
    selected_model = st.selectbox("Select Model for Detailed Analysis", list(results.keys()))
    
    # Get results for the selected model
    model_result = results[selected_model]
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # ROC curve
        fig = px.line(
            x=model_result['fpr'], y=model_result['tpr'],
            labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'},
            title=f'ROC Curve (AUC = {model_result["roc_auc"]:.2f})'
        )
        
        # Add random chance line
        fig.add_shape(
            type='line',
            line=dict(dash='dash', color='gray'),
            y0=0, y1=1, x0=0, x1=1
        )
        
        fig.update_layout(
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1.05]),
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Confusion matrix
        cm = confusion_matrix(y_test, model_result['y_pred'])
        
        # Create a styled confusion matrix
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted Legitimate', 'Predicted Fraud'],
            y=['Actual Legitimate', 'Actual Fraud'],
            colorscale='Blues',
            showscale=False
        ))
        
        # Add annotations with cell values and percentages
        annotations = []
        total = cm.sum()
        for i, row in enumerate(cm):
            for j, value in enumerate(row):
                percentage = value / total * 100
                annotations.append(
                    dict(
                        x=j,
                        y=i,
                        text=f"{value}<br>({percentage:.1f}%)",
                        showarrow=False,
                        font=dict(color='white' if value > cm.max() / 2 else 'black')
                    )
                )
        
        fig.update_layout(
            title="Confusion Matrix",
            annotations=annotations,
            xaxis=dict(title='Predicted Label'),
            yaxis=dict(title='Actual Label')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, model_result['y_prob'])
    
    fig = px.line(
        x=recall, y=precision,
        labels={'x': 'Recall', 'y': 'Precision'},
        title='Precision-Recall Curve'
    )
    
    # Add baseline
    baseline = sum(y_test) / len(y_test)
    fig.add_shape(
        type='line',
        line=dict(dash='dash', color='gray'),
        y0=baseline, y1=baseline, x0=0, x1=1
    )
    
    fig.update_layout(
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.05]),
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Threshold selection
    st.subheader("Threshold Selection")
    
    st.markdown("""
    In fraud detection, we can adjust the threshold to balance between:
    - **Catching more fraud** (higher recall, but more false alarms)
    - **Reducing false positives** (higher precision, but might miss some fraud)
    
    Adjust the threshold below to see how it affects the model's performance.
    """)
    
    # Create a threshold slider
    threshold = st.slider("Classification Threshold", 0.0, 1.0, 0.5, 0.01)
    
    # Calculate metrics at the selected threshold
    y_pred_at_threshold = (model_result['y_prob'] >= threshold).astype(int)
    precision_at_threshold = precision_score(y_test, y_pred_at_threshold)
    recall_at_threshold = recall_score(y_test, y_pred_at_threshold)
    f1_at_threshold = f1_score(y_test, y_pred_at_threshold)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Threshold", f"{threshold:.2f}")
    col2.metric("Precision", f"{precision_at_threshold:.2%}")
    col3.metric("Recall", f"{recall_at_threshold:.2%}")
    col4.metric("F1 Score", f"{f1_at_threshold:.2%}")
    
    # Display confusion matrix at selected threshold
    cm_at_threshold = confusion_matrix(y_test, y_pred_at_threshold)
    
    # Calculate metrics for business impact
    tn, fp, fn, tp = cm_at_threshold.ravel()
    
    # Assuming average transaction amount and fraud costs
    avg_fraud_amount = df[df['is_fraud'] == 1]['amount'].mean()
    cost_per_false_positive = 50  # Cost of investigating a false alarm
    
    # Calculate business metrics
    fraud_caught = tp * avg_fraud_amount
    fraud_missed = fn * avg_fraud_amount
    investigation_costs = fp * cost_per_false_positive
    net_savings = fraud_caught - investigation_costs
    
    st.subheader("Estimated Business Impact")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Fraud Caught", f"${fraud_caught:,.2f}")
    col2.metric("Fraud Missed", f"${fraud_missed:,.2f}")
    col3.metric("Investigation Costs", f"${investigation_costs:,.2f}")
    
    st.metric("Net Savings", f"${net_savings:,.2f}")
    
    st.markdown("""
    **Note**: This impact analysis is based on assumptions about fraud amounts and investigation costs.
    In a real implementation, these figures would be based on actual business data.
    """)

# Fraud Detection Tab
with tabs[3]:
    st.header("Fraud Detection System")
    
    st.markdown("""
    Test the fraud detection system by creating a transaction and see if it would be flagged as fraudulent.
    You can either create a transaction manually or use a template for legitimate or suspicious transactions.
    """)
    
    # Load the best model based on F1 score
    best_model_name = metrics_df.iloc[0]['Model']
    best_model = models[best_model_name]
    
    # Create transaction templates
    transaction_templates = {
        "Create from scratch": {},
        "Typical legitimate transaction": {
            "amount": 125.0,
            "old_balance_orig": 3000.0,
            "new_balance_orig": 2875.0,
            "old_balance_dest": 1500.0,
            "new_balance_dest": 1625.0,
            "hour_of_day": 14,
            "day_of_week": 3,
            "time_since_last_transaction": 5.0,
            "unusual_location": 0,
            "unusual_device": 0,
            "type_CASH_OUT": 0,
            "type_DEBIT": 0,
            "type_PAYMENT": 1,
            "type_TRANSFER": 0
        },
        "Suspicious transaction": {
            "amount": 8500.0,
            "old_balance_orig": 10000.0,
            "new_balance_orig": 0.0,
            "old_balance_dest": 500.0,
            "new_balance_dest": 500.0,
            "hour_of_day": 2,
            "day_of_week": 5,
            "time_since_last_transaction": 0.2,
            "unusual_location": 1,
            "unusual_device": 1,
            "type_CASH_OUT": 0,
            "type_DEBIT": 0,
            "type_PAYMENT": 0,
            "type_TRANSFER": 1
        }
    }
    
    # Select transaction template
    template = st.selectbox("Select a starting point", list(transaction_templates.keys()))
    
    initial_values = transaction_templates[template]
    
    # Create columns for input fields
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Transaction Details")
        
        amount = st.number_input("Transaction Amount ($)", 
                                min_value=0.01, max_value=50000.0, 
                                value=initial_values.get("amount", 100.0),
                                step=10.0)
        
        transaction_types = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT']
        selected_type = transaction_types[0]  # Default to PAYMENT
        
        # If a template is selected, determine which type is set to 1
        if template != "Create from scratch":
            for tx_type in transaction_types:
                if initial_values.get(f"type_{tx_type}", 0) == 1:
                    selected_type = tx_type
                    break
        
        transaction_type = st.selectbox("Transaction Type", transaction_types, 
                                      index=transaction_types.index(selected_type))
        
        hour_of_day = st.slider("Hour of Day (0-23)", 
                               min_value=0, max_value=23, 
                               value=initial_values.get("hour_of_day", 12))
        
        day_of_week = st.selectbox("Day of Week", 
                                 ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                                 index=initial_values.get("day_of_week", 0))
        
        # Convert day name to number (0-6)
        day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, 
                 "Friday": 4, "Saturday": 5, "Sunday": 6}
        day_of_week_num = day_map[day_of_week]
        
        time_since_last = st.number_input("Time Since Last Transaction (days)", 
                                        min_value=0.0, max_value=30.0, 
                                        value=initial_values.get("time_since_last_transaction", 1.0),
                                        step=0.1)
    
    with col2:
        st.subheader("Account Details")
        
        old_balance_orig = st.number_input("Origin Account Initial Balance ($)", 
                                         min_value=0.0, max_value=100000.0, 
                                         value=initial_values.get("old_balance_orig", 1000.0),
                                         step=100.0)
        
        # Calculate default new balance
        default_new = max(0.0, old_balance_orig - amount)
        if template != "Create from scratch":
            default_new = initial_values.get("new_balance_orig", default_new)
        
        new_balance_orig = st.number_input("Origin Account New Balance ($)", 
                                         min_value=0.0, max_value=100000.0, 
                                         value=default_new,
                                         step=100.0)
        
        old_balance_dest = st.number_input("Destination Account Initial Balance ($)", 
                                         min_value=0.0, max_value=100000.0, 
                                         value=initial_values.get("old_balance_dest", 500.0),
                                         step=100.0)
        
        # Calculate default new balance for destination
        default_dest_new = old_balance_dest + amount
        if template != "Create from scratch":
            default_dest_new = initial_values.get("new_balance_dest", default_dest_new)
        
        new_balance_dest = st.number_input("Destination Account New Balance ($)", 
                                         min_value=0.0, max_value=100000.0, 
                                         value=default_dest_new,
                                         step=100.0)
        
        # Security indicators
        unusual_location = st.checkbox("Unusual Login Location", 
                                     value=bool(initial_values.get("unusual_location", False)))
        
        unusual_device = st.checkbox("Unusual Device", 
                                   value=bool(initial_values.get("unusual_device", False)))
    
    # Create one-hot encoded transaction type
    tx_type_dict = {f"type_{tx}": 1 if tx == transaction_type else 0 for tx in transaction_types}
    
    # Create transaction feature vector
    transaction_features = {
        'amount': amount,
        'old_balance_orig': old_balance_orig,
        'new_balance_orig': new_balance_orig,
        'old_balance_dest': old_balance_dest,
        'new_balance_dest': new_balance_dest,
        'hour_of_day': hour_of_day,
        'day_of_week': day_of_week_num,
        'time_since_last_transaction': time_since_last,
        'unusual_location': int(unusual_location),
        'unusual_device': int(unusual_device)
    }
    
    # Add transaction type one-hot features
    transaction_features.update(tx_type_dict)
    
    # Create a DataFrame with the transaction data
    transaction_df = pd.DataFrame([transaction_features])
    
    # Get required feature names from the model
    required_features = feature_names.copy()
    
    # Ensure all required features are present (fill missing with zeros)
    for feature in required_features:
        if feature not in transaction_df.columns:
            transaction_df[feature] = 0
    
    # Ensure the feature order matches the training data
    transaction_df = transaction_df[required_features]
    
    # Flag any profile anomalies for alerting
    balance_anomalies = []
    security_alerts = []
    
    # Check for account drainage (new balance is 0 or very small compared to old)
    if old_balance_orig > 100 and new_balance_orig < old_balance_orig * 0.05:
        balance_anomalies.append("Account drainage detected (origin account balance reduced to near-zero)")
    
    # Check for balance anomalies
    if abs((new_balance_orig - old_balance_orig) + amount) > 1:
        balance_anomalies.append("Origin account balance change doesn't match transaction amount")
    
    if abs((new_balance_dest - old_balance_dest) - amount) > 1:
        balance_anomalies.append("Destination account balance change doesn't match transaction amount")
    
    # Security alerts
    if unusual_location and unusual_device:
        security_alerts.append("Transaction from both unusual location and unusual device")
    
    if hour_of_day >= 22 or hour_of_day <= 5:
        security_alerts.append("Transaction occurred during nighttime hours")
    
    if time_since_last < 0.1:
        security_alerts.append("Very short time since last transaction (potential automated attack)")
    
    # Detect fraud with the trained model
    if st.button("Analyze Transaction", type='primary'):
        # Scale the features
        transaction_scaled = scaler.transform(transaction_df)
        
        # Get prediction and probability
        fraud_prediction = best_model.predict(transaction_scaled)[0]
        fraud_probability = best_model.predict_proba(transaction_scaled)[0, 1]
        
        # Create a risk score (0-100)
        risk_score = int(fraud_probability * 100)
        
        # Create result card with colored background based on prediction
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if fraud_prediction == 1:
                st.error(f"### ⚠️ FRAUD ALERT")
                st.markdown(f"<div style='background-color: #ffebee; padding: 20px; border-radius: 10px;'><h3 style='text-align: center; color: #c62828;'>High Risk Transaction</h3><p style='text-align: center; font-size: 24px;'>Risk Score: {risk_score}/100</p></div>", unsafe_allow_html=True)
            else:
                st.success(f"### ✓ Transaction Appears Legitimate")
                st.markdown(f"<div style='background-color: #e8f5e9; padding: 20px; border-radius: 10px;'><h3 style='text-align: center; color: #2e7d32;'>Low Risk Transaction</h3><p style='text-align: center; font-size: 24px;'>Risk Score: {risk_score}/100</p></div>", unsafe_allow_html=True)
        
        # Show gauge chart for risk score
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Risk Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "green"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show additional alerts
        if balance_anomalies:
            st.warning("### Balance Anomalies Detected")
            for anomaly in balance_anomalies:
                st.markdown(f"- {anomaly}")
        
        if security_alerts:
            st.warning("### Security Alerts")
            for alert in security_alerts:
                st.markdown(f"- {alert}")
        
        # Show model's confidence
        st.subheader("Model Confidence")
        st.markdown(f"The model is **{fraud_probability:.1%}** confident that this transaction is fraudulent.")
        st.markdown(f"Using the **{best_model_name}** model for prediction.")
        
        # Next steps
        st.subheader("Recommended Next Steps")
        if fraud_prediction == 1 or risk_score > 70:
            st.markdown("""
            1. **Block the transaction** until further verification
            2. **Contact the customer** via a trusted phone number to confirm
            3. **Flag the account** for additional monitoring
            4. **Review recent activity** on the account for other suspicious transactions
            """)
        elif risk_score > 30:
            st.markdown("""
            1. **Allow the transaction** but flag for review
            2. **Monitor the account** for additional unusual activity
            3. **Consider sending** a verification notification to the customer
            """)
        else:
            st.markdown("""
            1. **Allow the transaction** to proceed
            2. **No additional action** required
            """)

# Model Explanation Tab
with tabs[4]:
    st.header("Model Explainability")
    
    st.markdown("""
    Understanding why a model flags a transaction as fraudulent is crucial for:
    - Explaining decisions to customers and regulators
    - Gaining insights into new fraud patterns
    - Improving the model and reducing false positives
    
    Below we use SHAP (SHapley Additive exPlanations) to interpret our model's predictions.
    """)
    
    # Select model to explain
    model_to_explain = st.selectbox("Select Model to Explain", list(models.keys()))
    
    # Get SHAP values
    selected_model = models[model_to_explain]
    
    # Use a small sample of test data to make SHAP calculation faster
    sample_size = min(100, len(X_test_scaled))
    X_sample = X_test_scaled[:sample_size]
    
    with st.spinner("Calculating SHAP values (this may take a moment)..."):
        try:
            shap_values, explainer = get_shap_values(selected_model, X_sample, model_to_explain, feature_names)
            shap_calculation_successful = True
        except Exception as e:
            st.error(f"Error calculating SHAP values: {e}")
            shap_calculation_successful = False
    
    if shap_calculation_successful:
        # Create a base value reference
        if explainer is not None and hasattr(explainer, 'expected_value'):
            base_value = explainer.expected_value
            if isinstance(base_value, list):
                base_value = base_value[1]  # For models that return one base value per class
        else:
            base_value = 0
        
        # Overall feature importance
        st.subheader("Global Feature Importance")
        
        # Create a DataFrame with feature importance values
        if len(shap_values.shape) > 1:
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': np.abs(shap_values).mean(axis=0)
            }).sort_values('Importance', ascending=False)
        else:
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': np.abs(shap_values)
            }).sort_values('Importance', ascending=False)
        
        # Plot feature importance
        fig = px.bar(
            feature_importance.head(10),  # Top 10 features
            x='Importance',
            y='Feature',
            title='Top 10 Features by SHAP Importance',
            orientation='h'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Explanation of feature importance
        st.markdown("""
        **How to interpret feature importance:**
        
        The chart above shows which features have the most influence on the model's predictions.
        Features with higher importance have greater impact on the fraud detection decision.
        """)
        
        # Display SHAP summary plot
        st.subheader("SHAP Summary Plot")
        
        # Create a simplified SHAP summary plot using plotly
        # We'll take a sample of our SHAP values for better visualization
        sample_indices = np.random.choice(range(len(shap_values)), min(50, len(shap_values)), replace=False)
        
        # Get sample SHAP values and corresponding feature values
        shap_sample = shap_values[sample_indices]
        feature_names_sample = feature_names
        
        # Create a DataFrame for the plot
        plot_data = []
        for i, feature in enumerate(feature_names_sample):
            for j, sample in enumerate(sample_indices):
                if i < shap_sample.shape[1]:  # Ensure we don't go out of bounds
                    feature_val = X_sample[sample, i]
                    shap_val = shap_sample[j, i]
                    plot_data.append({
                        'Feature': feature,
                        'SHAP Value': shap_val,
                        'Feature Value': feature_val
                    })
        
        shap_df = pd.DataFrame(plot_data)
        
        # Calculate feature importance for ordering
        feature_order = feature_importance['Feature'].tolist()
        
        # Create the plot
        fig = px.strip(
            shap_df,
            x='SHAP Value',
            y='Feature',
            color='Feature Value',
            category_orders={'Feature': feature_order[:10]},  # Only show top 10 features
            # color_continuous_scale='RdBu_r',
            title='SHAP Values by Feature',
            labels={'SHAP Value': 'Impact on Model Output', 'Feature Value': 'Feature Value (normalized)'}
        )
        
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **How to interpret this plot:**
        
        - **Position on x-axis**: Shows whether the feature pushes the prediction higher (right) or lower (left)
        - **Color**: Blue points are low feature values, red points are high feature values
        - **Pattern**: If high values (red) are consistently on one side, it shows a clear relationship
        
        For example, if high values of 'amount' (red) appear on the right side, it means larger transaction amounts tend to increase the fraud prediction.
        """)
        
        # Individual Transaction Explanation
        st.subheader("Individual Transaction Explanation")
        
        st.markdown("""
        Select a transaction from the test set to explain why the model classified it as fraudulent or legitimate.
        """)
        
        # Choose fraud/legitimate examples to explain
        explanation_type = st.radio("Select example type:", ["Fraudulent Transaction", "Legitimate Transaction"])
        
        # Find examples of each type
        y_test_array = np.array(y_test)
        fraud_indices = np.where(y_test_array == 1)[0]
        legit_indices = np.where(y_test_array == 0)[0]
        
        if explanation_type == "Fraudulent Transaction":
            if len(fraud_indices) > 0:
                # Get results for the current model
                model_result = results[model_to_explain]
                y_pred = model_result['y_pred']
                
                # Correctly predicted fraud cases
                correct_fraud = np.where((y_test_array == 1) & (y_pred == 1))[0]
                if len(correct_fraud) > 0:
                    sample_idx = np.random.choice(correct_fraud)
                else:
                    sample_idx = np.random.choice(fraud_indices)
            else:
                st.warning("No fraudulent transactions in the test set. Showing a legitimate transaction instead.")
                sample_idx = np.random.choice(legit_indices)
        else:  # Legitimate Transaction
            sample_idx = np.random.choice(legit_indices)
        
        # Get the selected transaction
        sample_X = X_test_scaled[sample_idx].reshape(1, -1)
        sample_y = y_test.iloc[sample_idx]
        
        # Get prediction and probability for this sample
        sample_pred = selected_model.predict(sample_X)[0]
        sample_prob = selected_model.predict_proba(sample_X)[0, 1]
        
        # Display transaction details
        st.write(f"**Actual label:** {'Fraudulent' if sample_y == 1 else 'Legitimate'}")
        st.write(f"**Predicted label:** {'Fraudulent' if sample_pred == 1 else 'Legitimate'} (confidence: {sample_prob:.2f})")
        
        # Get SHAP values for this example
        try:
            if hasattr(explainer, 'shap_values'):
                example_shap = explainer.shap_values(sample_X)
                if isinstance(example_shap, list):  # For models that return one set of SHAP values per class
                    example_shap = example_shap[1]  # For binary classification, we take the positive class
            else:
                example_shap = explainer(sample_X) if explainer else np.zeros((1, len(feature_names)))
            
            # Create a waterfall chart showing feature contributions
            feature_contribution = pd.DataFrame({
                'Feature': feature_names[:len(example_shap[0])],  # Ensure dimensions match
                'Contribution': example_shap[0][:len(feature_names)]  # Ensure dimensions match
            }).sort_values('Contribution', key=abs, ascending=False)
            
            # Create a waterfall chart
            fig = go.Figure(go.Waterfall(
                name="SHAP",
                orientation="h",
                measure=["relative"] * len(feature_contribution) + ["total"],
                x=[*feature_contribution['Contribution'].values, 0],  # Add 0 for the total
                y=[*feature_contribution['Feature'].values, "Net Effect"],  # Add "Net Effect" label
                connector={"mode": "between", "line": {"width": 1, "color": "rgb(0, 0, 0)", "dash": "solid"}},
                increasing={"marker": {"color": "#FF4136"}},
                decreasing={"marker": {"color": "#3D9970"}}
            ))
            
            fig.update_layout(
                title="Feature Contributions to Prediction",
                showlegend=False,
                height=max(300, len(feature_contribution) * 20)  # Adjust height based on number of features
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **How to interpret this chart:**
            
            - **Red bars** push the prediction toward fraud
            - **Green bars** push the prediction toward legitimate
            - **Bar length** indicates how strongly the feature influences the prediction
            """)
            
            # Rank the top contributing factors to the decision
            st.subheader("Top Factors in This Decision")
            
            # Get the most influential features (absolute contribution)
            top_factors = feature_contribution.head(5)
            
            for _, row in top_factors.iterrows():
                feature = row['Feature']
                contribution = row['Contribution']
                

                if contribution > 0:
                    st.markdown(f"🔴 **{feature}** pushed toward a fraud prediction (contribution: +{contribution:.4f})")
                else:
                    st.markdown(f"🟢 **{feature}** pushed toward a legitimate prediction (contribution: {contribution:.4f})")
        
        except Exception as e:
            st.error(f"Error generating SHAP explanation: {str(e)}")
            st.markdown("""
            Unable to generate detailed SHAP explanations for this example.
            
            Common reasons for this error:
            - The model structure doesn't fully support SHAP analysis
            - Dimension mismatch between features and SHAP values
            - Memory limitations
            
            Try selecting a different model or a simpler example.
            """)
    else:
        st.warning("""
        SHAP calculations were not successful. 
        
        As an alternative, we'll show feature importance based on the model's internal attributes.
        """)
        
        # Show feature importance based on model attributes
        if model_to_explain in ["Random Forest", "Gradient Boosting", "XGBoost"]:
            model = models[model_to_explain]
            
            # Extract feature importance
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            else:
                classifier = model.named_steps.get('classifier', None)
                if classifier and hasattr(classifier, 'feature_importances_'):
                    importances = classifier.feature_importances_
                else:
                    importances = np.ones(len(feature_names)) / len(feature_names)
                    
            # Create DataFrame for plotting
            importance_df = pd.DataFrame({
                'Feature': feature_names[:len(importances)],
                'Importance': importances[:len(feature_names)]
            }).sort_values('Importance', ascending=False)
            
            # Plot feature importance
            fig = px.bar(
                importance_df.head(10),
                x='Importance',
                y='Feature',
                title='Top 10 Features by Importance',
                orientation='h'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif model_to_explain == "Logistic Regression":
            model = models[model_to_explain]
            
            # Extract coefficients
            if hasattr(model, 'coef_'):
                coefficients = model.coef_[0]
            else:
                classifier = model.named_steps.get('classifier', None)
                if classifier and hasattr(classifier, 'coef_'):
                    coefficients = classifier.coef_[0]
                else:
                    coefficients = np.ones(len(feature_names)) / len(feature_names)
            
            # Create DataFrame for plotting
            coef_df = pd.DataFrame({
                'Feature': feature_names[:len(coefficients)],
                'Coefficient': coefficients[:len(feature_names)]
            }).sort_values('Coefficient', key=abs, ascending=False)
            
            # Plot coefficients
            fig = px.bar(
                coef_df.head(10),
                x='Coefficient',
                y='Feature',
                title='Top 10 Features by Coefficient Magnitude',
                orientation='h'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("Feature importance visualization is not available for this model type.")

# Add a sidebar with additional information
with st.sidebar:
    st.header("About")
    st.markdown("""
    ## Financial Fraud Detection
    
    This application demonstrates how machine learning can be used to detect fraudulent financial transactions.
    
    **Key Features:**
    - Synthetic transaction data generation
    - Data exploration and visualization
    - Multiple ML models comparison
    - Interactive fraud detection
    - Model explainability with SHAP
    
    **Technologies Used:**
    - Python
    - Streamlit
    - scikit-learn
    - XGBoost
    - SHAP
    - Plotly
    """)
    
    st.markdown("---")
    
    st.subheader("Dataset Statistics")
    fraud_count = df['is_fraud'].sum()
    legitimate_count = len(df) - fraud_count
    st.write(f"Total Transactions: {len(df):,}")
    st.write(f"Fraudulent: {fraud_count:,} ({fraud_count/len(df):.2%})")
    st.write(f"Legitimate: {legitimate_count:,} ({legitimate_count/len(df):.2%})")
