# Streamlit Loan Approval Binary Classification Application

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="Loan Approval Predictor", layout="wide")

# Generate synthetic data for demonstration
np.random.seed(42)

@st.cache_data
def generate_loan_data(n_samples=1000):
    """Generate synthetic loan application data."""
    # Features
    age = np.random.normal(35, 10, n_samples)
    income = np.random.normal(60000, 15000, n_samples)
    loan_amount = np.random.normal(200000, 100000, n_samples)
    credit_score = np.random.normal(700, 100, n_samples)
    debt_to_income = np.random.normal(0.3, 0.1, n_samples)
    
    # Create a DataFrame
    data = pd.DataFrame({
        'Age': age,
        'Income': income,
        'LoanAmount': loan_amount,
        'CreditScore': credit_score,
        'DebtToIncome': debt_to_income
    })
    
    # Generate target variable (loan approval) based on a rule
    probability = 1 / (1 + np.exp(-(0.01 * (data['CreditScore'] - 650) + 
                                   0.00001 * (data['Income'] - 40000) - 
                                   5 * (data['DebtToIncome'] - 0.4))))
    data['Approved'] = (np.random.random(n_samples) < probability).astype(int)
    
    return data

@st.cache_resource
def train_model():
    """Train the loan approval model."""
    # Generate data
    loan_data = generate_loan_data()
    
    # Split features and target
    X = loan_data.drop('Approved', axis=1)
    y = loan_data['Approved']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train the model
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions on test set
    y_pred = model.predict(X_test_scaled)
    
    return model, scaler, X_test, y_test, y_pred, loan_data

# Train model and get results
model, scaler, X_test, y_test, y_pred, loan_data = train_model()

# Main application
st.title("Loan Approval Predictor")
st.write("""
This application demonstrates binary classification in machine learning by predicting
whether a loan application will be approved based on applicant information.
""")

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["Make Prediction", "Model Performance", "Data Exploration"])

with tab1:
    st.header("Loan Application Form")
    
    # Create form for user input
    with st.form("loan_application_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            income = st.number_input("Annual Income ($)", min_value=0, max_value=1000000, value=60000)
            loan_amount = st.number_input("Loan Amount ($)", min_value=1000, max_value=1000000, value=200000)
        
        with col2:
            credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
            debt_to_income = st.number_input("Debt-to-Income Ratio (0-1)", min_value=0.0, max_value=1.0, value=0.3, format="%.2f")
        
        submit_button = st.form_submit_button("Predict Loan Approval", type="primary")
    
    if submit_button:
        # Create a DataFrame for the new application
        new_application = pd.DataFrame({
            'Age': [age],
            'Income': [income],
            'LoanAmount': [loan_amount],
            'CreditScore': [credit_score],
            'DebtToIncome': [debt_to_income]
        })
        
        # Scale the features
        new_application_scaled = scaler.transform(new_application)
        
        # Make prediction
        prediction = model.predict(new_application_scaled)[0]
        probability = model.predict_proba(new_application_scaled)[0][1]
        
        # Display result
        st.subheader("Loan Application Result")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if prediction == 1:
                st.success("✅ APPROVED")
            else:
                st.error("❌ REJECTED")
                
            st.metric("Approval Probability", f"{probability:.1%}")
        
        with col2:
            # Create a gauge chart for probability
            fig, ax = plt.subplots(figsize=(4, 0.8))
            ax.barh(0, probability, color='green', height=0.4)
            ax.barh(0, 1-probability, left=probability, color='red', height=0.4)
            ax.set_xlim(0, 1)
            ax.set_yticks([])
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
            plt.tight_layout()
            st.pyplot(fig)
        
        # Show feature importance
        st.subheader("Feature Impact")
        feature_importance = pd.DataFrame({
            'Feature': new_application.columns,
            'Value': new_application.values[0],
            'Coefficient': model.coef_[0]
        })
        
        # Normalize feature values to compare importance
        feature_importance['Normalized_Value'] = (new_application.values[0] - scaler.mean_) / scaler.scale_
        feature_importance['Impact'] = feature_importance['Normalized_Value'] * feature_importance['Coefficient']
        
        # Sort by absolute impact
        feature_importance = feature_importance.sort_values(by='Impact', key=abs, ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        colors = ['green' if x > 0 else 'red' for x in feature_importance['Impact']]
        sns.barplot(x='Impact', y='Feature', data=feature_importance, palette=colors, ax=ax)
        ax.set_title('Feature Impact on Prediction')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        st.pyplot(fig)
        
        # Explanation of feature importance
        st.info("""
        The chart above shows how each feature influenced the prediction. 
        Green bars mean the feature increased the probability of approval, 
        while red bars decreased it.
        """)

with tab2:
    st.header("Model Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Model accuracy and metrics
        accuracy = accuracy_score
        

with tab2:
    st.header("Model Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Model accuracy and metrics
        accuracy = accuracy_score(y_test, y_pred)
        st.metric("Model Accuracy", f"{accuracy:.2f}")
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.subheader("Classification Report")
        st.dataframe(report_df.style.format("{:.2f}"))
    
    with col2:
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        ax.set_xticklabels(['Rejected', 'Approved'])
        ax.set_yticklabels(['Rejected', 'Approved'])
        st.pyplot(fig)
    
    # Feature importance
    st.subheader("Feature Importance")
    feature_importance = pd.DataFrame({
        'Feature': X_test.columns,
        'Importance': np.abs(model.coef_[0])
    }).sort_values('Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax)
    ax.set_title('Feature Importance (Absolute Coefficient Values)')
    st.pyplot(fig)
    
    # ROC curve
    from sklearn.metrics import roc_curve, auc
    y_scores = model.predict_proba(scaler.transform(X_test))[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    st.pyplot(fig)
    
    st.info("""
    **Understanding the metrics:**
    - **Accuracy**: Overall correctness of the model
    - **Precision**: How many of the predicted approvals were actually approved
    - **Recall**: How many of the actual approvals were correctly predicted
    - **F1-score**: Harmonic mean of precision and recall
    - **ROC Curve**: Shows the performance of the model at different threshold settings
    """)

with tab3:
    st.header("Dataset Exploration")
    
    # Show dataset sample
    st.subheader("Sample Data")
    st.dataframe(loan_data.sample(10))
    
    # Dataset statistics
    st.subheader("Dataset Statistics")
    st.dataframe(loan_data.describe())
    
    # Distribution of target variable
    st.subheader("Loan Approval Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    approval_counts = loan_data['Approved'].value_counts()
    sns.barplot(x=approval_counts.index.map({0: 'Rejected', 1: 'Approved'}), y=approval_counts.values, ax=ax)
    ax.set_xlabel('Loan Status')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Loan Approval Status')
    st.pyplot(fig)
    
    # Correlation heatmap
    st.subheader("Feature Correlation")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(loan_data.corr(), annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Matrix')
    st.pyplot(fig)
    
    # Feature distributions by approval status
    st.subheader("Feature Distributions by Approval Status")
    
    # Let user select a feature
    selected_feature = st.selectbox(
        "Select feature to visualize its distribution:",
        options=X_test.columns
    )
    
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(data=loan_data, x=selected_feature, hue='Approved', 
                 kde=True, element='step', common_norm=False,
                 palette=['red', 'green'], 
                 hue_order=[0, 1], ax=ax)
    ax.set_title(f'{selected_feature} Distribution by Approval Status')
    st.pyplot(fig)
    
    # Scatter plot of two features
    st.subheader("Feature Relationship")
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_feature = st.selectbox('Select X-axis feature:', options=X_test.columns, key='x_feature')
    
    with col2:
        y_feature = st.selectbox('Select Y-axis feature:', options=X_test.columns, key='y_feature')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = sns.scatterplot(data=loan_data, x=x_feature, y=y_feature, hue='Approved', 
                     palette=['red', 'green'], alpha=0.6, ax=ax)
    ax.set_title(f'{x_feature} vs {y_feature} by Approval Status')
    
    # Add a legend with custom labels
    handles, labels = scatter.get_legend_handles_labels()
    ax.legend(handles, ['Rejected', 'Approved'])
    
    st.pyplot(fig)

# Add a sidebar with additional information
with st.sidebar:
    st.title("About")
    st.info("""
    **Loan Approval Predictor**
    
    This application demonstrates a binary classification machine learning model for 
    loan approval prediction. The model is trained on synthetic data to illustrate 
    the core principles of machine learning classification.
    
    **Features used:**
    - Age
    - Income
    - Loan Amount
    - Credit Score
    - Debt-to-Income Ratio
    
    **Model:** Logistic Regression
    """)
    
    st.subheader("How it works")
    st.write("""
    1. The model was trained on synthetic data that mimics real loan applications
    2. It learned patterns associated with loan approval/rejection
    3. When you input your information, the model:
       - Standardizes your data
       - Calculates the probability of approval
       - Provides a binary decision (approve/reject)
    """)
    
    # Dataset size information
    st.subheader("Dataset Information")
    st.write(f"Total samples: {len(loan_data)}")
    st.write(f"Approved loans: {loan_data['Approved'].sum()} ({loan_data['Approved'].mean():.1%})")
    st.write(f"Rejected loans: {len(loan_data) - loan_data['Approved'].sum()} ({1 - loan_data['Approved'].mean():.1%})")
