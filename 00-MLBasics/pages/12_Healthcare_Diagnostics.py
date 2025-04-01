# Healthcare Diagnostics ML Application - Streamlit

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, 
                            roc_curve, precision_recall_curve, classification_report)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
import xgboost as xgb
import shap
import time
import json
import pickle
from PIL import Image
import io
import requests
from io import BytesIO
import warnings
import base64

# Suppress warnings
warnings.filterwarnings("ignore")

# Set page configuration
st.set_page_config(
    page_title="Healthcare ML Diagnostics",
    page_icon="🏥",
    layout="wide"
)

# Functions for data loading and processing
@st.cache_data
def load_heart_disease_data():
    """
    Load the heart disease dataset from UCI repository
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    
    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 
        'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    
    try:
        df = pd.read_csv(url, names=column_names, na_values='?')
        
        # Clean the data
        df['ca'] = pd.to_numeric(df['ca'], errors='coerce')
        df['thal'] = pd.to_numeric(df['thal'], errors='coerce')
        
        # Replace '?' with NaN and handle missing values
        df = df.replace('?', np.nan)
        
        # Convert target to binary (0 = no disease, 1 = disease)
        df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Return a mock dataset if loading fails
        return create_mock_heart_disease_data()

@st.cache_data
def load_diabetes_data():
    """
    Load the Pima Indians Diabetes dataset
    """
    try:
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        column_names = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
        ]
        df = pd.read_csv(url, names=column_names)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Return a mock dataset if loading fails
        return create_mock_diabetes_data()

def create_mock_heart_disease_data():
    """
    Create a mock heart disease dataset if the original cannot be loaded
    """
    np.random.seed(42)
    n = 303  # Same size as original dataset
    
    # Generate synthetic data
    age = np.random.randint(25, 80, n)
    sex = np.random.randint(0, 2, n)
    cp = np.random.randint(0, 4, n)  # Chest pain type
    trestbps = np.random.randint(90, 200, n)  # Resting blood pressure
    chol = np.random.randint(120, 400, n)  # Cholesterol
    fbs = np.random.randint(0, 2, n)  # Fasting blood sugar
    restecg = np.random.randint(0, 3, n)  # Rest ECG
    thalach = np.random.randint(80, 220, n)  # Max heart rate
    exang = np.random.randint(0, 2, n)  # Exercise induced angina
    oldpeak = np.round(np.random.uniform(0, 6, n), 1)  # ST depression
    slope = np.random.randint(0, 3, n)  # ST slope
    ca = np.random.randint(0, 4, n)  # Major vessels
    thal = np.random.randint(1, 4, n)  # Thalassemia
    
    # Create target variable with correlation to features
    target = np.zeros(n, dtype=int)
    for i in range(n):
        # Simple model: higher risk with age, chest pain, and cholesterol
        risk = (
            0.01 * age[i] +
            0.5 * cp[i] +
            0.005 * chol[i] -
            0.005 * thalach[i] +
            0.5 * exang[i] +
            oldpeak[i] +
            0.5 * ca[i]
        )
        
        # Apply sigmoid to get probability and convert to binary
        prob = 1 / (1 + np.exp(-risk + 5))  # +5 to center the distribution
        target[i] = 1 if np.random.random() < prob else 0
    
    # Create DataFrame
    df = pd.DataFrame({
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal, 'target': target
    })
    
    return df

def create_mock_diabetes_data():
    """
    Create a mock diabetes dataset if the original cannot be loaded
    """
    np.random.seed(42)
    n = 768  # Same size as original dataset
    
    # Generate synthetic data
    pregnancies = np.random.randint(0, 18, n)
    glucose = np.random.randint(50, 200, n)
    blood_pressure = np.random.randint(30, 120, n)
    skin_thickness = np.random.randint(5, 60, n)
    insulin = np.random.randint(10, 400, n)
    bmi = np.round(np.random.uniform(15, 50, n), 1)
    diabetes_pedigree = np.round(np.random.uniform(0.05, 2.5, n), 3)
    age = np.random.randint(21, 81, n)
    
    # Create target variable with correlation to features
    outcome = np.zeros(n, dtype=int)
    for i in range(n):
        # Simple model: higher risk with glucose, BMI, and age
        risk = (
            0.01 * glucose[i] +
            0.05 * bmi[i] +
            0.01 * age[i] +
            0.2 * diabetes_pedigree[i]
        )
        
        # Apply sigmoid to get probability and convert to binary
        prob = 1 / (1 + np.exp(-risk + 5))  # +5 to center the distribution
        outcome[i] = 1 if np.random.random() < prob else 0
    
    # Create DataFrame
    df = pd.DataFrame({
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': diabetes_pedigree,
        'Age': age,
        'Outcome': outcome
    })
    
    return df

@st.cache_data
def get_heart_disease_description():
    """Return information about heart disease features"""
    return {
        "age": "Age in years",
        "sex": "Sex (1 = male; 0 = female)",
        "cp": "Chest pain type (0-3)",
        "trestbps": "Resting blood pressure in mm Hg",
        "chol": "Serum cholesterol in mg/dl",
        "fbs": "Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)",
        "restecg": "Resting ECG results (0-2)",
        "thalach": "Maximum heart rate achieved",
        "exang": "Exercise induced angina (1 = yes; 0 = no)",
        "oldpeak": "ST depression induced by exercise relative to rest",
        "slope": "Slope of the peak exercise ST segment (0-2)",
        "ca": "Number of major vessels (0-3) colored by fluoroscopy",
        "thal": "Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect)",
        "target": "Heart disease (1 = present; 0 = absent)"
    }

@st.cache_data
def get_diabetes_description():
    """Return information about diabetes features"""
    return {
        "Pregnancies": "Number of times pregnant",
        "Glucose": "Plasma glucose concentration (mg/dL)",
        "BloodPressure": "Diastolic blood pressure (mm Hg)",
        "SkinThickness": "Triceps skin fold thickness (mm)",
        "Insulin": "2-Hour serum insulin (mu U/ml)",
        "BMI": "Body mass index (weight in kg/(height in m)²)",
        "DiabetesPedigreeFunction": "Diabetes pedigree function (genetic score)",
        "Age": "Age in years",
        "Outcome": "Class variable (0 = no diabetes, 1 = diabetes)"
    }

def preprocess_data(df, dataset_type):
    """
    Preprocess the data for modeling
    """
    # Handle missing values first
    imputer = SimpleImputer(strategy='median')
    
    if dataset_type == 'heart_disease':
        # Split features and target
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Impute missing values
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        
    else:  # diabetes
        # Split features and target
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        
        # Replace zeros with NaN for certain features
        for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
            X[col] = X[col].replace(0, np.nan)
        
        # Impute missing values
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X, X_scaled, y, X_train, X_test, y_train, y_test, scaler

@st.cache_resource
def train_models(X_train, y_train, X_test, y_test):
    """
    Train various machine learning models and return their performances
    """
    # Define models to train
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "XGBoost": xgb.XGBClassifier(n_estimators=100, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Neural Network": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    }
    
    # Train each model and collect results
    results = {}
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        # Store results
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'y_pred': y_pred,
            'y_proba': y_proba
        }
    
    return results

# @st.cache_resource
def get_feature_importance(model_name, model, X):
    """
    Get feature importance based on the model type
    """
    # @st.cache_data  # Change from st.cache to st.cache_data
    def get_importance(model_name, coef_or_importance, feature_names):
        if model_name in ["Random Forest", "Gradient Boosting", "XGBoost"]:
            return pd.Series(coef_or_importance, index=feature_names)
        elif model_name in ["Logistic Regression", "SVM"]:
            return pd.Series(np.abs(coef_or_importance), index=feature_names)
        return None

    feature_importance = None
    
    if model_name in ["Random Forest", "Gradient Boosting", "XGBoost"]:
        # Tree-based models have feature_importance_ attribute
        if hasattr(model, 'feature_importances_'):
            feature_importance = get_importance(
                model_name, 
                model.feature_importances_,
                X.columns
            )
    elif model_name == "Logistic Regression":
        # Linear models have coef_ attribute
        if hasattr(model, 'coef_'):
            feature_importance = get_importance(
                model_name,
                model.coef_[0],
                X.columns
            )
    elif model_name == "SVM" and hasattr(model, 'coef_'):
        # Linear SVM has coef_
        feature_importance = get_importance(
            model_name,
            model.coef_[0],
            X.columns
        )
    
    return feature_importance


# def get_feature_importance(model_name, model, X):
#     """
#     Get feature importance based on the model type
#     """
#     feature_importance = None
    
#     if model_name in ["Random Forest", "Gradient Boosting", "XGBoost"]:
#         # Tree-based models have feature_importance_ attribute
#         if hasattr(model, 'feature_importances_'):
#             feature_importance = model.feature_importances_
#     elif model_name == "Logistic Regression":
#         # Linear models have coef_ attribute
#         if hasattr(model, 'coef_'):
#             feature_importance = np.abs(model.coef_[0])  # Take absolute values for importance
#     elif model_name == "SVM" and hasattr(model, 'coef_'):
#         # Linear SVM has coef_
#         feature_importance = np.abs(model.coef_[0])
    
#     return feature_importance

# @st.cache_resource
def calculate_shap_values(model, X_sample, model_name):
    """
    Calculate SHAP values for model interpretation
    """
    try:
        # For different model types, use appropriate explainer
        if model_name in ["Random Forest", "Gradient Boosting", "XGBoost"]:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # For binary classification, take the positive class
        elif model_name in ["Logistic Regression", "SVM"]:
            explainer = shap.LinearExplainer(model, X_sample)
            shap_values = explainer.shap_values(X_sample)
        else:
            # For other models, use KernelExplainer but limit to fewer samples for performance
            background = shap.kmeans(X_sample, 10)
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values = explainer.shap_values(X_sample[:50])[1]
        
        return shap_values, explainer
    except Exception as e:
        st.warning(f"SHAP calculation failed: {e}. Using default feature importance instead.")
        return None, None

def get_normal_ranges(dataset_type):
    """
    Return normal ranges for health metrics
    """
    if dataset_type == 'heart_disease':
        return {
            'age': (18, 100),
            'sex': (0, 1),
            'cp': (0, 3),
            'trestbps': (90, 120),  # Normal resting blood pressure
            'chol': (125, 200),     # Normal cholesterol
            'fbs': (0, 1),
            'restecg': (0, 2),
            'thalach': (85, 185),   # Normal max heart rate
            'exang': (0, 1),
            'oldpeak': (0, 4),
            'slope': (0, 2),
            'ca': (0, 3),
            'thal': (1, 3)
        }
    else:  # diabetes
        return {
            'Pregnancies': (0, 20),
            'Glucose': (70, 99),      # Normal fasting glucose
            'BloodPressure': (60, 80), # Normal diastolic BP
            'SkinThickness': (10, 40), # Normal triceps skin fold
            'Insulin': (16, 166),      # Normal insulin levels
            'BMI': (18.5, 24.9),       # Normal BMI
            'DiabetesPedigreeFunction': (0.1, 2.5),
            'Age': (18, 100)
        }

# Main application
def main():
    st.title("🏥 Healthcare Diagnostics with Machine Learning")
    
    # App description
    st.markdown("""
    This application demonstrates how machine learning can be used to assist in healthcare diagnostics.
    The system analyzes patient data to predict the likelihood of heart disease or diabetes.
    
    **Note:** This is a demonstration using publicly available datasets and should not be used for actual medical diagnosis.
    Always consult with healthcare professionals for medical advice.
    """)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Go to", 
        ["Introduction", "Heart Disease Prediction", "Diabetes Prediction", "About ML in Healthcare"])
    
    # Load datasets
    heart_df = load_heart_disease_data()
    diabetes_df = load_diabetes_data()
    
    # Get feature descriptions
    heart_desc = get_heart_disease_description()
    diabetes_desc = get_diabetes_description()
    
    # Introduction
    if app_mode == "Introduction":
        show_introduction()
    
    # Heart Disease Prediction
    elif app_mode == "Heart Disease Prediction":
        st.header("Heart Disease Prediction")
        tabs = st.tabs(["Make Prediction", "Explore Data", "Model Performance"])
        
        # Handle data and model training
        X_heart, X_heart_scaled, y_heart, X_heart_train, X_heart_test, y_heart_train, y_heart_test, heart_scaler = preprocess_data(heart_df, 'heart_disease')
        heart_results = train_models(X_heart_train, y_heart_train, X_heart_test, y_heart_test)
        
        # Make Prediction tab
        with tabs[0]:
            show_heart_disease_prediction(heart_results, X_heart, heart_scaler, heart_desc)
        
        # Explore Data tab
        with tabs[1]:
            explore_heart_data(heart_df, heart_desc)
        
        # Model Performance tab
        with tabs[2]:
            show_model_performance(heart_results, "heart disease", X_heart, X_heart_test, y_heart_test, heart_df.columns[:-1])
    
    # Diabetes Prediction
    elif app_mode == "Diabetes Prediction":
        st.header("Diabetes Prediction")
        tabs = st.tabs(["Make Prediction", "Explore Data", "Model Performance"])
        
        # Handle data and model training
        X_diabetes, X_diabetes_scaled, y_diabetes, X_diabetes_train, X_diabetes_test, y_diabetes_train, y_diabetes_test, diabetes_scaler = preprocess_data(diabetes_df, 'diabetes')
        diabetes_results = train_models(X_diabetes_train, y_diabetes_train, X_diabetes_test, y_diabetes_test)
        
        # Make Prediction tab
        with tabs[0]:
            show_diabetes_prediction(diabetes_results, X_diabetes, diabetes_scaler, diabetes_desc)
        
        # Explore Data tab
        with tabs[1]:
            explore_diabetes_data(diabetes_df, diabetes_desc)
        
        # Model Performance tab
        with tabs[2]:
            show_model_performance(diabetes_results, "diabetes", X_diabetes, X_diabetes_test, y_diabetes_test, diabetes_df.columns[:-1])
    
    # About ML in Healthcare
    elif app_mode == "About ML in Healthcare":
        show_about_ml_healthcare()
    
    # Footer
    st.markdown("---")
    st.caption("This application is for educational purposes only. Not for medical use.")

def show_introduction():
    st.header("Machine Learning in Healthcare Diagnostics")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        Machine learning is transforming healthcare by enabling earlier detection, more accurate diagnosis, 
        and personalized treatment plans. This application demonstrates two common use cases:
        
        ### 1. Heart Disease Prediction
        
        Early detection of heart disease risk factors can significantly improve patient outcomes.
        Our model analyzes clinical parameters like:
        - Age and demographic information
        - Cholesterol levels
        - Blood pressure measurements
        - ECG results
        - Exercise test data
        
        ### 2. Diabetes Prediction
        
        Diabetes affects millions worldwide, and early intervention can prevent complications.
        The diabetes prediction model considers:
        - Glucose levels
        - BMI (Body Mass Index)
        - Blood pressure
        - Age and other personal factors
        - Family history through diabetes pedigree function
        
        ### How to Use This Application
        
        - Navigate using the sidebar menu
        - Enter patient parameters to get predictions
        - Explore the datasets to understand risk factors
        - Review model performance metrics
        
        **Remember:** This is a demonstration tool. Real medical decisions should involve healthcare professionals.
        """)
    
    with col2:
        st.image("https://img.freepik.com/premium-vector/doctor-patient-laptop-with-healthcare-medical-icons-monitor_333239-21.jpg", 
                caption="Healthcare ML Applications", use_container_width=True)
        
        st.markdown("""
        ### Benefits of ML in Healthcare
        
        - **Early Detection**: Identify disease risks before symptoms appear
        - **Decision Support**: Assist healthcare providers with evidence-based insights
        - **Personalized Medicine**: Tailor treatments to individual patient profiles
        - **Resource Optimization**: Focus healthcare resources where they're needed most
        - **Continuous Improvement**: Models can learn from new data and outcomes
        """)

def show_heart_disease_prediction(results, X, scaler, feature_desc):
    st.subheader("Patient Data Input")
    st.write("Enter patient information to predict heart disease risk")
    
    # Select best model (by AUC)
    best_model_name = max(results.items(), key=lambda x: x[1]['auc'])[0]
    
    # Option to choose model
    model_name = st.selectbox("Select model for prediction", 
                            list(results.keys()), 
                            index=list(results.keys()).index(best_model_name))
    
    model = results[model_name]['model']
    
    # Get normal ranges
    normal_ranges = get_normal_ranges('heart_disease')
    
    # Create form for input
    with st.form(key="heart_disease_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", 
                                min_value=18, max_value=100, 
                                value=50, 
                                help=feature_desc['age'])
            
            sex = st.radio("Sex", 
                        [0, 1], 
                        format_func=lambda x: "Female" if x == 0 else "Male",
                        horizontal=True,
                        help=feature_desc['sex'])
            
            cp = st.selectbox("Chest Pain Type", 
                           [0, 1, 2, 3], 
                           format_func=lambda x: ["Typical Angina", "Atypical Angina", 
                                                "Non-anginal Pain", "Asymptomatic"][x],
                           help=feature_desc['cp'])
            
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 
                                    min_value=80, max_value=220, 
                                    value=120,
                                    help=feature_desc['trestbps'])
            
            chol = st.number_input("Serum Cholesterol (mg/dl)", 
                                min_value=100, max_value=600, 
                                value=200,
                                help=feature_desc['chol'])
        
        with col2:
            fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", 
                        [0, 1], 
                        format_func=lambda x: "No" if x == 0 else "Yes",
                        horizontal=True,
                        help=feature_desc['fbs'])
            
            restecg = st.selectbox("Resting ECG Results", 
                                [0, 1, 2], 
                                format_func=lambda x: ["Normal", "ST-T Wave Abnormality", 
                                                     "Left Ventricular Hypertrophy"][x],
                                help=feature_desc['restecg'])
            
            thalach = st.number_input("Maximum Heart Rate", 
                                    min_value=60, max_value=220, 
                                    value=150,
                                    help=feature_desc['thalach'])
            
            exang = st.radio("Exercise-Induced Angina", 
                           [0, 1], 
                           format_func=lambda x: "No" if x == 0 else "Yes",
                           horizontal=True,
                           help=feature_desc['exang'])
            
            oldpeak = st.number_input("ST Depression Induced by Exercise", 
                                    min_value=0.0, max_value=10.0, 
                                    value=1.0, step=0.1,
                                    help=feature_desc['oldpeak'])
        
        with col3:
            slope = st.selectbox("Slope of Peak Exercise ST Segment", 
                              [0, 1, 2], 
                              format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x],
                              help=feature_desc['slope'])
            
            ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", 
                           [0, 1, 2, 3, 4],
                           help=feature_desc['ca'])
            
            # thal = st.selectbox("Thalassemia", 
            #                  [1, 2, 3], 
            #                  format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect"][x],
            #                  help=feature_desc['thal'])
            
            thal = st.selectbox("Thalassemia", 
                                [0, 1, 2], 
                                format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect"][x],
                                help=feature_desc['thal'])
            
            
        submitted = st.form_submit_button(label="Predict Heart Disease Risk", type="primary")
        
        
    
    # When form is submitted
    if  submitted:
        # Create input array for prediction
        input_data = np.array([
            age, sex, cp, trestbps, chol, fbs, restecg, thalach, 
            exang, oldpeak, slope, ca, thal
        ]).reshape(1, -1)
        
        # Create a DataFrame for visualization
        input_df = pd.DataFrame([{
            'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
            'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
            'exang': exang, 'oldpeak': oldpeak, 'slope': slope,
            'ca': ca, 'thal': thal
        }])
        
        # Scale the input
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction_proba = model.predict_proba(input_scaled)[0, 1]
        prediction = 1 if prediction_proba >= 0.5 else 0
        
        # Display prediction
        st.markdown("### Prediction Result")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if prediction == 1:
                st.error("⚠️ High Risk of Heart Disease")
                risk_level = "High Risk"
            else:
                st.success("✅ Low Risk of Heart Disease")
                risk_level = "Low Risk"
            
            risk_percentage = f"{prediction_proba:.1%}"
            st.metric("Risk Probability", risk_percentage)
            st.caption(f"Using {model_name} model")
        
        with col2:
            # Create gauge chart for risk visualization
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prediction_proba * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Heart Disease Risk", 'font': {'size': 24}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 25], 'color': 'green'},
                        {'range': [25, 50], 'color': 'yellow'},
                        {'range': [50, 75], 'color': 'orange'},
                        {'range': [75, 100], 'color': 'red'},
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            
            fig.update_layout(
                height=250,
                margin=dict(l=20, r=20, t=30, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': "darkblue", 'family': "Arial"}
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Show risk factors
        st.subheader("Patient Risk Factor Analysis")
        
        # Get feature importance for this model
        feature_importance = get_feature_importance(model_name, model, X)
        
        # Highlight abnormal values
        abnormal_features = []
        for feature, value in input_df.iloc[0].items():
            if feature in normal_ranges:
                low, high = normal_ranges[feature]
                if value < low or value > high:
                    abnormal_features.append(feature)
        

        if abnormal_features:
            st.warning("The following values are outside normal ranges:")
            abnormal_text = ", ".join([f"**{f}** ({input_df[f].iloc[0]})" for f in abnormal_features])
            st.markdown(abnormal_text)
        
        # Show top features contributing to prediction
        if feature_importance is not None:
            sorted_idx = np.argsort(feature_importance)[::-1]
            top_features = [X.columns[i] for i in sorted_idx[:5]]  # Top 5 features
            
            st.subheader("Top Contributing Factors")
            
            # Create contribution chart
            contrib_df = pd.DataFrame({
                'Feature': [feature_desc.get(f, f) for f in top_features],
                'Importance': feature_importance[sorted_idx[:5]],
                'Value': [input_df[f].iloc[0] for f in top_features]
            })
            
            fig = px.bar(contrib_df, x='Importance', y='Feature', 
                       text='Value',
                       orientation='h',
                       title="Feature Importance for Heart Disease Risk",
                       color='Importance'
                    #    color_continuous_scale='Reds'
                       )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Health recommendations
        st.subheader("Recommendations")
        
        if prediction == 1:
            st.markdown("""
            Based on the high risk assessment:
            
            - **Consult a cardiologist** for comprehensive evaluation
            - **Monitor blood pressure** and heart rate regularly
            - **Consider stress test or ECG** as recommended by your doctor
            - **Review medication** and treatment options
            - **Lifestyle changes** may include:
              - Heart-healthy diet low in saturated fats
              - Regular exercise appropriate for your condition
              - Smoking cessation if applicable
              - Stress reduction techniques
            """)
        else:
            st.markdown("""
            While your current risk appears low:
            
            - **Continue regular check-ups** with your healthcare provider
            - **Maintain heart-healthy habits**:
              - Regular physical activity (aim for 150 minutes/week)
              - Balanced diet rich in fruits, vegetables, and whole grains
              - Limited alcohol intake
              - Adequate sleep (7-8 hours)
            - **Monitor your numbers** (blood pressure, cholesterol) annually
            """)
        
        # Disclaimer
        st.info("**Disclaimer:** This prediction is for educational purposes only and should not replace professional medical advice.")

def show_diabetes_prediction(results, X, scaler, feature_desc):
    st.subheader("Patient Data Input")
    st.write("Enter patient information to predict diabetes risk")
    
    # Select best model (by AUC)
    best_model_name = max(results.items(), key=lambda x: x[1]['auc'])[0]
    
    # Option to choose model
    model_name = st.selectbox("Select model for prediction", 
                            list(results.keys()), 
                            index=list(results.keys()).index(best_model_name))
    
    model = results[model_name]['model']
    
    # Get normal ranges
    normal_ranges = get_normal_ranges('diabetes')
    
    # Create form for input
    with st.form(key="diabetes_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            pregnancies = st.number_input("Number of Pregnancies", 
                                        min_value=0, max_value=20, 
                                        value=2, 
                                        help=feature_desc['Pregnancies'])
            
            glucose = st.number_input("Plasma Glucose Concentration (mg/dL)", 
                                    min_value=40, max_value=300, 
                                    value=120,
                                    help=feature_desc['Glucose'])
            
            blood_pressure = st.number_input("Diastolic Blood Pressure (mm Hg)", 
                                          min_value=40, max_value=130, 
                                          value=70,
                                          help=feature_desc['BloodPressure'])
            
            skin_thickness = st.number_input("Triceps Skin Fold Thickness (mm)", 
                                          min_value=5, max_value=100, 
                                          value=20,
                                          help=feature_desc['SkinThickness'])
        
        with col2:
            insulin = st.number_input("2-Hour Serum Insulin (mu U/ml)", 
                                    min_value=0, max_value=700, 
                                    value=80,
                                    help=feature_desc['Insulin'])
            
            bmi = st.number_input("Body Mass Index (kg/m²)", 
                                min_value=15.0, max_value=60.0, 
                                value=25.0, step=0.1,
                                help=feature_desc['BMI'])
            
            dpf = st.number_input("Diabetes Pedigree Function", 
                                min_value=0.05, max_value=2.5, 
                                value=0.5, step=0.01,
                                help=feature_desc['DiabetesPedigreeFunction'])
            
            age = st.number_input("Age (years)", 
                                min_value=18, max_value=100, 
                                value=35,
                                help=feature_desc['Age'])
        
        submit_button = st.form_submit_button(label="Predict Diabetes Risk", type="primary")
    
    # When form is submitted
    if submit_button:
        # Create input array for prediction
        input_data = np.array([
            pregnancies, glucose, blood_pressure, skin_thickness, 
            insulin, bmi, dpf, age
        ]).reshape(1, -1)
        
        # Create a DataFrame for visualization
        input_df = pd.DataFrame([{
            'Pregnancies': pregnancies, 'Glucose': glucose, 
            'BloodPressure': blood_pressure, 'SkinThickness': skin_thickness,
            'Insulin': insulin, 'BMI': bmi, 
            'DiabetesPedigreeFunction': dpf, 'Age': age
        }])
        
        # Scale the input
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction_proba = model.predict_proba(input_scaled)[0, 1]
        prediction = 1 if prediction_proba >= 0.5 else 0
        
        # Display prediction
        st.markdown("### Prediction Result")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if prediction == 1:
                st.error("⚠️ High Risk of Diabetes")
                risk_level = "High Risk"
            else:
                st.success("✅ Low Risk of Diabetes")
                risk_level = "Low Risk"
            
            risk_percentage = f"{prediction_proba:.1%}"
            st.metric("Risk Probability", risk_percentage)
            st.caption(f"Using {model_name} model")
        
        with col2:
            # Create gauge chart for risk visualization
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prediction_proba * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Diabetes Risk", 'font': {'size': 24}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 25], 'color': 'green'},
                        {'range': [25, 50], 'color': 'yellow'},
                        {'range': [50, 75], 'color': 'orange'},
                        {'range': [75, 100], 'color': 'red'},
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            
            fig.update_layout(
                height=250,
                margin=dict(l=20, r=20, t=30, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': "darkblue", 'family': "Arial"}
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Show risk factors
        st.subheader("Patient Risk Factor Analysis")
        
        # Get feature importance for this model
        feature_importance = get_feature_importance(model_name, model, X)
        
        # Highlight abnormal values
        abnormal_features = []
        for feature, value in input_df.iloc[0].items():
            if feature in normal_ranges:
                low, high = normal_ranges[feature]
                if value < low or value > high:
                    abnormal_features.append(feature)
        
        if abnormal_features:
            st.warning("The following values are outside normal ranges:")
            abnormal_text = ", ".join([f"**{f}** ({input_df[f].iloc[0]})" for f in abnormal_features])
            st.markdown(abnormal_text)
        
        # Show top features contributing to prediction
        if feature_importance is not None:
            sorted_idx = np.argsort(feature_importance)[::-1]
            top_features = [X.columns[i] for i in sorted_idx[:5]]  # Top 5 features
            
            st.subheader("Top Contributing Factors")
            
            # Create contribution chart
            contrib_df = pd.DataFrame({
                'Feature': [feature_desc.get(f, f) for f in top_features],
                'Importance': feature_importance[sorted_idx[:5]],
                'Value': [input_df[f].iloc[0] for f in top_features]
            })
            
            fig = px.bar(contrib_df, x='Importance', y='Feature', 
                       text='Value',
                       orientation='h',
                       title="Feature Importance for Diabetes Risk",
                       color='Importance',
                    #    color_continuous_scale='Blues'
                       )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Health recommendations
        st.subheader("Recommendations")
        
        if prediction == 1:
            st.markdown("""
            Based on the high risk assessment:
            
            - **Consult a healthcare provider** for proper diabetes testing
            - **Monitor blood glucose levels** regularly
            - **Review diet** with focus on:
              - Limiting simple carbohydrates and sugars
              - Increasing fiber intake
              - Controlling portion sizes
            - **Increase physical activity** (aim for at least 150 minutes/week)
            - **Maintain healthy weight** through diet and exercise
            - **Regular check-ups** to monitor kidney function, eye health, and circulation
            """)
        else:
            st.markdown("""
            While your current risk appears low:
            
            - **Maintain a healthy lifestyle**:
              - Balanced diet rich in vegetables, fruits, and whole grains
              - Regular physical activity
              - Healthy weight management
            - **Regular check-ups** including blood glucose screening
            - **Stay well-hydrated** and limit sugary beverages
            - **Learn about diabetes symptoms** for early detection
            """)
        
        # Disclaimer
        st.info("**Disclaimer:** This prediction is for educational purposes only and should not replace professional medical advice.")

def explore_heart_data(df, feature_desc):
    st.subheader("Heart Disease Dataset Exploration")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write(f"Total patients: {len(df)}")
        st.write(f"Heart disease cases: {df['target'].sum()} ({df['target'].mean()*100:.1f}%)")
        st.write(f"Healthy cases: {len(df) - df['target'].sum()} ({(1-df['target'].mean())*100:.1f}%)")
        
        st.markdown("### Feature Descriptions")
        for feature, description in feature_desc.items():
            st.markdown(f"**{feature}**: {description}")
    
    with col2:
        # Show distribution of heart disease by age
        fig = px.histogram(df, x="age", color="target", 
                         barmode="group", 
                         color_discrete_map={0: "#3498db", 1: "#e74c3c"},
                         labels={"target": "Heart Disease", "age": "Age"},
                         title="Heart Disease Distribution by Age")
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Show correlation heatmap
    st.subheader("Feature Correlations")
    
    # Calculate correlation matrix
    corr = df.corr()
    
    # Create heatmap
    fig = px.imshow(corr, text_auto=".2f", aspect="auto")
    fig.update_layout(title="Correlation Matrix", height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Interactive feature exploration
    st.subheader("Feature Exploration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        feature1 = st.selectbox("Select First Feature", options=df.columns[:-1], index=0)
    
    with col2:
        feature2 = st.selectbox("Select Second Feature", options=df.columns[:-1], index=4)
    
    # Create scatter plot
    fig = px.scatter(df, x=feature1, y=feature2, color="target",
                   color_discrete_map={0: "#3498db", 1: "#e74c3c"},
                   labels={"target": "Heart Disease"},
                   title=f"Relationship between {feature1} and {feature2}")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show distribution of target by categorical features
    categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    
    st.subheader("Disease Distribution by Categorical Features")
    
    # Choose a feature to display
    selected_cat = st.selectbox("Select Categorical Feature", options=categorical_features)
    
    # Special formatting for categorical features
    if selected_cat == "sex":
        category_names = ["Female", "Male"]
    elif selected_cat == "cp":
        category_names = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
    elif selected_cat == "fbs":
        category_names = ["≤ 120 mg/dl", "> 120 mg/dl"]
    elif selected_cat == "restecg":
        category_names = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"]
    elif selected_cat == "exang":
        category_names = ["No", "Yes"]
    elif selected_cat == "slope":
        category_names = ["Upsloping", "Flat", "Downsloping"]
    elif selected_cat == "thal":
        category_names = ["Normal", "Fixed Defect", "Reversible Defect"]
    else:
        category_names = [str(i) for i in range(5)]
    
    # Count data for each category
    # Convert to integer first to handle potential float values
    df[selected_cat] = df[selected_cat].astype(int)
    counts = df.groupby([selected_cat, 'target']).size().unstack(fill_value=0)
    
    # Normalize to get percentages
    percentages = counts.div(counts.sum(axis=1), axis=0) * 100
    
    # Create a bar chart
    fig = go.Figure()
    
    for target, color in zip([0, 1], ["#3498db", "#e74c3c"]):
        if target in percentages.columns:
            fig.add_trace(go.Bar(
                x=[category_names[i] if i < len(category_names) else str(i) for i in percentages.index],
                y=percentages[target],
                name="No Disease" if target == 0 else "Heart Disease",
                marker_color=color
            ))
    
    fig.update_layout(
        barmode='group',
        title=f"Heart Disease Distribution by {selected_cat}",
        xaxis_title=selected_cat,
        yaxis_title="Percentage",
        legend_title="Diagnosis"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def explore_diabetes_data(df, feature_desc):
    st.subheader("Diabetes Dataset Exploration")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write(f"Total patients: {len(df)}")
        st.write(f"Diabetes cases: {df['Outcome'].sum()} ({df['Outcome'].mean()*100:.1f}%)")
        st.write(f"Non-diabetic cases: {len(df) - df['Outcome'].sum()} ({(1-df['Outcome'].mean())*100:.1f}%)")
        
        st.markdown("### Feature Descriptions")
        for feature, description in feature_desc.items():
            st.markdown(f"**{feature}**: {description}")
    
    with col2:
        # Show distribution of diabetes by age
        fig = px.histogram(df, x="Age", color="Outcome", 
                         barmode="group", 
                         color_discrete_map={0: "#3498db", 1: "#e74c3c"},
                         labels={"Outcome": "Diabetes", "Age": "Age"},
                         title="Diabetes Distribution by Age")
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Show correlation heatmap
    st.subheader("Feature Correlations")
    
    # Calculate correlation matrix
    corr = df.corr()
    
    # Create heatmap
    fig = px.imshow(corr, text_auto=".2f", aspect="auto")
    fig.update_layout(title="Correlation Matrix", height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Interactive feature exploration
    st.subheader("Feature Exploration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        feature1 = st.selectbox("Select First Feature", options=df.columns[:-1], index=1)  # Default to Glucose
    
    with col2:
        feature2 = st.selectbox("Select Second Feature", options=df.columns[:-1], index=5)  # Default to BMI
    
    # Create scatter plot
    fig = px.scatter(df, x=feature1, y=feature2, color="Outcome",
                   color_discrete_map={0: "#3498db", 1: "#e74c3c"},
                   labels={"Outcome": "Diabetes"},
                   title=f"Relationship between {feature1} and {feature2}")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show boxplots for numerical features
    st.subheader("Feature Distribution by Diabetes Status")
    
    selected_num = st.selectbox("Select Numerical Feature", 
                             options=['Glucose', 'BMI', 'Age', 'BloodPressure', 
                                     'Insulin', 'DiabetesPedigreeFunction'])
    
    fig = px.box(df, x="Outcome", y=selected_num, 
               color="Outcome", 
               color_discrete_map={0: "#3498db", 1: "#e74c3c"},
               labels={"Outcome": "Diabetes"},
               title=f"Distribution of {selected_num} by Diabetes Status",
               category_orders={"Outcome": [0, 1]})
    
    fig.update_layout(xaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=['No Diabetes', 'Diabetes']))
    st.plotly_chart(fig, use_container_width=True)
    
    # Potential risk factors
    st.subheader("Potential Risk Factors")
    
    # Create a parallel coordinates plot to visualize multiple features at once
    dimensions = [dict(range=[df[col].min(), df[col].max()],
                      label=col, values=df[col]) for col in df.columns[:-1]]
    
    fig = go.Figure(data=
        go.Parcoords(
            line=dict(color=df['Outcome'],
                    colorscale=[[0, '#3498db'], [1, '#e74c3c']],
                    showscale=True,
                    colorbar=dict(title="Diabetes")),
            dimensions=dimensions
        )
    )
    
    fig.update_layout(
        title="Parallel Coordinates Plot of Features",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_model_performance(results, condition, X, X_test, y_test, feature_names):
    st.subheader("Model Performance Comparison")
    
    # Create a comparison table
    comparison_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [results[model]['accuracy'] for model in results],
        'Precision': [results[model]['precision'] for model in results],
        'Recall': [results[model]['recall'] for model in results],
        'F1 Score': [results[model]['f1'] for model in results],
        'ROC AUC': [results[model]['auc'] for model in results]
    })
    
    # Sort by AUC
    comparison_df = comparison_df.sort_values('ROC AUC', ascending=False).reset_index(drop=True)
    
    # Format percentages
    formatted_df = comparison_df.copy()
    for col in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']:
        formatted_df[col] = formatted_df[col].map(lambda x: f"{x:.1%}")
    
    st.table(formatted_df)
    
    # Show best model details
    best_model_name = comparison_df.iloc[0]['Model']
    st.success(f"✅ Best Model: **{best_model_name}** with AUC of {comparison_df.iloc[0]['ROC AUC']}")
    
    # Metrics explanation
    with st.expander("Understanding Performance Metrics"):
        st.markdown("""
        - **Accuracy**: Overall proportion of correct predictions (TP + TN) / Total
        - **Precision**: Proportion of true positives among positive predictions TP / (TP + FP)
        - **Recall (Sensitivity)**: Proportion of true positives correctly identified TP / (TP + FN)
        - **F1 Score**: Harmonic mean of precision and recall, balances both metrics
        - **ROC AUC**: Area under the Receiver Operating Characteristic curve, measures the model's ability to distinguish between classes
        
        For clinical applications, high sensitivity (recall) is often prioritized to minimize false negatives (missed diagnoses).
        """)
    
    # Let user select a model to analyze
    st.subheader("Detailed Model Analysis")
    selected_model = st.selectbox("Select Model", 
                                list(results.keys()), 
                                index=list(results.keys()).index(best_model_name))
    
    model_result = results[selected_model]
    model = model_result['model']
    
    # Show confusion matrix
    st.markdown(f"#### Confusion Matrix for {selected_model}")
    
    cm = confusion_matrix(y_test, model_result['y_pred'])
    
    # Extract the values from the confusion matrix
    tn, fp, fn, tp = cm.ravel()
    
    # Create a heatmap
    fig = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x="Predicted", y="Actual"),
        x=["No Disease", "Disease"],
        y=["No Disease", "Disease"],
        # color_continuous_scale="Blues"
    )
    
    fig.update_layout(title=f"Confusion Matrix - {selected_model}")
    st.plotly_chart(fig, use_container_width=True)
    
    # ROC Curve
    st.markdown(f"#### ROC Curve for {selected_model}")
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_test, model_result['y_proba'])
    auc_score = roc_auc_score(y_test, model_result['y_proba'])
    
    fig = go.Figure()
    
    # Add ROC curve
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC (AUC = {auc_score:.3f})',
        line=dict(color='darkorange', width=2)
    ))
    
    # Add diagonal line (random classifier)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(color='navy', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=f'ROC Curve - {selected_model}',
        xaxis=dict(title='False Positive Rate'),
        yaxis=dict(title='True Positive Rate'),
        legend=dict(x=0.01, y=0.99),
        width=600,
        height=400
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Understanding ROC Curve")
        st.markdown("""
        The ROC curve shows the trade-off between:
        - **True Positive Rate (Sensitivity)**: Proportion of actual positives correctly identified
        - **False Positive Rate (1-Specificity)**: Proportion of actual negatives incorrectly classified
        
        A model with perfect classification would have:
        - **AUC = 1.0**: The curve would reach the top-left corner
        - **Random guessing = 0.5**: The diagonal line
        """)
    
    # Feature importance
    st.subheader("Feature Importance")
    
    feature_importance = get_feature_importance(selected_model, model, X)
    
    if feature_importance is not None:
        # Create a DataFrame for the feature importance
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        # Create bar chart
        fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title=f'Feature Importance - {selected_model}',
            color='Importance',
            # color_continuous_scale='viridis'
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"Feature importance visualization not available for {selected_model}")
    
    # Try to calculate SHAP values (with error handling)
    try:
        # Take a small sample for SHAP analysis (for performance reasons)
        X_sample = X_test[:100]
        
        st.subheader("SHAP Analysis")
        st.markdown("""
        SHAP (SHapley Additive exPlanations) values help explain how each feature contributes to predictions.
        """)
        
        with st.spinner("Calculating SHAP values (this may take a moment)..."):
            shap_values, explainer = calculate_shap_values(model, X_sample, selected_model)
        
        if shap_values is not None:
            # Create a DataFrame for visualization
            feature_names_list = list(feature_names)
            shap_df = pd.DataFrame()
            
            # Collect SHAP values for each feature
            for i, feature in enumerate(feature_names_list):
                if i < shap_values.shape[1]:  # Ensure we don't go out of bounds
                    shap_df[feature] = shap_values[:, i]
            
            # Calculate mean absolute SHAP values for each feature
            mean_shap = shap_df.abs().mean().sort_values(ascending=False)
            
            # Create bar chart for mean SHAP values
            fig = px.bar(
                x=mean_shap.values[:10],  # Top 10 features
                y=mean_shap.index[:10],
                orientation='h',
                title="Mean |SHAP| Value (Feature Impact)",
                labels={'x': 'mean |SHAP|', 'y': 'Feature'},
                color=mean_shap.values[:10],
                # color_continuous_scale='viridis'
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Create summary plot alternative using Plotly
            st.markdown("### SHAP Summary Plot")
            st.markdown("This plot shows how each feature affects the prediction.")
            
            # Get top features by mean SHAP
            top_features = mean_shap.index[:5].tolist()  # Top 5
            
            # Create a long-form DataFrame for the summary plot
            summary_data = []
            
            for feature in top_features:
                for i in range(len(X_sample)):
                    summary_data.append({
                        'Feature': feature,
                        'SHAP Value': shap_df[feature][i],
                        'Feature Value': X_sample[:, feature_names_list.index(feature)][i]
                    })
            
            summary_df = pd.DataFrame(summary_data)
            
            # Create the summary plot
            fig = px.strip(
                summary_df,
                x='SHAP Value',
                y='Feature',
                color='Feature Value',
                # color_continuous_scale='RdBu_r',
                stripmode='overlay',
                title="SHAP Summary Plot"
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # SHAP explanation
            st.markdown("""
            **How to interpret SHAP values:**
            - **Positive SHAP values** (right side) increase the likelihood of the condition
            - **Negative SHAP values** (left side) decrease the likelihood
            - **Color** represents the feature value (red = high, blue = low)
            
            For example, high glucose levels (red dots) typically have positive SHAP values, indicating they increase the predicted risk of diabetes.
            """)
    
    except Exception as e:
        st.warning(f"Could not calculate SHAP values: {e}")

def show_about_ml_healthcare():
    st.header("Machine Learning in Healthcare")
    
    col1, col2 = st.columns([3, 2])
    

    with col1:
        st.markdown("""
        ### Transforming Medical Diagnostics

        Machine learning is revolutionizing healthcare by enhancing diagnostic accuracy, improving treatment plans, and increasing operational efficiency. These technologies are becoming essential tools for healthcare providers seeking to deliver personalized and precise care.

        #### Key Applications in Healthcare

        1. **Disease Diagnosis and Prediction**
           - Early detection of diseases like cancer, diabetes, and heart disease
           - Identification of high-risk patients for preventive interventions
           - Analysis of medical images (X-rays, MRIs, CT scans) to detect abnormalities

        2. **Treatment Optimization**
           - Personalized treatment recommendations based on patient characteristics
           - Medication effectiveness prediction
           - Adverse event and readmission risk assessment

        3. **Healthcare Operations**
           - Resource allocation and scheduling optimization
           - Fraud detection in healthcare claims
           - Patient flow management and hospital capacity planning

        #### Benefits of ML in Healthcare

        - **Improved Accuracy**: ML models can analyze vast amounts of data to identify patterns invisible to human observers
        - **Early Detection**: Algorithms can detect subtle indicators of disease before symptoms are evident
        - **Reduced Costs**: More efficient diagnostics and targeted treatments reduce healthcare expenses
        - **Personalized Care**: Tailored approaches based on individual patient characteristics
        - **Accessible Expertise**: ML can extend specialized medical knowledge to underserved areas
        """)
    
    with col2:
        st.image("https://img.freepik.com/free-vector/medical-technology-science-background-vector-blue-with-digital-healthcare_53876-117739.jpg", caption="AI in Healthcare", use_container_width=True)
        
        st.markdown("""
        ### ML Technologies in Medical Use Today
        
        - **IBM Watson for Oncology**: Treatment recommendation
        - **Google DeepMind**: Medical imaging analysis
        - **PathAI**: Pathology diagnosis assistance
        - **Aidoc**: Radiology diagnostic support
        - **Arterys**: Cardiac analysis software
        """)
    
    st.subheader("Types of Machine Learning in Healthcare")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### Supervised Learning
        
        **Applications:**
        - Disease classification
        - Mortality risk prediction
        - Treatment outcome prediction
        
        **Examples:**
        - Logistic regression for disease risk
        - Random forests for readmission prediction
        - Neural networks for image analysis
        """)
    
    with col2:
        st.markdown("""
        #### Unsupervised Learning
        
        **Applications:**
        - Patient segmentation
        - Anomaly detection in medical data
        - Disease subtype discovery
        
        **Examples:**
        - Clustering for patient stratification
        - Dimensionality reduction for genomic data
        - Association rules for comorbidity patterns
        """)
    
    with col3:
        st.markdown("""
        #### Reinforcement Learning
        
        **Applications:**
        - Treatment optimization
        - Adaptive clinical trials
        - Personalized dosing regimens
        
        **Examples:**
        - Dynamic treatment regimes
        - Automated insulin delivery systems
        - Robotic surgery assistance
        """)
    
    st.subheader("Ethical Considerations and Challenges")
    
    st.markdown("""
    While ML offers tremendous potential, its implementation in healthcare faces important challenges:
    
    - **Data Privacy and Security**: Patient data requires strict protection under regulations like HIPAA
    - **Algorithm Bias**: ML models can perpetuate or amplify existing healthcare disparities
    - **Transparency and Explainability**: Healthcare professionals need to understand AI recommendations
    - **Regulatory Approval**: Medical ML applications require rigorous validation and approval
    - **Integration with Workflow**: Technology must enhance, not burden, clinical workflow
    - **Human Oversight**: ML should augment, not replace, healthcare professional judgment
    """)
    
    st.subheader("Future Directions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Emerging Trends
        
        - **Federated Learning**: Training models across institutions without sharing raw data
        - **Multimodal Models**: Integrating diverse data types (images, text, genomics)
        - **Continuous Learning Systems**: Models that adapt to new medical evidence
        - **Edge Computing**: Bringing ML capabilities to medical devices
        - **Explainable AI**: More transparent algorithms for clinical decision support
        """)
    
    with col2:
        st.markdown("""
        #### On the Horizon
        
        - **Digital Twin Technology**: Patient-specific models for treatment simulation
        - **Ambient Clinical Intelligence**: AI assistants during patient-doctor interactions
        - **Real-time Monitoring and Intervention**: ML-powered wearables and IoT devices
        - **Precision Prevention**: Personalized health maintenance programs
        - **Cross-modal Transfer Learning**: Applying learning from one medical domain to another
        """)
    
    st.info("""
    **Note:** Despite these advancements, the healthcare industry emphasizes that machine learning tools are designed to support, 
    not replace, the clinical judgment of healthcare professionals. The human element of care remains essential.
    """)

if __name__ == "__main__":
    main()
